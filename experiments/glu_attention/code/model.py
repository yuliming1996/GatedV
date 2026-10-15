"""GatedV and GLU value-gating controls in the same causal Transformer.

GLU uses independent content/gate projections, content * SiLU(gate), before
aggregation. Only its value width varies; Q/K, RoPE, FFN, normalization and
training numerics are retained from the portable GatedV implementation.
"""
from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("baseline", "v_silu", "v_glu_full", "v_glu_reduced")
GLU_ARMS = ("v_glu_full", "v_glu_reduced")


@dataclass
class Config:
    vocab: int = 8192
    width: int = 128
    layers: int = 6
    heads: int = 4
    hidden: int = 384
    context: int = 256
    arm: str = "baseline"
    qk_norm: bool = False
    projection_hidden: int = 0
    glu_width_policy: str = "exact"

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm: {self.arm}")
        for name in ("vocab", "width", "layers", "heads", "hidden", "context"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.projection_hidden, bool)
                or not isinstance(self.projection_hidden, int)
                or self.projection_hidden < 0):
            raise ValueError("projection_hidden must be a nonnegative integer")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("head dimension must be an even integer")
        if self.projection_hidden != 0:
            raise ValueError("GLU comparison does not use projection_hidden")
        if self.glu_width_policy not in ("exact", "paper_floor", "nearest"):
            raise ValueError("glu_width_policy must be exact, paper_floor, or nearest")
        if self.arm == "v_glu_reduced":
            if self.glu_width_policy == "exact" and (2 * self.width) % (3 * self.heads):
                raise ValueError("Exact GLU parameter matching requires 2*width divisible by 3*heads; choose an explicit rounding policy otherwise")
            if self.value_width <= 0:
                raise ValueError("Reduced Value head dimension must be positive")

    @property
    def value_width(self):
        if self.arm != "v_glu_reduced":
            return self.width
        per_head, remainder = divmod(2 * self.width, 3 * self.heads)
        if self.glu_width_policy == "nearest" and 2 * remainder >= 3 * self.heads:
            per_head += 1
        return self.heads * per_head

    @property
    def resolved_projection_hidden(self):
        return self.projection_hidden or 2 * self.width


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


def rotate(x, cos, sin):
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.cfg, self.index = cfg, index
        # Preserve the legacy inert buffer so baseline/GatedV state remains
        # directly testable against the retained model. No calibration is used.
        self.register_buffer("v_calibration", torch.tensor([1., 0., 1., 0.]))
        d, m = cfg.width, cfg.value_width
        self.qkv = nn.Linear(d, 2 * d + 2 * m if cfg.arm in GLU_ARMS else 3 * d, bias=False)
        self.out = nn.Linear(m, d, bias=False)
        if cfg.qk_norm:
            self.qnorm = RMSNorm(d // cfg.heads)
            self.knorm = RMSNorm(d // cfg.heads)

    def project(self, x, interventions=()):
        if interventions:
            raise ValueError("GLU comparison has no inference-time interventions")
        if self.cfg.arm in GLU_ARMS:
            d, m = self.cfg.width, self.cfg.value_width
            q, k, content, gate = self.qkv(x).split((d, d, m, m), dim=-1)
            return q, k, content * F.silu(gate)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return q, k, F.silu(v) if self.cfg.arm == "v_silu" else v

    def forward(self, x, cos, sin, interventions=()):
        b, t, d = x.shape
        h, dh = self.cfg.heads, d // self.cfg.heads
        q, k, v = self.project(x, interventions)
        q, k = (z.reshape(b, t, h, dh).transpose(1, 2) for z in (q, k))
        v = v.reshape(b, t, h, self.cfg.value_width // h).transpose(1, 2)
        if self.cfg.qk_norm:
            q, k = self.qnorm(q), self.knorm(k)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, self.cfg.value_width))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.up = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.width), RMSNorm(cfg.width)
        self.attn = Attention(cfg, index)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin, interventions):
        x = x + self.attn(self.norm1(x), cos, sin, interventions)
        return x + self.ffn(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.layers))
        self.norm = RMSNorm(cfg.width)
        dh = cfg.width // cfg.heads
        angle = torch.outer(torch.arange(max(4096, cfg.context)),
                            10000. ** (-torch.arange(0, dh, 2).float() / dh))
        self.register_buffer("rope_cos", angle.cos(), persistent=False)
        self.register_buffer("rope_sin", angle.sin(), persistent=False)

    def forward(self, tokens, interventions=()):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, nonempty sequence]")
        if interventions:
            raise ValueError(f"unsupported interventions: {interventions}")
        t = tokens.shape[1]
        if t > len(self.rope_cos):
            raise ValueError("context exceeds RoPE table")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, self.rope_cos[:t], self.rope_sin[:t], interventions)
        return F.linear(self.norm(x), self.embedding.weight)

    def initialize(self, seed):
        def fill_matrix(name, p, multiplier=1.):
            local_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % (2**63 - 1)
            generator = torch.Generator(device="cpu").manual_seed(local_seed)
            scale = multiplier / math.sqrt(p.shape[1])
            p.copy_(torch.randn(p.shape, dtype=p.dtype, generator=generator) * scale)

        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.ndim == 1:
                    p.fill_(1.)
                elif name.endswith("attn.qkv.weight") and self.cfg.arm in GLU_ARMS:
                    prefix = name.rsplit(".", 2)[0]
                    d, m = self.cfg.width, self.cfg.value_width
                    for label, chunk in zip(("q", "k", "v", "v_gate"), p.split((d, d, m, m), dim=0)):
                        fill_matrix(f"{prefix}.{label}.weight", chunk)
                elif name.endswith(("attn.qkv.weight", "attn.kv.weight")):
                    # Virtual names preserve identical baseline K/V when only Q
                    # changes, without allocating unused linear-query rows.
                    labels = ("q", "k", "v") if name.endswith("qkv.weight") else ("k", "v")
                    prefix = name.rsplit(".", 2)[0]
                    for label, chunk in zip(labels, p.chunk(len(labels), dim=0)):
                        fill_matrix(f"{prefix}.{label}.weight", chunk)
                else:
                    multiplier = 1.
                    if name.endswith(("attn.out.weight", "ffn.down.weight")):
                        multiplier = 1 / math.sqrt(2 * self.cfg.layers)
                    fill_matrix(name, p, multiplier)
        return self

    def optimizer(self, lr, weight_decay=.1):
        decay, no_decay = [], []
        for p in self.parameters():
            (no_decay if p.ndim < 2 else decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.}],
            lr=lr, betas=(.9, .95), eps=1e-8, foreach=False,
        )

    def metadata(self):
        parameters = sum(p.numel() for p in self.parameters())
        embedding_parameters = self.embedding.weight.numel()
        matrix_macs = sum(p.numel() for n, p in self.named_parameters()
                          if p.ndim == 2 and n != "embedding.weight")
        return {
            "config": asdict(self.cfg),
            "parameters": parameters,
            "embedding_parameters": embedding_parameters,
            "nonembedding_parameters": parameters - embedding_parameters,
            "resolved_projection_hidden": self.cfg.resolved_projection_hidden,
            "tied_embedding": True,
            "value_projection": {
                "operator": "content * SiLU(gate)" if self.cfg.arm in GLU_ARMS else ("SiLU(V)" if self.cfg.arm == "v_silu" else "V"),
                "qk_head_width": self.cfg.width // self.cfg.heads,
                "value_head_width": self.cfg.value_width // self.cfg.heads,
                "value_width": self.cfg.value_width,
                "width_policy": self.cfg.glu_width_policy if self.cfg.arm == "v_glu_reduced" else "full",
                "bias": False,
                "parameter_delta_vs_baseline": self.cfg.layers * (
                    3 * self.cfg.width * self.cfg.value_width - 2 * self.cfg.width ** 2
                ) if self.cfg.arm in GLU_ARMS else 0,
                "exact_parameter_match": self.cfg.arm not in GLU_ARMS or 3 * self.cfg.value_width == 2 * self.cfg.width,
                "pairing_scope": "All identical-shape shared matrices use canonical name seeds. Full GLU content shares V initialization. Reduced V/output shapes differ; full-model initialization is not identical across GLU and GatedV.",
            },
            "backbone_matrix_macs_per_token": matrix_macs,
            "mac_count_scope": "Linear matrices only; excludes tied vocabulary readout, attention products, activations, and norms.",
            "initialization": {
                "version": "name_seeded_fan_in_v1",
                "matrix_distribution": "Gaussian std=1/sqrt(fan_in), including embedding with fan_in=width",
                "residual_output_multiplier": 1 / math.sqrt(2 * self.cfg.layers),
                "residual_output_matrices": ["attn.out", "ffn.down"],
                "fused_qkv_seeds": "Canonical virtual attn.q/k/v.weight names plus independent attn.v_gate.weight for GLU",
                "projection_scale_caveat": "Fan-in Gaussian initialization, output scaling based on actual value width. No extra normalization or calibration of GLU.",
                "historical_compatibility": "Baseline and GatedV retain original equations and parameter initialization; new runtime and checkpoint identities are separate.",
            },
        }

