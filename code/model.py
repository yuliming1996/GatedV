"""Causal language-model controls for nonlinear Q/K/V projections.

The backbone is pre-RMSNorm, RoPE, SwiGLU, and a tied embedding/readout.
Every matrix uses name-seeded fan-in Gaussian initialization, rather than the
historical experiment's fixed 0.02 standard deviation. Two explicitly named
mechanism controls use frozen train-only initialization moment coefficients.

The q_silu/v_silu/qkv_silu arms retain the baseline fused projection and apply
only a parameter-free SiLU to selected branches before head reshaping/RoPE.
"""
from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


ARMS = (
    "baseline", "q_mlp", "q_gated", "qkv_mlp", "qkv_gated",
    "qkv_linear", "qkv_norm", "shared_mlp", "output_gate",
    "q_silu", "v_silu", "qkv_silu", "v_affine_match", "v_silu_standardized",
)
SHARED_DOWN_INITIAL_SCALE = 1e-3


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


class TokenProjection(nn.Module):
    """Independent d -> m -> d projection; all matrix operations have no bias."""

    def __init__(self, width, hidden, kind):
        super().__init__()
        if kind not in ("mlp", "gated", "linear", "norm"):
            raise ValueError(f"unknown projection kind: {kind}")
        self.kind = kind
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        if kind == "gated":
            self.gate = nn.Linear(width, hidden, bias=False)
        elif kind == "norm":
            self.norm = RMSNorm(hidden)

    def forward(self, x):
        z = self.up(x)
        if self.kind == "gated":
            z = F.silu(self.gate(x)) * z
        elif self.kind == "mlp":
            z = F.silu(z)
        elif self.kind == "norm":
            z = self.norm(z)
        return self.down(z)


class Attention(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.cfg, self.index = cfg, index
        # Frozen scalar coefficients, not optimizer parameters. All four study
        # arms share the same state schema and name-seeded parameter weights.
        self.register_buffer("v_calibration", torch.tensor([1., 0., 1., 0.]))
        self.calibration_applied = False
        d, m = cfg.width, cfg.resolved_projection_hidden
        if cfg.arm in ("q_mlp", "q_gated"):
            self.q_proj = TokenProjection(d, m, cfg.arm.removeprefix("q_"))
            self.kv = nn.Linear(d, 2 * d, bias=False)
        elif cfg.arm in ("qkv_mlp", "qkv_gated", "qkv_linear", "qkv_norm"):
            kind = cfg.arm.removeprefix("qkv_")
            self.q_proj = TokenProjection(d, m, kind)
            self.k_proj = TokenProjection(d, m, kind)
            self.v_proj = TokenProjection(d, m, kind)
        else:
            self.qkv = nn.Linear(d, 3 * d, bias=False)
        if cfg.arm == "shared_mlp":
            self.shared_up = nn.Linear(d, m, bias=False)
            self.shared_down = nn.Linear(m, d, bias=False)
        elif cfg.arm == "output_gate":
            self.output_gate = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        if cfg.qk_norm:
            self.qnorm = RMSNorm(d // cfg.heads)
            self.knorm = RMSNorm(d // cfg.heads)

    def project(self, x, interventions=()):
        if hasattr(self, "shared_up") and "no_shared" not in interventions:
            x = x + self.shared_down(F.silu(self.shared_up(x)))
        if hasattr(self, "qkv"):
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            if self.cfg.arm in ("q_silu", "qkv_silu"):
                q = F.silu(q)
            if self.cfg.arm == "qkv_silu":
                k = F.silu(k)
            if self.cfg.arm in ("v_silu", "qkv_silu"):
                v = F.silu(v)
            elif self.cfg.arm in ("v_affine_match", "v_silu_standardized"):
                if not self.calibration_applied:
                    raise RuntimeError("Mechanism control requires frozen calibration")
                a, b, c, d = self.v_calibration.unbind()
                v = a * v + b if self.cfg.arm == "v_affine_match" else c * F.silu(v) + d
            return q, k, v
        if hasattr(self, "kv"):
            k, v = self.kv(x).chunk(2, dim=-1)
            return self.q_proj(x), k, v
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def forward(self, x, cos, sin, interventions=()):
        b, t, d = x.shape
        h, dh = self.cfg.heads, d // self.cfg.heads
        q, k, v = (z.reshape(b, t, h, dh).transpose(1, 2)
                   for z in self.project(x, interventions))
        if self.cfg.qk_norm:
            q, k = self.qnorm(q), self.knorm(k)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if hasattr(self, "output_gate") and "no_gate" not in interventions:
            gate = 2 * self.output_gate(x).reshape(b, t, h, dh).transpose(1, 2).sigmoid()
            y = y * gate
        return self.out(y.transpose(1, 2).reshape(b, t, d))


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
        if set(interventions) - {"no_gate", "no_shared"}:
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
                if name.endswith("attn.output_gate.weight"):
                    p.zero_()
                elif p.ndim == 1:
                    p.fill_(1.)
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
                    elif name.endswith("attn.shared_down.weight"):
                        multiplier = SHARED_DOWN_INITIAL_SCALE
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

    def apply_calibration(self, constants):
        if len(constants) != self.cfg.layers:
            raise ValueError("Calibration layer count differs")
        for i, (block, row) in enumerate(zip(self.blocks, constants)):
            if row["layer"] != i or any(not math.isfinite(float(row[k])) for k in ("a", "b", "c", "d")):
                raise ValueError("Invalid calibration coefficients")
            with torch.no_grad():
                block.attn.v_calibration.copy_(torch.tensor([row[k] for k in ("a", "b", "c", "d")],
                                                          dtype=torch.float32, device=block.attn.v_calibration.device))
            block.attn.calibration_applied = True
        return self

    def verify_calibration(self, constants):
        if len(constants) != self.cfg.layers:
            raise ValueError("Calibration layer count differs")
        for block, row in zip(self.blocks, constants):
            expected = torch.tensor([row[k] for k in ("a", "b", "c", "d")], dtype=torch.float32)
            if not block.attn.calibration_applied or not torch.equal(block.attn.v_calibration.detach().cpu(), expected):
                raise ValueError("Checkpoint calibration buffer differs from frozen constants")

    @staticmethod
    def _is_mechanism(name):
        return ".attn." in name and any(part in name for part in (
            ".q_proj.", ".k_proj.", ".v_proj.", ".shared_up.",
            ".shared_down.", ".output_gate.",
        ))

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
            "mechanism_parameters": {n: p.numel() for n, p in self.named_parameters()
                                     if self._is_mechanism(n)},
            "backbone_matrix_macs_per_token": matrix_macs,
            "mac_count_scope": "Linear matrices only; excludes tied vocabulary readout, attention products, activations, and norms.",
            "initialization": {
                "version": "name_seeded_fan_in_v1",
                "matrix_distribution": "Gaussian std=1/sqrt(fan_in), including embedding with fan_in=width",
                "residual_output_multiplier": 1 / math.sqrt(2 * self.cfg.layers),
                "residual_output_matrices": ["attn.out", "ffn.down"],
                "shared_down_multiplier": SHARED_DOWN_INITIAL_SCALE,
                "output_gate": "Zero weights: initial 2*sigmoid(0)=1",
                "fused_qkv_seeds": "Separate virtual parameter names attn.q/k/v.weight; Q-only KV uses identical K/V seeds",
                "projection_scale_caveat": "v_affine_match and v_silu_standardized use frozen train-only baseline-initialization scalar moments; other arms are unchanged. No adaptive/output RMS calibration is applied.",
                "historical_compatibility": "New baselines are trained with this fan-in initialization; historical fixed-0.02 results are not paired controls.",
            },
        }

    def mechanism_stats(self):
        result = {}
        for name, p in self.named_parameters():
            if self._is_mechanism(name):
                z = p.detach().float().cpu()
                result[name] = {"rms": z.square().mean().sqrt().item(),
                                "mean": z.mean().item(), "min": z.min().item(),
                                "max": z.max().item()}
        return result
