"""Portable I/O, deterministic numerical policy, and run identity helpers."""
import hashlib
import json
import os
from pathlib import Path
import platform

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

MODEL_KEYS = ('arm', 'vocab', 'width', 'layers', 'heads', 'hidden', 'context', 'qk_norm', 'projection_hidden')
MAIN_ARMS = ('baseline', 'q_silu', 'v_silu', 'qkv_silu')
ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_POLICY = {'dtype': 'float32', 'autocast': False, 'tf32_matmul': False,
                    'tf32_cudnn': False, 'matmul_precision': 'highest',
                    'cudnn_benchmark': False, 'deterministic_algorithms': True,
                    'sdpa': 'math', 'compile': False, 'adamw_foreach': False,
                    'cublas_workspace_config': ':4096:8'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def configure(device):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('This reference supports CPU or CUDA; MPS is not the study numerical policy')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU is suitable only for bounded verification')
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device


def environment(device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'machine': platform.machine(), 'system': platform.system(),
            'device': str(device), 'cuda_runtime': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'numerical_policy': NUMERICAL_POLICY}


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode() + b'\0')
        h.update(json.dumps(list(x.shape), separators=(',', ':')).encode() + b'\0')
        h.update(x.numpy().astype('<f4', copy=False).tobytes())
    return h.hexdigest()


def load_recipe(path):
    cfg = read_json(path)
    cfg.pop('note', None)
    for key in ('steps', 'batch', 'eval_tokens', 'save_every', 'log_every', 'threads'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('warmup', 'eval_every'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if not np.isfinite(cfg['lr']) or cfg['lr'] <= 0 or cfg['warmup'] >= cfg['steps']:
        raise ValueError('Invalid learning rate or warmup budget')
    return cfg
