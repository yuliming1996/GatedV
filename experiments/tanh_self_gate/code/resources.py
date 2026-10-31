"""Per-process allocator ceiling and sampled training resource evidence."""
import os
import time

import psutil
import torch

from common import write_json

CUDA_LIMIT = 12 * 1024 ** 3


def limit_allocator(device):
    if device.type == 'cuda':
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., CUDA_LIMIT / total), device)


class ResourceMonitor:
    def __init__(self, output, device):
        self.output, self.device = output, device
        self.process = psutil.Process(os.getpid())
        self.created = self.process.create_time()
        self.last = 0.
        self.stop_reason = None

    def sample(self, step, force=False):
        now = time.monotonic()
        if not force and now - self.last < 15:
            return self.stop_reason
        self.last = now
        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
        row = {'pid': self.process.pid, 'create_time': self.created,
               'updated_unix': time.time(), 'step': step,
               'rss': self.process.memory_info().rss,
               'available_memory': memory.available, 'swap_used': swap.used,
               'cuda_allocator_limit': CUDA_LIMIT}
        if self.device.type == 'cuda':
            row.update(cuda_allocated=torch.cuda.memory_allocated(self.device),
                       cuda_reserved=torch.cuda.memory_reserved(self.device),
                       cuda_peak_allocated=torch.cuda.max_memory_allocated(self.device),
                       cuda_peak_reserved=torch.cuda.max_memory_reserved(self.device))
            if max(row[k] for k in ('cuda_allocated', 'cuda_reserved',
                                   'cuda_peak_allocated', 'cuda_peak_reserved')) > CUDA_LIMIT:
                self.stop_reason = 'cuda_allocator_limit'
        row['stop_reason'] = self.stop_reason
        write_json(self.output / 'resource.json', row)
        return self.stop_reason
