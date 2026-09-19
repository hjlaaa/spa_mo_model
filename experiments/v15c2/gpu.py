"""GPU readiness checks in fresh processes, without models or training."""
from __future__ import annotations

import json
import subprocess
import sys
import time

from .plan import ROOT

PROBE_CODE = """
import json
import numpy as np
import torch
import faiss
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA; check driver/device visibility and execution sandbox'
assert hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0, 'FAISS GPU unavailable'
x = torch.tensor([1., 2., 3.], device='cuda')
assert (x*x).sum().item() == 14.0
torch.cuda.synchronize()
resources = faiss.StandardGpuResources()
resources.setTempMemory(16 * 1024 * 1024)
index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatIP(2))
data = np.eye(2, dtype=np.float32)
index.add(data)
scores, indices = index.search(data, 1)
assert indices[:, 0].tolist() == [0, 1]
print(json.dumps({'status': 'ready', 'gpu': torch.cuda.get_device_name(0),
    'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'torch_cuda_compute': 'passed', 'faiss_gpu_search': 'passed'}))
"""


def probe_gpu(python, *, attempts=3, delay_seconds=5):
    """Retry visibility/compute checks; never fall back to CPU training."""
    if attempts < 1 or delay_seconds < 0:
        raise ValueError("Invalid GPU probe retry settings.")
    failures = []
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run([python, "-c", PROBE_CODE], cwd=ROOT,
                                    capture_output=True, text=True, timeout=45)
            if result.returncode == 0:
                report = json.loads(result.stdout)
                if report.get("status") != "ready":
                    raise ValueError("GPU probe did not report readiness.")
                return {**report, "attempt": attempt, "previous_failures": failures}
            error = result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}"
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        failures.append({"attempt": attempt, "error": error})
        print(f"GPU check {attempt}/{attempts} failed: {error}", file=sys.stderr, flush=True)
        if attempt < attempts:
            time.sleep(delay_seconds)
    raise RuntimeError(f"GPU unavailable after {attempts} checks; no CPU fallback, no training started. {failures}")
