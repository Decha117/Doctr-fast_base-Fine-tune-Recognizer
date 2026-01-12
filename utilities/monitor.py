from __future__ import annotations

from typing import Dict

import psutil
import pynvml


_DEF_GPU_NAME = "Unavailable"


def _get_gpu_info() -> Dict:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "name": name,
            "memory_used_mb": round(memory.used / (1024 * 1024), 2),
            "memory_total_mb": round(memory.total / (1024 * 1024), 2),
        }
    except Exception:
        return {"name": _DEF_GPU_NAME, "memory_used_mb": 0, "memory_total_mb": 0}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            return {"name": _DEF_GPU_NAME, "memory_used_mb": 0, "memory_total_mb": 0}


def get_system_snapshot() -> Dict:
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu": _get_gpu_info(),
    }
