"""Wkv kernel bindings: standalone .so (nvcc) + ctypes + torch wrappers.

The kernel is built once into ``rwkv7_engine/_wkv7.so`` (or the directory named
by ``SIPHON_RWKV7_SO``) with a plain nvcc invocation targeting the local
architectures, so it does not depend on the torch CUDA build.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sysconfig
from pathlib import Path

import torch

__all__ = ["wkv7", "build_wkv_lib", "WkvState"]

_KERNEL_SRC = Path(__file__).with_name("wkv_kernel.cu")


def _so_path() -> Path:
    env = os.environ.get("SIPHON_RWKV7_SO")
    if env:
        return Path(env)
    return Path(__file__).with_name("_wkv7.so")


def _detect_arch_flags() -> list[str]:
    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        archs = [a.replace("compute_", "").replace("sm_", "").strip()
                 for a in env.split(";") if a.strip()]
    else:
        major, minor = torch.cuda.get_device_capability()
        archs = [f"{major}{minor}"]
    # nvcc >= 12.8 wants the underscore form (compute_70, code=sm_70)
    return [f"-gencode=arch=compute_{a.replace('.', '')},code=sm_{a.replace('.', '')}"
            for a in archs]


def build_wkv_lib(force: bool = False) -> Path:
    """Compile wkv_kernel.cu to a shared object. Returns the .so path."""
    so = _so_path()
    if so.exists() and not force:
        return so
    nvcc = os.environ.get("NVCC")
    if not nvcc:
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
    cmd = [
        nvcc, "-O3", "-std=c++17", "--use_fast_math",
        "-Xcompiler", "-fPIC", "-shared",
        *_detect_arch_flags(),
        str(_KERNEL_SRC), "-o", str(so),
    ]
    subprocess.run(cmd, check=True)
    return so


class _WkvLib:
    _inst: "_WkvLib | None" = None

    def __init__(self, so: Path) -> None:
        self.lib = ctypes.CDLL(str(so))
        self.lib.wkv7_serial_launch.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.lib.wkv7_serial_launch.restype = None

    @classmethod
    def get(cls) -> "_WkvLib":
        if cls._inst is None:
            cls._inst = cls(build_wkv_lib())
        return cls._inst


def wkv7(
    r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor,
    s_in: torch.Tensor | None, s_out: torch.Tensor | None,
) -> torch.Tensor:
    """Run the RWKV7 Wkv recurrence.

    r, w, k, v, a, b: [T, H*D] fp16 (head-major rows), contiguous.
    s_in / s_out:     [H*D] fp32 (optional); s_in=None starts from zero state.
    Returns:          [T, H*D] fp32 with the per-head scalar replicated over D.
    """
    T, C = r.shape
    D = 64
    H = C // D
    out = torch.empty((T, C), dtype=torch.float32, device=r.device)
    stream = torch.cuda.current_stream(r.device).cuda_stream
    lib = _WkvLib.get().lib
    lib.wkv7_serial_launch(
        ctypes.c_int(T), ctypes.c_int(H),
        r.data_ptr(), w.data_ptr(), k.data_ptr(), v.data_ptr(),
        a.data_ptr(), b.data_ptr(),
        s_in.data_ptr() if s_in is not None else 0,
        s_out.data_ptr() if s_out is not None else 0,
        out.data_ptr(),
        ctypes.c_void_p(stream),
    )
    return out
