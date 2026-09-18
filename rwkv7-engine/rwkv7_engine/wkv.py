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

__all__ = ["wkv7", "gemv16", "wkv7_post", "lora_gate", "lora_gates4",
           "lora_h1", "lora_gates", "build_wkv_lib", "WkvState"]

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
        self.lib.wkv7_split_launch.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.lib.wkv7_split_launch.restype = None
        self.lib.gemv16_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.gemv16_launch.restype = None
        self.lib.wkv7_post_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_void_p,
        ]
        self.lib.wkv7_post_launch.restype = None
        self.lib.lora_gate_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        self.lib.lora_gate_launch.restype = None
        self.lib.lora_gates4_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.lib.lora_gates4_launch.restype = None
        self.lib.lora_h1_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.lora_h1_launch.restype = None
        self.lib.lora_gates_launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.lib.lora_gates_launch.restype = None
        self._ws = None          # (device_index, tensor)
        self._gws = None         # (device_index, tensor) LoRA gate workspace

    def workspace(self, device) -> torch.Tensor:
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        if self._ws is None or self._ws[0] != idx:
            self._ws = (idx, torch.zeros(48 * 1024 * 1024 // 4,
                                         dtype=torch.float32, device=dev))
        return self._ws[1]

    def gate_ws(self, device, C: int) -> torch.Tensor:
        """Persistent fp32 workspace for LoRA split-R partials (graph safe)."""
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        need = 8 * 64 * C + 4 * 32 * 512  # gates [4][64][C] + h1 [4][32][512]
        if self._gws is None or self._gws[0] != idx or self._gws[1].numel() < need:
            self._gws = (idx, torch.zeros(need, dtype=torch.float32, device=dev))
        return self._gws[1]

    @classmethod
    def get(cls) -> "_WkvLib":
        if cls._inst is None:
            cls._inst = cls(build_wkv_lib())
        return cls._inst


def gemv16(
    W: torch.Tensor, x: torch.Tensor, out_f32: bool = False,
) -> torch.Tensor:
    """Custom split-K GEMV for decode:  y[n] = sum_k x[k]*W[k,n].

    W: [K, N] fp16 row-major (contiguous).  x: [K] fp16 (contiguous).
    Returns [N] fp16 (out_f32=False) or fp32 (out_f32=True).
    Falls back to torch.matmul when shapes are unsuitable.
    """
    K, N = W.shape
    if K % 8 or N % 8 or W.dtype != torch.float16 or x.dtype != torch.float16:
        y = torch.matmul(x.unsqueeze(0), W)
        return y[0].float() if out_f32 else y[0]
    lib = _WkvLib.get()
    ws = lib.workspace(W.device)
    y = torch.empty((N,), dtype=torch.float32 if out_f32 else torch.float16,
                    device=W.device)
    stream = torch.cuda.current_stream(W.device).cuda_stream
    lib.lib.gemv16_launch(
        W.data_ptr(), x.data_ptr(), y.data_ptr(),
        K, N, 1 if out_f32 else 0,
        ws.data_ptr(), ws.numel(),
        ctypes.c_void_p(stream),
    )
    return y


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
    launcher = lib.wkv7_split_launch if T == 1 else lib.wkv7_serial_launch
    launcher(
        ctypes.c_int(T), ctypes.c_int(H),
        r.data_ptr(), w.data_ptr(), k.data_ptr(), v.data_ptr(),
        a.data_ptr(), b.data_ptr(),
        s_in.data_ptr() if s_in is not None else 0,
        s_out.data_ptr() if s_out is not None else 0,
        out.data_ptr(),
        ctypes.c_void_p(stream),
    )
    return out


def wkv7_post(
    o_wkv: torch.Tensor, k: torch.Tensor, r: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, gn_w: torch.Tensor, gn_b: torch.Tensor, rk_w: torch.Tensor,
    H: int, D: int, eps: float,
) -> torch.Tensor:
    """Fused group_norm + rk + o*v + o*g epilogue (decode, T==1).

    o_wkv [C] fp32 (Wkv output); k/r/v/g [C] fp16; gn_w/gn_b/rk_w [C] fp32.
    Returns og [C] fp16 = (group_norm(o_wkv) + rk*v) * g.
    """
    C = o_wkv.numel()
    og = torch.empty((C,), dtype=torch.float16, device=o_wkv.device)
    stream = torch.cuda.current_stream(o_wkv.device).cuda_stream
    lib = _WkvLib.get().lib
    lib.wkv7_post_launch(
        o_wkv.data_ptr(), k.data_ptr(), r.data_ptr(), v.data_ptr(),
        g.data_ptr(), gn_w.data_ptr(), gn_b.data_ptr(), rk_w.data_ptr(),
        og.data_ptr(), H, D, float(eps), ctypes.c_void_p(stream),
    )
    return og


def _gate_workspace(device, C: int) -> torch.Tensor:
    return _WkvLib.get().gate_ws(device, C)


def lora_h1(
    xs: "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]",
    W1s: "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]",
    acts: "tuple[int, int, int, int]" = (1, 0, 2, 0),
    phases: int = 32,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """The four LoRA down-projections h1 = act1(x @ W1) in ONE launch (decode).

    xs:  (xw, xa, xg, xv) each [C] fp16 (xv may be None -> zero-size output).
    W1s: (W1w, W1a, W1g, W1v) each [C, R] fp16 (lora.0 transposed).
    acts: act1 per gate (1=tanh, 0=identity, 2=sigmoid).
    Returns (h1w, h1a, h1g, h1v) fp16, each [R].
    """
    dev = W1s[0].device
    C = xs[0].numel()
    R = [0 if w is None else w.shape[1] for w in W1s]
    outs = [torch.empty((r,), dtype=torch.float16, device=dev) for r in R]
    zero = 0
    ptrs_x = [zero if t is None else t.data_ptr() for t in xs]
    ptrs_w = [zero if t is None else t.data_ptr() for t in W1s]
    ws = _gate_workspace(dev, C)
    stream = torch.cuda.current_stream(dev).cuda_stream
    lib = _WkvLib.get().lib
    acts_c = (ctypes.c_int * 4)(*acts)
    lib.lora_h1_launch(
        *ptrs_x, *ptrs_w, R[0], R[1], R[2], R[3], C, phases,
        ctypes.cast(acts_c, ctypes.c_void_p), ws.data_ptr(),
        *(outs[i].data_ptr() for i in range(4)),
        ctypes.c_void_p(stream),
    )
    return outs[0], outs[1], outs[2], outs[3]


def lora_gates(
    h1s: "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]",
    W2s: "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]",
    b2s: "tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]",
    w_scale: float = 1.0, splitR: int = 16,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """The four LoRA up-projections + activations in ONE launch (decode).

    Returns (w, a, g, t): w = w_scale*sigmoid(b2w + W2w @ h1w), etc.
    """
    dev = W2s[0].device
    C = W2s[0].shape[1]
    R = [0 if w is None else w.shape[0] for w in W2s]
    outs = [torch.empty((C,), dtype=torch.float16, device=dev) for _ in range(4)]
    zero = 0
    ptrs_w = [zero if t is None else t.data_ptr() for t in W2s]
    ptrs_h = [zero if t is None else t.data_ptr() for t in h1s]
    ptrs_b = [zero if t is None else t.data_ptr() for t in b2s]
    ws = _gate_workspace(dev, C)
    stream = torch.cuda.current_stream(dev).cuda_stream
    lib = _WkvLib.get().lib
    lib.lora_gates_launch(
        *ptrs_w, *ptrs_h, *ptrs_b, float(w_scale),
        R[0], R[1], R[2], R[3], C, int(splitR),
        *(outs[i].data_ptr() for i in range(4)),
        ws.data_ptr(), ctypes.c_void_p(stream),
    )
    return outs[0], outs[1], outs[2], outs[3]


def lora_gate(
    h1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor | None,
    scale: float = 1.0, act1: int = 0, act2: int = 1,
) -> torch.Tensor:
    """LoRA gate GEMV (decode): out[n] = act2(b2[n] + sum_r h1[r]*W2[r,n]).

    h1 [R] fp16, W2 [R,C] fp16 (native lora.2 layout), b2 [C] fp16 or None.
    act2: 0 = identity, 1 = sigmoid.
    """
    C = W2.shape[1]
    R = W2.shape[0]
    out = torch.empty((C,), dtype=torch.float16, device=W2.device)
    ws = _gate_workspace(W2.device, C)
    stream = torch.cuda.current_stream(W2.device).cuda_stream
    lib = _WkvLib.get().lib
    lib.lora_gate_launch(
        h1.data_ptr(), W2.data_ptr(),
        b2.data_ptr() if b2 is not None else 0,
        out.data_ptr(), ws.data_ptr(), C, R, float(scale), act1, act2,
        ctypes.c_void_p(stream),
    )
    return out


def lora_gates4(
    h1w: torch.Tensor, h1a: torch.Tensor, h1g: torch.Tensor,
    h1v: torch.Tensor | None,
    W2w: torch.Tensor, W2a: torch.Tensor, W2g: torch.Tensor,
    W2v: torch.Tensor | None,
    b2w: torch.Tensor, b2a: torch.Tensor, b2v: torch.Tensor | None,
    w_scale: float = 1.0,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """The four RWKV7 attention LoRA gates in one call (decode).

    w = w_scale*sigmoid(b2w + W2w @ h1w)   (h1w = tanh(x @ w_l0))
    a =            sigmoid(b2a + W2a @ h1a)
    g =                      W2g @ h1g     (h1g = sigmoid(x @ g_l0))
    t =            sigmoid(b2v + W2v @ h1v)   (identity when h1v is None)
    """
    C = W2w.shape[1]
    dev = W2w.device
    out_w = torch.empty((C,), dtype=torch.float16, device=dev)
    out_a = torch.empty((C,), dtype=torch.float16, device=dev)
    out_g = torch.empty((C,), dtype=torch.float16, device=dev)
    out_v = torch.empty((C,), dtype=torch.float16, device=dev)
    ws = _gate_workspace(dev, C)
    stream = torch.cuda.current_stream(dev).cuda_stream
    lib = _WkvLib.get().lib
    lib.lora_gates4_launch(
        h1w.data_ptr(), h1a.data_ptr(), h1g.data_ptr(),
        h1v.data_ptr() if h1v is not None else 0,
        W2w.data_ptr(), W2a.data_ptr(), W2g.data_ptr(),
        W2v.data_ptr() if W2v is not None else 0,
        b2w.data_ptr(), b2a.data_ptr(),
        b2v.data_ptr() if b2v is not None else 0,
        float(w_scale), C, h1w.numel(), h1a.numel(), h1g.numel(),
        h1v.numel() if h1v is not None else 0,
        out_w.data_ptr(), out_a.data_ptr(), out_g.data_ptr(), out_v.data_ptr(),
        ws.data_ptr(), ctypes.c_void_p(stream),
    )
    return out_w, out_a, out_g, out_v
