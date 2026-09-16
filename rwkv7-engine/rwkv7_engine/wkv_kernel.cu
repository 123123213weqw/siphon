// Siphon RWKV7 (G1x) Wkv kernel
//
// The RWKV-7 Wkv (time mix) recurrence. Per (sequence, head) there is a state
// MATRIX S in R^{D x D} (D = head_dim) that is updated token by token.  Each
// row i of S is updated independently, sharing the per-head D-vectors w, k, b
// and the per-element scalar v_i:
//
//   sa_i  = a_t . S_{t-1, i}                  // a = -kk (l2-normalized key)
//   S_{t, i} = S_{t-1, i} * exp(w_t) + k_t * v_{t,i} + sa_i * b_t   // b = kk*a
//   o_{t,i}  = S_{t, i} . r_t                 // D distinct values per head
//
// Note: the state is a full D x D matrix per head (NOT a per-element scalar);
// the output o has D distinct values per head (row_i . r), matching
// llama.cpp's rwkv_wkv7_f32 and the FLA fused_mul_recurrent_rwkv7 kernel.
//
// fp16 inputs, fp32 state (the V100-friendly "fp32io16" mode). Serial
// recurrence over the sequence: one CUDA block per head, one thread per state
// ROW (D threads/block). D = 64 -> 64 threads (2 warps) per block.
//
// Memory layout (row-major):
//   r, w, k, v, a, b : [T, H*D] fp16, head-major (head h occupies [h*D,(h+1)*D))
//   s_in, s_out      : [H*D*D] fp32, s[h*D*D + i*D + j]
//   out              : [T, H*D] fp32

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

#define WKV_D 64

extern "C" __global__ void wkv7_serial_kernel(
    const int T, const int H,
    const __half* __restrict__ r, const __half* __restrict__ w,
    const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ a, const __half* __restrict__ b,
    const float* __restrict__ s_in,
    float* __restrict__ s_out,
    float* __restrict__ out)
{
    const int tid = threadIdx.x;           // 0 .. WKV_D-1  (state row index)
    const int head_i = blockIdx.x;         // 0 .. H-1
    const int C = H * WKV_D;
    const long row_base = (long)head_i * WKV_D * WKV_D + (long)tid * WKV_D;

    // This thread owns row `tid` of the D x D state matrix for head `head_i`.
    float state[WKV_D];
#pragma unroll
    for (int j = 0; j < WKV_D; ++j)
        state[j] = s_in ? s_in[row_base + j] : 0.f;

    __shared__ float sr[WKV_D], sw[WKV_D], sk[WKV_D], sa_[WKV_D], sb[WKV_D];

    for (int t = 0; t < T; ++t) {
        const long base = (long)t * C + (long)head_i * WKV_D;
        sr[tid]  = __half2float(r[base + tid]);
        sw[tid]  = __half2float(w[base + tid]);
        sk[tid]  = __half2float(k[base + tid]);
        sa_[tid] = __half2float(a[base + tid]);
        sb[tid]  = __half2float(b[base + tid]);
        __syncthreads();
        const float v_i = __half2float(v[base + tid]);

        // sa = a . state_row  (dot over the D-vector)
        float sdot = 0.f;
#pragma unroll
        for (int j = 0; j < WKV_D; ++j) sdot += sa_[j] * state[j];

        // state update + output o = state_row . r  (dot over the D-vector)
        float o = 0.f;
#pragma unroll
        for (int j = 0; j < WKV_D; ++j) {
            state[j] = state[j] * __expf(sw[j]) + sk[j] * v_i + sdot * sb[j];
            o += state[j] * sr[j];
        }
        out[base + tid] = o;
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < WKV_D; ++j)
        if (s_out) s_out[row_base + j] = state[j];
}

// Host-side launcher (ctypes calls this; a __global__ fn cannot be launched
// directly from a foreign ABI).
extern "C" void wkv7_serial_launch(
    const int T, const int H,
    const __half* r, const __half* w,
    const __half* k, const __half* v,
    const __half* a, const __half* b,
    const float* s_in,
    float* s_out,
    float* out,
    cudaStream_t stream)
{
    dim3 grid(H);
    dim3 block(WKV_D);
    wkv7_serial_kernel<<<grid, block, 0, stream>>>(
        T, H, r, w, k, v, a, b, s_in, s_out, out);
}
