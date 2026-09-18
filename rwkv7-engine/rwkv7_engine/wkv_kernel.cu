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
// Split-J variant of the Wkv recurrence, for the decode case where the
// serial-in-t kernel leaves the GPU nearly idle: with one thread per state row
// there are only H blocks of D threads (40 x 64 = 2 warps/SM on V100, i.e.
// ~3% occupancy), so the two D-length dot products run as long dependent
// chains with nothing to hide the latency.
//
// Here WKV_SPLIT_J threads cooperate on one row (each owning a slice of the
// j dimension) and each block covers a row-group of one head, so the same
// work is spread over WKV_RG times as many blocks and WKV_SPLIT_J times as
// many threads. The per-row dot products are combined with warp shuffles.
// The state slice a thread owns is fixed across tokens, so it stays in
// registers exactly as in the serial kernel.
#define WKV_SPLIT_J 4
#define WKV_RG 2

extern "C" __global__ void wkv7_split_kernel(
    const int T, const int H,
    const __half* __restrict__ r, const __half* __restrict__ w,
    const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ a, const __half* __restrict__ b,
    const float* __restrict__ s_in,
    float* __restrict__ s_out,
    float* __restrict__ out)
{
    constexpr int JN = WKV_D / WKV_SPLIT_J;     // 16 j's per thread
    constexpr int ROWS = WKV_D / WKV_RG;        // 32 rows per block
    const int head = blockIdx.x / WKV_RG;
    const int rg = blockIdx.x % WKV_RG;
    const int lane = threadIdx.x / WKV_SPLIT_J; // row within the group
    const int js = threadIdx.x % WKV_SPLIT_J;   // which j slice
    const int row = rg * ROWS + lane;
    const int j0 = js * JN;
    const int C = H * WKV_D;
    const long row_base = (long)head * WKV_D * WKV_D + (long)row * WKV_D;

    float st[JN];
#pragma unroll
    for (int i = 0; i < JN; ++i)
        st[i] = s_in ? s_in[row_base + j0 + i] : 0.f;

    __shared__ float sr[WKV_D], sw[WKV_D], sk[WKV_D], sa_[WKV_D], sb[WKV_D],
                     swe[WKV_D];

    for (int t = 0; t < T; ++t) {
        const long base = (long)t * C + (long)head * WKV_D;
        if (threadIdx.x < WKV_D) {
            const int i = threadIdx.x;
            sr[i]  = __half2float(r[base + i]);
            sw[i]  = __half2float(w[base + i]);
            sk[i]  = __half2float(k[base + i]);
            sa_[i] = __half2float(a[base + i]);
            sb[i]  = __half2float(b[base + i]);
            swe[i] = __expf(sw[i]);      // exp once per block, not per thread
        }
        __syncthreads();
        const float v_i = __half2float(v[base + row]);

        float sdot = 0.f;
#pragma unroll
        for (int i = 0; i < JN; ++i) sdot += sa_[j0 + i] * st[i];
        sdot += __shfl_xor_sync(0xffffffffu, sdot, 1);
        sdot += __shfl_xor_sync(0xffffffffu, sdot, 2);

        float o = 0.f;
#pragma unroll
        for (int i = 0; i < JN; ++i) {
            const int j = j0 + i;
            st[i] = st[i] * swe[j] + sk[j] * v_i + sdot * sb[j];
            o += st[i] * sr[j];
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if (js == 0) out[base + row] = o;
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < JN; ++i)
        if (s_out) s_out[row_base + j0 + i] = st[i];
}

extern "C" void wkv7_split_launch(
    const int T, const int H,
    const __half* r, const __half* w, const __half* k, const __half* v,
    const __half* a, const __half* b,
    const float* s_in, float* s_out, float* out, cudaStream_t stream)
{
    const int threads = (WKV_D / WKV_RG) * WKV_SPLIT_J;
    wkv7_split_kernel<<<H * WKV_RG, threads, 0, stream>>>(
        T, H, r, w, k, v, a, b, s_in, s_out, out);
}

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

// ---------------------------------------------------------------------------
// GEMV for the decode path:  y[n] = sum_k x[k] * W[k, n]
//   W : [K, N] row-major fp16   (weights are stored transposed, [K,N])
//   x : [K] fp16
//   y : [N] fp16 or fp32
//
// One block handles 2048 consecutive outputs (256 threads x GEMV_VEC);
// each thread strides K in half8 (16B) steps. Consecutive threads read
// consecutive 16B chunks of W => fully coalesced. Split-K (blockIdx.y)
// spreads the reduction; partials go to a fp32 workspace, a second kernel
// reduces. Requires K % 8 == 0 and N % 8 == 0 (true for all RWKV7 dims).
#define GEMV_VEC 8

template <typename TOut>
__global__ void gemv_partial_kernel(
    const __half* __restrict__ W,   // [K, N]
    const __half* __restrict__ x,   // [K]
    TOut* __restrict__ y,           // [N], used when splitK == 1
    float* __restrict__ ws,         // [splitK, N], used when splitK > 1
    int K, int N, int splitK, int k_chunk)
{
    const int n0 = blockIdx.x * (blockDim.x * GEMV_VEC);
    const int n  = n0 + threadIdx.x * GEMV_VEC;   // this thread's output base
    const int s  = blockIdx.y;
    const int k_begin = s * k_chunk;
    const int k_end = min(k_begin + k_chunk, K);

    float acc[GEMV_VEC];
#pragma unroll
    for (int i = 0; i < GEMV_VEC; ++i) acc[i] = 0.f;

    if (n >= N) return;
    for (int k = k_begin; k < k_end; k += GEMV_VEC) {
        uint4 xraw = *reinterpret_cast<const uint4*>(x + k);
        float xf[GEMV_VEC];
        {
            const __half2* xh = reinterpret_cast<const __half2*>(&xraw);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 t = __half22float2(xh[i]);
                xf[2 * i] = t.x;
                xf[2 * i + 1] = t.y;
            }
        }
        const __half* wp = W + (long)k * N + n;
        uint4 wraw[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            wraw[i] = *reinterpret_cast<const uint4*>(wp + (long)i * N);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half2* wh = reinterpret_cast<const __half2*>(&wraw[i]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 wf = __half22float2(wh[j]);
                acc[2 * j]     += xf[i] * wf.x;
                acc[2 * j + 1] += xf[i] * wf.y;
            }
        }
        wraw[0] = *reinterpret_cast<const uint4*>(wp + (long)4 * N);
        wraw[1] = *reinterpret_cast<const uint4*>(wp + (long)5 * N);
        wraw[2] = *reinterpret_cast<const uint4*>(wp + (long)6 * N);
        wraw[3] = *reinterpret_cast<const uint4*>(wp + (long)7 * N);
#pragma unroll
        for (int i = 4; i < 8; ++i) {
            const __half2* wh = reinterpret_cast<const __half2*>(&wraw[i - 4]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 wf = __half22float2(wh[j]);
                acc[2 * j]     += xf[i] * wf.x;
                acc[2 * j + 1] += xf[i] * wf.y;
            }
        }
    }

    if (splitK == 1) {
#pragma unroll
        for (int i = 0; i < GEMV_VEC; ++i)
            if (n + i < N)
                y[n + i] = static_cast<TOut>(acc[i]);
    } else {
#pragma unroll
        for (int i = 0; i < GEMV_VEC; ++i)
            if (n + i < N)
                ws[(long)s * N + n + i] = acc[i];
    }
}

template <typename TOut>
__global__ void gemv_reduce_kernel(
    const float* __restrict__ ws,   // [splitK, N]
    TOut* __restrict__ y,           // [N]
    int N, int splitK)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    // Four independent accumulators: the partials are strided by N, so a
    // single accumulator leaves the thread stalled on one load chain.
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= splitK; i += 4) {
        s0 += ws[(long)(i + 0) * N + n];
        s1 += ws[(long)(i + 1) * N + n];
        s2 += ws[(long)(i + 2) * N + n];
        s3 += ws[(long)(i + 3) * N + n];
    }
    for (; i < splitK; ++i) s0 += ws[(long)i * N + n];
    y[n] = static_cast<TOut>((s0 + s1) + (s2 + s3));
}

extern "C" void gemv16_launch(
    const void* W, const void* x, void* y,
    int K, int N, int out_f32,
    float* ws, int ws_floats,   // pre-allocated workspace, total floats
    cudaStream_t stream)
{
    const int outputs_per_block = 256 * GEMV_VEC;
    int blocks_n = (N + outputs_per_block - 1) / outputs_per_block;
    int splitK = (1280 + blocks_n - 1) / blocks_n;
    int cap = (K + 63) / 64;
    if (cap > 80) cap = 80;
    if (cap < 20) cap = 20;
    if (splitK > cap) splitK = cap;
    if (splitK < 1) splitK = 1;
    int k_chunk = (K + splitK - 1) / splitK;
    k_chunk = (k_chunk + GEMV_VEC - 1) / GEMV_VEC * GEMV_VEC;
    splitK = (K + k_chunk - 1) / k_chunk;
    int ws_cap = ws_floats / N;
    if (ws_cap < 1) ws_cap = 1;
    if (splitK > ws_cap) {
        splitK = ws_cap;
        k_chunk = (K + splitK - 1) / splitK;
        k_chunk = (k_chunk + GEMV_VEC - 1) / GEMV_VEC * GEMV_VEC;
        splitK = (K + k_chunk - 1) / k_chunk;
    }

    dim3 grid(blocks_n, splitK);
    dim3 block(256);
    if (out_f32) {
        float* yf = (float*)y;
        gemv_partial_kernel<float><<<grid, block, 0, stream>>>(
            (const __half*)W, (const __half*)x, yf, ws, K, N, splitK, k_chunk);
        if (splitK > 1) {
            gemv_reduce_kernel<float><<<(N + 255) / 256, 256, 0, stream>>>(
                ws, yf, N, splitK);
        }
    } else {
        __half* yh = (__half*)y;
        gemv_partial_kernel<__half><<<grid, block, 0, stream>>>(
            (const __half*)W, (const __half*)x, yh, ws, K, N, splitK, k_chunk);
        if (splitK > 1) {
            gemv_reduce_kernel<__half><<<(N + 255) / 256, 256, 0, stream>>>(
                ws, yh, N, splitK);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused post-Wkv attention epilogue (decode, T==1).
//
// Replaces, per layer:
//   o  = group_norm(o_wkv)              // H groups of D
//   rk = (k * r * r_k).sum(-1)          // per head
//   o  = o + rk * v
//   og = o * g
// with a single kernel.  o_wkv is the fp32 Wkv output [C]; k, r, v, g are
// fp16 [C] (head-major, head h = [h*D,(h+1)*D)); gn_w/gn_b/r_k are fp32 [C].
// Output og [C] fp16, so the model then does  x = residual + og @ o_w.
// One block per head, D threads.
extern "C" __global__ void wkv7_post_kernel(
    const int H, const int D, const float eps,
    const float* __restrict__ o_wkv,
    const __half* __restrict__ k, const __half* __restrict__ r,
    const __half* __restrict__ v, const __half* __restrict__ g,
    const __half* __restrict__ gn_w, const __half* __restrict__ gn_b,
    const __half* __restrict__ rk_w,
    __half* __restrict__ og)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;              // 0 .. D-1
    const int base = h * D + d;

    __shared__ float ssum[WKV_D], ssumsq[WKV_D], sprod[WKV_D];

    const float ov = o_wkv[base];
    ssum[d]   = ov;
    ssumsq[d] = ov * ov;
    sprod[d]  = __half2float(k[base]) * __half2float(r[base]) * __half2float(rk_w[base]);
    __syncthreads();

    float sum = 0.f, sumsq = 0.f, rk = 0.f;
#pragma unroll 4
    for (int i = 0; i < D; ++i) { sum += ssum[i]; sumsq += ssumsq[i]; rk += sprod[i]; }

    const float mean = sum / (float)D;
    float var = sumsq / (float)D - mean * mean;
    if (var < 0.f) var = 0.f;
    const float nv = (ov - mean) * rsqrtf(var + eps);

    const float vval = __half2float(v[base]);
    const float gval = __half2float(g[base]);
    const float gnw  = __half2float(gn_w[base]);
    const float gnb  = __half2float(gn_b[base]);
    og[base] = __float2half((nv * gnw + gnb + rk * vval) * gval);
}

extern "C" void wkv7_post_launch(
    const float* o_wkv, const __half* k, const __half* r, const __half* v,
    const __half* g, const __half* gn_w, const __half* gn_b, const __half* rk_w,
    __half* og, int H, int D, float eps, cudaStream_t stream)
{
    wkv7_post_kernel<<<H, D, 0, stream>>>(H, D, eps, o_wkv, k, r, v, g,
                                          gn_w, gn_b, rk_w, og);
}

// ---------------------------------------------------------------------------
// LoRA gates, batched across the four gates (decode, T==1).
//
// RWKV7 has four LoRA-gated vectors per layer:
//   w = w_scale*sigmoid(b2w + W2w @ tanh(xw @ W1w))     (decay)
//   a =            sigmoid(b2a + W2a @ h1a)             (in-context lr)
//   g =                      W2g @ sigmoid(xg @ W1g)    (output gate)
//   t =            sigmoid(b2v + W2v @ h1v)             (v_first mix)
// The four are independent, and each is tiny (rank 64..320), so issuing them
// as separate kernels left the GPU nearly idle (measured 82 GB/s effective).
// Here the gate index is blockIdx.z, which quadruples the resident blocks per
// launch and removes six launches per layer.
//
// Launch order per layer: h1 partial -> h1 reduce (applies act1) ->
//                         gates partial -> gates reduce (bias+act2+scale).
#define LORA_VEC 8
#define LORA_RMAX 512      // max rank (g1j: w96 a96 g320 v64)
#define LORA_SPLIT_R 16    // r-chunks per gate

// --- h1 = act1(x @ W1): 4 skinny GEMVs, one launch -------------------------
extern "C" __global__ void lora_h1_partial_kernel(
    const int C, const int splitK, const int k_chunk,
    const __half* __restrict__ xw, const __half* __restrict__ xa,
    const __half* __restrict__ xg, const __half* __restrict__ xv,
    const __half* __restrict__ W1w, const __half* __restrict__ W1a,
    const __half* __restrict__ W1g, const __half* __restrict__ W1v,
    const int Rw, const int Ra, const int Rg, const int Rv,
    float* __restrict__ ws)                  // [4][splitK][LORA_RMAX]
{
    const int z = blockIdx.z;                // gate
    const int R = z == 0 ? Rw : (z == 1 ? Ra : (z == 2 ? Rg : Rv));
    if (R <= 0) return;
    const __half* x  = z == 0 ? xw  : (z == 1 ? xa  : (z == 2 ? xg  : xv));
    const __half* W  = z == 0 ? W1w : (z == 1 ? W1a : (z == 2 ? W1g : W1v));
    const int n = blockIdx.x * blockDim.x * LORA_VEC + threadIdx.x * LORA_VEC;
    if (n >= R) return;
    const int s = blockIdx.y;
    const int kb = s * k_chunk;
    const int ke = min(kb + k_chunk, C);

    float acc[LORA_VEC];
#pragma unroll
    for (int i = 0; i < LORA_VEC; ++i) acc[i] = 0.f;
    for (int k = kb; k < ke; ++k) {
        const float xv_ = __half2float(x[k]);
        uint4 w = *reinterpret_cast<const uint4*>(W + (long)k * R + n);
        const __half2* wh = reinterpret_cast<const __half2*>(&w);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(wh[j]);
            acc[2 * j]     += xv_ * f.x;
            acc[2 * j + 1] += xv_ * f.y;
        }
    }
#pragma unroll
    for (int i = 0; i < LORA_VEC; ++i)
        ws[((long)z * splitK + s) * LORA_RMAX + n + i] = acc[i];
}

// --- h1 reduce + act1 (one block per gate) ---------------------------------
extern "C" __global__ void lora_h1_reduce_kernel(
    const int splitK,
    const int Rw, const int Ra, const int Rg, const int Rv,
    const int act1w, const int act1a, const int act1g, const int act1v,
    const float* __restrict__ ws,            // [4][splitK][LORA_RMAX]
    __half* __restrict__ h1w, __half* __restrict__ h1a,
    __half* __restrict__ h1g, __half* __restrict__ h1v)
{
    const int z = blockIdx.x;
    const int R = z == 0 ? Rw : (z == 1 ? Ra : (z == 2 ? Rg : Rv));
    const int r = threadIdx.x;
    if (r >= R) return;
    const int act = z == 0 ? act1w : (z == 1 ? act1a : (z == 2 ? act1g : act1v));
    float acc = 0.f;
    for (int s = 0; s < splitK; ++s)
        acc += ws[((long)z * splitK + s) * LORA_RMAX + r];
    if (act == 1) acc = tanhf(acc);
    else if (act == 2) acc = 1.f / (1.f + __expf(-acc));
    __half* out = z == 0 ? h1w : (z == 1 ? h1a : (z == 2 ? h1g : h1v));
    out[r] = __float2half(acc);
}

// --- gates = act2(b2 + W2 @ h1): 4 GEMVs, one launch -----------------------
extern "C" __global__ void lora_gates_partial_kernel(
    const int C, const int splitR,
    const int rcw, const int rca, const int rcg, const int rcv,
    const __half* __restrict__ W2w, const __half* __restrict__ W2a,
    const __half* __restrict__ W2g, const __half* __restrict__ W2v,
    const __half* __restrict__ h1w, const __half* __restrict__ h1a,
    const __half* __restrict__ h1g, const __half* __restrict__ h1v,
    const int Rw, const int Ra, const int Rg, const int Rv,
    float* __restrict__ ws)                  // [4][splitR][C]
{
    const int z = blockIdx.z;
    const int ZR = z == 0 ? Rw : (z == 1 ? Ra : (z == 2 ? Rg : Rv));
    if (ZR <= 0) return;
    const __half* W2 = z == 0 ? W2w : (z == 1 ? W2a : (z == 2 ? W2g : W2v));
    const __half* h1 = z == 0 ? h1w : (z == 1 ? h1a : (z == 2 ? h1g : h1v));
    const int r_chunk = z == 0 ? rcw : (z == 1 ? rca : (z == 2 ? rcg : rcv));
    const int rb = blockIdx.y * r_chunk;
    const int re = min(rb + r_chunk, ZR);
    const int n = blockIdx.x * blockDim.x * LORA_VEC + threadIdx.x * LORA_VEC;
    if (n >= C) return;

    float acc[LORA_VEC];
#pragma unroll
    for (int i = 0; i < LORA_VEC; ++i) acc[i] = 0.f;
    for (int r = rb; r < re; ++r) {
        const float hv = __half2float(h1[r]);
        uint4 w = *reinterpret_cast<const uint4*>(W2 + (long)r * C + n);
        const __half2* wh = reinterpret_cast<const __half2*>(&w);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(wh[j]);
            acc[2 * j]     += hv * f.x;
            acc[2 * j + 1] += hv * f.y;
        }
    }
#pragma unroll
    for (int i = 0; i < LORA_VEC; ++i)
        ws[((long)z * splitR + blockIdx.y) * C + n + i] = acc[i];
}

// --- gates reduce: bias + act2 + scale, all four gates ---------------------
extern "C" __global__ void lora_gates_reduce_kernel(
    const int C, const int splitR,
    const int Rw, const int Ra, const int Rg, const int Rv,
    const __half* __restrict__ b2w, const __half* __restrict__ b2a,
    const __half* __restrict__ b2v, const float w_scale,
    const float* __restrict__ ws,            // [4][splitR][C]
    __half* __restrict__ out_w, __half* __restrict__ out_a,
    __half* __restrict__ out_g, __half* __restrict__ out_v)
{
    const int z = blockIdx.y;
    const int ZR = z == 0 ? Rw : (z == 1 ? Ra : (z == 2 ? Rg : Rv));
    if (ZR <= 0) return;                    // gate absent (layer 0 has no v_lora)
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= C) return;
    float acc = 0.f;
    for (int s = 0; s < splitR; ++s) acc += ws[((long)z * splitR + s) * C + n];
    if (z == 0) {
        acc += __half2float(b2w[n]);
        acc = w_scale / (1.f + __expf(-acc));
        out_w[n] = __float2half(acc);
    } else if (z == 1) {
        acc += __half2float(b2a[n]);
        out_a[n] = __float2half(1.f / (1.f + __expf(-acc)));
    } else if (z == 2) {
        out_g[n] = __float2half(acc);
    } else {
        acc += __half2float(b2v[n]);
        out_v[n] = __float2half(1.f / (1.f + __expf(-acc)));
    }
}

extern "C" void lora_h1_launch(
    const __half* xw, const __half* xa, const __half* xg, const __half* xv,
    const __half* W1w, const __half* W1a, const __half* W1g, const __half* W1v,
    const int Rw, const int Ra, const int Rg, const int Rv, int C, int splitK,
    const int* act1, float* ws_h1, __half* h1w, __half* h1a, __half* h1g,
    __half* h1v, cudaStream_t stream)
{
    int k_chunk = (C + splitK - 1) / splitK;
    splitK = (C + k_chunk - 1) / k_chunk;
    const int threads = 128;
    int rmax = Rw;
    if (Ra > rmax) rmax = Ra;
    if (Rg > rmax) rmax = Rg;
    if (Rv > rmax) rmax = Rv;
    dim3 grid((rmax + threads * LORA_VEC - 1) / (threads * LORA_VEC), splitK, 4);
    lora_h1_partial_kernel<<<grid, threads, 0, stream>>>(
        C, splitK, k_chunk, xw, xa, xg, xv, W1w, W1a, W1g, W1v,
        Rw, Ra, Rg, Rv, ws_h1);
    lora_h1_reduce_kernel<<<4, 512, 0, stream>>>(
        splitK, Rw, Ra, Rg, Rv, act1[0], act1[1], act1[2], act1[3],
        ws_h1, h1w, h1a, h1g, h1v);
}

extern "C" void lora_gates_launch(
    const __half* W2w, const __half* W2a, const __half* W2g, const __half* W2v,
    const __half* h1w, const __half* h1a, const __half* h1g, const __half* h1v,
    const __half* b2w, const __half* b2a, const __half* b2v, float w_scale,
    const int Rw, const int Ra, const int Rg, const int Rv, int C, int splitR,
    __half* out_w, __half* out_a, __half* out_g, __half* out_v,
    float* ws, cudaStream_t stream)
{
    if (splitR < 1) splitR = 1;
    int rcw = (Rw + splitR - 1) / splitR;
    int rca = (Ra + splitR - 1) / splitR;
    int rcg = (Rg + splitR - 1) / splitR;
    int rcv = (Rv + splitR - 1) / splitR;
    const int threads = 128;
    dim3 grid((C + threads * LORA_VEC - 1) / (threads * LORA_VEC), splitR, 4);
    lora_gates_partial_kernel<<<grid, threads, 0, stream>>>(
        C, splitR, rcw, rca, rcg, rcv, W2w, W2a, W2g, W2v,
        h1w, h1a, h1g, h1v, Rw, Ra, Rg, Rv, ws);
    dim3 rgrid((C + 255) / 256, 4);
    lora_gates_reduce_kernel<<<rgrid, 256, 0, stream>>>(
        C, splitR, Rw, Ra, Rg, Rv, b2w, b2a, b2v, w_scale, ws,
        out_w, out_a, out_g, out_v);
}
// ---------------------------------------------------------------------------
// LoRA gate GEMV (decode, T==1), split-R vectorised.
//
//   out[n] = act2( b2[n] + sum_r h1[r] * W2[r, n] )
//
// W2 is [R, C] fp16 (the native lora.2 layout).  The shapes are "fat and
// short" (R = 96..320), so the plain one-thread-per-output form launches only
// C threads and streams 2 bytes per thread per step: that measured 62 GB/s.
// Here each thread owns VEC outputs (16B) and the R dimension is split
// SPLIT_R ways across blockIdx.y, so the grid grows by SPLIT_R and every
// thread reads full 16B words.  Partials land in an fp32 workspace; the
// reduce kernel folds them and applies bias + activation + scale, so the
// whole gate costs one partial pass plus one tiny reduce.
#define LORA_VEC 8
#define LORA_SPLIT_R 32

extern "C" __global__ void lora_gate_partial_kernel(
    const int R, const int C, const int r_chunk, const int act1,
    const __half* __restrict__ W2,      // [R, C]
    const __half* __restrict__ h1,      // [R]
    float* __restrict__ ws)             // [splitR, C]
{
    const int n = blockIdx.x * blockDim.x * LORA_VEC + threadIdx.x * LORA_VEC;
    if (n >= C) return;
    const int s = blockIdx.y;
    const int rb = s * r_chunk;
    const int re = min(rb + r_chunk, R);

    float acc[LORA_VEC];
#pragma unroll
    for (int i = 0; i < LORA_VEC; ++i) acc[i] = 0.f;

    // W2 rows are contiguous in n: consecutive threads cover consecutive 16B
    // words, so a warp reads full cache lines.
    for (int r = rb; r < re; ++r) {
        float hv = __half2float(h1[r]);
        if (act1 == 1) hv = tanhf(hv);
        else if (act1 == 2) hv = 1.f / (1.f + __expf(-hv));
        uint4 w = *reinterpret_cast<const uint4*>(W2 + (long)r * C + n);
        const __half2* wh = reinterpret_cast<const __half2*>(&w);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(wh[j]);
            acc[2 * j]     += hv * f.x;
            acc[2 * j + 1] += hv * f.y;
        }
    }
#pragma unroll
    for (int i = 0; i < LORA_VEC; ++i)
        ws[(long)s * C + n + i] = acc[i];
}

extern "C" __global__ void lora_gate_reduce_kernel(
    const int C, const int splitR, const float scale,
    const int act2,                      // 0 = identity, 1 = sigmoid
    const __half* __restrict__ b2,       // [C] fp16 or null
    const float* __restrict__ ws,        // [splitR, C]
    __half* __restrict__ out)            // [C]
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= C) return;
    float acc = b2 ? __half2float(b2[n]) : 0.f;
    for (int s = 0; s < splitR; ++s) acc += ws[(long)s * C + n];
    if (act2 == 1) acc = 1.f / (1.f + __expf(-acc));
    out[n] = __float2half(scale * acc);
}

extern "C" void lora_gate_launch(
    const __half* h1, const __half* W2, const __half* b2,
    __half* out, float* ws, int C, int R, float scale, int act1, int act2,
    cudaStream_t stream)
{
    int splitR = LORA_SPLIT_R;
    int r_chunk = (R + splitR - 1) / splitR;
    splitR = (R + r_chunk - 1) / r_chunk;
    if (splitR < 1) splitR = 1;
    int threads = 128;
    int blocks_n = (C + threads * LORA_VEC - 1) / (threads * LORA_VEC);
    dim3 grid(blocks_n, splitR);
    lora_gate_partial_kernel<<<grid, threads, 0, stream>>>(
        R, C, r_chunk, act1, W2, h1, ws);
    lora_gate_reduce_kernel<<<(C + 255) / 256, 256, 0, stream>>>(
        C, splitR, scale, act2, b2, ws, out);
}

// Four gates in one launch: w, a, g, t.  Each gate has its own R and bias
// (g has no bias), so they are issued back-to-back with disjoint workspace
// slices and one reduce per gate.  Fusing them into a single kernel was tried
// and measured slower; the win here is grid size, not launch count.
extern "C" void lora_gates4_launch(
    const __half* h1w, const __half* h1a, const __half* h1g, const __half* h1v,
    const __half* W2w, const __half* W2a, const __half* W2g, const __half* W2v,
    const __half* b2w, const __half* b2a, const __half* b2v,
    float w_scale, int C, int Rw, int Ra, int Rg, int Rv,
    __half* out_w, __half* out_a, __half* out_g, __half* out_v,
    float* ws, cudaStream_t stream)
{
    if (Rw > 0)   // h1w -> tanh, then sigmoid on the output
        lora_gate_launch(h1w, W2w, b2w, out_w, ws, C, Rw, w_scale, 1, 1, stream);
    if (Ra > 0)
        lora_gate_launch(h1a, W2a, b2a, out_a, ws, C, Ra, 1.f, 0, 1, stream);
    if (Rg > 0)   // h1g -> sigmoid, identity output
        lora_gate_launch(h1g, W2g, NULL, out_g, ws, C, Rg, 1.f, 2, 0, stream);
    if (Rv > 0)
        lora_gate_launch(h1v, W2v, b2v, out_v, ws, C, Rv, 1.f, 0, 1, stream);
}
