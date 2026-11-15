// src/models/qwen4exp/prefill/gemm_wht128.cu - orthonormal WHT-128 + int8 tensor-core GEMM for batched native-weight
// prompt projections.
//
// An orthonormal 128-point Walsh-Hadamard transform is applied identically to each 128-element chunk of X and of the
// dequantized FP16 weights W (H_128 is symmetric and orthonormal, so (X H)(W H)^T = X W^T).  Each 128-element rotated
// chunk is symmetrically quantized to int8 with one FP32 scale and pre-packed into m16n8k32 fragment planes so the
// mainloop streams coalesced 16-byte cp.async loads without shared-memory swizzling.  Everything lives inside Gemm's
// existing reusable scratch buffer.
#include "gemm_wht128.hpp"

#include "basalt/kernels/dequant_bf16.hpp"
#include "basalt/log.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>

namespace basalt::prefill {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "prefill gemm wht128: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16_zfill(void* dst_smem, const void* src_gmem, bool pred) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    const int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src_gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma_k32_acc(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float wht32_lane(float v, int lane) {
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const float u = __shfl_xor_sync(0xffffffffu, v, d);
        v = (lane & d) ? (u - v) : (v + u);
    }
    return v * 0.1767766953f;
}
#endif

template <bool IS_B>
__device__ __forceinline__ void quant_wht128_pack_stage(int local_tile, int kc, int tid,
                                                        const __half* __restrict__ src, int rows, int K,
                                                        int tile_offset, int total_tiles, uint32_t* __restrict__ P,
                                                        float* __restrict__ S, int8_t (&sm_q)[4][16][36]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int wp = tid >> 5;
    const int lane = tid & 31;
    const int r = local_tile * 16 + wp;
    const int dst_tile = tile_offset + local_tile;

    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (r < rows) {
        constexpr uint32_t kSigns[4] = {0x9E3779B9u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au};
        const __half* rp = src + (size_t) r * K + kc * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float raw = __half2float(rp[j * 32 + lane]);
            const float signed_v = __int_as_float(__float_as_int(raw) ^ (((kSigns[j] >> lane) & 1u) << 31));
            v[j] = wht32_lane(signed_v, lane);
        }
        const float s01 = v[0] + v[1], d01 = v[0] - v[1];
        const float s23 = v[2] + v[3], d23 = v[2] - v[3];
        v[0] = (s01 + s23) * 0.5f;
        v[1] = (d01 + d23) * 0.5f;
        v[2] = (s01 - s23) * 0.5f;
        v[3] = (d01 - d23) * 0.5f;
    }

    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

#pragma unroll
    for (int j = 0; j < 4; ++j) sm_q[j][wp][lane] = (int8_t) __float2int_rn(v[j] * inv);
    if (lane == 0) S[((size_t) kc * total_tiles + dst_tile) * 16 + wp] = amax * (1.0f / 127.0f);
    __syncthreads();

    if (wp < 4) {
        const int j = wp;
        const int kb = kc * 4 + j;
        const int g = lane >> 2;
        const int t = lane & 3;
        uint32_t o0, o1, o2, o3;
        if constexpr (!IS_B) {
            o0 = *reinterpret_cast<const uint32_t*>(&sm_q[j][g][4 * t]);
            o1 = *reinterpret_cast<const uint32_t*>(&sm_q[j][g + 8][4 * t]);
            o2 = *reinterpret_cast<const uint32_t*>(&sm_q[j][g][16 + 4 * t]);
            o3 = *reinterpret_cast<const uint32_t*>(&sm_q[j][g + 8][16 + 4 * t]);
        } else {
            o0 = *reinterpret_cast<const uint32_t*>(&sm_q[j][g][4 * t]);
            o1 = *reinterpret_cast<const uint32_t*>(&sm_q[j][g][16 + 4 * t]);
            o2 = *reinterpret_cast<const uint32_t*>(&sm_q[j][g + 8][4 * t]);
            o3 = *reinterpret_cast<const uint32_t*>(&sm_q[j][g + 8][16 + 4 * t]);
        }
        uint32_t* dst = P + ((size_t) kb * total_tiles + dst_tile) * 128;
        dst[0 * 32 + lane] = o0;
        dst[1 * 32 + lane] = o1;
        dst[2 * 32 + lane] = o2;
        dst[3 * 32 + lane] = o3;
    }
#else
    (void) local_tile; (void) kc; (void) tid; (void) src; (void) rows; (void) K;
    (void) tile_offset; (void) total_tiles; (void) P; (void) S; (void) sm_q;
#endif
}

template <bool IS_B>
__global__ void __launch_bounds__(512)
quant_wht128_pack_kernel(const __half* __restrict__ src, int rows, int K, int tile_offset, int total_tiles,
                         uint32_t* __restrict__ P, float* __restrict__ S) {
    __shared__ alignas(16) int8_t sm_q[4][16][36];
    quant_wht128_pack_stage<IS_B>((int) blockIdx.x, (int) blockIdx.y, (int) threadIdx.x, src, rows, K, tile_offset,
                                  total_tiles, P, S, sm_q);
}

__device__ __forceinline__ void gemm_wht128_128x64_stage(int bx, int by, int tid, const uint32_t* __restrict__ Wp,
                                                         const float* __restrict__ Ws,
                                                         const uint32_t* __restrict__ Xp,
                                                         const float* __restrict__ Xs, float* __restrict__ Y, int T,
                                                         int N, int K, int ldy, float beta,
                                                         uint32_t (&smA)[4][8][128], uint32_t (&smB)[4][4][128],
                                                         float (&smSa)[2][128], float (&smSb)[2][64]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int STAGES = 4;
    const int num_pid_m = (N + 127) >> 7;
    const int num_pid_n = (T + 63) >> 6;
    const int pid = bx + by * num_pid_m;
    constexpr int GROUP_M = 8;
    const int num_pid_in_group = GROUP_M * num_pid_n;
    const int group_id = pid / num_pid_in_group;
    const int first_pid_m = group_id * GROUP_M;
    const int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
    const int pid_m = first_pid_m + (pid % group_size_m);
    const int pid_n = (pid % num_pid_in_group) / group_size_m;

    const int bm = pid_m * 8;
    const int bn = pid_n * 4;
    const int m_tiles = (N + 15) >> 4;
    const int n_tiles = (T + 15) >> 4;
    const int k_blocks = K >> 5;
    const int k_chunks = K >> 7;

    const int wp = tid >> 5;
    const int lane = tid & 31;
    const int wm = (wp >> 1) * 2;
    const int wn = (wp & 1) * 2;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int m_rem = min(8, m_tiles - bm);
    const int n_rem = min(4, n_tiles - bn);

    auto issue_scales = [&](int kc, int sbuf) {
        if (kc < k_chunks) {
            if (wp == 4) {
                const int tm = min(lane >> 2, max(m_rem - 1, 0));
                cp16_zfill(&smSa[sbuf][lane * 4], &Ws[((size_t) kc * m_tiles + bm + tm) * 16 + (lane & 3) * 4],
                           (lane >> 2) < m_rem);
            } else if (wp == 5 && lane < 16) {
                const int tn = min(lane >> 2, max(n_rem - 1, 0));
                cp16_zfill(&smSb[sbuf][lane * 4], &Xs[((size_t) kc * n_tiles + bn + tn) * 16 + (lane & 3) * 4],
                           (lane >> 2) < n_rem);
            }
        }
    };

    auto issue_k = [&](int kb, int stg) {
        if (kb < k_blocks) {
            cp16_zfill(&smA[stg][wp][lane * 4],
                       &Wp[((size_t) kb * m_tiles + bm + min(wp, max(m_rem - 1, 0))) * 128 + lane * 4], wp < m_rem);
            if (wp < 4) {
                cp16_zfill(&smB[stg][wp][lane * 4],
                           &Xp[((size_t) kb * n_tiles + bn + min(wp, max(n_rem - 1, 0))) * 128 + lane * 4],
                           wp < n_rem);
            }
        }
        cp_commit();
    };

    float acc[2][4][4] = {};
    issue_scales(0, 0);
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) issue_k(s, s);

    for (int kc = 0; kc < k_chunks; ++kc) {
        const int sbuf = kc & 1;
        int tot[2][4][4] = {};

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int kb = kc * 4 + j;
            const int stg = kb & 3;
            cp_wait<STAGES - 2>();
            __syncthreads();

            uint32_t a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                a[mi][0] = smA[stg][wm + mi][0 * 32 + lane];
                a[mi][1] = smA[stg][wm + mi][1 * 32 + lane];
                a[mi][2] = smA[stg][wm + mi][2 * 32 + lane];
                a[mi][3] = smA[stg][wm + mi][3 * 32 + lane];
            }
            uint32_t b00[2], b01[2], b10[2], b11[2];
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                b00[ni] = smB[stg][wn + ni][0 * 32 + lane];
                b01[ni] = smB[stg][wn + ni][1 * 32 + lane];
                b10[ni] = smB[stg][wn + ni][2 * 32 + lane];
                b11[ni] = smB[stg][wn + ni][3 * 32 + lane];
            }

            if (j == 0) issue_scales(kc + 1, sbuf ^ 1);
            issue_k(kb + STAGES - 1, (kb + STAGES - 1) & 3);

#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    mma_k32_acc(tot[mi][2 * ni + 0], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b00[ni], b01[ni]);
                    mma_k32_acc(tot[mi][2 * ni + 1], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b10[ni], b11[ni]);
                }
            }
        }

        const float sa0[2] = {smSa[sbuf][(wm + 0) * 16 + g], smSa[sbuf][(wm + 1) * 16 + g]};
        const float sa1[2] = {smSa[sbuf][(wm + 0) * 16 + g + 8], smSa[sbuf][(wm + 1) * 16 + g + 8]};
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const float sb00 = smSb[sbuf][(wn + ni) * 16 + 2 * t];
            const float sb01 = smSb[sbuf][(wn + ni) * 16 + 2 * t + 1];
            const float sb10 = smSb[sbuf][(wn + ni) * 16 + 8 + 2 * t];
            const float sb11 = smSb[sbuf][(wn + ni) * 16 + 8 + 2 * t + 1];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                acc[mi][2 * ni + 0][0] = fmaf((float) tot[mi][2 * ni + 0][0], sa0[mi] * sb00, acc[mi][2 * ni + 0][0]);
                acc[mi][2 * ni + 0][1] = fmaf((float) tot[mi][2 * ni + 0][1], sa0[mi] * sb01, acc[mi][2 * ni + 0][1]);
                acc[mi][2 * ni + 0][2] = fmaf((float) tot[mi][2 * ni + 0][2], sa1[mi] * sb00, acc[mi][2 * ni + 0][2]);
                acc[mi][2 * ni + 0][3] = fmaf((float) tot[mi][2 * ni + 0][3], sa1[mi] * sb01, acc[mi][2 * ni + 0][3]);

                acc[mi][2 * ni + 1][0] = fmaf((float) tot[mi][2 * ni + 1][0], sa0[mi] * sb10, acc[mi][2 * ni + 1][0]);
                acc[mi][2 * ni + 1][1] = fmaf((float) tot[mi][2 * ni + 1][1], sa0[mi] * sb11, acc[mi][2 * ni + 1][1]);
                acc[mi][2 * ni + 1][2] = fmaf((float) tot[mi][2 * ni + 1][2], sa1[mi] * sb10, acc[mi][2 * ni + 1][2]);
                acc[mi][2 * ni + 1][3] = fmaf((float) tot[mi][2 * ni + 1][3], sa1[mi] * sb11, acc[mi][2 * ni + 1][3]);
            }
        }
    }
    cp_wait<0>();

#pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
        const int tok0 = (bn + wn) * 16 + nb * 8 + 2 * t;
        const int tok1 = tok0 + 1;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int col0 = (bm + wm + mi) * 16 + g;
            const int col1 = col0 + 8;
            if (beta == 0.0f) {
                if (col0 < N) {
                    if (tok0 < T) Y[(size_t) tok0 * ldy + col0] = acc[mi][nb][0];
                    if (tok1 < T) Y[(size_t) tok1 * ldy + col0] = acc[mi][nb][1];
                }
                if (col1 < N) {
                    if (tok0 < T) Y[(size_t) tok0 * ldy + col1] = acc[mi][nb][2];
                    if (tok1 < T) Y[(size_t) tok1 * ldy + col1] = acc[mi][nb][3];
                }
            } else {
                if (col0 < N) {
                    if (tok0 < T) Y[(size_t) tok0 * ldy + col0] = fmaf(beta, Y[(size_t) tok0 * ldy + col0], acc[mi][nb][0]);
                    if (tok1 < T) Y[(size_t) tok1 * ldy + col0] = fmaf(beta, Y[(size_t) tok1 * ldy + col0], acc[mi][nb][1]);
                }
                if (col1 < N) {
                    if (tok0 < T) Y[(size_t) tok0 * ldy + col1] = fmaf(beta, Y[(size_t) tok0 * ldy + col1], acc[mi][nb][2]);
                    if (tok1 < T) Y[(size_t) tok1 * ldy + col1] = fmaf(beta, Y[(size_t) tok1 * ldy + col1], acc[mi][nb][3]);
                }
            }
        }
    }
#else
    (void) bx; (void) by; (void) tid; (void) Wp; (void) Ws; (void) Xp; (void) Xs;
    (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) beta;
    (void) smA; (void) smB; (void) smSa; (void) smSb;
#endif
}

__global__ void __launch_bounds__(256, 2)
gemm_wht128_128x64_kernel(const uint32_t* __restrict__ Wp, const float* __restrict__ Ws,
                          const uint32_t* __restrict__ Xp, const float* __restrict__ Xs, float* __restrict__ Y, int T,
                          int N, int K, int ldy, float beta) {
    __shared__ alignas(16) uint32_t smA[4][8][128];
    __shared__ alignas(16) uint32_t smB[4][4][128];
    __shared__ alignas(16) float smSa[2][128];
    __shared__ alignas(16) float smSb[2][64];
    gemm_wht128_128x64_stage((int) blockIdx.x, (int) blockIdx.y, (int) threadIdx.x, Wp, Ws, Xp, Xs, Y, T, N, K, ldy,
                             beta, smA, smB, smSa, smSb);
}

std::atomic<int> g_override{-1};

bool device_ok() {
    static std::mutex mu;
    static int8_t ok[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return false;
    std::lock_guard<std::mutex> lk(mu);
    if (ok[dev] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaFuncAttributes fa{};
        const bool ready = major >= 8 &&
                           cudaFuncGetAttributes(&fa, gemm_wht128_128x64_kernel) == cudaSuccess &&
                           fa.ptxVersion >= 80;
        cudaGetLastError();
        ok[dev] = ready ? 1 : -1;
    }
    return ok[dev] > 0;
}

}  // namespace

void gemm_wht128_set_override(int mode) { g_override.store(mode); }

bool gemm_wht128_native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N,
                        int64_t K, int64_t ldy, float beta, uint16_t* scratch, int64_t scratch_elems, void* stream) {
    if (T <= 0 || N <= 0) return true;
    const int ov = g_override.load();
    if (ov == 0) return false;
    if (scratch == nullptr || scratch_elems <= 0 || K < 128 || (K & 127) != 0 ||
        T > (1ll << 30) || N > (1ll << 30) || K > (1ll << 30))
        return false;
    if (ov < 0) {
        const bool above_crossover = (N >= 1024 && T >= 768) || (T >= 1536 && N * T >= (1ll << 20));
        if (!above_crossover) return false;
    }
    if (!device_ok()) return false;

    if (ldy <= 0) ldy = N;
    const size_t cap = (size_t) scratch_elems * 2;
    const int k_blocks = (int) (K >> 5);
    const int k_chunks = (int) (K >> 7);
    const size_t wp_per_mtile = (size_t) k_blocks * 512;
    const size_t ws_per_mtile = (size_t) k_chunks * 64;
    const size_t xp_per_ntile = (size_t) k_blocks * 512;
    const size_t xs_per_ntile = (size_t) k_chunks * 64;

    const size_t min_deq = 32 * (size_t) K;   // 16 FP16 rows
    const size_t min_x = 4 * xp_per_ntile + align256(4 * xs_per_ntile);
    const size_t min_top = std::max(min_deq, min_x);

    auto w_need = [&](int64_t m_tiles) -> size_t {
        return (size_t) m_tiles * wp_per_mtile + align256((size_t) m_tiles * ws_per_mtile);
    };

    const int64_t m_all = (N + 15) >> 4;
    int64_t m_slice = m_all;
    if (w_need(m_all) + std::max(min_top, cap >> 2) > cap) {
        if (w_need(8) + min_top > cap) return false;
        const size_t budget = cap - std::max(min_top, cap >> 2);
        int64_t mt = (int64_t) ((budget >= 256 ? budget - 256 : 0) / (wp_per_mtile + ws_per_mtile));
        mt &= ~int64_t(7);
        if (mt < 8) mt = 8;
        m_slice = std::min(m_all, mt);
    }
    const int64_t n_rows_slice = m_slice * 16;

    const cudaStream_t s = (cudaStream_t) stream;
    uint8_t* const base = reinterpret_cast<uint8_t*>(scratch);

    for (int64_t r_base = 0; r_base < N; r_base += n_rows_slice) {
        const int64_t N_cur = std::min<int64_t>(n_rows_slice, N - r_base);
        const int m_tiles = (int) ((N_cur + 15) >> 4);
        const size_t wp_bytes = (size_t) m_tiles * wp_per_mtile;
        const size_t ws_bytes = align256((size_t) m_tiles * ws_per_mtile);
        uint32_t* const Wp = reinterpret_cast<uint32_t*>(base);
        float* const Ws = reinterpret_cast<float*>(base + wp_bytes);
        uint8_t* const top = base + wp_bytes + ws_bytes;
        const size_t rem = cap - (wp_bytes + ws_bytes);

        int64_t deq_rows = (int64_t) (rem / ((size_t) K * 2));
        if (deq_rows < N_cur) deq_rows &= ~int64_t(15);
        if (deq_rows <= 0) return false;

        for (int64_t r_sub = 0; r_sub < N_cur; r_sub += deq_rows) {
            const int64_t nr = std::min<int64_t>(deq_rows, N_cur - r_sub);
            basalt::kernels::dequant_f16(ggml_type, W_blocks, r_base + r_sub, nr, K,
                                         reinterpret_cast<uint16_t*>(top), stream);
            const dim3 grid_w((unsigned) ((nr + 15) >> 4), (unsigned) k_chunks);
            quant_wht128_pack_kernel<false><<<grid_w, 512, 0, s>>>(
                reinterpret_cast<const __half*>(top), (int) nr, (int) K, (int) (r_sub >> 4), m_tiles, Wp, Ws);
        }

        const int64_t n_all = (T + 15) >> 4;
        int64_t max_ntiles = (int64_t) ((rem >= 256 ? rem - 256 : 0) / (xp_per_ntile + xs_per_ntile));
        if (max_ntiles < n_all) max_ntiles &= ~int64_t(3);
        if (max_ntiles <= 0) return false;
        const int64_t T_step = max_ntiles * 16;

        for (int64_t t0 = 0; t0 < T; t0 += T_step) {
            const int64_t T_cur = std::min<int64_t>(T_step, T - t0);
            const int n_tiles = (int) ((T_cur + 15) >> 4);
            const size_t xp_bytes = (size_t) n_tiles * xp_per_ntile;
            uint32_t* const Xp = reinterpret_cast<uint32_t*>(top);
            float* const Xs = reinterpret_cast<float*>(top + xp_bytes);

            const dim3 grid_x((unsigned) n_tiles, (unsigned) k_chunks);
            quant_wht128_pack_kernel<true><<<grid_x, 512, 0, s>>>(
                reinterpret_cast<const __half*>(X + t0 * K), (int) T_cur, (int) K, 0, n_tiles, Xp, Xs);

            const dim3 grid_g((unsigned) ((N_cur + 127) >> 7), (unsigned) ((T_cur + 63) >> 6));
            gemm_wht128_128x64_kernel<<<grid_g, 256, 0, s>>>(
                Wp, Ws, Xp, Xs, Y + t0 * ldy + r_base, (int) T_cur, (int) N_cur, (int) K, (int) ldy, beta);
        }
    }
    ck(cudaGetLastError(), "gemm_wht128_native");
    return true;
}

}  // namespace basalt::prefill
