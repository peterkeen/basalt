// src/models/qwen4exp/prefill/moe_fused_v2.cuh - shared device stage for the v2 native fused prompt expert kernels.
// No __global__ kernel lives here; each moe_fused_v2_*.cu instantiates one thin wrapper and returns its V2Spec.
#pragma once

#include "moe_fused_v2.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "basalt/ggml/ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"

namespace basalt::prefill::fused {
namespace v2_detail {

constexpr int AB = 80;
constexpr int MAGIC = 0x4B400000;
constexpr float MAGICF = 12582912.0f;
constexpr int GU_ROWS_K = 2560, D_ROWS_K = 640;

constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_XS = GGML_TYPE_IQ4_XS,
              T_IQ4_NL = GGML_TYPE_IQ4_NL, T_Q2_0 = GGML_TYPE_Q2_0;

__host__ __device__ constexpr int sb_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110
         : t == T_IQ4_XS ? 136 : 18;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4
         : t == T_IQ3_S ? 512 * 4 : 0;
}

constexpr int V2_WR = 256;
constexpr int V2_NR = 128;
constexpr int V2_CSTR = 144;
constexpr int V2_ACT = V2_NR * V2_CSTR + 4 * V2_NR * 4;
constexpr int V2_PF = 2;
static_assert(kV2Threads == V2_WR, "one thread per weight row for L2 prefetches");

template <int WT, bool GU, int WU = WT>
struct V2 {
    static constexpr bool MIX = GU && (WU != WT);
    static constexpr bool SPLIT2 = GU && (WT == T_IQ3_S || WT == T_IQ4_XS);
    static constexpr bool K16 = per16(WT) || (MIX && per16(WU));
    static constexpr int K = GU ? GU_ROWS_K : D_ROWS_K;
    static constexpr int NC = K / 128;
    static constexpr int CPR = (GU && !SPLIT2) ? 2 : 1;
    static constexpr int NU = NC / CPR;

    // Contiguous unit bytes when !SPLIT2; for SPLIT2, two sub-spans from the 256-value superblock of size SB.
    static constexpr int SB = sb_bytes(WT);
    static constexpr int RB = GU ? SB : (WT == T_IQ4_NL ? 72 : 36);
    static constexpr int RB_U = MIX ? sb_bytes(WU) : RB;

    // SPLIT2 sub-spans: IQ3_S copies [32*h .. +34) into [0..48) and [66 .. +44) into [48..112);
    // IQ4_XS copies [0 .. 8) into [0..32) and [8+64*h .. +64) into [32..112).
    static constexpr int NP_A = WT == T_IQ3_S ? 3 : 2;
    static constexpr int LEN_A = WT == T_IQ3_S ? 34 : 8;
    static constexpr int OFF_B_SM = NP_A * 16;
    static constexpr int LEN_B = WT == T_IQ3_S ? 44 : 64;

    static constexpr int SLOT = SPLIT2 ? 112
                              : (RB <= 66 && RB_U <= 66) ? 80
                              : (RB <= 82 && RB_U <= 82 && GU) ? 96
                              : 112;
    static constexpr int NP = SLOT / 16;
    static constexpr int NFB = (GU ? 1280 : 2560) / V2_WR;
    static constexpr int ACT_LD = K / 64 * AB;
    static constexpr int OFF_RAW = 2 * V2_ACT;
    static constexpr int RAWBUF = V2_WR * SLOT;
    static constexpr int OFF_SROW = OFF_RAW + 2 * RAWBUF;
    static constexpr int OFF_GRID = OFF_SROW + V2_NR * 4;
    static constexpr int OFF_GRID_U = OFF_GRID + grid_bytes(WT);
    static constexpr int OFF_MISC = OFF_GRID_U + (MIX ? grid_bytes(WU) : 0);
    static constexpr int SMEM = OFF_MISC + 16;

    static_assert(SLOT % 16 == 0 && NC % CPR == 0, "v2 layout");
    static_assert(SPLIT2 || (SLOT >= 14 + RB && SLOT >= 14 + RB_U), "v2 raw slot");
    static_assert(!SPLIT2 || (NP_A * 16 >= 14 + LEN_A && SLOT - OFF_B_SM >= 14 + LEN_B), "v2 split slot");
    static_assert(V2_WR * NP % kV2Threads == 0 && V2_NR * 12 % kV2Threads == 0, "v2 copy loops");
    static_assert(8 * V2_NR * 4 <= V2_ACT, "H scale exchange fits in a chunk stage");
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC));
}
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
}

__device__ __forceinline__ uint32_t lds32_2a(const uint8_t* sm, uint32_t off) {
    const uint32_t* q = (const uint32_t*) (sm + (off & ~3u));
    return __byte_perm(q[0], q[1], (off & 2u) ? 0x5432u : 0x3210u);
}
__device__ __forceinline__ uint32_t lds16(const uint8_t* sm, uint32_t off) { return *(const uint16_t*) (sm + off); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }

// Four codebook bytes with sign bits `nib` (bit k: byte k negative), safe when a codebook byte is zero.
__device__ __forceinline__ uint32_t signed4_z(uint32_t gw, uint32_t nib) {
    const uint32_t m = ((nib * 0x00204081u) & 0x01010101u) * 0xFFu;
    return __vsub4(gw ^ m, m);
}

__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}

template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else return nullptr;
}

template <int T> __device__ __forceinline__ float unit_d_scale(float d) {
    if constexpr (T == T_IQ2_XXS || T == T_IQ2_XS || T == T_IQ2_S) return d * 0.125f;
    else if constexpr (T == T_IQ3_XXS) return d * 0.25f;
    else return d;
}

// Decode lane t's two 4-byte A-fragment words (lo: 4t..4t+3, hi: 16+4t..16+4t+3) and scales (dw0, dw1).
template <int T>
__device__ __forceinline__ void dec_sub(const uint8_t* sm, uint32_t ro_a, [[maybe_unused]] uint32_t ro_b,
                                        [[maybe_unused]] int ib, [[maybe_unused]] int j, uint32_t t,
                                        [[maybe_unused]] const uint32_t* grid,
                                        [[maybe_unused]] const uint32_t (&kv)[4], [[maybe_unused]] float dd,
                                        uint32_t& lo, uint32_t& hi, float& dw0, float& dw1) {
    if constexpr (T == T_IQ2_XXS) {
        const uint32_t hn = t & 1u, la = t >> 1, lb = 2u + la;
        const uint32_t ia = sm[ro_a + 2 + 8 * ib + la], ib_idx = sm[ro_a + 2 + 8 * ib + lb];
        const uint32_t w1 = lds32_2a(sm, ro_a + 6 + 8 * ib);
        const uint32_t sh = 7u * la;
        const uint32_t sa = (w1 >> sh) & 127u, sb = (w1 >> (sh + 14u)) & 127u;
        lo = signed4_z(grid[2u * ia + hn], ((sa >> (4u * hn)) & 15u) | (((uint32_t) __popc(sa) & hn) << 3));
        hi = signed4_z(grid[2u * ib_idx + hn], ((sb >> (4u * hn)) & 15u) | (((uint32_t) __popc(sb) & hn) << 3));
        dw0 = dw1 = dd * (float) ((w1 >> 27) | 1u);
    } else if constexpr (T == T_IQ2_XS) {
        const uint32_t hn = t & 1u, la = t >> 1, lb = 2u + la;
        const uint32_t ca = lds16(sm, ro_a + 2 + 8 * ib + 2 * la);
        const uint32_t cb = lds16(sm, ro_a + 2 + 8 * ib + 2 * lb);
        const uint32_t sa = (ca >> 9) & 127u, sb = (cb >> 9) & 127u;
        lo = signed4_z(grid[2u * (ca & 511u) + hn], ((sa >> (4u * hn)) & 15u) | (((uint32_t) __popc(sa) & hn) << 3));
        hi = signed4_z(grid[2u * (cb & 511u) + hn], ((sb >> (4u * hn)) & 15u) | (((uint32_t) __popc(sb) & hn) << 3));
        const uint32_t sc = sm[ro_a + 66 + ib];
        dw0 = dd * (float) (2u * (sc & 15u) + 1u);
        dw1 = dd * (float) (2u * (sc >> 4) + 1u);
    } else if constexpr (T == T_IQ2_S) {
        const uint32_t hn = t & 1u, la = t >> 1, lb = 2u + la;
        const uint32_t qh = sm[ro_a + 66 + ib], sc = sm[ro_a + 74 + ib];
        const uint32_t ia = (uint32_t) sm[ro_a + 2 + 4 * ib + la] | (((qh >> (2u * la)) & 3u) << 8);
        const uint32_t ib_idx = (uint32_t) sm[ro_a + 2 + 4 * ib + lb] | (((qh >> (2u * lb)) & 3u) << 8);
        const uint32_t sa = sm[ro_a + 34 + 4 * ib + la], sb = sm[ro_a + 34 + 4 * ib + lb];
        lo = signed4_z(grid[2u * ia + hn], (sa >> (4u * hn)) & 15u);
        hi = signed4_z(grid[2u * ib_idx + hn], (sb >> (4u * hn)) & 15u);
        dw0 = dd * (float) (2u * (sc & 15u) + 1u);
        dw1 = dd * (float) (2u * (sc >> 4) + 1u);
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t qa = sm[ro_a + 2 + 8 * ib + t], qb = sm[ro_a + 6 + 8 * ib + t];
        const uint32_t w2 = lds32_2a(sm, ro_a + 66 + 4 * ib);
        const uint32_t hn = t & 1u, sh = 7u * (t >> 1);
        const uint32_t sa = (w2 >> sh) & 127u, sb = (w2 >> (sh + 14u)) & 127u;
        lo = signed4_z(grid[qa], ((sa >> (4u * hn)) & 15u) | (((uint32_t) __popc(sa) & hn) << 3));
        hi = signed4_z(grid[qb], ((sb >> (4u * hn)) & 15u) | (((uint32_t) __popc(sb) & hn) << 3));
        dw0 = dw1 = dd * (float) (2u * (w2 >> 28) + 1u);
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t qh = sm[ro_b + ib];
        const uint32_t ia = (uint32_t) sm[ro_a + 2 + 8 * j + t] | (((qh >> t) & 1u) << 8);
        const uint32_t ib_idx = (uint32_t) sm[ro_a + 6 + 8 * j + t] | (((qh >> (4u + t)) & 1u) << 8);
        const uint32_t w2 = lds32_2a(sm, ro_b + 8 + 4 * ib);
        lo = signed4_z(grid[ia], (w2 >> (4u * t)) & 15u);
        hi = signed4_z(grid[ib_idx], (w2 >> (16u + 4u * t)) & 15u);
        const uint32_t sc = (sm[ro_b + 40 + (ib >> 1)] >> (4 * (ib & 1))) & 15u;
        dw0 = dw1 = dd * (float) (1u + 2u * sc);
    } else if constexpr (T == T_IQ4_XS) {
        table16(lds32_2a(sm, ro_b + 16 * j + 4 * t), kv, lo, hi);
        const uint32_t ls = ((sm[ro_a + 4 + (ib >> 1)] >> (4 * (ib & 1))) & 15u) |
                            (((lds16(sm, ro_a + 2) >> (2 * ib)) & 3u) << 4);
        dw0 = dw1 = dd * (float) ((int) ls - 32);
    } else if constexpr (T == T_IQ4_NL) {
        const uint32_t bo = ro_a + 18u * (uint32_t) j;
        table16(lds32_2a(sm, bo + 2u + 4u * t), kv, lo, hi);
        dw0 = dw1 = half_at(lds16(sm, bo));
    } else {   // T_Q2_0
        const uint32_t bo = ro_a + 18u * (uint32_t) (j >> 1);
        const uint32_t sub = (uint32_t) (j & 1);
        const uint32_t qa = sm[bo + 2u + 8u * sub + t], qb = sm[bo + 6u + 8u * sub + t];
        const uint32_t ua = (((qa & 0x33u) * 0x00001001u) & 0x00030003u) |
                            ((((qa >> 2) & 0x33u) * 0x00100100u) & 0x03000300u);
        const uint32_t ub = (((qb & 0x33u) * 0x00001001u) & 0x00030003u) |
                            ((((qb >> 2) & 0x33u) * 0x00100100u) & 0x03000300u);
        lo = __vsub4(ua, 0x01010101u);
        hi = __vsub4(ub, 0x01010101u);
        dw0 = dw1 = half_at(lds16(sm, bo));
    }
}
#endif

template <int WT, bool GU, int WU = WT>
__device__ __forceinline__ void v2_stage(int tid, const Batch& b, const NativeGeom& geo, const Tables& tb,
                                         const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
                                         uint8_t* __restrict__ out, float* __restrict__ dm, int* __restrict__ ctr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = V2<WT, GU, WU>;
    extern __shared__ __align__(16) uint8_t smem[];
    int* srow = (int*) (smem + C::OFF_SROW);
    int* misc = (int*) (smem + C::OFF_MISC);
    [[maybe_unused]] const uint32_t* sgrid = (const uint32_t*) (smem + C::OFF_GRID);
    [[maybe_unused]] const uint32_t* sgrid_u = (const uint32_t*) (smem + (C::MIX ? C::OFF_GRID_U : C::OFF_GRID));
    const int lane = tid & 31, warp = tid >> 5, g = lane >> 2;
    const uint32_t t = lane & 3;

    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += kV2Threads) ((uint32_t*) (smem + C::OFF_GRID))[i] = gs[i];
    }
    if constexpr (C::MIX && grid_bytes(WU) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WU>();
        for (int i = tid; i < grid_bytes(WU) / 4; i += kV2Threads) ((uint32_t*) (smem + C::OFF_GRID_U))[i] = gs[i];
    }
    [[maybe_unused]] uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL || (C::MIX && WU == T_IQ4_XS)) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }

    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * C::NFB;
    const int b_lane = ((lane & 7) + 8 * (lane >> 4)) * V2_CSTR + 16 * ((lane >> 3) & 1);
    auto row_ptr = [&](const uint8_t* bl, int f, int r) -> const uint8_t* {
        if constexpr (GU) {
            const bool up = (r & 8) != 0;
            const size_t rstride = (C::MIX && up) ? geo.up_row : geo.gu_row;
            return bl + (up ? geo.up_off : 0) + (size_t) (128 * f + 8 * (r >> 4) + (r & 7)) * rstride;
        } else {
            return bl + geo.down_off + (size_t) (V2_WR * f + r) * geo.d_row;
        }
    };
    auto prefetch_unit = [&](const uint8_t* p, int r, int u) {
        if constexpr (C::SPLIT2) {
            if ((u & 1) == 0) {
                const uint8_t* sbp = p + (u >> 1) * C::SB;
                pf_l2(sbp);
                if constexpr (C::SB > 112) pf_l2(sbp + 64);
                pf_l2(sbp + C::SB - 1);
            }
        } else {
            const int rb = (C::MIX && (r & 8)) ? C::RB_U : C::RB;
            pf_l2(p + u * rb);
            pf_l2(p + u * rb + rb - 1);
        }
    };

    if (tid == 0) misc[0] = atomicAdd(ctr, 1);
    __syncthreads();
    int w = misc[0];
    while (w < nwork) {
        const int2 tl = tb.tiles[t0 + w / C::NFB];
        const int fb = w % C::NFB, e = tl.x, row0 = tl.y;
        if (((row0 - tb.off[e]) & (V2_NR - 1)) != 0) {
            __syncthreads();
            if (tid == 0) misc[0] = atomicAdd(ctr, 1);
            __syncthreads();
            w = misc[0];
            continue;
        }
        const int nrows = min(V2_NR, tb.off[e + 1] - row0), npairs = (nrows + 15) >> 4;
        const uint8_t* blob = b.blob[e - b.e0];
        int nxt = 0;
        if (tid == 0) nxt = atomicAdd(ctr, 1);
        if (tid < V2_NR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : 0;
        uint32_t lo4[2][2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
                lo4[i][hh] = (uint32_t) (uintptr_t) row_ptr(blob, fb, 32 * warp + 16 * i + 8 * hh + g) & 15u;
        const uint8_t* myrow = row_ptr(blob, fb, tid);
        __syncthreads();

        auto load_raw = [&](int u) {
            const int dst = C::OFF_RAW + (u & 1) * C::RAWBUF;
#pragma unroll
            for (int k = 0; k < V2_WR * C::NP / kV2Threads; ++k) {
                const int idx = tid + k * kV2Threads, r = idx / C::NP, p = idx - r * C::NP;
                if constexpr (C::SPLIT2) {
                    const uint8_t* sbp = row_ptr(blob, fb, r) + (u >> 1) * C::SB;
                    const int h = u & 1;
                    if (p < C::NP_A) {
                        const uint8_t* a = WT == T_IQ3_S ? (sbp + 32 * h) : sbp;
                        const uint32_t mis = (uint32_t) (uintptr_t) a & 15u;
                        if (p < (int) ((mis + C::LEN_A + 15) >> 4))
                            cp16(smem + dst + r * C::SLOT + 16 * p, a - mis + 16 * p);
                    } else {
                        const int pb = p - C::NP_A;
                        const uint8_t* a = WT == T_IQ3_S ? (sbp + 66) : (sbp + 8 + 64 * h);
                        const uint32_t mis = (uint32_t) (uintptr_t) a & 15u;
                        if (pb < (int) ((mis + C::LEN_B + 15) >> 4))
                            cp16(smem + dst + r * C::SLOT + C::OFF_B_SM + 16 * pb, a - mis + 16 * pb);
                    }
                } else {
                    const int rb = (C::MIX && (r & 8)) ? C::RB_U : C::RB;
                    const uint8_t* a = row_ptr(blob, fb, r) + u * rb;
                    const uint32_t mis = (uint32_t) (uintptr_t) a & 15u;
                    if (p < (int) ((mis + rb + 15) >> 4))
                        cp16(smem + dst + r * C::SLOT + 16 * p, a - mis + 16 * p);
                }
            }
        };
        auto load_act = [&](int c) {
            const int so = (c & 1) * V2_ACT;
#pragma unroll
            for (int k = 0; k < V2_NR * 12 / kV2Threads; ++k) {
                const int idx = tid + k * kV2Threads, r = idx / 12, p = idx - r * 12;
                if (r >= nrows) break;
                const uint8_t* s = act + (size_t) srow[r] * C::ACT_LD + (size_t) (2 * c) * AB;
                if (p < 8) cp16(smem + so + r * V2_CSTR + 16 * p, s + (p >> 2) * AB + 16 * (p & 3));
                else cp4(smem + so + V2_NR * V2_CSTR + 4 * ((p - 8) * V2_NR + r),
                         s + ((p - 8) >> 1) * AB + 64 + 4 * (p & 1));
            }
        };
        load_raw(0);
        load_act(0);
        cp_commit();
#pragma unroll
        for (int u = 1; u <= V2_PF; ++u)
            if (u < C::NU) prefetch_unit(myrow, tid, u);

        float acc[2][16][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int n = 0; n < 16; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        uint32_t pr_a[2][2], pr_b[2][2];
        [[maybe_unused]] float dd[2][2];
#pragma unroll 1
        for (int c = 0; c < C::NC; ++c) {
            cp_wait<0>();
            __syncthreads();
            if (c + 1 < C::NC) {
                load_act(c + 1);
                if ((c + 1) % C::CPR == 0) {
                    const int u = (c + 1) / C::CPR;
                    load_raw(u);
                    if (u + V2_PF < C::NU) prefetch_unit(myrow, tid, u + V2_PF);
                }
            }
            cp_commit();
            if (c == 1) {
                const int wn = misc[1];
                if (wn < nwork) {
                    const int2 tn = tb.tiles[t0 + wn / C::NFB];
                    if (((tn.y - tb.off[tn.x]) & (V2_NR - 1)) == 0) {
                        const uint8_t* nrow = row_ptr(b.blob[tn.x - b.e0], wn % C::NFB, tid);
#pragma unroll
                        for (int u = 0; u < V2_PF; ++u)
                            if (u < C::NU) prefetch_unit(nrow, tid, u);
                    }
                }
            }
            if (c % C::CPR == 0) {
                const int u = c / C::CPR;
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        const uint32_t sbase = C::OFF_RAW + (u & 1) * C::RAWBUF +
                                               (32 * warp + 16 * i + 8 * hh + g) * C::SLOT;
                        if constexpr (C::SPLIT2) {
                            const int sb = u >> 1;
                            const uint32_t mis_sb = (lo4[i][hh] + (uint32_t) (sb * C::SB)) & 15u;
                            pr_a[i][hh] = sbase + mis_sb;
                            pr_b[i][hh] = sbase + C::OFF_B_SM + ((mis_sb + (WT == T_IQ3_S ? 2u : 8u)) & 15u);
                            if (WT == T_IQ4_XS || (u & 1) == 0)
                                dd[i][hh] = unit_d_scale<WT>(half_at(lds16(smem, pr_a[i][hh])));
                        } else {
                            const int rb = (C::MIX && hh == 1) ? C::RB_U : C::RB;
                            pr_a[i][hh] = sbase + ((lo4[i][hh] + (uint32_t) (u * rb)) & 15u);
                            pr_b[i][hh] = 0;
                            if constexpr (GU) {
                                const float d = half_at(lds16(smem, pr_a[i][hh]));
                                dd[i][hh] = (C::MIX && hh == 1) ? unit_d_scale<WU>(d) : unit_d_scale<WT>(d);
                            }
                        }
                    }
            }
            const int so = (c & 1) * V2_ACT;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t a[2][4];
                float dw0[2][2], nm0[2][2];
                [[maybe_unused]] float dw1[2][2], nm1[2][2];
                const int ib = 4 * (c & 1) + j;
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        uint32_t lo, hi;
                        float s0, s1;
                        if constexpr (C::MIX) {
                            if (hh == 0)
                                dec_sub<WT>(smem, pr_a[i][0], pr_b[i][0], ib, j, t, sgrid, kv, dd[i][0], lo, hi, s0, s1);
                            else
                                dec_sub<WU>(smem, pr_a[i][1], pr_b[i][1], ib, j, t, sgrid_u, kv, dd[i][1], lo, hi, s0, s1);
                        } else {
                            dec_sub<WT>(smem, pr_a[i][hh], pr_b[i][hh], ib, j, t, sgrid, kv, dd[i][hh], lo, hi, s0, s1);
                        }
                        a[i][hh] = lo;
                        a[i][2 + hh] = hi;
                        dw0[i][hh] = s0;
                        nm0[i][hh] = -MAGICF * s0;
                        if constexpr (C::K16) {
                            dw1[i][hh] = s1;
                            nm1[i][hh] = -MAGICF * s1;
                        }
                    }
                const float* sdx = (const float*) (smem + so + V2_NR * V2_CSTR) + j * V2_NR;
#pragma unroll
                for (int np = 0; np < 8; ++np) {
                    if (np >= npairs) break;
                    uint32_t bq[4];
                    ldsm4(bq, smem + so + b_lane + np * 16 * V2_CSTR + 32 * j);
#pragma unroll
                    for (int nn = 0; nn < 2; ++nn) {
                        const float2 dx = *(const float2*) (sdx + 16 * np + 8 * nn + 2 * t);
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            float* ac = acc[i][2 * np + nn];
                            if constexpr (C::K16) {
                                int d0[4], d1[4];
                                mma16(d0, a[i][0], a[i][1], bq[2 * nn]);
                                mma16(d1, a[i][2], a[i][3], bq[2 * nn + 1]);
                                const float v0 = __fmaf_rn(__int_as_float(d0[0]), dw0[i][0], nm0[i][0]) +
                                                 __fmaf_rn(__int_as_float(d1[0]), dw1[i][0], nm1[i][0]);
                                const float v1 = __fmaf_rn(__int_as_float(d0[1]), dw0[i][0], nm0[i][0]) +
                                                 __fmaf_rn(__int_as_float(d1[1]), dw1[i][0], nm1[i][0]);
                                const float v2 = __fmaf_rn(__int_as_float(d0[2]), dw0[i][1], nm0[i][1]) +
                                                 __fmaf_rn(__int_as_float(d1[2]), dw1[i][1], nm1[i][1]);
                                const float v3 = __fmaf_rn(__int_as_float(d0[3]), dw0[i][1], nm0[i][1]) +
                                                 __fmaf_rn(__int_as_float(d1[3]), dw1[i][1], nm1[i][1]);
                                ac[0] = __fmaf_rn(v0, dx.x, ac[0]);
                                ac[1] = __fmaf_rn(v1, dx.y, ac[1]);
                                ac[2] = __fmaf_rn(v2, dx.x, ac[2]);
                                ac[3] = __fmaf_rn(v3, dx.y, ac[3]);
                            } else {
                                int d[4];
                                mma32(d, a[i], bq[2 * nn], bq[2 * nn + 1]);
                                ac[0] = __fmaf_rn(__fmaf_rn(__int_as_float(d[0]), dw0[i][0], nm0[i][0]), dx.x, ac[0]);
                                ac[1] = __fmaf_rn(__fmaf_rn(__int_as_float(d[1]), dw0[i][0], nm0[i][0]), dx.y, ac[1]);
                                ac[2] = __fmaf_rn(__fmaf_rn(__int_as_float(d[2]), dw0[i][1], nm0[i][1]), dx.x, ac[2]);
                                ac[3] = __fmaf_rn(__fmaf_rn(__int_as_float(d[3]), dw0[i][1], nm0[i][1]), dx.y, ac[3]);
                            }
                        }
                    }
                }
            }
            if (c == 0 && tid == 0) misc[1] = nxt;
        }

        if constexpr (GU) {
            float* sam = (float*) (smem + (C::NC & 1) * V2_ACT);
#pragma unroll
            for (int n = 0; n < 16; ++n)
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    float am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const float gt = acc[i][n][cc], up = acc[i][n][2 + cc];
                        const float h = gt / (1.0f + __expf(-gt)) * up;
                        acc[i][n][cc] = h;
                        am = fmaxf(am, fabsf(h));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    acc[0][n][2 + cc] = am;
                    if (g == 0) sam[warp * V2_NR + 8 * n + 2 * t + cc] = am;
                }
            __syncthreads();
            const int blk = 4 * fb + (warp >> 1), hh = blk & 1;
#pragma unroll
            for (int n = 0; n < 16; ++n)
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int r = 8 * n + 2 * (int) t + cc;
                    const float am = fmaxf(acc[0][n][2 + cc], sam[(warp ^ 1) * V2_NR + r]);
                    if (r < nrows) {
                        const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
#pragma unroll
                        for (int i = 0; i < 2; ++i)
                            o[32 * hh + 16 * (warp & 1) + 8 * i + g] =
                                (uint8_t) (int8_t) __float2int_rn(acc[i][n][cc] * inv);
                        if ((warp & 1) == 0 && g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
        } else {
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int n = 0; n < 16; ++n)
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = 8 * n + 2 * (int) t + (q & 1);
                        if (r < nrows)
                            dm[(size_t) (row0 + r) * 2560 + V2_WR * fb + 32 * warp + 16 * i + 8 * (q >> 1) + g] =
                                acc[i][n][q];
                    }
        }
        __syncthreads();
        w = misc[1];
    }
#else
    (void) tid; (void) b; (void) geo; (void) tb; (void) act; (void) src; (void) out; (void) dm; (void) ctr;
#endif
}

}  // namespace v2_detail
}  // namespace basalt::prefill::fused
