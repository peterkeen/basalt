// src/kernels/cpu/kq_avx512.cpp - AVX-512 VNNI multi-token dot products for K-quant and legacy expert
// formats (Q4_K, Q5_K, Q6_K, Q5_1, Q8_0), bit for bit equal to ggml's single-token dots.
//
// K-quant blocks (256 weights) unpack their weights and sub-block scales into 512-bit vectors once per block and run
// 512-bit VNNI dots across the active tokens; at one token, four weight rows share each activation load.  Q5_1 and
// Q8_0 blocks (32 weights) pack two weight rows into the low and high 256-bit halves of each 512-bit vector and run
// 512-bit VNNI byte dots across row pairs and tokens.  Every lane's integer sum and float accumulation match ggml's
// single-token dot in order, so outputs are bit-identical regardless of token count or row-slice boundaries.
#include "basalt/kernels/cpu/kq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

constexpr size_t prefetch_ahead = 2048;

inline uint16_t u16(const void* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

inline uint32_t u32(const void* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}

inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

inline float swiglu_eval(float g, float u, float sg, float su, bool scaled, float limit) {
    if (scaled) {
        g *= sg;
        u *= su;
    }
    if (limit > 0.f) {
        g = std::fmin(g, limit);
        u = std::fmax(std::fmin(u, limit), -limit);
    }
    return (g / (1.f + std::exp(-g))) * u;
}

// ---- Q4_K / Q5_K / Q6_K x Q8_K ----

struct KBlockUnpack {
    __m512i q[4];
    __m512i sc[4];
    __m128i mins;
    float xd;
    float xdmin;
};

inline void unpack_q4k_scales(const uint8_t* scales12, __m128i& mins, __m512i sc[4]) {
    constexpr uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
    uint32_t utmp[4] = {};
    std::memcpy(utmp, scales12, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const __m256i ms = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
    mins = _mm256_extracti128_si256(ms, 1);
    const __m128i sc16 = _mm256_castsi256_si128(ms);
    const __m256i sc8 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi16(sc16, sc16)),
                                                _mm_unpackhi_epi16(sc16, sc16), 1);
    const __m512i sc512 = _mm512_castsi256_si512(sc8);
    sc[0] = _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1), sc512);
    sc[1] = _mm512_permutexvar_epi32(_mm512_setr_epi32(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3), sc512);
    sc[2] = _mm512_permutexvar_epi32(_mm512_setr_epi32(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5), sc512);
    sc[3] = _mm512_permutexvar_epi32(_mm512_setr_epi32(6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7), sc512);
}

template <int T>
inline void unpack_k_block(const uint8_t* raw, KBlockUnpack& b) {
    const __m512i m4 = _mm512_set1_epi8(0x0f);
    if constexpr (T == 12) {
        const auto* x = (const block_q4_K*) raw;
        b.xd = half_at(x, 0);
        b.xdmin = half_at(x, 1);
        unpack_q4k_scales(x->scales, b.mins, b.sc);
        for (int j = 0; j < 4; ++j) {
            const __m512i v = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (x->qs + 32 * j)));
            b.q[j] = _mm512_and_si512(_mm512_mask_srli_epi16(v, 0xFFFF0000u, v, 4), m4);
        }
    } else if constexpr (T == 13) {
        const auto* x = (const block_q5_K*) raw;
        b.xd = half_at(x, 0);
        b.xdmin = half_at(x, 1);
        unpack_q4k_scales(x->scales, b.mins, b.sc);
        const __m512i hb = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) x->qh));
        const __m512i m10 = _mm512_set1_epi8(0x10);
        const __m512i sh01 = _mm512_mask_set1_epi16(_mm512_set1_epi16(4), 0xFFFF0000u, 3);
        const __m512i sh23 = _mm512_mask_set1_epi16(_mm512_set1_epi16(2), 0xFFFF0000u, 1);
        const __m512i sh67 = _mm512_mask_set1_epi16(_mm512_set1_epi16(2), 0xFFFF0000u, 3);
        const __m512i h[4] = {
            _mm512_and_si512(_mm512_sllv_epi16(hb, sh01), m10),
            _mm512_and_si512(_mm512_sllv_epi16(hb, sh23), m10),
            _mm512_and_si512(_mm512_mask_srli_epi16(hb, 0xFFFF0000u, hb, 1), m10),
            _mm512_and_si512(_mm512_srlv_epi16(hb, sh67), m10),
        };
        for (int j = 0; j < 4; ++j) {
            const __m512i v = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (x->qs + 32 * j)));
            b.q[j] = _mm512_or_si512(_mm512_and_si512(_mm512_mask_srli_epi16(v, 0xFFFF0000u, v, 4), m4), h[j]);
        }
    } else if constexpr (T == 14) {
        const auto* x = (const block_q6_K*) raw;
        b.xd = h2f(x->d);
        b.xdmin = 0.f;
        b.mins = _mm_setzero_si128();
        const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) x->scales));
        const __m512i sc32 = _mm512_mullo_epi32(_mm512_cvtepu16_epi32(sc16), _mm512_set1_epi32(0x00010001));
        b.sc[0] = _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3), sc32);
        b.sc[1] = _mm512_permutexvar_epi32(_mm512_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7), sc32);
        b.sc[2] = _mm512_permutexvar_epi32(_mm512_setr_epi32(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11), sc32);
        b.sc[3] = _mm512_permutexvar_epi32(_mm512_setr_epi32(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15), sc32);
        const __m512i m30 = _mm512_set1_epi8(0x30);
        const __m512i sh_lo = _mm512_mask_set1_epi16(_mm512_set1_epi16(4), 0xFFFF0000u, 2);
        for (int j = 0; j < 2; ++j) {
            const __m512i ql = _mm512_loadu_si512(x->ql + 64 * j);
            const __m512i qh = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (x->qh + 32 * j)));
            const __m512i h01 = _mm512_and_si512(_mm512_sllv_epi16(qh, sh_lo), m30);
            const __m512i h23 = _mm512_and_si512(_mm512_mask_srli_epi16(qh, 0xFFFF0000u, qh, 2), m30);
            b.q[2 * j + 0] = _mm512_or_si512(_mm512_and_si512(ql, m4), h01);
            b.q[2 * j + 1] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(ql, 4), m4), h23);
        }
    }
}

template <int T>
constexpr size_t k_block_bytes() {
    if constexpr (T == 12) return sizeof(block_q4_K);
    if constexpr (T == 13) return sizeof(block_q5_K);
    return sizeof(block_q6_K);
}

// R contiguous weight rows starting at row0 against NT tokens: R > 1 with NT == 1 hoists activation loads across the
// R rows; R == 1 with NT >= 1 hoists weight unpacking across the NT tokens.
template <int T, int R, int NT>
inline void k_rows_pass(const uint8_t* row0, size_t row_bytes, int nb, const block_q8_K* const* y, float out_s[R][NT],
                        const uint8_t*& pf) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i m32s = _mm512_set1_epi8(32);
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc[R][NT];
    __m128 acc_m[R][NT];
    float summs[R][NT];
    for (int m = 0; m < R; ++m)
        for (int t = 0; t < NT; ++t) {
            acc[m][t] = _mm256_setzero_ps();
            acc_m[m][t] = _mm_setzero_ps();
            summs[m][t] = 0.f;
        }
    constexpr size_t bsz = k_block_bytes<T>();
    const size_t pf_ahead = R > 1 ? (size_t) R * row_bytes : prefetch_ahead;
    if (pf < row0) pf = (const uint8_t*) ((uintptr_t) row0 & ~(uintptr_t) 63);
    if constexpr (NT == 1) {
        for (int i = 0; i < nb; ++i) {
            for (const uint8_t* lim = row0 + (size_t) (i + 1) * R * bsz + pf_ahead; pf < lim; pf += 64)
                _mm_prefetch((const char*) pf, _MM_HINT_T0);
            const block_q8_K* yi = y[0] + i;
            const float yd = yi->d;
            const __m512i q8_0 = _mm512_loadu_si512(yi->qs + 0);
            const __m512i q8_1 = _mm512_loadu_si512(yi->qs + 64);
            const __m512i q8_2 = _mm512_loadu_si512(yi->qs + 128);
            const __m512i q8_3 = _mm512_loadu_si512(yi->qs + 192);
            __m128i q8s = _mm_setzero_si128();
            __m512i q8sub0, q8sub1, q8sub2, q8sub3;
            if constexpr (T == 12 || T == 13) {
                const __m256i bs = _mm256_loadu_si256((const __m256i*) yi->bsums);
                q8s = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256(bs, 1));
            } else {
                q8sub0 = _mm512_maddubs_epi16(m32s, q8_0);
                q8sub1 = _mm512_maddubs_epi16(m32s, q8_1);
                q8sub2 = _mm512_maddubs_epi16(m32s, q8_2);
                q8sub3 = _mm512_maddubs_epi16(m32s, q8_3);
            }
            for (int m = 0; m < R; ++m) {
                KBlockUnpack b;
                unpack_k_block<T>(row0 + (size_t) m * row_bytes + (size_t) i * bsz, b);
                if constexpr (T == 12) {
                    const __m128i prod = _mm_madd_epi16(b.mins, q8s);
                    acc_m[m][0] = _mm_fmadd_ps(_mm_set1_ps(-yd * b.xdmin), _mm_cvtepi32_ps(prod), acc_m[m][0]);
                } else if constexpr (T == 13) {
                    const __m128i prod = _mm_madd_epi16(b.mins, q8s);
                    const __m128i hs = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
                    summs[m][0] += (-yd * b.xdmin) * _mm_extract_epi32(hs, 0);
                }
                __m512i p0 = _mm512_maddubs_epi16(b.q[0], q8_0);
                __m512i p1 = _mm512_maddubs_epi16(b.q[1], q8_1);
                __m512i p2 = _mm512_maddubs_epi16(b.q[2], q8_2);
                __m512i p3 = _mm512_maddubs_epi16(b.q[3], q8_3);
                if constexpr (T == 14) {
                    p0 = _mm512_sub_epi16(p0, q8sub0);
                    p1 = _mm512_sub_epi16(p1, q8sub1);
                    p2 = _mm512_sub_epi16(p2, q8sub2);
                    p3 = _mm512_sub_epi16(p3, q8sub3);
                }
                const __m512i s01 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(zero, b.sc[0], p0), b.sc[1], p1);
                const __m512i s23 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(zero, b.sc[2], p2), b.sc[3], p3);
                const __m512i s512 = _mm512_add_epi32(s01, s23);
                const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(s512), _mm512_extracti64x4_epi64(s512, 1));
                acc[m][0] = _mm256_fmadd_ps(_mm256_set1_ps(yd * b.xd), _mm256_cvtepi32_ps(sumi), acc[m][0]);
            }
        }
    } else {
        for (int i = 0; i < nb; ++i) {
            for (const uint8_t* lim = row0 + (size_t) (i + 1) * R * bsz + pf_ahead; pf < lim; pf += 64)
                _mm_prefetch((const char*) pf, _MM_HINT_T0);
            for (int m = 0; m < R; ++m) {
                KBlockUnpack b;
                unpack_k_block<T>(row0 + (size_t) m * row_bytes + (size_t) i * bsz, b);
                for (int t = 0; t < NT; ++t) {
                    const block_q8_K* yi = y[t] + i;
                    const float yd = yi->d;
                    if constexpr (T == 12) {
                        const __m256i bs = _mm256_loadu_si256((const __m256i*) yi->bsums);
                        const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256(bs, 1));
                        const __m128i prod = _mm_madd_epi16(b.mins, q8s);
                        acc_m[m][t] = _mm_fmadd_ps(_mm_set1_ps(-yd * b.xdmin), _mm_cvtepi32_ps(prod), acc_m[m][t]);
                    } else if constexpr (T == 13) {
                        const __m256i bs = _mm256_loadu_si256((const __m256i*) yi->bsums);
                        const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256(bs, 1));
                        const __m128i prod = _mm_madd_epi16(b.mins, q8s);
                        const __m128i hs = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
                        summs[m][t] += (-yd * b.xdmin) * _mm_extract_epi32(hs, 0);
                    }
                    const __m512i q8_0 = _mm512_loadu_si512(yi->qs + 0);
                    const __m512i q8_1 = _mm512_loadu_si512(yi->qs + 64);
                    const __m512i q8_2 = _mm512_loadu_si512(yi->qs + 128);
                    const __m512i q8_3 = _mm512_loadu_si512(yi->qs + 192);
                    __m512i p0 = _mm512_maddubs_epi16(b.q[0], q8_0);
                    __m512i p1 = _mm512_maddubs_epi16(b.q[1], q8_1);
                    __m512i p2 = _mm512_maddubs_epi16(b.q[2], q8_2);
                    __m512i p3 = _mm512_maddubs_epi16(b.q[3], q8_3);
                    if constexpr (T == 14) {
                        p0 = _mm512_sub_epi16(p0, _mm512_maddubs_epi16(m32s, q8_0));
                        p1 = _mm512_sub_epi16(p1, _mm512_maddubs_epi16(m32s, q8_1));
                        p2 = _mm512_sub_epi16(p2, _mm512_maddubs_epi16(m32s, q8_2));
                        p3 = _mm512_sub_epi16(p3, _mm512_maddubs_epi16(m32s, q8_3));
                    }
                    const __m512i s01 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(zero, b.sc[0], p0), b.sc[1], p1);
                    const __m512i s23 = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(zero, b.sc[2], p2), b.sc[3], p3);
                    const __m512i s512 = _mm512_add_epi32(s01, s23);
                    const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(s512), _mm512_extracti64x4_epi64(s512, 1));
                    acc[m][t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * b.xd), _mm256_cvtepi32_ps(sumi), acc[m][t]);
                }
            }
        }
    }
    for (int m = 0; m < R; ++m)
        for (int t = 0; t < NT; ++t) {
            if constexpr (T == 12) {
                __m128 mv = acc_m[m][t];
                mv = _mm_add_ps(mv, _mm_movehl_ps(mv, mv));
                mv = _mm_add_ss(mv, _mm_movehdup_ps(mv));
                out_s[m][t] = hsum_float_8(acc[m][t]) + _mm_cvtss_f32(mv);
            } else if constexpr (T == 13) {
                out_s[m][t] = hsum_float_8(acc[m][t]) + summs[m][t];
            } else {
                out_s[m][t] = hsum_float_8(acc[m][t]);
            }
        }
}

template <int T, int NT>
void k_dot_rows_tok(const uint8_t* w, size_t row_bytes, int nb, float sd, bool scaled, const block_q8_K* const* y,
                    float* const* out, int r0, int r1) {
    const size_t pf_init = NT == 1 ? 4 * row_bytes : prefetch_ahead;
    const uint8_t* pf = (const uint8_t*) ((uintptr_t) (w + (size_t) r0 * row_bytes + pf_init) & ~(uintptr_t) 63);
    int r = r0;
    if constexpr (NT == 1) {
        for (; r + 4 <= r1; r += 4) {
            float s[4][1];
            k_rows_pass<T, 4, 1>(w + (size_t) r * row_bytes, row_bytes, nb, y, s, pf);
            for (int m = 0; m < 4; ++m) out[0][r + m] = scaled ? s[m][0] * sd : s[m][0];
        }
    }
    for (; r < r1; ++r) {
        float s[1][NT];
        k_rows_pass<T, 1, NT>(w + (size_t) r * row_bytes, row_bytes, nb, y, s, pf);
        for (int t = 0; t < NT; ++t) out[t][r] = scaled ? s[0][t] * sd : s[0][t];
    }
}

template <int T, int NT>
void k_gu_rows_tok(const uint8_t* blob, size_t gu_row, size_t up_off, int nb, float sg, float su, bool scaled,
                   float limit, const block_q8_K* const* y, float* const* ff, int r0, int r1) {
    // Walk contiguous gate rows, then contiguous up rows, so each projection streams sequentially from DRAM instead
    // of alternating across `up_off` every block.
    constexpr int kTile = 32;
    float ub[NT][kTile];
    float* up[NT];
    for (int t = 0; t < NT; ++t) up[t] = ub[t];
    for (int a = r0; a < r1; a += kTile) {
        const int b = std::min(a + kTile, r1);
        k_dot_rows_tok<T, NT>(blob, gu_row, nb, 1.f, false, y, ff, a, b);
        k_dot_rows_tok<T, NT>(blob + up_off + (size_t) a * gu_row, gu_row, nb, 1.f, false, y, up, 0, b - a);
        for (int r = a; r < b; ++r)
            for (int t = 0; t < NT; ++t)
                ff[t][r] = swiglu_eval(ff[t][r], ub[t][r - a], sg, su, scaled, limit);
    }
}

// ---- Q5_1 x Q8_1 and Q8_0 x Q8_0 (two rows per 512-bit vector) ----

template <int RP, int NT, bool TAIL>
inline void q5_1_pass(const uint8_t* row0, size_t row_bytes, int nb, float sd, bool scaled, const block_q8_1* const* y,
                      float* const* out, int r, const uint8_t*& pf) {
    const __m512i m4 = _mm512_set1_epi8(0x0f);
    const __m512i dup = _mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3);
    const __m512i dx_perm = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2);
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc[RP][NT];
    float summs[RP][2][NT];
    for (int p = 0; p < RP; ++p)
        for (int t = 0; t < NT; ++t) {
            acc[p][t] = _mm512_setzero_ps();
            summs[p][0][t] = 0.f;
            summs[p][1][t] = 0.f;
        }
    constexpr int rows_step = TAIL ? 1 : (2 * RP);
    constexpr size_t pf_ahead = (size_t) rows_step * 768 > prefetch_ahead ? (size_t) rows_step * 768 : prefetch_ahead;
    for (int ib = 0; ib < nb; ++ib) {
        for (const uint8_t* lim = row0 + (size_t) (ib + 1) * rows_step * sizeof(block_q5_1) + pf_ahead;
             pf < lim; pf += 64)
            _mm_prefetch((const char*) pf, _MM_HINT_T0);
        __m512i qy[NT];
        __m512 dyv[NT];
        float ys[NT];
        for (int t = 0; t < NT; ++t) {
            const block_q8_1* yi = y[t] + ib;
            qy[t] = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) yi->qs));
            const __m128 ds = _mm_cvtph_ps(_mm_cvtsi32_si128((int) u32(yi)));
            dyv[t] = _mm512_broadcastss_ps(ds);
            ys[t] = _mm_cvtss_f32(_mm_shuffle_ps(ds, ds, 0x55));
        }
        for (int p = 0; p < RP; ++p) {
            const auto* b0 = (const block_q5_1*) (row0 + (size_t) (2 * p) * row_bytes) + ib;
            const auto* b1 = TAIL ? b0 : (const block_q5_1*) ((const uint8_t*) b0 + row_bytes);
            const __m256i q01 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) b0->qs)),
                                                        _mm_loadu_si128((const __m128i*) b1->qs), 1);
            const __m512i bits = _mm512_permutexvar_epi64(dup, _mm512_castsi256_si512(q01));
            const __m512i ql = _mm512_and_si512(_mm512_mask_srli_epi16(bits, 0xFF00FF00u, bits, 4), m4);
            const __mmask64 hm = (__mmask64) u32(b0->qh) | ((__mmask64) u32(b1->qh) << 32);
            const __m512i qx = _mm512_or_si512(ql, _mm512_maskz_set1_epi8(hm, 0x10));
            const __m128 dm = _mm_cvtph_ps(_mm_cvtsi64_si128((int64_t) ((uint64_t) u32(b0) | ((uint64_t) u32(b1) << 32))));
            const __m512 dx = _mm512_permutexvar_ps(dx_perm, _mm512_castps128_ps512(dm));
            const float xm0 = _mm_cvtss_f32(_mm_shuffle_ps(dm, dm, 0x55));
            const float xm1 = _mm_cvtss_f32(_mm_shuffle_ps(dm, dm, 0xFF));
            for (int t = 0; t < NT; ++t) {
                summs[p][0][t] += xm0 * ys[t];
                if constexpr (!TAIL) summs[p][1][t] += xm1 * ys[t];
                const __m512 q = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(zero, qx, qy[t]));
                acc[p][t] = _mm512_fmadd_ps(q, _mm512_mul_ps(dx, dyv[t]), acc[p][t]);
            }
        }
    }
    for (int p = 0; p < RP; ++p)
        for (int t = 0; t < NT; ++t) {
            const float v0 = hsum_float_8(_mm512_castps512_ps256(acc[p][t])) + summs[p][0][t];
            out[t][r + 2 * p] = scaled ? v0 * sd : v0;
            if constexpr (!TAIL) {
                const float v1 = hsum_float_8(_mm512_extractf32x8_ps(acc[p][t], 1)) + summs[p][1][t];
                out[t][r + 2 * p + 1] = scaled ? v1 * sd : v1;
            }
        }
}

template <int RP, int NT>
void q5_1_rows_tok(const uint8_t* w, size_t row_bytes, int nb, float sd, bool scaled, const block_q8_1* const* y,
                   float* const* out, int r0, int r1) {
    constexpr size_t pf_init = (size_t) (2 * RP) * 768 > prefetch_ahead ? (size_t) (2 * RP) * 768 : prefetch_ahead;
    const uint8_t* pf = (const uint8_t*) ((uintptr_t) (w + (size_t) r0 * row_bytes + pf_init) & ~(uintptr_t) 63);
    int r = r0;
    for (; r + 2 * RP <= r1; r += 2 * RP)
        q5_1_pass<RP, NT, false>(w + (size_t) r * row_bytes, row_bytes, nb, sd, scaled, y, out, r, pf);
    for (; r + 2 <= r1; r += 2)
        q5_1_pass<1, NT, false>(w + (size_t) r * row_bytes, row_bytes, nb, sd, scaled, y, out, r, pf);
    if (r < r1)
        q5_1_pass<1, NT, true>(w + (size_t) r * row_bytes, row_bytes, nb, sd, scaled, y, out, r, pf);
}

template <int RP, int NT, bool TAIL>
inline void q8_0_pass(const uint8_t* row0, size_t row_bytes, int nb, float sd, bool scaled, const block_q8_0* const* y,
                      float* const* out, int r, const uint8_t*& pf) {
    const __m512i dx_perm = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m512i zero = _mm512_setzero_si512();
    __m512 acc[RP][NT];
    for (int p = 0; p < RP; ++p)
        for (int t = 0; t < NT; ++t) acc[p][t] = _mm512_setzero_ps();
    constexpr int rows_step = TAIL ? 1 : (2 * RP);
    constexpr size_t pf_ahead = (size_t) rows_step * 768 > prefetch_ahead ? (size_t) rows_step * 768 : prefetch_ahead;
    for (int ib = 0; ib < nb; ++ib) {
        for (const uint8_t* lim = row0 + (size_t) (ib + 1) * rows_step * sizeof(block_q8_0) + pf_ahead;
             pf < lim; pf += 64)
            _mm_prefetch((const char*) pf, _MM_HINT_T0);
        __m512i qy[NT];
        __m512 dyv[NT];
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yi = y[t] + ib;
            qy[t] = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) yi->qs));
            dyv[t] = _mm512_set1_ps(h2f(yi->d));
        }
        for (int p = 0; p < RP; ++p) {
            const auto* b0 = (const block_q8_0*) (row0 + (size_t) (2 * p) * row_bytes) + ib;
            const auto* b1 = TAIL ? b0 : (const block_q8_0*) ((const uint8_t*) b0 + row_bytes);
            const __m512i qx = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) b0->qs)),
                                                  _mm256_loadu_si256((const __m256i*) b1->qs), 1);
            const __m512i ax = _mm512_abs_epi8(qx);
            const __mmask64 neg = _mm512_movepi8_mask(qx);
            const __m128 df = _mm_cvtph_ps(_mm_cvtsi32_si128((int) ((uint32_t) u16(b0) | ((uint32_t) u16(b1) << 16))));
            const __m512 dx = _mm512_permutexvar_ps(dx_perm, _mm512_castps128_ps512(df));
            for (int t = 0; t < NT; ++t) {
                const __m512i sy = _mm512_mask_sub_epi8(qy[t], neg, zero, qy[t]);
                const __m512 q = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(zero, ax, sy));
                acc[p][t] = _mm512_fmadd_ps(_mm512_mul_ps(dx, dyv[t]), q, acc[p][t]);
            }
        }
    }
    for (int p = 0; p < RP; ++p)
        for (int t = 0; t < NT; ++t) {
            const float v0 = hsum_float_8(_mm512_castps512_ps256(acc[p][t]));
            out[t][r + 2 * p] = scaled ? v0 * sd : v0;
            if constexpr (!TAIL) {
                const float v1 = hsum_float_8(_mm512_extractf32x8_ps(acc[p][t], 1));
                out[t][r + 2 * p + 1] = scaled ? v1 * sd : v1;
            }
        }
}

template <int RP, int NT>
void q8_0_rows_tok(const uint8_t* w, size_t row_bytes, int nb, float sd, bool scaled, const block_q8_0* const* y,
                   float* const* out, int r0, int r1) {
    constexpr size_t pf_init = (size_t) (2 * RP) * 768 > prefetch_ahead ? (size_t) (2 * RP) * 768 : prefetch_ahead;
    const uint8_t* pf = (const uint8_t*) ((uintptr_t) (w + (size_t) r0 * row_bytes + pf_init) & ~(uintptr_t) 63);
    int r = r0;
    for (; r + 2 * RP <= r1; r += 2 * RP)
        q8_0_pass<RP, NT, false>(w + (size_t) r * row_bytes, row_bytes, nb, sd, scaled, y, out, r, pf);
    for (; r + 2 <= r1; r += 2)
        q8_0_pass<1, NT, false>(w + (size_t) r * row_bytes, row_bytes, nb, sd, scaled, y, out, r, pf);
    if (r < r1)
        q8_0_pass<1, NT, true>(w + (size_t) r * row_bytes, row_bytes, nb, sd, scaled, y, out, r, pf);
}

template <int T>
void dispatch_k_gu(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, bool scaled,
                   float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) {
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = std::min(4, nt - t0);
        const block_q8_K* y[4];
        for (int t = 0; t < k; ++t) y[t] = (const block_q8_K*) act[t0 + t];
        switch (k) {
            case 1: k_gu_rows_tok<T, 1>(blob, gu_row, up_off, nb, sg, su, scaled, limit, y, ff + t0, r0, r1); break;
            case 2: k_gu_rows_tok<T, 2>(blob, gu_row, up_off, nb, sg, su, scaled, limit, y, ff + t0, r0, r1); break;
            case 3: k_gu_rows_tok<T, 3>(blob, gu_row, up_off, nb, sg, su, scaled, limit, y, ff + t0, r0, r1); break;
            default: k_gu_rows_tok<T, 4>(blob, gu_row, up_off, nb, sg, su, scaled, limit, y, ff + t0, r0, r1); break;
        }
    }
}

template <int T>
void dispatch_k_rows(const uint8_t* w, size_t row_bytes, int n, float sd, bool scaled, const void* const* act, int nt,
                     float* const* out, int r0, int r1) {
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = std::min(4, nt - t0);
        const block_q8_K* y[4];
        for (int t = 0; t < k; ++t) y[t] = (const block_q8_K*) act[t0 + t];
        switch (k) {
            case 1: k_dot_rows_tok<T, 1>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
            case 2: k_dot_rows_tok<T, 2>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
            case 3: k_dot_rows_tok<T, 3>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
            default: k_dot_rows_tok<T, 4>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
        }
    }
}

}  // namespace

bool kq512_supported(int type) noexcept {
    return type == 7 || type == 8 || type == 12 || type == 13 || type == 14;
}

bool kq512_fits(int type, int n) noexcept {
    if (n <= 0) return false;
    if (type == 7 || type == 8) return n % 32 == 0;
    if (type == 12 || type == 13 || type == 14) return n % QK_K == 0;
    return false;
}

void kq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su,
                   bool scaled, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) {
    switch (type) {
        case 12: dispatch_k_gu<12>(blob, gu_row, up_off, n, sg, su, scaled, limit, act, nt, ff, r0, r1); return;
        case 13: dispatch_k_gu<13>(blob, gu_row, up_off, n, sg, su, scaled, limit, act, nt, ff, r0, r1); return;
        case 14: dispatch_k_gu<14>(blob, gu_row, up_off, n, sg, su, scaled, limit, act, nt, ff, r0, r1); return;
        case 7:
        case 8: {
            constexpr int kTile = 64, kTok = 4;
            float gb[kTok][kTile], ub[kTok][kTile];
            for (int t0 = 0; t0 < nt; t0 += kTok) {
                const int m = std::min(kTok, nt - t0);
                for (int a = r0; a < r1; a += kTile) {
                    const int b = std::min(r1, a + kTile);
                    float* gp[kTok];
                    float* up[kTok];
                    for (int t = 0; t < m; ++t) {
                        gp[t] = gb[t];
                        up[t] = ub[t];
                    }
                    kq512_rows(type, blob + (size_t) a * gu_row, gu_row, n, 1.f, false, act + t0, m, gp, 0, b - a);
                    kq512_rows(type, blob + up_off + (size_t) a * gu_row, gu_row, n, 1.f, false, act + t0, m, up, 0, b - a);
                    for (int t = 0; t < m; ++t)
                        for (int r = a; r < b; ++r)
                            ff[t0 + t][r] = swiglu_eval(gb[t][r - a], ub[t][r - a], sg, su, scaled, limit);
                }
            }
            return;
        }
        default: return;
    }
}

void kq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, float sd, bool scaled,
                const void* const* act, int nt, float* const* out, int r0, int r1) {
    switch (type) {
        case 12: dispatch_k_rows<12>(w, row_bytes, n, sd, scaled, act, nt, out, r0, r1); return;
        case 13: dispatch_k_rows<13>(w, row_bytes, n, sd, scaled, act, nt, out, r0, r1); return;
        case 14: dispatch_k_rows<14>(w, row_bytes, n, sd, scaled, act, nt, out, r0, r1); return;
        case 7: {
            const int nb = n / QK5_1;
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = std::min(4, nt - t0);
                const block_q8_1* y[4];
                for (int t = 0; t < k; ++t) y[t] = (const block_q8_1*) act[t0 + t];
                switch (k) {
                    case 1: q5_1_rows_tok<4, 1>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
                    case 2: q5_1_rows_tok<2, 2>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
                    case 3: q5_1_rows_tok<2, 3>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
                    default: q5_1_rows_tok<2, 4>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
                }
            }
            return;
        }
        case 8: {
            const int nb = n / QK8_0;
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = std::min(4, nt - t0);
                const block_q8_0* y[4];
                for (int t = 0; t < k; ++t) y[t] = (const block_q8_0*) act[t0 + t];
                switch (k) {
                    case 1: q8_0_rows_tok<4, 1>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
                    case 2: q8_0_rows_tok<2, 2>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
                    case 3: q8_0_rows_tok<2, 3>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
                    default: q8_0_rows_tok<2, 4>(w, row_bytes, nb, sd, scaled, y, out + t0, r0, r1); break;
                }
            }
            return;
        }
        default: return;
    }
}

}  // namespace basalt::kernels::cpu
