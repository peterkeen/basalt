// src/kernels/cpu/expert.cpp - the AVX-512 Q2_0 expert rows and activation quantizer.
//
// See include/basalt/kernels/cpu/expert.hpp first; it says why each piece is shaped this way.
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/log.hpp"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

namespace basalt::kernels::cpu {

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    if (n < 0 || n % QKA != 0 || n > kActMaxWidth) {
        basalt::log::error("basalt", "act_quant_q8_1: width %d; at most %d, in chunks of %d", n, kActMaxWidth, QKA);
        std::abort();
    }
    a.nchunks = n / QKA;
    // Round half away from zero (`t + copysign(0.5, t)`, truncated), matching `lround` without a per-element call or
    // `_mm512_roundscale_ps`'s half-to-even ties.
    const __m512 half = _mm512_set1_ps(0.5f), mhalf = _mm512_set1_ps(-0.5f), zero = _mm512_setzero_ps();
    const __m512i lo = _mm512_set1_epi32(-127), hi = _mm512_set1_epi32(127);
    const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        const __m512 x0 = _mm512_loadu_ps(xb), x1 = _mm512_loadu_ps(xb + 16);
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_and_ps(x0, absmask), _mm512_and_ps(x1, absmask)));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m512 vinv = _mm512_set1_ps(inv);
        const __m512 t0 = _mm512_mul_ps(x0, vinv), t1 = _mm512_mul_ps(x1, vinv);
        const __m512 r0 = _mm512_add_ps(t0, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t0, zero, _CMP_GE_OQ), mhalf, half));
        const __m512 r1 = _mm512_add_ps(t1, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t1, zero, _CMP_GE_OQ), mhalf, half));
        __m512i v0 = _mm512_cvttps_epi32(r0), v1 = _mm512_cvttps_epi32(r1);
        v0 = _mm512_min_epi32(_mm512_max_epi32(v0, lo), hi);
        v1 = _mm512_min_epi32(_mm512_max_epi32(v1, lo), hi);
        _mm_storeu_si128((__m128i*) (a.q + k * QKA), _mm512_cvtepi32_epi8(v0));
        _mm_storeu_si128((__m128i*) (a.q + k * QKA + 16), _mm512_cvtepi32_epi8(v1));
        const int32_t sum = _mm512_reduce_add_epi32(_mm512_add_epi32(v0, v1));
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

// ================================ Q2_0 rows in the GGUF block layout ================================
//
// The IQ model files keep their down projections in Q2_0 GGUF blocks (fp16 d, then 16 code bytes: 18 bytes per
// 64 weights, interleaved), which ggml-cpu computes with a scalar loop on x86.  One unpack (16 code bytes -> 64 codes
// via VPMULTISHIFTQB, argument order (control, data)), one VNNI dot over 64 activations, and one FMA per token.
namespace {

inline __m512i unpack64_q2_0(const uint8_t* codes) {
    const __m128i packed = _mm_loadu_si128((const __m128i*) codes);          // 8 u16 = 64 codes
    const __m512i lanes = _mm512_cvtepu16_epi64(packed);                    // 8 qwords, 8 codes each
    const __m512i ctrl = _mm512_set1_epi64((long long) 0x0E0C0A0806040200ULL);
    return _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, lanes), _mm512_set1_epi8(3));
}

template<int NT>
inline void q2g_row_multi(const uint8_t* row, const ActQ* const* a, const __m512* as, const float* hxs, int nblocks,
                          float* res) {
    __m512 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm512_setzero_ps(); corr[t] = 0.f; }
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        uint16_t dh;
        std::memcpy(&dh, blk, 2);
        const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) dh)));
        const __m512 dv = _mm512_set1_ps(d);
        const __m512i w = unpack64_q2_0(blk + 2);
        for (int t = 0; t < NT; ++t) {
            const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w,
                                                    _mm512_load_si512((const void*) (a[t]->q + b * QK)));
            const __m512 sv = _mm512_mul_ps(dv, as[t * (MAXC / 2) + b]);
            acc[t] = _mm512_fmadd_ps(sv, _mm512_cvtepi32_ps(dot), acc[t]);
            corr[t] += d * hxs[t * (MAXC / 2) + b];
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]) - corr[t];
}

template<int NT>
void q2g_rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    __m512 as[NT * (MAXC / 2)];
    float hxs[NT * (MAXC / 2)];
    for (int t = 0; t < NT; ++t)
        for (int b = 0; b < nblocks; ++b) {
            as[t * (MAXC / 2) + b] = _mm512_permutexvar_ps(base, _mm512_castps128_ps512(_mm_loadl_pi(_mm_setzero_ps(), (const __m64*) (a[t]->scale + 2 * b))));
            hxs[t * (MAXC / 2) + b] = a[t]->hx[2 * b] + a[t]->hx[2 * b + 1];
        }
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        q2g_row_multi<NT>(w + (size_t) r * row_bytes, a, as, hxs, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

}  // namespace

void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1) {
    if (nblocks < 0 || nblocks > MAXC / 2) {
        basalt::log::error("basalt", "q2_0_gguf_rows_multi: %d blocks; at most %d", nblocks, MAXC / 2);
        std::abort();
    }
    switch (nt) {
        case 1: q2g_rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: q2g_rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: q2g_rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: q2g_rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 5: q2g_rows<5>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 6: q2g_rows<6>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 7: q2g_rows<7>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 8: q2g_rows<8>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 8) {
                const int k = nt - t0 < 8 ? nt - t0 : 8;
                q2_0_gguf_rows_multi(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

}  // namespace basalt::kernels::cpu
