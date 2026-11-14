// src/kernels/cpu/iq_avx512.cpp - the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once and every token applies
// them with a masked subtract, a `maddubs` and a `dpwssd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic`
// references): integer sums per block, times d_x * d_y * the format's constant - only the order of the float additions
// differs.
//
// IQ1_S (19) has an exact kernel, every token bit for bit ggml's single-token dot; so have IQ2_XXS (16), IQ2_XS (17),
// IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23) and IQ1_M (29), so limited-SwiGLU layers and IQ4_XS / IQ1_M layers
// keep ggml's exact output while decoding each block once across tokens.
// Multi-token formats on the 512-bit float accumulator: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).
//
// The sign masks of IQ2_XXS, IQ2_XS and IQ3_XXS are built from their 7-bit indices with BMI2 (pdep, and pext for
// IQ2_XS) and a byte parity rather than eight table loads.  IQ3_XXS and IQ3_S pack their 4-byte codebook entries into
// nibble pairs and decode all 64 indices of a 256-weight block at once in vector registers with AVX-512 VBMI
// (`vpermi2b` and `vpmultishiftqb`).  IQ4_NL (20) down rows pack two rows into the 256-bit halves of each ZMM and use
// unsigned VNNI dot products with precomputed activation biases.
#include "basalt/kernels/cpu/iq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"
#include "basalt/log.hpp"
#include "basalt/platform/compiler.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// the index of the lowest set bit of a nonzero mask: the compiler's own scan
inline int lowest_bit(unsigned m) { return platform::ctz32(m); }

// software prefetch distance in bytes: the rows are streamed from DRAM, so ask for them a few blocks ahead
constexpr int prefetch_ahead = 2048;

// ksigns_iq2xs[i] for eight 7-bit sign indices at once: the table is i | parity(i) << 7.  `x` holds the indices back to
// back (index l in bits 7l..7l+6); byte l of the result is sign byte l, as the table lookups place it.
inline uint64_t ksigns8(uint64_t x) {
    const uint64_t s = _pdep_u64(x, 0x7F7F7F7F7F7F7F7Full);
    uint64_t p = s ^ (s >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    return s | ((p & 0x0101010101010101ull) << 7);
}

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// Nibble-packed VBMI lookup tables for IQ3_XXS (256 entries) and IQ3_S (512 entries).  Initialized lazily inside
// AVX-512 entry points so no constructor runs before main() on CPUs without AVX-512.
struct alignas(64) Iq3xxsLut {
    uint8_t t01[256];
    uint8_t t23[256];
};

const Iq3xxsLut& iq3xxs_lut() {
    static const Iq3xxsLut lut = [] {
        Iq3xxsLut t{};
        for (int i = 0; i < 256; ++i) {
            const uint32_t w = iq3xxs_grid[i];
            const uint32_t b0 = (w >> 3) & 7u, b1 = (w >> 11) & 7u;
            const uint32_t b2 = (w >> 19) & 7u, b3 = (w >> 27) & 7u;
            t.t01[i] = (uint8_t) (b0 | (b1 << 4));
            t.t23[i] = (uint8_t) (b2 | (b3 << 4));
        }
        return t;
    }();
    return lut;
}

struct alignas(64) Iq3sLut {
    uint8_t t01[512];
    uint8_t t23[512];
};

const Iq3sLut& iq3s_lut() {
    static const Iq3sLut lut = [] {
        Iq3sLut t{};
        for (int i = 0; i < 512; ++i) {
            const uint32_t w = iq3s_grid[i];
            t.t01[i] = (uint8_t) ((w & 0x0Fu) | ((w >> 4) & 0xF0u));
            t.t23[i] = (uint8_t) (((w >> 16) & 0x0Fu) | ((w >> 20) & 0xF0u));
        }
        return t;
    }();
    return lut;
}

// Unpack chunk j (16 4-byte entries, stored as 16 uint16 nibble-quartets in half j of (w_lo, w_hi)) into 64 bytes.
inline __m512i unpack_nibbles_chunk(__m512i w_lo, __m512i w_hi, int j) {
    const __m512i mshift = _mm512_set1_epi64(0x1C1814100C080400LL);
    const __m512i m0f = _mm512_set1_epi8(0x0F);
    __m256i w256;
    if (j == 0) w256 = _mm512_castsi512_si256(w_lo);
    else if (j == 1) w256 = _mm512_extracti64x4_epi64(w_lo, 1);
    else if (j == 2) w256 = _mm512_castsi512_si256(w_hi);
    else w256 = _mm512_extracti64x4_epi64(w_hi, 1);
    return _mm512_and_si512(_mm512_multishift_epi64_epi8(mshift, _mm512_cvtepu32_epi64(w256)), m0f);
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    struct Ctx {};
    struct Block {
        Block(const uint8_t*, const Ctx&) {}
        inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) const {
            const uint8_t* q = b + 2 + 16 * j;
            const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
            g = _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                                 (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                                 (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                                 (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
            const uint64_t s = ksigns8((uint64_t) (a1 & 0x0FFFFFFFu) | ((uint64_t) (b1 & 0x0FFFFFFFu) << 28));
            m = _cvtu64_mask64(s);
            const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
            sc = scales4(sa, sa, sb, sb);
        }
    };
    static Ctx make_ctx() { return {}; }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    struct Ctx {};
    struct Block {
        Block(const uint8_t*, const Ctx&) {}
        inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) const {
            const uint8_t* q = b + 2 + 16 * j;
            uint16_t v[8];
            std::memcpy(v, q, 16);
            g = _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                                 (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                                 (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                                 (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
            // the top 7 bits of each u16 are its sign index
            const uint64_t s = ksigns8(_pext_u64(u64(q), 0xFE00FE00FE00FE00ull) | (_pext_u64(u64(q + 8), 0xFE00FE00FE00FE00ull) << 28));
            m = _cvtu64_mask64(s);
            const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
            sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
        }
    };
    static Ctx make_ctx() { return {}; }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    struct Ctx {};
    struct Block {
        Block(const uint8_t*, const Ctx&) {}
        inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) const {
            const uint8_t* qs = b + 2 + 8 * j;
            const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
            const uint64_t a0 = _pdep_u64(u32(qs), 0x00FF00FF00FF00FFull) | _pdep_u64(h0, 0x0300030003000300ull);
            const uint64_t a1 = _pdep_u64(u32(qs + 4), 0x00FF00FF00FF00FFull) | _pdep_u64(h1, 0x0300030003000300ull);
#define G2(a, k) (long long) iq2s_grid[(uint16_t) ((a) >> (16 * (k)))]
            g = _mm512_set_epi64(G2(a1, 3), G2(a1, 2), G2(a1, 1), G2(a1, 0), G2(a0, 3), G2(a0, 2), G2(a0, 1), G2(a0, 0));
#undef G2
            m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
            const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
            sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
        }
    };
    static Ctx make_ctx() { return {}; }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    using Ctx = const Iq3xxsLut*;
    struct Block {
        __m512i w_lo, w_hi, val_lut;
        inline Block(const uint8_t* b, Ctx lut) {
            const __m512i ql = _mm512_loadu_si512((const void*) (b + 2));
            const __mmask64 m7 = _mm512_movepi8_mask(ql);
            const __m512i z01_0 = _mm512_load_si512((const void*) (lut->t01 + 0));
            const __m512i z01_1 = _mm512_load_si512((const void*) (lut->t01 + 64));
            const __m512i z01_2 = _mm512_load_si512((const void*) (lut->t01 + 128));
            const __m512i z01_3 = _mm512_load_si512((const void*) (lut->t01 + 192));
            const __m512i z23_0 = _mm512_load_si512((const void*) (lut->t23 + 0));
            const __m512i z23_1 = _mm512_load_si512((const void*) (lut->t23 + 64));
            const __m512i z23_2 = _mm512_load_si512((const void*) (lut->t23 + 128));
            const __m512i z23_3 = _mm512_load_si512((const void*) (lut->t23 + 192));
            const __m512i p01 = _mm512_mask_blend_epi8(m7, _mm512_permutex2var_epi8(z01_0, ql, z01_1),
                                                       _mm512_permutex2var_epi8(z01_2, ql, z01_3));
            const __m512i p23 = _mm512_mask_blend_epi8(m7, _mm512_permutex2var_epi8(z23_0, ql, z23_1),
                                                       _mm512_permutex2var_epi8(z23_2, ql, z23_3));
            w_lo = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(p01)),
                                   _mm512_slli_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(p23)), 8));
            w_hi = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(p01, 1)),
                                   _mm512_slli_epi16(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(p23, 1)), 8));
            val_lut = _mm512_broadcast_i32x4(_mm_setr_epi8(4, 12, 20, 28, 36, 44, 52, 62, 0, 0, 0, 0, 0, 0, 0, 0));
        }
        inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) const {
            g = _mm512_shuffle_epi8(val_lut, unpack_nibbles_chunk(w_lo, w_hi, j));
            const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
            const uint64_t s = ksigns8((uint64_t) (a & 0x0FFFFFFFu) | ((uint64_t) (c & 0x0FFFFFFFu) << 28));
            m = _cvtu64_mask64(s);
            const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
            sc = scales4(sa, sa, sb, sb);
        }
    };
    static Ctx make_ctx() { return &iq3xxs_lut(); }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    using Ctx = const Iq3sLut*;
    struct Block {
        __m512i w_lo, w_hi;
        inline Block(const uint8_t* b, Ctx lut) {
            const __m512i ql = _mm512_loadu_si512((const void*) (b + 2));
            const __mmask64 m7 = _mm512_movepi8_mask(ql);
            const __mmask64 mh = _cvtu64_mask64(u64(b + 66));
            auto look = [&](const uint8_t* tab) {
                const __m512i lo = _mm512_mask_blend_epi8(
                    m7, _mm512_permutex2var_epi8(_mm512_load_si512((const void*) (tab + 0)), ql,
                                                 _mm512_load_si512((const void*) (tab + 64))),
                    _mm512_permutex2var_epi8(_mm512_load_si512((const void*) (tab + 128)), ql,
                                             _mm512_load_si512((const void*) (tab + 192))));
                const __m512i hi = _mm512_mask_blend_epi8(
                    m7, _mm512_permutex2var_epi8(_mm512_load_si512((const void*) (tab + 256)), ql,
                                                 _mm512_load_si512((const void*) (tab + 320))),
                    _mm512_permutex2var_epi8(_mm512_load_si512((const void*) (tab + 384)), ql,
                                             _mm512_load_si512((const void*) (tab + 448))));
                return _mm512_mask_blend_epi8(mh, lo, hi);
            };
            const __m512i p01 = look(lut->t01);
            const __m512i p23 = look(lut->t23);
            w_lo = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(p01)),
                                   _mm512_slli_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(p23)), 8));
            w_hi = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(p01, 1)),
                                   _mm512_slli_epi16(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(p23, 1)), 8));
        }
        inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) const {
            g = unpack_nibbles_chunk(w_lo, w_hi, j);
            m = _cvtu64_mask64(u64(b + 74 + 8 * j));
            const uint8_t s = b[106 + j];
            const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
            sc = scales4(sa, sa, sb, sb);
        }
    };
    static Ctx make_ctx() { return &iq3s_lut(); }
};

template <> struct Fmt<23> {   // IQ4_XS: d, scales_h, scales_l[4], qs[128]
    static constexpr int bytes = 136;
    static constexpr float K = 1.0f;
    struct Ctx {};
    struct Block {
        Block(const uint8_t*, const Ctx&) {}
        inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) const {
            const __m512i values = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kvalues_iq4nl));
            const __m512i dup = _mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3);
            const __m512i raw = _mm512_permutexvar_epi64(
                dup, _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) (b + 8 + 32 * j))));
            const __m512i nib = _mm512_and_si512(_mm512_mask_srli_epi16(raw, 0xFF00FF00u, raw, 4), _mm512_set1_epi8(0x0f));
            const __m512i w = _mm512_shuffle_epi8(values, nib);
            g = _mm512_abs_epi8(w);
            m = _mm512_movepi8_mask(w);
            const uint8_t sl = b[4 + j];
            const uint16_t sh = (uint16_t) (u16(b + 2) >> (4 * j));
            const int ls1 = ((sl & 0x0f) | ((sh << 4) & 0x30)) - 32;
            const int ls2 = ((sl >> 4) | ((sh << 2) & 0x30)) - 32;
            sc = scales4(ls1, ls1, ls2, ls2);
        }
    };
    static Ctx make_ctx() { return {}; }
};


// ---- IQ1_S (19), NT tokens: per token ggml's AVX2 dot with the 64-value steps widened to 512 bits.  The integer sums
// are exact and folded to ggml's eight int32 lanes, and each token's float accumulation keeps ggml's order, so every
// token's result is bit for bit ggml's single-token dot.  The block is decoded once for all NT tokens.
inline float hsum8(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

template <int NT>
inline void iq1s_row_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res) {
    // lane i of the 32 grid indices of a block takes its high bits from qh word i/4, shifted by 3 * (i % 4)
    const __m512i rep = _mm512_set_epi16(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1,
                                         0, 0, 0, 0);
    const __m512i sh = _mm512_set1_epi64(0x0009000600030000LL);
    const __m512i c0 = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i ones = _mm512_set1_epi16(1);
    __m256 accum[NT];
    float accum1[NT];
    for (int t = 0; t < NT; ++t) {
        accum[t] = _mm256_setzero_ps();
        accum1[t] = 0;
    }
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * sizeof(block_iq1_s);
        _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
        const __m128i qh = _mm_loadu_si128((const __m128i*) (blk + 34));
        const __m512i hv = _mm512_permutexvar_epi16(rep, _mm512_castsi128_si512(qh));
        const __m512i hi = _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(hv, sh), _mm512_set1_epi16(7)), 8);
        const __m512i idx16 = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (blk + 2))), hi);
        // per group scale 2 * ((qh >> 12) & 7) + 1 (int16 lanes 0..7) and the sign of the delta term
        const __m128i ls = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 12), _mm_set1_epi16(7)), 1), _mm_set1_epi16(1));
        const __m128i sg = _mm_srai_epi16(qh, 15);
        const __m256i lss = _mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_xor_si128(ls, sg), sg));
        const __m512i ls512 = _mm512_castsi128_si512(ls);
        __m512i acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = zero;
        alignas(64) uint16_t ix[32];
        _mm512_store_si512((void*) ix, idx16);
        for (int j = 0; j < 4; ++j) {
            const uint16_t* q = ix + 8 * j;
            const __m512i g = _mm512_set_epi64((long long) iq1s_grid[q[7]], (long long) iq1s_grid[q[6]], (long long) iq1s_grid[q[5]],
                                               (long long) iq1s_grid[q[4]], (long long) iq1s_grid[q[3]], (long long) iq1s_grid[q[2]],
                                               (long long) iq1s_grid[q[1]], (long long) iq1s_grid[q[0]]);
            const __mmask64 gm = _mm512_movepi8_mask(g);
            const __m512i ga = _mm512_abs_epi8(g);
            const __m512i lsj = _mm512_permutexvar_epi16(_mm512_add_epi16(c0, _mm512_set1_epi16((short) (2 * j))), ls512);
            // (the permute index: lanes 0-15 -> word 2j, lanes 16-31 -> word 2j + 1)
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, gm, zero, yv);
                acc[t] = _mm512_dpwssd_epi32(acc[t], _mm512_maddubs_epi16(ga, ys), lsj);
            }
        }
        const float dw = h2f(u16(blk));
        for (int t = 0; t < NT; ++t) {
            const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(acc[t]), _mm512_extracti64x4_epi64(acc[t], 1));
            // the delta term: sum over the 8 groups of bsums pair * sign * scale (exact integers)
            const __m256i pairs = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) y[t][i].bsums), _mm512_castsi512_si256(ones));
            const __m256i s8 = _mm256_mullo_epi32(pairs, lss);
            __m128i t4 = _mm_add_epi32(_mm256_castsi256_si128(s8), _mm256_extracti128_si256(s8, 1));
            t4 = _mm_add_epi32(t4, _mm_unpackhi_epi64(t4, t4));
            const int sumi1 = _mm_cvtsi128_si32(_mm_add_epi32(t4, _mm_shuffle_epi32(t4, 1)));
            const float d = y[t][i].d * dw;
            accum[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum[t]);
            accum1[t] += d * sumi1;
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accum[t]) + IQ1S_DELTA * accum1[t];
}

// ---- IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, NT tokens: integer sums folded to ggml's eight int32 lanes, its
// float accumulation order per token, so every token's result is bit for bit ggml's single-token dot.
template <int TY, int NT>
inline void row_dot_exact(const uint8_t* row, int nb, const block_q8_K* const* y, float* res,
                          typename Fmt<TY>::Ctx ctx) {
    __m256 accum[NT];
    for (int t = 0; t < NT; ++t) accum[t] = _mm256_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
        const typename Fmt<TY>::Block st(blk, ctx);
        __m512i acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = zero;
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            st.decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acc[t] = _mm512_dpwssd_epi32(acc[t], _mm512_maddubs_epi16(g, ys), sc);
            }
        }
        const float dw = h2f(u16(blk));
        for (int t = 0; t < NT; ++t) {
            const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(acc[t]), _mm512_extracti64x4_epi64(acc[t], 1));
            accum[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][i].d), _mm256_cvtepi32_ps(sumi), accum[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const float s = hsum8(accum[t]);
        if constexpr (Fmt<TY>::K == 1.0f) res[t] = s;
        else res[t] = Fmt<TY>::K * s;
    }
}

// ---- IQ1_M (29), NT tokens, each bit for bit ggml's dot: the same folding, with the grid-shift term in its own lanes.
template <int NT>
inline void iq1m_row_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i one8 = _mm512_set1_epi8(1);
    __m256 accum1[NT], accum2[NT];
    for (int t = 0; t < NT; ++t) accum1[t] = accum2[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const block_iq1_m* blk = (const block_iq1_m*) (row + (size_t) i * sizeof(block_iq1_m));
        _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
        uint16_t sc[4];
        std::memcpy(sc, blk->scales, 8);
        iq1m_scale_t scale;
        scale.u16 = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        __m512i acc1[NT], acc2[NT];
        for (int t = 0; t < NT; ++t) acc1[t] = acc2[t] = zero;
        for (int j = 0; j < 4; ++j) {
            const uint32_t qh = u32(blk->qh + 4 * j);
            const uint8_t* qs = blk->qs + 8 * j;
            // grid index k: qs[k] | (low 3 bits of qh nibble k) << 8; the nibble's top bit is the group's shift sign
            const uint64_t i0 = _pdep_u64(u32(qs), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh & 0x7777u, 0x0f000f000f000f00ULL);
            const uint64_t i1 = _pdep_u64(u32(qs + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64((qh >> 16) & 0x7777u, 0x0f000f000f000f00ULL);
            const __m512i g = _mm512_set_epi64((long long) iq1s_grid[i1 >> 48], (long long) iq1s_grid[(i1 >> 32) & 0xffff],
                                               (long long) iq1s_grid[(i1 >> 16) & 0xffff], (long long) iq1s_grid[i1 & 0xffff],
                                               (long long) iq1s_grid[i0 >> 48], (long long) iq1s_grid[(i0 >> 32) & 0xffff],
                                               (long long) iq1s_grid[(i0 >> 16) & 0xffff], (long long) iq1s_grid[i0 & 0xffff]);
            const __mmask64 gm = _mm512_movepi8_mask(g);
            const __m512i ga = _mm512_abs_epi8(g);
            const __mmask64 ds = _cvtu64_mask64(_pdep_u64(_pext_u32(qh, 0x88888888u), 0x0101010101010101ULL) * 0xff);
            const int w = sc[j];
            const __m512i ls = scales4(2 * (w & 7) + 1, 2 * ((w >> 3) & 7) + 1, 2 * ((w >> 6) & 7) + 1, 2 * ((w >> 9) & 7) + 1);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i dot = _mm512_maddubs_epi16(ga, _mm512_mask_sub_epi8(yv, gm, zero, yv));
                const __m512i dd = _mm512_maddubs_epi16(one8, _mm512_mask_sub_epi8(yv, ds, zero, yv));
                acc1[t] = _mm512_dpwssd_epi32(acc1[t], dot, ls);
                acc2[t] = _mm512_dpwssd_epi32(acc2[t], dd, ls);
            }
        }
        const float dw = h2f(scale.u16);
        for (int t = 0; t < NT; ++t) {
            const __m256 d = _mm256_set1_ps(y[t][i].d * dw);
            accum1[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_add_epi32(_mm512_castsi512_si256(acc1[t]), _mm512_extracti64x4_epi64(acc1[t], 1))), accum1[t]);
            accum2[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_add_epi32(_mm512_castsi512_si256(acc2[t]), _mm512_extracti64x4_epi64(acc2[t], 1))), accum2[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accum1[t]) + IQ1M_DELTA * hsum8(accum2[t]);
}

template <int TY> struct ExactCtx {
    using Type = typename Fmt<TY>::Ctx;
    static Type make() { return Fmt<TY>::make_ctx(); }
};
template <> struct ExactCtx<19> {
    struct Type {};
    static Type make() { return {}; }
};
template <> struct ExactCtx<29> {
    struct Type {};
    static Type make() { return {}; }
};

// the exact dots of one row for NT tokens
template <int TY, int NT>
inline void exact_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res, typename ExactCtx<TY>::Type ctx) {
    if constexpr (TY == 19) { (void) ctx; iq1s_row_dot<NT>(row, nb, y, res); }
    else if constexpr (TY == 29) { (void) ctx; iq1m_row_dot<NT>(row, nb, y, res); }
    else row_dot_exact<TY, NT>(row, nb, y, res, ctx);
}

// gate/up rows r0..r1 of NT tokens at once, then each token's scale, limit and SwiGLU as one token alone takes them
template <int TY, int NT>
void exact_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int nb, float sg, float su, bool scaled, float limit,
                   const void* const* act, float* const* ff, int r0, int r1) {
    const auto ctx = ExactCtx<TY>::make();
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        exact_dot<TY, NT>(blob + (size_t) r * gu_row, nb, y, g, ctx);
        exact_dot<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u, ctx);
        for (int t = 0; t < NT; ++t) {
            if (scaled) {
                g[t] *= sg;
                u[t] *= su;
            }
            if (limit > 0.f) {
                g[t] = std::fmin(g[t], limit);
                u[t] = std::fmax(std::fmin(u[t], limit), -limit);
            }
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

template <int TY, int NT>
void exact_rows(const uint8_t* w, size_t row_bytes, int nb, const void* const* act, float* const* out, int r0, int r1) {
    const auto ctx = ExactCtx<TY>::make();
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        exact_dot<TY, NT>(w + (size_t) r * row_bytes, nb, y, res, ctx);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

// k = 1..4 tokens (the accumulators of four tokens stay in registers)
template <int TY>
void exact_gu_rows_k(int k, const uint8_t* blob, size_t gu_row, size_t up_off, int nb, float sg, float su, bool scaled,
                     float limit, const void* const* act, float* const* ff, int r0, int r1) {
    switch (k) {
        case 1: exact_gu_rows<TY, 1>(blob, gu_row, up_off, nb, sg, su, scaled, limit, act, ff, r0, r1); break;
        case 2: exact_gu_rows<TY, 2>(blob, gu_row, up_off, nb, sg, su, scaled, limit, act, ff, r0, r1); break;
        case 3: exact_gu_rows<TY, 3>(blob, gu_row, up_off, nb, sg, su, scaled, limit, act, ff, r0, r1); break;
        default: exact_gu_rows<TY, 4>(blob, gu_row, up_off, nb, sg, su, scaled, limit, act, ff, r0, r1); break;
    }
}

template <int TY>
void exact_rows_k(int k, const uint8_t* w, size_t row_bytes, int nb, const void* const* act, float* const* out, int r0, int r1) {
    switch (k) {
        case 1: exact_rows<TY, 1>(w, row_bytes, nb, act, out, r0, r1); break;
        case 2: exact_rows<TY, 2>(w, row_bytes, nb, act, out, r0, r1); break;
        case 3: exact_rows<TY, 3>(w, row_bytes, nb, act, out, r0, r1); break;
        default: exact_rows<TY, 4>(w, row_bytes, nb, act, out, r0, r1); break;
    }
}

// tokens in groups of up to four
void exact_gu_rows_nt(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, bool scaled,
                      float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) {
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (type) {
        case 19: exact_gu_rows_k<19>(k, blob, gu_row, up_off, nb, sg, su, scaled, limit, act + t0, ff + t0, r0, r1); break;
        case 16: exact_gu_rows_k<16>(k, blob, gu_row, up_off, nb, sg, su, scaled, limit, act + t0, ff + t0, r0, r1); break;
        case 17: exact_gu_rows_k<17>(k, blob, gu_row, up_off, nb, sg, su, scaled, limit, act + t0, ff + t0, r0, r1); break;
        case 18: exact_gu_rows_k<18>(k, blob, gu_row, up_off, nb, sg, su, scaled, limit, act + t0, ff + t0, r0, r1); break;
        case 21: exact_gu_rows_k<21>(k, blob, gu_row, up_off, nb, sg, su, scaled, limit, act + t0, ff + t0, r0, r1); break;
        case 22: exact_gu_rows_k<22>(k, blob, gu_row, up_off, nb, sg, su, scaled, limit, act + t0, ff + t0, r0, r1); break;
        case 23: exact_gu_rows_k<23>(k, blob, gu_row, up_off, nb, sg, su, scaled, limit, act + t0, ff + t0, r0, r1); break;
        case 29: exact_gu_rows_k<29>(k, blob, gu_row, up_off, nb, sg, su, scaled, limit, act + t0, ff + t0, r0, r1); break;
        default: break;
        }
    }
}

void exact_rows_nt(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                   int r0, int r1) {
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (type) {
        case 19: exact_rows_k<19>(k, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
        case 16: exact_rows_k<16>(k, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
        case 17: exact_rows_k<17>(k, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
        case 18: exact_rows_k<18>(k, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
        case 21: exact_rows_k<21>(k, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
        case 22: exact_rows_k<22>(k, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
        case 23: exact_rows_k<23>(k, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
        case 29: exact_rows_k<29>(k, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
        default: break;
        }
    }
}

template <int TY, int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res,
                    typename Fmt<TY>::Ctx ctx) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        // E-2: the weight rows stream from DRAM (4 KB pages when large pages are refused): ask for the bytes a few
        // blocks ahead (the next row's start included) before the decode needs them
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        const typename Fmt<TY>::Block st(blk, ctx);
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            st.decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_dpwssd_epi32(acci[t], _mm512_maddubs_epi16(g, ys), sc);
            }
        }
        const float dx = h2f(u16(blk)) * Fmt<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

template <int TY, int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const auto ctx = Fmt<TY>::make_ctx();
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(blob + (size_t) r * gu_row, nb, y, g, ctx);
        row_dot<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u, ctx);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const auto ctx = Fmt<TY>::make_ctx();
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res, ctx);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 8) {
            const int k = nt - t0 < 8 ? nt - t0 : 8;
            if (k == 8) gu_rows<TY, 8>(blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1);
            else gu_rows_nt<TY>(k, blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1);
        }
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

// ---- IQ4_NL (type 20) down rows in 512-bit lanes: two 32-value blocks (rows r + 2p and r + 2p + 1) occupy the low and
// high 256-bit halves of each ZMM, sharing the broadcast Q8_0 activation and unsigned VNNI bias seed across up to 8 rows
// and 4 tokens simultaneously.  Each 256-bit half reduces in the same order as iq4nl256_down_rows_vnni, so every output
// word is bit for bit identical.
constexpr int iq4nl512_max_blocks = 128;

// kvalues_iq4nl + 128: unsigned operand of _mm512_dpbusd_epi32
alignas(16) constexpr uint8_t iq4nl_u8[16] = {
    1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241
};

inline float hsum8_hadd(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

template <int RP, int NT, bool TAIL>
inline void iq4nl512_pass(const uint8_t* row0, size_t row_bytes, int nb, const block_q8_0* const* y,
                          const __m256i* seed, const float* dy, float* const* out, int r, const uint8_t*& pf) {
    const __m512i values = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) iq4nl_u8));
    const __m512i m4b = _mm512_set1_epi8(0x0f);
    const __m512i dup = _mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3);
    const __m512i dx_perm = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    __m512 acc[RP][NT];
    for (int p = 0; p < RP; ++p)
        for (int t = 0; t < NT; ++t) acc[p][t] = _mm512_setzero_ps();
    constexpr int rows_step = TAIL ? 1 : (2 * RP);
    constexpr size_t pf_ahead = (size_t) rows_step * 512 > prefetch_ahead ? (size_t) rows_step * 512 : prefetch_ahead;
    for (int ib = 0; ib < nb; ++ib) {
        for (const uint8_t* lim = row0 + (size_t) (ib + 1) * rows_step * sizeof(block_iq4_nl) + pf_ahead;
             pf < lim; pf += 64)
            _mm_prefetch((const char*) pf, _MM_HINT_T0);
        __m512i q8[NT], sd[NT];
        __m512 dyv[NT];
        for (int t = 0; t < NT; ++t) {
            const int idx = ib * NT + t;
            q8[t] = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) y[t][ib].qs));
            sd[t] = _mm512_broadcast_i64x4(seed[idx]);
            dyv[t] = _mm512_set1_ps(dy[idx]);
        }
        for (int p = 0; p < RP; ++p) {
            const uint8_t* blk0 = row0 + (size_t) (2 * p) * row_bytes + (size_t) ib * sizeof(block_iq4_nl);
            const uint8_t* blk1 = TAIL ? blk0 : (blk0 + row_bytes);
            const __m256i b01 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) (blk0 + 2))),
                                                        _mm_loadu_si128((const __m128i*) (blk1 + 2)), 1);
            const __m512i bits = _mm512_permutexvar_epi64(dup, _mm512_castsi256_si512(b01));
            const __m512i q4 = _mm512_shuffle_epi8(
                values, _mm512_and_si512(_mm512_mask_srli_epi16(bits, 0xFF00FF00u, bits, 4), m4b));
            const __m128 df = _mm_cvtph_ps(_mm_cvtsi32_si128((int) ((uint32_t) u16(blk0) | ((uint32_t) u16(blk1) << 16))));
            const __m512 dx = _mm512_permutexvar_ps(dx_perm, _mm512_castps128_ps512(df));
            for (int t = 0; t < NT; ++t) {
                const __m512i prod = _mm512_dpbusd_epi32(sd[t], q4, q8[t]);
                acc[p][t] = _mm512_fmadd_ps(_mm512_mul_ps(dx, dyv[t]), _mm512_cvtepi32_ps(prod), acc[p][t]);
            }
        }
    }
    for (int p = 0; p < RP; ++p)
        for (int t = 0; t < NT; ++t) {
            out[t][r + 2 * p] = hsum8_hadd(_mm512_castps512_ps256(acc[p][t]));
            if constexpr (!TAIL) out[t][r + 2 * p + 1] = hsum8_hadd(_mm512_extractf32x8_ps(acc[p][t], 1));
        }
}

template <int RP, int NT>
void iq4nl512_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
                   int r0, int r1) {
    const int nb = n / QK4_NL;
    alignas(32) __m256i seed[NT * iq4nl512_max_blocks];
    alignas(32) float dy[NT * iq4nl512_max_blocks];
    const __m256i bias = _mm256_set1_epi8((char) 0x80);
    const __m256i zero = _mm256_setzero_si256();
    for (int ib = 0; ib < nb; ++ib)
        for (int t = 0; t < NT; ++t) {
            const int idx = ib * NT + t;
            dy[idx] = h2f(y[t][ib].d);
            const __m256i q8 = _mm256_loadu_si256((const __m256i*) y[t][ib].qs);
            seed[idx] = _mm256_sub_epi32(zero, _mm256_dpbusd_epi32(zero, bias, q8));
        }
    constexpr size_t pf_init = (size_t) (2 * RP) * 512 > prefetch_ahead ? (size_t) (2 * RP) * 512 : prefetch_ahead;
    const uint8_t* pf = (const uint8_t*) ((uintptr_t) (w + (size_t) r0 * row_bytes + pf_init) & ~(uintptr_t) 63);
    int r = r0;
    for (; r + 2 * RP <= r1; r += 2 * RP)
        iq4nl512_pass<RP, NT, false>(w + (size_t) r * row_bytes, row_bytes, nb, y, seed, dy, out, r, pf);
    for (; r + 2 <= r1; r += 2)
        iq4nl512_pass<1, NT, false>(w + (size_t) r * row_bytes, row_bytes, nb, y, seed, dy, out, r, pf);
    if (r < r1)
        iq4nl512_pass<1, NT, true>(w + (size_t) r * row_bytes, row_bytes, nb, y, seed, dy, out, r, pf);
}

}  // namespace

void quantize_q8_K_avx512(const float* x, void* dst, int64_t k) {
    block_q8_K* y = (block_q8_K*) dst;
    const int64_t nb = k / QK_K;
    const __m512i top = _mm512_set1_epi32(127);
    const __m512i one8 = _mm512_set1_epi8(1);
    const __m512i perm = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m512 v[QK_K / 16], av[QK_K / 16];
        __m512 am0 = _mm512_setzero_ps(), am1 = _mm512_setzero_ps();
        for (int j = 0; j < QK_K / 16; j += 2) {
            v[j] = _mm512_loadu_ps(x + 16 * j);
            v[j + 1] = _mm512_loadu_ps(x + 16 * (j + 1));
            av[j] = _mm512_abs_ps(v[j]);
            av[j + 1] = _mm512_abs_ps(v[j + 1]);
            am0 = _mm512_max_ps(am0, av[j]);
            am1 = _mm512_max_ps(am1, av[j + 1]);
        }
        am0 = _mm512_max_ps(am0, am1);
        const __m256 a256 = _mm256_max_ps(_mm512_castps512_ps256(am0), _mm512_extractf32x8_ps(am0, 1));
        __m128 a128 = _mm_max_ps(_mm256_castps256_ps128(a256), _mm256_extractf128_ps(a256, 1));
        a128 = _mm_max_ps(a128, _mm_movehl_ps(a128, a128));
        const float amax = _mm_cvtss_f32(_mm_max_ss(a128, _mm_movehdup_ps(a128)));
        if (amax == 0.f) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        // the signed value of the first element of largest magnitude fixes the scale (branchless across the 16 chunks)
        const __m512 amv = _mm512_set1_ps(amax);
        unsigned mw = 0;
        int mbase = 0;
        for (int g = 7; g >= 0; --g) {
            const unsigned m0 = (unsigned) _mm512_cmp_ps_mask(av[2 * g], amv, _CMP_EQ_OQ);
            const unsigned m1 = (unsigned) _mm512_cmp_ps_mask(av[2 * g + 1], amv, _CMP_EQ_OQ);
            const unsigned m = m0 | (m1 << 16);
            if (m != 0) { mw = m; mbase = 32 * g; }
        }
        const float mx = x[mbase + lowest_bit(mw)];
        const float iscale = -127.f / mx;
        const __m512 is = _mm512_set1_ps(iscale);
        __m128i bs[4];
        for (int g = 0; g < 4; ++g) {
            const int j = 4 * g;
            const __m512i q0 = _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(is, v[j + 0])), top);
            const __m512i q1 = _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(is, v[j + 1])), top);
            const __m512i q2 = _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(is, v[j + 2])), top);
            const __m512i q3 = _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(is, v[j + 3])), top);
            // per 128-bit lane: four bytes from each of q0..q3; permute puts the 64 codes in order
            const __m512i b = _mm512_packs_epi16(_mm512_packs_epi32(q0, q1), _mm512_packs_epi32(q2, q3));
            _mm512_storeu_si512((void*) (y[i].qs + 16 * j), _mm512_permutexvar_epi32(perm, b));
            const __m512i s4 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), one8, b);
            const __m256i s2 = _mm256_add_epi32(_mm512_castsi512_si256(s4), _mm512_extracti64x4_epi64(s4, 1));
            bs[g] = _mm_add_epi32(_mm256_castsi256_si128(s2), _mm256_extracti128_si256(s2, 1));
        }
        _mm_storeu_si128((__m128i*) (y[i].bsums + 0), _mm_packs_epi32(bs[0], bs[1]));
        _mm_storeu_si128((__m128i*) (y[i].bsums + 8), _mm_packs_epi32(bs[2], bs[3]));
        y[i].d = 1 / iscale;
    }
}

bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

bool iq1s_512_fits(int n) noexcept { return n > 0 && n % QK_K == 0; }

void iq1s_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, bool scaled,
                      float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) {
    exact_gu_rows_nt(19, blob, gu_row, up_off, n, sg, su, scaled, limit, act, nt, ff, r0, r1);
}

void iq1s_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                   int r0, int r1) {
    exact_rows_nt(19, w, row_bytes, n, act, nt, out, r0, r1);
}

bool iq_exact_512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23 || type == 29;
}

void iq_exact_512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su,
                          bool scaled, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) {
    exact_gu_rows_nt(type, blob, gu_row, up_off, n, sg, su, scaled, limit, act, nt, ff, r0, r1);
}

void iq_exact_512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                       float* const* out, int r0, int r1) {
    exact_rows_nt(type, w, row_bytes, n, act, nt, out, r0, r1);
}

bool iq4nl512_fits(int n) noexcept { return n > 0 && n % QK4_NL == 0 && n / QK4_NL <= iq4nl512_max_blocks; }

void iq4nl512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    if (!iq4nl512_fits(n)) {
        basalt::log::error("basalt", "iq4nl512_down_rows: rows of %d values; at most %d, in blocks of %d", n,
                           iq4nl512_max_blocks * QK4_NL, QK4_NL);
        std::exit(1);
    }
    if (nt > 4) {
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            iq4nl512_down_rows(w, row_bytes, n, hq + t0, k, out + t0, r0, r1);
        }
        return;
    }
    const block_q8_0* y[4];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
    switch (nt) {
        case 1: iq4nl512_rows<4, 1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: iq4nl512_rows<2, 2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: iq4nl512_rows<2, 3>(w, row_bytes, n, y, out, r0, r1); break;
        default: iq4nl512_rows<2, 4>(w, row_bytes, n, y, out, r0, r1); break;
    }
}

}  // namespace basalt::kernels::cpu
