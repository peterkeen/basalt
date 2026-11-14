// tests/kernels/cpu/iq_avx512_test.cpp - the AVX-512 i-quant rows against their frozen output hashes and a plain
// double-precision reference.
//
//     build/iq_avx512_test [--print] [--bench]
//
// Gate/up rows (`iq512_gu_rows`) and plain dot rows (`iq512_rows`) of IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S and IQ2_S at the
// model's row geometry, 1 to 8 tokens, on random weights, on weights pinned to their extreme codes (every grid index and
// sign bit set, and every one clear) and with saturating activations.  Checks: (1) every dot-row word hashes to the value
// the kernels gave when the hash was frozen (the decode may change, the bits may not); (2) a token's rows do not depend on
// how many tokens share the call; (3) the dot rows and the gate/up rows against a double reference (dequantize, then sum in
// double), whose error is reported.  The dot rows are hashed: their integer decode and float reduction are the same on
// every compiler.  The gate/up rows carry the silu, whose expf comes from the C library and does not agree between MSVC
// and glibc, so they are checked against the reference instead of a frozen bit pattern.  The fixture uses only the raw
// generator output, so it is the same on every standard library.
// `--print` prints the hashes, `--bench` one in-cache thread's rate per format and token count.
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/kq_avx512.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int H = cpu::H, FF = cpu::FF, MAXT = 8, MODES = 4;

struct Frozen {
    int type;
    uint64_t hash;
};
// FNV-1a over every dot-row word of every mode and token count, in the order main() computes them
constexpr Frozen kFrozen[] = {
    {16, 0x5863fdcf90fa4725ull}, {17, 0x3acf759ebce5f2a1ull}, {18, 0x5861282d89bf5a41ull},
    {21, 0x931a4e9ae9822c6aull}, {22, 0xacaaeba7bf448c1eull},
};

// mode 0 random bytes, 1 every code bit set, 2 random, 3 every code bit clear; a valid small fp16 scale at the start of
// each block (IQ2_XS / IQ3_S / IQ2_S scale nibbles stay in the code bytes)
void fill(std::vector<uint8_t>& v, int type, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i + 8 <= v.size(); i += 8) {
        const uint64_t x = mode == 1 ? ~0ull : mode == 3 ? 0ull : rng();
        std::memcpy(v.data() + i, &x, 8);
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t b = 0; b + bs <= v.size(); b += bs) {
        const uint16_t d = (uint16_t) (0x2800 + (rng() & 0xff));
        if (type == 14) {
            std::memcpy(&v[b + bs - 2], &d, 2);
        } else {
            std::memcpy(&v[b], &d, 2);
            if (type == 7 || type == 12 || type == 13) {
                const uint16_t dm = (uint16_t) (0x2400 + (rng() & 0xff));
                std::memcpy(&v[b + 2], &dm, 2);
            }
        }
    }
}

// block_q8_K: float d, int8 qs[256], int16 bsums[16]
void dequant_q8_K(const uint8_t* p, float* y, int n) {
    for (int b = 0; b < n / 256; ++b, p += 292) {
        float d;
        std::memcpy(&d, p, 4);
        for (int i = 0; i < 256; ++i) *y++ = d * (float) (int8_t) p[4 + i];
    }
}

void fnv(uint64_t& h, const float* p, size_t n) {
    const auto* b = (const unsigned char*) p;
    for (size_t i = 0; i < 4 * n; ++i) h = (h ^ b[i]) * 1099511628211ull;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    bool print = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        print |= a == "--print";
        bench |= a == "--bench";
    }
    if (!cpu::cpu_avx512_ok()) {
        std::printf("iq_avx512_test SKIPPED: this CPU lacks the AVX-512 set the kernels need\n");
        return 0;
    }
    std::printf("iq_avx512_test: %s\n", cpu::cpu_name().c_str());
    int bad = 0;
    for (const Frozen& fz : kFrozen) {
        const int type = fz.type;
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(type, 20, H, FF, f, err)) { std::printf("native_fmt(%d): %s\n", type, err.c_str()); return 1; }
        const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
        uint64_t h = 1469598103934665603ull;
        double max_rel = 0, max_gu = 0;
        size_t tok_diff = 0;
        std::mt19937 rng(11 + (unsigned) type);
        for (int mode = 0; mode < MODES; ++mode) {
            std::vector<uint8_t> blob(2 * f.up_off);
            fill(blob, type, 100 + (uint64_t) type * 7 + (uint64_t) mode, mode);
            std::vector<std::vector<uint8_t>> act(MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
            std::vector<std::vector<float>> actf(MAXT, std::vector<float>(H));
            std::vector<float> x(H);
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : x) v = mode == 2 ? ((rng() & 1) ? 1.f : -1.f) : (float) ((int) (rng() % 2001) - 1000) / 400.f;
                cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
                dequant_q8_K(act[(size_t) t].data(), actf[(size_t) t].data(), H);
            }
            // each token alone: the rows every wider call must reproduce
            std::vector<std::vector<float>> one_ff(MAXT, std::vector<float>(FF)), one_d(MAXT, std::vector<float>(FF));
            for (int t = 0; t < MAXT; ++t) {
                const void* a1[1] = {act[(size_t) t].data()};
                float* f1[1] = {one_ff[(size_t) t].data()};
                float* d1[1] = {one_d[(size_t) t].data()};
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, a1, 1, f1, 0, FF);
                cpu::iq512_rows(type, blob.data(), f.gu_row, H, a1, 1, d1, 0, FF);
            }
            for (int nt = 1; nt <= MAXT; ++nt) {
                const void* ap[MAXT];
                std::vector<std::vector<float>> ff(nt, std::vector<float>(FF)), dd(nt, std::vector<float>(FF));
                float *fp[MAXT], *dp[MAXT];
                for (int t = 0; t < nt; ++t) {
                    ap[t] = act[(size_t) t].data();
                    fp[t] = ff[(size_t) t].data();
                    dp[t] = dd[(size_t) t].data();
                }
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fp, 0, FF);
                cpu::iq512_rows(type, blob.data(), f.gu_row, H, ap, nt, dp, 0, FF);
                for (int t = 0; t < nt; ++t) {
                    fnv(h, dp[t], FF);
                    for (int r = 0; r < FF; ++r)
                        tok_diff += std::memcmp(&fp[t][r], &one_ff[(size_t) t][(size_t) r], 4) != 0 ||
                                    std::memcmp(&dp[t][r], &one_d[(size_t) t][(size_t) r], 4) != 0;
                }
                std::vector<float> wrow(H), wup(H);
                for (int r : {0, 1, 97, FF - 1}) {
                    tr->to_float(blob.data() + (size_t) r * f.gu_row, wrow.data(), H);
                    tr->to_float(blob.data() + f.up_off + (size_t) r * f.gu_row, wup.data(), H);
                    for (int t = 0; t < nt; ++t) {
                        double want = 0, up = 0;
                        for (int i = 0; i < H; ++i) {
                            const double a = (double) actf[(size_t) t][(size_t) i];
                            want += (double) wrow[(size_t) i] * a;
                            up += (double) wup[(size_t) i] * a;
                        }
                        max_rel = std::max(max_rel, std::fabs(dp[t][r] - want) / std::max(1e-3, std::fabs(want)));
                        const double gu = (want / (1.0 + std::exp(-want))) * up;   // silu(gate_r . a) * (up_r . a)
                        max_gu = std::max(max_gu, std::fabs(fp[t][r] - gu) / std::max(1e-3, std::fabs(gu)));
                    }
                }
            }
        }
        const bool hash_ok = h == fz.hash;
        std::printf("  type %2d: hash %016llx %s, token-count dependence %zu words, dot rows vs double: max rel %.3g, "
                    "gate/up rows vs double: max rel %.3g\n",
                    type, (unsigned long long) h, hash_ok ? "(frozen)" : "(DIFFERS from the frozen hash)", tok_diff, max_rel,
                    max_gu);
        if (print) std::printf("    {%d, 0x%016llxull},\n", type, (unsigned long long) h);
        bad += !hash_ok;
        bad += tok_diff != 0;
        if (max_rel > 1e-3) { std::printf("REFERENCE ERROR too large for type %d (dot rows)\n", type); ++bad; }
        if (max_gu > 1e-4) { std::printf("REFERENCE ERROR too large for type %d (gate/up rows)\n", type); ++bad; }
    }

    // IQ4_NL (type 20) 512-bit row-pair down kernel: bit for bit equal to iq4nl256_down_rows_vnni across token counts,
    // extreme weight/activation patterns, and odd/even row-range splits.
    {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(23, 20, H, FF, f, err)) { std::printf("native_fmt(23, 20): %s\n", err.c_str()); return 1; }
        size_t diff_vnni = 0, tok_diff = 0, split_diff = 0;
        std::mt19937 rng(2020);
        for (int mode = 0; mode < MODES; ++mode) {
            std::vector<uint8_t> w(f.d_row * H);
            fill(w, 20, 900 + (uint64_t) mode, mode);
            std::vector<std::vector<uint8_t>> hq(MAXT, std::vector<uint8_t>(cpu::kNativeHBytes));
            std::vector<float> h_in(FF);
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : h_in) v = mode == 2 ? ((rng() & 1) ? 1e3f : -1e3f) : (float) ((int) (rng() % 2001) - 1000) / 400.f;
                cpu::native_quant_h(f, h_in.data(), hq[(size_t) t].data());
            }
            std::vector<std::vector<float>> one(MAXT, std::vector<float>(H));
            for (int t = 0; t < MAXT; ++t) {
                const void* p1[1] = {hq[(size_t) t].data()};
                float* o1[1] = {one[(size_t) t].data()};
                cpu::iq4nl512_down_rows(w.data(), f.d_row, FF, p1, 1, o1, 0, H);
            }
            for (int nt = 1; nt <= MAXT; ++nt) {
                const void* hp[MAXT];
                std::vector<std::vector<float>> got(nt, std::vector<float>(H)), ref(nt, std::vector<float>(H));
                float *gp[MAXT], *rp[MAXT];
                for (int t = 0; t < nt; ++t) {
                    hp[t] = hq[(size_t) t].data();
                    gp[t] = got[(size_t) t].data();
                    rp[t] = ref[(size_t) t].data();
                }
                cpu::iq4nl512_down_rows(w.data(), f.d_row, FF, hp, nt, gp, 0, H);
                cpu::iq4nl256_down_rows_vnni(w.data(), f.d_row, FF, hp, nt, rp, 0, H);
                for (int t = 0; t < nt; ++t)
                    for (int r = 0; r < H; ++r) {
                        diff_vnni += std::memcmp(&gp[t][r], &rp[t][r], 4) != 0;
                        tok_diff += std::memcmp(&gp[t][r], &one[(size_t) t][(size_t) r], 4) != 0;
                    }
                if (mode == 0 && (nt == 1 || nt == 4 || nt == 7)) {
                    std::vector<std::vector<float>> sp(nt, std::vector<float>(H, -999.f));
                    float* spp[MAXT];
                    for (int t = 0; t < nt; ++t) spp[t] = sp[(size_t) t].data();
                    const int cuts[] = {0, 1, 2, 9, 12, 23, 31, 40, H};
                    for (size_t c = 0; c + 1 < sizeof(cuts) / sizeof(cuts[0]); ++c)
                        cpu::iq4nl512_down_rows(w.data(), f.d_row, FF, hp, nt, spp, cuts[c], cuts[c + 1]);
                    for (int t = 0; t < nt; ++t)
                        for (int r = 0; r < H; ++r)
                            split_diff += std::memcmp(&spp[t][r], &gp[t][r], 4) != 0;
                }
            }
        }
        std::printf("  type 20 (IQ4_NL down 512): vs vnni256 %zu diffs, token-count %zu diffs, row-split %zu diffs\n",
                    diff_vnni, tok_diff, split_diff);
        bad += diff_vnni != 0 || tok_diff != 0 || split_diff != 0;
    }

    // K-quant and legacy AVX-512 VNNI kernels (Q4_K=12, Q5_K=13, Q6_K=14, Q5_1=7, Q8_0=8): bit for bit equal to
    // ggml's single-token AVX-512 dot across 1..8 tokens, extreme byte patterns, SwiGLU limit/scales, mixed gate/up,
    // and odd/even row-range splits.
    {
        for (int type : {12, 13, 14, 7, 8}) {
            const bool is_k = (type == 12 || type == 13 || type == 14);
            const int cols = is_k ? H : FF;
            const int rows = is_k ? FF : H;
            const size_t rb = ggml_row_size((ggml_type) type, cols);
            const auto* tc = ggml_get_type_traits_cpu_avx512((ggml_type) type);
            const ggml_type at = tc->vec_dot_type;
            const auto* ta = ggml_get_type_traits_cpu_avx512(at);
            size_t dot_diff = 0, gu_diff = 0, split_diff = 0;
            std::mt19937 rng(3000 + type);
            for (int mode = 0; mode < MODES; ++mode) {
                std::vector<uint8_t> blob(2 * rb * (size_t) rows);
                fill(blob, type, 1200 + (uint64_t) (type * 10 + mode), mode);
                std::vector<std::vector<uint8_t>> act(MAXT, std::vector<uint8_t>(ggml_row_size(at, cols)));
                std::vector<float> xin(cols);
                const void* ap[MAXT];
                for (int t = 0; t < MAXT; ++t) {
                    for (auto& v : xin)
                        v = mode == 2 ? ((rng() & 1) ? 1e3f : -1e3f) : (float) ((int) (rng() % 2001) - 1000) / 400.f;
                    if (at == GGML_TYPE_Q8_K) cpu::quantize_q8_K_avx512(xin.data(), act[(size_t) t].data(), cols);
                    else ta->from_float(xin.data(), act[(size_t) t].data(), cols);
                    ap[t] = act[(size_t) t].data();
                }
                const float sg = 0.75f, su = 1.25f, sd = 0.5f, lim = 6.0f;
                const bool sc_on = (mode == 2);
                const float lim_on = (mode == 1) ? lim : 0.f;
                std::vector<std::vector<float>> ref_d(MAXT, std::vector<float>(rows)), ref_gu(MAXT, std::vector<float>(rows));
                for (int t = 0; t < MAXT; ++t)
                    for (int r = 0; r < rows; ++r) {
                        float g = 0.f, u = 0.f;
                        tc->vec_dot(cols, &g, 0, blob.data() + (size_t) r * rb, 0, ap[t], 0, 1);
                        tc->vec_dot(cols, &u, 0, blob.data() + (size_t) rows * rb + (size_t) r * rb, 0, ap[t], 0, 1);
                        ref_d[(size_t) t][(size_t) r] = sc_on ? g * sd : g;
                        if (sc_on) { g *= sg; u *= su; }
                        if (lim_on > 0.f) { g = std::fmin(g, lim_on); u = std::fmax(std::fmin(u, lim_on), -lim_on); }
                        ref_gu[(size_t) t][(size_t) r] = (g / (1.f + std::exp(-g))) * u;
                    }
                for (int nt = 1; nt <= MAXT; ++nt) {
                    std::vector<std::vector<float>> gd(nt, std::vector<float>(rows)), ggu(nt, std::vector<float>(rows));
                    float *p_d[MAXT], *p_gu[MAXT];
                    for (int t = 0; t < nt; ++t) { p_d[t] = gd[(size_t) t].data(); p_gu[t] = ggu[(size_t) t].data(); }
                    cpu::kq512_rows(type, blob.data(), rb, cols, sd, sc_on, ap, nt, p_d, 0, rows);
                    cpu::kq512_gu_rows(type, blob.data(), rb, (size_t) rows * rb, cols, sg, su, sc_on, lim_on, ap, nt, p_gu, 0, rows);
                    for (int t = 0; t < nt; ++t)
                        for (int r = 0; r < rows; ++r) {
                            dot_diff += std::memcmp(&p_d[t][r], &ref_d[(size_t) t][(size_t) r], 4) != 0;
                            gu_diff += std::memcmp(&p_gu[t][r], &ref_gu[(size_t) t][(size_t) r], 4) != 0;
                        }
                    if (mode == 0 && (nt == 1 || nt == 4 || nt == 7)) {
                        std::vector<std::vector<float>> sp(nt, std::vector<float>(rows, -999.f));
                        float* spp[MAXT];
                        for (int t = 0; t < nt; ++t) spp[t] = sp[(size_t) t].data();
                        const int cuts[] = {0, 1, 3, 10, 19, 32, rows};
                        for (size_t c = 0; c + 1 < sizeof(cuts) / sizeof(cuts[0]); ++c)
                            cpu::kq512_rows(type, blob.data(), rb, cols, sd, sc_on, ap, nt, spp, cuts[c], cuts[c + 1]);
                        for (int t = 0; t < nt; ++t)
                            for (int r = 0; r < rows; ++r)
                                split_diff += std::memcmp(&spp[t][r], &p_d[t][r], 4) != 0;
                    }
                }
            }
            std::printf("  type %2d (kq512 %-5s): vs ggml dot %zu diffs, gu %zu diffs, row-split %zu diffs\n",
                        type, ggml_type_name((ggml_type) type), dot_diff, gu_diff, split_diff);
            bad += dot_diff != 0 || gu_diff != 0 || split_diff != 0;
        }
        // Mixed gate/up (Q4_K gate + Q5_K up) through native_gu_rows
        cpu::NativeFmt fm;
        std::string ferr;
        if (!cpu::native_fmt(12, 7, H, FF, fm, ferr, 6.0f, 13)) {
            std::printf("native_fmt(12/13, 7): %s\n", ferr.c_str());
            ++bad;
        } else {
            std::vector<uint8_t> blob(fm.bytes);
            fill(blob, 12, 4100, 0);
            std::vector<uint8_t> upb(fm.down_off - fm.up_off);
            fill(upb, 13, 4101, 0);
            std::memcpy(blob.data() + fm.up_off, upb.data(), upb.size());
            std::mt19937 rng(42);
            std::vector<std::vector<uint8_t>> act(4, std::vector<uint8_t>(cpu::kNativeActBytes));
            std::vector<float> xin(H);
            const void* ap[4];
            for (int t = 0; t < 4; ++t) {
                for (auto& v : xin) v = (float) ((int) (rng() % 2001) - 1000) / 400.f;
                cpu::native_quant_act(fm, xin.data(), act[(size_t) t].data());
                ap[t] = act[(size_t) t].data();
            }
            std::vector<std::vector<float>> f4(4, std::vector<float>(FF)), f1(4, std::vector<float>(FF));
            float* pf4[4] = {f4[0].data(), f4[1].data(), f4[2].data(), f4[3].data()};
            cpu::native_gu_rows(fm, blob.data(), ap, 4, pf4, 0, FF);
            size_t mix_diff = 0;
            for (int t = 0; t < 4; ++t) {
                const void* a1[1] = {ap[t]};
                float* p1[1] = {f1[(size_t) t].data()};
                cpu::native_gu_rows(fm, blob.data(), a1, 1, p1, 0, FF);
                for (int r = 0; r < FF; ++r) mix_diff += std::memcmp(&f4[(size_t) t][(size_t) r], &f1[(size_t) t][(size_t) r], 4) != 0;
            }
            bad += mix_diff != 0;
        }
    }

    std::printf("iq_avx512_test: %s\n", bad ? "FAIL" : "PASS");

    if (bench) {   // 200 gate/up rows (the pool's task) per call, the weights cache-resident: one thread's decode rate
        std::mt19937 rng(5);
        for (const Frozen& fz : kFrozen) {
            const int type = fz.type;
            cpu::NativeFmt f;
            std::string err;
            cpu::native_fmt(type, 20, H, FF, f, err);
            std::vector<uint8_t> blob(2 * f.up_off);
            fill(blob, type, 5, 0);
            std::vector<uint8_t> act(cpu::kNativeActBytes);
            std::vector<float> x(H);
            for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 400.f;
            cpu::native_quant_act(f, x.data(), act.data());
            for (int nt : {1, 2, 4}) {
                const void* ap[MAXT];
                std::vector<std::vector<float>> ff(nt, std::vector<float>(FF));
                float* fp[MAXT];
                for (int t = 0; t < nt; ++t) { ap[t] = act.data(); fp[t] = ff[(size_t) t].data(); }
                const int reps = 400;
                double best = 1e30;
                for (int k = 0; k < 5; ++k) {
                    const double t0 = now_ms();
                    for (int r = 0; r < reps; ++r) cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fp, 0, 200);
                    best = std::min(best, now_ms() - t0);
                }
                std::printf("  bench type %2d nt %d: %.2f us per 200 rows, %.2f GB/s (in cache, one thread)\n", type, nt,
                            best * 1e3 / reps, (double) reps * 400.0 * (double) f.gu_row / (best * 1e6));
            }
        }
    }
    return bad == 0 ? 0 : 1;
}
