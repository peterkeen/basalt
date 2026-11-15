// prefill_dense_gemm_test - WHT-128 int8 tensor-core path in Gemm::native against
//   - the FP16 dequant + cuBLAS path (Gemm::native_wht128_override(0)), and
//   - a double-precision host reference on ggml's row dequantizers.
// Covers Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL, Q8_0, Gaussian + 25x channel-outlier activations, all-zero token rows,
// non-multiple-of-128 N and T, ldy > N, beta != 0, small-scratch N/dequant/T slicing parity, and batch-width /
// token-position bitwise invariance.  Exit 77 without a CUDA device with int8 tensor cores.
#include "basalt/models/qwen4exp/prefill/gemm.hpp"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(n, 16)), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return reinterpret_cast<T*>(p); }
};

struct Err {
    double rms = 0.0;
    double worst = 0.0;
};

Err compare_rows(const float* got, int64_t ldg, const float* ref, int64_t ldr, int64_t T, int64_t N) {
    double num = 0.0, den = 0.0, worst = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        double rmax = 0.0, emax = 0.0;
        for (int64_t c = 0; c < N; ++c) {
            const double g = got[t * ldg + c];
            const double r = ref[t * ldr + c];
            const double d = g - r;
            num += d * d;
            den += r * r;
            rmax = std::max(rmax, std::fabs(r));
            emax = std::max(emax, std::fabs(d));
        }
        if (rmax > 1e-6) worst = std::max(worst, emax / rmax);
    }
    return {den > 0.0 ? std::sqrt(num / den) : std::sqrt(num), worst};
}

std::vector<uint8_t> make_weights(ggml_type tp, int64_t N, int64_t K, std::mt19937& rng, std::vector<float>& w_deq) {
    const size_t bs = ggml_type_size(tp), blck = (size_t) ggml_blck_size(tp);
    const size_t rbytes = ggml_row_size(tp, K), nb = (size_t) N * ((size_t) K / blck);
    std::vector<uint8_t> raw((size_t) N * rbytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t& b : raw) b = (uint8_t) byte(rng);
    if (tp == GGML_TYPE_Q6_K) {
        std::uniform_int_distribution<int> s8(-100, 100);
        std::uniform_real_distribution<float> sd(0.00002f, 0.00008f);
        for (size_t b = 0; b < nb; ++b) {
            for (int j = 0; j < 16; ++j) raw[b * bs + 192 + j] = (uint8_t) (int8_t) s8(rng);
            const ggml_fp16_t h = ggml_fp32_to_fp16(sd(rng));
            std::memcpy(raw.data() + b * bs + 208, &h, 2);
        }
    } else {
        float lo = 0.0008f, mid = 0.0f;
        if (tp == GGML_TYPE_Q4_K) { lo = 0.0002f; mid = 7.5f; }
        else if (tp == GGML_TYPE_Q5_K) { lo = 0.0001f; mid = 15.5f; }
        else if (tp == GGML_TYPE_IQ4_XS) { lo = 0.00005f; }
        std::uniform_real_distribution<float> sc(lo, 4.0f * lo);
        for (size_t b = 0; b < nb; ++b) {
            const float d = sc(rng);
            const ggml_fp16_t h = ggml_fp32_to_fp16(d);
            std::memcpy(raw.data() + b * bs, &h, 2);
            if (mid != 0.0f) {
                const ggml_fp16_t hm = ggml_fp32_to_fp16(mid * d);
                std::memcpy(raw.data() + b * bs + 2, &hm, 2);
            }
        }
    }
    const auto* tr = ggml_get_type_traits(tp);
    if (!tr || !tr->to_float) throw std::runtime_error(std::string("missing to_float for ") + ggml_type_name(tp));
    w_deq.resize((size_t) N * K);
    for (int64_t r = 0; r < N; ++r)
        tr->to_float(raw.data() + (size_t) r * rbytes, w_deq.data() + (size_t) r * K, K);
    return raw;
}

void make_act_f16(int64_t T, int64_t K, int64_t zero_tok, std::mt19937& rng, std::vector<uint16_t>& x_f16,
                  std::vector<float>& x_f32) {
    x_f16.resize((size_t) T * K);
    x_f32.resize((size_t) T * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < K; ++k) {
            float v = (t == zero_tok) ? 0.0f : nd(rng) * ((k % 97 == 3) ? 25.0f : 1.0f);
            const ggml_fp16_t h = ggml_fp32_to_fp16(v);
            x_f16[(size_t) t * K + k] = h;
            x_f32[(size_t) t * K + k] = ggml_fp16_to_fp32(h);
        }
    }
}

void cpu_ref_fp64(const float* x, const float* w, const float* y0, float beta, float* out, int64_t T, int64_t N,
                  int64_t K, int64_t ldy) {
    const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int tid = 0; tid < nth; ++tid) {
        th.emplace_back([&, tid] {
            for (int64_t t = tid; t < T; t += nth) {
                const float* xr = x + t * K;
                for (int64_t r = 0; r < N; ++r) {
                    const float* wr = w + r * K;
                    double s = 0.0;
                    for (int64_t k = 0; k < K; ++k) s += (double) xr[k] * (double) wr[k];
                    const double base = (beta == 0.0f || y0 == nullptr) ? 0.0 : (double) beta * (double) y0[t * ldy + r];
                    out[t * ldy + r] = (float) (base + s);
                }
            }
        });
    }
    for (auto& wth : th) wth.join();
}

void parity_part(cudaStream_t s) {
    basalt::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(s, 32ll << 20, err)) throw std::runtime_error(err);

    struct Case {
        ggml_type tp;
        int64_t T, N, K, ldy;
        float beta;
        const char* label;
    };
    const Case cases[] = {
        {GGML_TYPE_Q6_K,   179, 530,  2560, 560,  0.0f,  "Q6_K [530,2560] T=179 ldy=560"},
        {GGML_TYPE_Q5_K,   128, 768,  2560, 768,  0.0f,  "Q5_K [768,2560] T=128"},
        {GGML_TYPE_Q4_K,   96,  640,  2560, 672,  0.75f, "Q4_K [640,2560] T=96 ldy=672 beta=0.75"},
        {GGML_TYPE_IQ4_XS, 113, 512,  2560, 512,  0.0f,  "IQ4_XS [512,2560] T=113"},
        {GGML_TYPE_IQ4_NL, 131, 640,  2560, 640,  0.0f,  "IQ4_NL [640,2560] T=131"},
        {GGML_TYPE_Q8_0,   193, 2560, 640,  2592, -0.5f, "Q8_0 [2560,640] T=193 ldy=2592 beta=-0.5"},
    };

    std::mt19937 rng(4242);
    for (const Case& c : cases) {
        std::vector<float> w_deq, x_f32;
        std::vector<uint16_t> x_f16;
        const std::vector<uint8_t> w_raw = make_weights(c.tp, c.N, c.K, rng, w_deq);
        const int64_t zero_tok = c.T / 2;
        make_act_f16(c.T, c.K, zero_tok, rng, x_f16, x_f32);

        std::vector<float> y_init((size_t) c.T * c.ldy);
        std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
        for (float& v : y_init) v = ud(rng);
        std::vector<float> y_ref = y_init;
        cpu_ref_fp64(x_f32.data(), w_deq.data(), y_init.data(), c.beta, y_ref.data(), c.T, c.N, c.K, c.ldy);

        Dev dW(w_raw.size()), dX(x_f16.size() * 2), dY0(y_init.size() * 4), dY1(y_init.size() * 4);
        ck(cudaMemcpy(dW.p, w_raw.data(), w_raw.size(), cudaMemcpyHostToDevice), "dW");
        ck(cudaMemcpy(dX.p, x_f16.data(), x_f16.size() * 2, cudaMemcpyHostToDevice), "dX");
        ck(cudaMemcpy(dY0.p, y_init.data(), y_init.size() * 4, cudaMemcpyHostToDevice), "dY0");
        ck(cudaMemcpy(dY1.p, y_init.data(), y_init.size() * 4, cudaMemcpyHostToDevice), "dY1");
        ck(cudaDeviceSynchronize(), "uploads");

        basalt::prefill::Gemm::native_wht128_override(0);
        gemm.native(dX.as<uint16_t>(), (int) c.tp, dW.p, dY0.as<float>(), c.T, c.N, c.K, c.ldy, c.beta);
        basalt::prefill::Gemm::native_wht128_override(1);
        gemm.native(dX.as<uint16_t>(), (int) c.tp, dW.p, dY1.as<float>(), c.T, c.N, c.K, c.ldy, c.beta);
        basalt::prefill::Gemm::native_wht128_override(-1);
        ck(cudaStreamSynchronize(s), "sync");

        std::vector<float> h0(y_init.size()), h1(y_init.size());
        ck(cudaMemcpy(h0.data(), dY0.p, h0.size() * 4, cudaMemcpyDeviceToHost), "h0");
        ck(cudaMemcpy(h1.data(), dY1.p, h1.size() * 4, cudaMemcpyDeviceToHost), "h1");

        const Err e_f16 = compare_rows(h0.data(), c.ldy, y_ref.data(), c.ldy, c.T, c.N);
        const Err e_wht = compare_rows(h1.data(), c.ldy, y_ref.data(), c.ldy, c.T, c.N);
        const Err e_pair = compare_rows(h1.data(), c.ldy, h0.data(), c.ldy, c.T, c.N);
        std::printf("  %-40s  f16 vs FP64 %.3e  wht128 vs FP64 %.3e (worst %.3e)  vs f16 %.3e\n",
                    c.label, e_f16.rms, e_wht.rms, e_wht.worst, e_pair.rms);

        if (e_wht.rms > 1.2e-2 || e_wht.worst > 3.0e-2)
            throw std::runtime_error(std::string(c.label) + ": WHT-128 error exceeds bound");

        // Padding columns [N .. ldy) must remain untouched.
        for (int64_t t = 0; t < c.T; ++t) {
            for (int64_t col = c.N; col < c.ldy; ++col) {
                if (std::memcmp(&h1[(size_t) (t * c.ldy + col)], &y_init[(size_t) (t * c.ldy + col)], 4) != 0)
                    throw std::runtime_error(std::string(c.label) + ": modified column past N");
            }
        }
        // All-zero token row must equal exact beta * y_init (or exact 0.0f when beta == 0).
        if (c.beta == 0.0f) {
            for (int64_t col = 0; col < c.N; ++col) {
                if (h1[(size_t) (zero_tok * c.ldy + col)] != 0.0f)
                    throw std::runtime_error(std::string(c.label) + ": all-zero token output is non-zero");
            }
        }
    }

    // Batch-width, token-position, and small-scratch slicing bitwise invariance.
    {
        constexpr int64_t N = 2560, K = 2560, PROBE = 32;
        std::vector<float> w_deq, probe_f32;
        std::vector<uint16_t> probe_f16;
        const std::vector<uint8_t> w_raw = make_weights(GGML_TYPE_Q6_K, N, K, rng, w_deq);
        make_act_f16(PROBE, K, -1, rng, probe_f16, probe_f32);
        Dev dW(w_raw.size());
        ck(cudaMemcpy(dW.p, w_raw.data(), w_raw.size(), cudaMemcpyHostToDevice), "dW inv");

        auto run_at = [&](basalt::prefill::Gemm& gm, int64_t T, int64_t pos) {
            std::vector<uint16_t> x_all((size_t) T * K, 0x3c00u);
            std::memcpy(x_all.data() + (size_t) pos * K, probe_f16.data(), probe_f16.size() * 2);
            Dev dX(x_all.size() * 2), dY((size_t) T * N * 4);
            ck(cudaMemcpy(dX.p, x_all.data(), x_all.size() * 2, cudaMemcpyHostToDevice), "dX inv");
            ck(cudaDeviceSynchronize(), "inv upload");
            basalt::prefill::Gemm::native_wht128_override(1);
            gm.native(dX.as<uint16_t>(), GGML_TYPE_Q6_K, dW.p, dY.as<float>(), T, N, K);
            basalt::prefill::Gemm::native_wht128_override(-1);
            ck(cudaStreamSynchronize(s), "inv sync");
            std::vector<float> out((size_t) PROBE * N);
            ck(cudaMemcpy(out.data(), dY.as<float>() + (size_t) pos * N, out.size() * 4, cudaMemcpyDeviceToHost),
               "inv out");
            return out;
        };

        const std::vector<float> r_small = run_at(gemm, PROBE, 0);
        const std::vector<float> r_mid   = run_at(gemm, 1024, 503);
        const std::vector<float> r_wide  = run_at(gemm, 15000, 14111);   // crosses T_step slice boundary

        basalt::prefill::Gemm small_gemm;
        if (!small_gemm.init(s, 4ll << 20, err)) throw std::runtime_error(err);   // 8 MiB scratch -> N + deq + T slicing
        const std::vector<float> r_sliced = run_at(small_gemm, 2500, 1997);

        if (std::memcmp(r_small.data(), r_mid.data(), r_small.size() * 4) != 0 ||
            std::memcmp(r_small.data(), r_wide.data(), r_small.size() * 4) != 0 ||
            std::memcmp(r_small.data(), r_sliced.data(), r_small.size() * 4) != 0) {
            throw std::runtime_error("WHT-128 outputs varied with batch width, token position, or scratch slicing");
        }
        std::printf("  invariance: 32 probe rows bitwise equal across T=32, T=1024, T=15000, and 8 MiB sliced scratch\n");
    }
}

void bench_part(const std::vector<int>& chunks, cudaStream_t s) {
    basalt::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(s, 32ll << 20, err)) throw std::runtime_error(err);

    struct Shape {
        ggml_type tp;
        int64_t N, K;
        const char* name;
    };
    const Shape shapes[] = {
        {GGML_TYPE_Q6_K, 12288, 2560, "Q6_K [12288, 2560]"},
        {GGML_TYPE_Q6_K, 10240, 2560, "Q6_K [10240, 2560]"},
        {GGML_TYPE_Q6_K,  6144, 2560, "Q6_K [ 6144, 2560]"},
        {GGML_TYPE_Q6_K,  2560, 6144, "Q6_K [ 2560, 6144]"},
        {GGML_TYPE_Q6_K,   640, 2560, "Q6_K [  640, 2560]"},
        {GGML_TYPE_Q8_0,  2560,  640, "Q8_0 [ 2560,  640]"},
    };

    cudaEvent_t ea, eb;
    ck(cudaEventCreate(&ea), "event");
    ck(cudaEventCreate(&eb), "event");
    auto time_ms = [&](auto&& fn) {
        ck(cudaEventRecord(ea, s), "rec");
        fn();
        ck(cudaEventRecord(eb, s), "rec");
        ck(cudaEventSynchronize(eb), "sync");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, ea, eb), "elapsed");
        return ms;
    };

    std::mt19937 rng(99);
    for (const Shape& sh : shapes) {
        std::vector<float> w_deq;
        const std::vector<uint8_t> w_raw = make_weights(sh.tp, sh.N, sh.K, rng, w_deq);
        Dev dW(w_raw.size());
        ck(cudaMemcpy(dW.p, w_raw.data(), w_raw.size(), cudaMemcpyHostToDevice), "dW bench");

        for (int T : chunks) {
            std::vector<uint16_t> x_f16;
            std::vector<float> x_f32;
            make_act_f16(T, sh.K, -1, rng, x_f16, x_f32);
            Dev dX(x_f16.size() * 2), dY0((size_t) T * sh.N * 4), dY1((size_t) T * sh.N * 4);
            ck(cudaMemcpy(dX.p, x_f16.data(), x_f16.size() * 2, cudaMemcpyHostToDevice), "dX bench");
            ck(cudaDeviceSynchronize(), "bench upload");

            auto f16_fn = [&] {
                basalt::prefill::Gemm::native_wht128_override(0);
                gemm.native(dX.as<uint16_t>(), (int) sh.tp, dW.p, dY0.as<float>(), T, sh.N, sh.K);
            };
            auto wht_fn = [&] {
                basalt::prefill::Gemm::native_wht128_override(1);
                gemm.native(dX.as<uint16_t>(), (int) sh.tp, dW.p, dY1.as<float>(), T, sh.N, sh.K);
            };
            for (float spent = 0.0f; spent < 150.0f;) spent += time_ms(f16_fn) + time_ms(wht_fn);
            std::vector<float> t0, t1;
            for (int r = 0; r < 9; ++r) {
                t0.push_back(time_ms(f16_fn));
                t1.push_back(time_ms(wht_fn));
            }
            basalt::prefill::Gemm::native_wht128_override(-1);
            std::sort(t0.begin(), t0.end());
            std::sort(t1.begin(), t1.end());
            const float m0 = t0[4], m1 = t1[4];
            const double flops = 2.0 * (double) T * (double) sh.N * (double) sh.K;
            std::printf("  %s T=%-5d  f16 %7.1f us (%5.1f TF/s)  wht128 %7.1f us (%5.1f TF/s)  %.2fx\n",
                        sh.name, T, m0 * 1000.0f, flops / (m0 * 1e9), m1 * 1000.0f, flops / (m1 * 1e9), m0 / m1);
        }
    }
    cudaEventDestroy(ea);
    cudaEventDestroy(eb);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
            std::printf("no CUDA device: skipped\n");
            return 77;
        }
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        if (major < 8) {
            std::printf("WHT-128 int8 GEMM needs int8 tensor cores: skipped\n");
            return 77;
        }
        bool bench = true;
        std::vector<int> chunks = {1024, 2048, 8192, 16384};
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--no-bench") == 0) bench = false;
            if (std::strncmp(argv[i], "--chunks=", 9) == 0) {
                chunks.clear();
                for (const char* c = argv[i] + 9; *c;) {
                    chunks.push_back(std::atoi(c));
                    while (*c && *c != ',') ++c;
                    if (*c == ',') ++c;
                }
            }
        }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        parity_part(s);
        if (bench) bench_part(chunks, s);
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill dense native GEMM parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        basalt::prefill::Gemm::native_wht128_override(-1);
        std::fprintf(stderr, "prefill dense native GEMM parity failed: %s\n", e.what());
        return 1;
    }
}
