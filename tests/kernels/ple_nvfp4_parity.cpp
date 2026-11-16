// A synthetic NVFP4_32 PLE table through PleTable: every element against an independent field-by-field
// E2M1 x UE4M3 x ple_scale decode, both with and without `qwen4exp.ple.scale`, across mapped reads,
// issue/collect, gather_rows and the Direct 4 KiB reader.
#include "basalt/models/qwen4exp/kernels/ngram.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

namespace k = basalt::kernels;

constexpr uint32_t kTypeNvfp4_32 = 47;
constexpr uint32_t kRows = 4096;

template <typename T>
void put(std::vector<uint8_t>& b, T v) {
    const auto* p = (const uint8_t*) &v;
    b.insert(b.end(), p, p + sizeof(T));
}

void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// Independent field-by-field E2M1 and UE4M3 decode.
float ref_e2m1(int nib) {
    const float sign = (nib & 8) ? -1.0f : 1.0f;
    const int e = (nib >> 1) & 3, m = nib & 1;
    if (e == 0) return m == 0 ? 0.0f : sign * 0.5f;
    return sign * std::ldexp(1.0f + 0.5f * (float) m, e - 1);
}

float ref_ue4m3(uint8_t raw) {
    const uint8_t u = raw & 0x7Fu;
    if (u == 0 || u == 0x7Fu) return 0.0f;
    const int e = (u >> 3) & 0xF, m = u & 0x7;
    if (e == 0) return std::ldexp((float) m, -9);
    return std::ldexp(1.0f + (float) m / 8.0f, e - 7);
}

void fill_row_bytes(uint32_t row, uint8_t out90[k::PLE_ROW_BYTES]) {
    if (row < 32) {
        // Cover every UE4M3 byte 0..255 across the first 26 rows (10 sub-blocks per row) and every E2M1 nibble.
        for (int b = 0; b < 5; ++b) {
            uint8_t* blk = out90 + b * 18;
            const int sb0 = (int) row * 10 + 2 * b;
            blk[0] = (uint8_t) (sb0 & 0xFF);
            blk[1] = (uint8_t) ((sb0 + 1) & 0xFF);
            for (int s = 0; s < 2; ++s) {
                for (int j = 0; j < 8; ++j) {
                    const int lo = (j + s * 3 + b + (int) row) & 0xF;
                    const int hi = (lo + 8 + s) & 0xF;
                    blk[2 + s * 8 + j] = (uint8_t) (lo | (hi << 4));
                }
            }
        }
        return;
    }
    std::mt19937 rng(row * 2654435761u + 29u);
    for (int i = 0; i < k::PLE_ROW_BYTES; ++i) out90[i] = (uint8_t) (rng() & 0xFFu);
}

void ref_decode_row(uint32_t row, float ple_scale, float out160[k::PLE_HEAD_DIM]) {
    uint8_t raw[k::PLE_ROW_BYTES];
    fill_row_bytes(row, raw);
    for (int b = 0; b < 5; ++b) {
        const uint8_t* blk = raw + b * 18;
        for (int s = 0; s < 2; ++s) {
            const double sc = 0.5 * (double) ref_ue4m3(blk[s]) * (double) ple_scale;
            const uint8_t* qs = blk + 2 + s * 8;
            for (int j = 0; j < 8; ++j) {
                const double c0 = 2.0 * (double) ref_e2m1(qs[j] & 0xF);
                const double c1 = 2.0 * (double) ref_e2m1(qs[j] >> 4);
                out160[b * 32 + s * 16 + j] = (float) (c0 * sc);
                out160[b * 32 + s * 16 + j + 8] = (float) (c1 * sc);
            }
        }
    }
}

bool close_enough(float got, float want) {
    if (want == 0.0f) return got == 0.0f;
    const float diff = std::fabs(got - want);
    const float tol = 2.0f * std::numeric_limits<float>::epsilon() * std::fabs(want);
    return diff <= tol;
}

bool write_gguf(const std::string& path, bool with_scale, float scale_val) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 1);                      // n_tensors
    put<uint64_t>(b, with_scale ? 1 : 0);     // n_kv
    if (with_scale) {
        put_str(b, "qwen4exp.ple.scale");
        put<uint32_t>(b, 6);                  // GGUF metadata type 6 = float32
        put<float>(b, scale_val);
    }
    put_str(b, "per_layer_token_embd.weight");
    put<uint32_t>(b, 2);
    put<uint64_t>(b, k::PLE_HEAD_DIM);
    put<uint64_t>(b, kRows);
    put<uint32_t>(b, kTypeNvfp4_32);
    put<uint64_t>(b, 0);
    b.resize((b.size() + 31) / 32 * 32);
    uint8_t row_buf[k::PLE_ROW_BYTES];
    for (uint32_t r = 0; r < kRows; ++r) {
        fill_row_bytes(r, row_buf);
        b.insert(b.end(), row_buf, row_buf + k::PLE_ROW_BYTES);
    }
    std::ofstream out(path, std::ios::binary);
    out.write((const char*) b.data(), (std::streamsize) b.size());
    return out.good();
}

void check_rows(const k::PleTable& table, const char* what, float scale, const std::vector<uint32_t>& rows,
                long long& checked, int& bad) {
    for (uint32_t r : rows) {
        float got[k::PLE_HEAD_DIM], want[k::PLE_HEAD_DIM];
        table.read_row(r, got);
        ref_decode_row(r, scale, want);
        for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
            ++checked;
            if (!close_enough(got[i], want[i])) {
                if (bad < 8)
                    std::fprintf(stderr, "%s row %u elem %d: got %.9g, want %.9g\n", what, r, i,
                                 (double) got[i], (double) want[i]);
                ++bad;
            }
        }
    }
}

int run_case(const std::string& path, bool with_scale, float scale, long long& checked, int& bad) {
    const float eff_scale = with_scale ? scale : 1.0f;
    if (!write_gguf(path, with_scale, scale)) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return 2;
    }

    std::vector<uint32_t> probes;
    for (uint32_t r = 0; r < 32; ++r) probes.push_back(r);
    probes.push_back(kRows - 1);
    std::mt19937 rng(19);
    for (int i = 0; i < 64; ++i) probes.push_back(rng() % kRows);

    std::string err;
    k::PleTable table;
    k::PleIoOptions mo;
    mo.mode = k::PleIo::Mmap;
    if (!table.open(path, err, mo)) {
        std::fprintf(stderr, "open: %s\n", err.c_str());
        std::filesystem::remove(path);
        return 1;
    }
    if (std::strcmp(table.format(), "NVFP4_32") != 0 || table.rows() != kRows) {
        std::fprintf(stderr, "open gave %s x %llu, expected NVFP4_32 x %u\n", table.format(),
                     (unsigned long long) table.rows(), kRows);
        std::filesystem::remove(path);
        return 1;
    }
    check_rows(table, with_scale ? "mapped(scaled)" : "mapped(unit)", eff_scale, probes, checked, bad);

    uint32_t batch[k::PLE_N_HEADS];
    for (int h = 0; h < k::PLE_N_HEADS; ++h) batch[h] = probes[(h * 7) % probes.size()];
    std::vector<float> out(k::PLE_N_HEADS * k::PLE_HEAD_DIM);
    if (!table.issue(batch) || !table.collect(out.data(), err)) {
        std::fprintf(stderr, "issue/collect: %s\n", err.c_str());
        std::filesystem::remove(path);
        return 1;
    }
    for (int h = 0; h < k::PLE_N_HEADS; ++h) {
        float want[k::PLE_HEAD_DIM];
        ref_decode_row(batch[h], eff_scale, want);
        for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
            ++checked;
            if (!close_enough(out[h * k::PLE_HEAD_DIM + i], want[i])) {
                if (bad < 8) std::fprintf(stderr, "batch row %u elem %d differs\n", batch[h], i);
                ++bad;
            }
        }
    }
    table.close();

    k::PleTable direct;
    k::PleIoOptions dopt;
    dopt.mode = k::PleIo::Direct;
    dopt.cache_rows = 1u << 16;
    if (!direct.open(path, err, dopt)) {
        std::fprintf(stderr, "direct open: %s\n", err.c_str());
        std::filesystem::remove(path);
        return 1;
    }
    check_rows(direct, with_scale ? "direct(scaled)" : "direct(unit)", eff_scale, probes, checked, bad);

    std::vector<uint32_t> multi(4 * k::PLE_N_HEADS);
    for (size_t i = 0; i < multi.size(); ++i) multi[i] = probes[(i * 13) % probes.size()];
    std::vector<float> gout(multi.size() * k::PLE_HEAD_DIM);
    if (!direct.gather_rows(multi.data(), 4, gout.data(), err)) {
        std::fprintf(stderr, "gather_rows: %s\n", err.c_str());
        std::filesystem::remove(path);
        return 1;
    }
    for (size_t m = 0; m < multi.size(); ++m) {
        float want[k::PLE_HEAD_DIM];
        ref_decode_row(multi[m], eff_scale, want);
        for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
            ++checked;
            if (!close_enough(gout[m * k::PLE_HEAD_DIM + i], want[i])) {
                if (bad < 8) std::fprintf(stderr, "gather_rows row %u elem %d differs\n", multi[m], i);
                ++bad;
            }
        }
    }
    direct.close();
    std::filesystem::remove(path);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;

    const std::string path = (std::filesystem::temp_directory_path() / "ple_nvfp4_parity.gguf").string();
    long long checked = 0;
    int bad = 0;

    if (const int rc = run_case(path, false, 1.0f, checked, bad)) return rc;
    if (const int rc = run_case(path, true, 3.3239148e-5f, checked, bad)) return rc;

    // Non-positive qwen4exp.ple.scale is refused at open.
    if (!write_gguf(path, true, 0.0f)) return 2;
    {
        std::string err;
        k::PleTable t;
        k::PleIoOptions mo;
        mo.mode = k::PleIo::Mmap;
        if (t.open(path, err, mo)) {
            std::fprintf(stderr, "zero qwen4exp.ple.scale unexpectedly accepted\n");
            ++bad;
        }
        std::filesystem::remove(path);
    }

    if (bench) {
        constexpr int kToks = 10000;
        std::vector<uint8_t> raw((size_t) k::PLE_N_HEADS * k::PLE_ROW_BYTES);
        for (int h = 0; h < k::PLE_N_HEADS; ++h) fill_row_bytes((uint32_t) (32 + h), raw.data() + h * k::PLE_ROW_BYTES);
        float out[k::NG_N_EMBD];
        volatile float sink = 0.0f;
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < kToks; ++it) {
            for (int h = 0; h < k::PLE_N_HEADS; ++h)
                k::nvfp4_ple_dequant_row(raw.data() + h * k::PLE_ROW_BYTES, 3.3239148e-5f, out + h * k::PLE_HEAD_DIM);
            sink = sink + out[it & 255];
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        (void) sink;
        std::printf("NVFP4_32 PLE decode bench: %.3f us / token (16 rows = 2560 values, %.2f Mrows/s)\n",
                    us / kToks, (double) (kToks * k::PLE_N_HEADS) / us);
    }

    std::printf("NVFP4_32 PLE parity: %u-row synthetic table, %lld elements checked, %s\n", kRows, checked,
                bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
