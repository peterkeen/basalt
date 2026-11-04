// tests/ngram/ple_reader_test.cpp - the direct SSD row reader against ground truth.
//
//   ple_reader_test --selftest [--dir D]        synthetic table file; CPU and disk only, no model, no GPU.
//                                               Includes the --ple-file cases: the table read out of a GGUF that is
//                                               not the model file, alone or among other tensors, and the files it
//                                               has to refuse.
//   ple_reader_test --gguf SHARD2 [--rows N]    the real table: Direct vs Mmap bytes for N random rows (+ the
//                                               16 rows of every token in --tokens FILE), with read latencies
//   ple_reader_test --cross-check A[,B,...]     the same rows of several real tables (a .basalt pack and/or plain
//                                               GGUFs), compared: same encoding must be bit-identical
//
// Every row the synthetic table holds encodes its own index, so a wrong offset, a straddle mishandled or a
// dedup slot mixed up shows as a mismatch rather than as plausible data.
#include "basalt/artifact/dequant.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/ngram/ple_reader.hpp"
#include "basalt/platform/direct_file.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ng = basalt::ngram;
namespace k = basalt::kernels;
using basalt::platform::now_us;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                            \
    do {                                                         \
        if (!(c)) {                                              \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                   \
            std::fprintf(stderr, "\n");                          \
            ++g_fail;                                            \
        }                                                        \
    } while (0)

constexpr uint64_t HEADER = 192;   // the real shard's data offset, so rows are misaligned the same way

void expected_row(uint32_t row, uint32_t rb, uint8_t* out) {
    for (uint32_t b = 0; b < rb; ++b) out[b] = (uint8_t) ((row * 2654435761u + b * 97u) >> 7);
    std::memcpy(out, &row, 4);
}

bool make_table(const std::string& path, uint32_t rows, uint32_t rb) {
    std::ofstream f(path, std::ios::binary);
    std::vector<uint8_t> head(HEADER, 0xAB);
    f.write((const char*) head.data(), (std::streamsize) head.size());
    std::vector<uint8_t> r(rb);
    for (uint32_t i = 0; i < rows; ++i) {
        expected_row(i, rb, r.data());
        f.write((const char*) r.data(), rb);
    }
    return (bool) f;
}

bool check_rows(ng::PleReader& rd, const std::vector<uint32_t>& rows, uint32_t n_rows, uint32_t rb, const char* what) {
    std::vector<uint8_t> out(rows.size() * rb, 0xCC);
    std::string err;
    const auto t = rd.issue(rows.data(), rows.size(), out.data());
    if (!rd.collect(t, err)) { CHECK(false, "%s: collect failed: %s", what, err.c_str()); return false; }
    std::vector<uint8_t> want(rb);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] >= n_rows) std::memset(want.data(), 0, rb);
        else expected_row(rows[i], rb, want.data());
        if (std::memcmp(want.data(), &out[i * rb], rb) != 0) {
            CHECK(false, "%s: row %u (index %zu) differs", what, rows[i], i);
            return false;
        }
    }
    return true;
}

// row_bytes: ng::ROW_BYTES (90, IQ4_NL) is the production default; 110 (Q5_0) and 170 (Q8_0) are
// run too, through the exact same generic row_bytes path -- nothing here is IQ4_NL-specific, so a second row
// size is the correctness evidence for the generic path.
int selftest(const std::string& dir, uint32_t rb) {
    const uint32_t N = 500000;                          // 45 MB: large enough for thousands of distinct pages
    const std::string path = dir + "/ple_reader_selftest_" + std::to_string(rb) + ".bin";
    if (!make_table(path, N, rb)) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return 2; }
    std::mt19937 rng(7);
    for (uint64_t cache : {0ull, 4096ull}) {
        for (uint32_t inflight : {1u, 8u, 64u}) {
            ng::PleReader rd;
            std::string err;
            CHECK(rd.open(path, HEADER, N, inflight, cache, err, rb), "open: %s", err.c_str());
            // decode-shaped tickets: 16 random rows
            for (int t = 0; t < 200; ++t) {
                std::vector<uint32_t> rows(16);
                for (auto& r : rows) r = rng() % N;
                check_rows(rd, rows, N, rb, "decode");
            }
            // rows that straddle a 4 KiB boundary: byte offset of row r is HEADER + rb * r
            std::vector<uint32_t> straddle;
            for (uint32_t r = 0; r < N && straddle.size() < 64; ++r) {
                const uint64_t a = HEADER + (uint64_t) r * rb;
                if (a / 4096 != (a + rb - 1) / 4096) straddle.push_back(r);
            }
            check_rows(rd, straddle, N, rb, "straddle");
            // duplicates, neighbours on one page, the first and last rows, and out-of-range rows
            check_rows(rd, {5, 5, 6, 7, 5, 0, N - 1, N, 0xFFFFFFFFu, 44, 45}, N, rb, "dedup/edges");
            // a prefill-shaped ticket much larger than the in-flight window
            std::vector<uint32_t> bulk(20000);
            for (auto& r : bulk) r = rng() % N;
            check_rows(rd, bulk, N, rb, "bulk");
            // two tickets in flight at once, collected in reverse order
            std::vector<uint32_t> a(16), b(16);
            for (auto& r : a) r = rng() % N;
            for (auto& r : b) r = rng() % N;
            std::vector<uint8_t> oa(16 * rb), ob(16 * rb);
            const auto ta = rd.issue(a.data(), 16, oa.data());
            const auto tb = rd.issue(b.data(), 16, ob.data());
            CHECK(rd.collect(tb, err) && rd.collect(ta, err), "two tickets: %s", err.c_str());
            std::vector<uint8_t> want(rb);
            for (int i = 0; i < 16; ++i) {
                expected_row(a[i], rb, want.data());
                CHECK(!std::memcmp(want.data(), &oa[i * rb], rb), "ticket a row %d", i);
                expected_row(b[i], rb, want.data());
                CHECK(!std::memcmp(want.data(), &ob[i * rb], rb), "ticket b row %d", i);
            }
            if (cache > 0) CHECK(rd.stats().cache_hits > 0, "the row cache never hit");
            CHECK(rd.cache_size() <= rd.cache_capacity(), "row cache exceeded its bound");
        }
    }
    // fault injection: a 3 ms delay must be observed, and must not change the bytes
    {
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 0, err, rb), "open: %s", err.c_str());
        rd.set_injected_delay_us(3000);
        std::vector<uint32_t> rows(16);
        for (auto& r : rows) r = rng() % N;
        const double t0 = now_us();
        check_rows(rd, rows, N, rb, "delayed");
        CHECK(now_us() - t0 >= 3000, "injected delay not observed (%.0f us)", now_us() - t0);
        CHECK(rd.stats().late_injected > 0, "no read was held back");
    }
    // keep-alive: while rows are asked for, a page goes out after `period` without a read; it stops once the
    // window after the last issue has passed, starts again with the next issue (even one the row cache serves),
    // and never shows up as a row read or changes a row
    {
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 4096, err, rb), "open: %s", err.c_str());
        rd.set_keepalive(20.0, 0.5);
        std::vector<uint32_t> rows(16);
        for (auto& r : rows) r = rng() % N;
        check_rows(rd, rows, N, rb, "keep-alive, first ticket");
        const uint64_t reads0 = rd.snapshot().reads;
        const auto nap = [](int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); };
        nap(300);
        const uint64_t k1 = rd.snapshot().keepalive_reads;
        CHECK(k1 >= 8 && k1 <= 20, "keep-alive: %llu reads in 300 ms at a 20 ms period", (unsigned long long) k1);
        CHECK(rd.snapshot().reads == reads0, "keep-alive reads were counted as row reads");
        nap(600);                                          // the 0.5 s window has passed
        const uint64_t k2 = rd.snapshot().keepalive_reads;
        nap(300);
        const uint64_t k3 = rd.snapshot().keepalive_reads;
        CHECK(k3 == k2, "keep-alive went on after the window (%llu -> %llu)", (unsigned long long) k2,
              (unsigned long long) k3);
        CHECK(k2 <= k1 + 20, "keep-alive: %llu reads by the end of a 0.5 s window", (unsigned long long) k2);
        check_rows(rd, rows, N, rb, "keep-alive, cached ticket");   // the row cache serves all 16: no row read
        CHECK(rd.snapshot().reads == reads0, "cached rows were read again");
        nap(200);
        CHECK(rd.snapshot().keepalive_reads >= k3 + 5, "the keep-alive did not start again (%llu -> %llu)",
              (unsigned long long) k3, (unsigned long long) rd.snapshot().keepalive_reads);
        std::vector<uint32_t> more(2000);
        for (auto& r : more) r = rng() % N;
        check_rows(rd, more, N, rb, "keep-alive, rows afterwards");
        rd.set_keepalive(0, 0.5);                           // off
        const uint64_t k4 = rd.snapshot().keepalive_reads;
        nap(200);
        CHECK(rd.snapshot().keepalive_reads <= k4 + 1, "keep-alive went on after it was turned off");
        // a window that lapsed with no request starts again on rearm_keepalive, with no issue
        rd.set_keepalive(20.0, 0.3);
        nap(500);
        const uint64_t k5 = rd.snapshot().keepalive_reads;
        nap(150);
        CHECK(rd.snapshot().keepalive_reads == k5, "keep-alive went on after the window lapsed");
        rd.rearm_keepalive();
        nap(150);
        CHECK(rd.snapshot().keepalive_reads >= k5 + 3, "rearm_keepalive did not start the window again (%llu -> %llu)",
              (unsigned long long) k5, (unsigned long long) rd.snapshot().keepalive_reads);
        rd.reset_stats();
        CHECK(rd.snapshot().keepalive_reads == 0 && rd.snapshot().keepalive_us_max == 0, "reset_stats kept keep-alive counts");
    }
    std::filesystem::remove(path);
    std::printf("ple_reader selftest (row_bytes=%u): %s\n", rb, g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

// A decode-shaped thread (prefetch a window's tokens, then gather exactly those) beside a prompt-shaped thread
// (large gather_rows requests) on one table: every result must equal the mapped table's, and neither side may
// see a ticket error.
void stress(k::PleTable& mm, k::PleTable& direct, double secs) {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> n_decode{0}, n_prompt{0};
    const uint64_t N = direct.rows();
    auto expect = [&](const uint32_t* rows, size_t tokens, const std::vector<float>& got, const char* who) {
        std::vector<float> ref(k::NG_N_EMBD);
        for (size_t t = 0; t < tokens; ++t) {
            mm.gather(rows + t * k::PLE_N_HEADS, ref.data());
            CHECK(!std::memcmp(ref.data(), got.data() + t * k::NG_N_EMBD, ref.size() * sizeof(float)), "%s: token %zu differs", who, t);
        }
    };
    std::thread dec([&] {
        std::mt19937_64 rng(1);
        while (!stop.load()) {
            const size_t T = 1 + rng() % 8;
            std::vector<uint32_t> rows(T * k::PLE_N_HEADS);
            for (auto& r : rows) r = (uint32_t) (rng() % N);
            for (size_t t = 0; t < T; ++t) direct.prefetch_rows(rows.data() + t * k::PLE_N_HEADS);
            std::vector<float> out(T * k::NG_N_EMBD);
            std::string err;
            if (!direct.gather_batch(rows.data(), T, out.data(), err)) { CHECK(false, "decode gather_batch: %s", err.c_str()); break; }
            expect(rows.data(), T, out, "decode");
            ++n_decode;
        }
    });
    std::thread pre([&] {
        std::mt19937_64 rng(2);
        while (!stop.load()) {
            const size_t T = 64 + rng() % 192;
            std::vector<uint32_t> rows(T * k::PLE_N_HEADS);
            for (auto& r : rows) r = (uint32_t) (rng() % N);
            std::vector<float> out(T * k::NG_N_EMBD);
            std::string err;
            if (!direct.gather_rows(rows.data(), T, out.data(), err)) { CHECK(false, "prompt gather_rows: %s", err.c_str()); break; }
            expect(rows.data(), T, out, "prompt");
            ++n_prompt;
        }
    });
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    stop.store(true);
    dec.join();
    pre.join();
    std::printf("stress %.0f s: %llu decode gathers, %llu prompt gathers\n", secs, (unsigned long long) n_decode.load(),
                (unsigned long long) n_prompt.load());
}

int real(const std::string& gguf, int n_random, const std::string& tokens_path, uint32_t inflight, bool direct_first,
         bool direct_only, double stress_secs) {
    k::PleTable mm, direct;
    std::string err;
    basalt::core::ModelFile container;   // a .basalt file serves its source tensors as a GGUF view
    if (gguf.size() > 7 && gguf.compare(gguf.size() - 7, 7, ".basalt") == 0 &&
        (!container.open(gguf, err) || !container.load_meta(err) || !container.register_gguf_view(err))) {
        std::fprintf(stderr, "container open: %s\n", err.c_str());
        return 2;
    }
    k::PleIoOptions mo;
    mo.mode = k::PleIo::Mmap;
    k::PleIoOptions dopt;
    dopt.cache_rows = 0;                               // measure the SSD, not the cache
    dopt.max_inflight = inflight;
    // `--direct-only` measures the direct path with NO mapping of the file alive anywhere in the process: a
    // live section on the same file forces the file system to keep cached and non-cached views coherent.
    if (!direct_only && !mm.open(gguf, err, mo)) { std::fprintf(stderr, "mmap open: %s\n", err.c_str()); return 2; }
    if (!direct.open(gguf, err, dopt)) { std::fprintf(stderr, "direct open: %s\n", err.c_str()); return 2; }
    std::vector<std::vector<uint32_t>> tickets;
    std::mt19937_64 rng(11);
    for (int t = 0; t < n_random / 16; ++t) {
        std::vector<uint32_t> r(16);
        for (auto& x : r) x = (uint32_t) (rng() % direct.rows());
        tickets.push_back(r);
    }
    if (!tokens_path.empty()) {
        std::ifstream f(tokens_path);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        for (char& c : text) if (c == ',') c = ' ';
        std::istringstream in(text);
        std::vector<int32_t> ids;
        for (int32_t v; in >> v;) ids.push_back(v);
        const k::PleConsts C = k::ple_artifact_consts();
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t prev[2] = {i >= 2 ? ids[i - 2] : k::TOKEN_NULL, i >= 1 ? ids[i - 1] : k::TOKEN_NULL};
            std::vector<uint32_t> r(16);
            k::ngram_rows(&ids[i], prev, 1, C, r.data());
            tickets.push_back(r);
        }
    }
    std::vector<float> a(k::NG_N_EMBD), b(k::NG_N_EMBD);
    double t_mm = 0, t_dir = 0, t_issue = 0;
    for (const auto& r : tickets) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool do_direct = (pass == 0) == direct_first;
            const double t0 = now_us();
            if (!do_direct) {
                if (direct_only) continue;
                mm.gather(r.data(), a.data());
                t_mm += now_us() - t0;
            } else {
                const double ti = now_us();
                const bool issued = direct.issue(r.data());
                t_issue += now_us() - ti;
                if (!issued || !direct.collect(b.data(), err)) {
                    std::fprintf(stderr, "direct: %s\n", err.c_str());
                    return 1;
                }
                t_dir += now_us() - t0;
            }
        }
        if (!direct_only)
            CHECK(!std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), "token rows differ between mmap and direct");
    }
    if (stress_secs > 0 && !direct_only) stress(mm, direct, stress_secs);
    std::printf("tokens %zu: mmap %.1f us/token, direct %.1f us/token (issue on this thread %.1f us)\n%s\n",
                tickets.size(), t_mm / (double) tickets.size(), t_dir / (double) tickets.size(),
                t_issue / (double) tickets.size(), direct.io_report().c_str());
    std::printf("ple_reader real-table check: %s\n", g_fail ? "FAILED" : "OK (bit-identical)");
    return g_fail ? 1 : 0;
}

// A Q8_0 PLE row (5 blocks of fp16 scale + 32 int8) decoded by the shared dequantizer against a plain
// reference, on random rows with random scales.
int q8_decode_check() {
    std::mt19937 rng(11);
    int bad = 0;
    for (int t = 0; t < 2000; ++t) {
        uint8_t row[k::PLE_ROW_BYTES_Q8_0];
        for (auto& b : row) b = (uint8_t) rng();
        for (int b = 0; b < k::PLE_HEAD_DIM / 32; ++b) {   // finite scales only
            const uint16_t d = (uint16_t) (0x2400 + (rng() % 0x1800));
            std::memcpy(row + b * 34, &d, 2);
        }
        float got[k::PLE_HEAD_DIM];
        for (int b = 0; b < k::PLE_HEAD_DIM / 32; ++b) basalt::dequantize_q8_0(row + b * 34, got + b * 32);
        for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
            uint16_t h;
            std::memcpy(&h, row + (i / 32) * 34, 2);
            const int e = (h >> 10) & 31, m = h & 1023;
            const float d = (e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m + 1024), e - 25)) *
                            ((h & 0x8000) ? -1.0f : 1.0f);
            const float want = d * (float) (int8_t) row[(i / 32) * 34 + 2 + i % 32];
            if (got[i] != want) ++bad;
        }
    }
    std::printf("Q8_0 PLE row decode: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}

// ---- a table in a GGUF of its own: what `--ple-file` opens -----------------------------------------------
//
// The table is normally a section of the model file.  `--ple-file` points at a GGUF that is not the model: the
// table alone in a plain GGUF (a real IQ3_S shard 2 is exactly that), or a table sitting among a model's other
// tensors (the Q8_0 n-gram GGUFs, and Swift 1.5's shard 1).  These cases are synthetic and small on purpose.
struct SynthTensor {
    std::string name;
    uint32_t type;                      // ggml type id: 2 Q4_0, 8 Q8_0, 20 IQ4_NL
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
};

// Writes a GGUF v3 holding `ts`, in order, each payload starting at a multiple of 32 from the data section's start
// (no metadata, so the reader's default alignment of 32 is what the layout assumes).  Tensor offsets in a GGUF are
// relative to the data section, which starts after the header: an offset of 0 is NOT byte 0 of the file.
bool write_gguf(const std::string& path, const std::vector<SynthTensor>& ts) {
    std::vector<uint8_t> h;
    put<uint32_t>(h, 0x46554747u);   // "GGUF"
    put<uint32_t>(h, 3);
    put<uint64_t>(h, ts.size());
    put<uint64_t>(h, 0);             // no metadata
    std::vector<uint64_t> offs;
    uint64_t off = 0;
    for (const auto& t : ts) { offs.push_back(off); off = (off + t.bytes.size() + 31) / 32 * 32; }
    for (size_t i = 0; i < ts.size(); ++i) {
        put<uint64_t>(h, ts[i].name.size());
        h.insert(h.end(), ts[i].name.begin(), ts[i].name.end());
        put<uint32_t>(h, (uint32_t) ts[i].shape.size());
        for (uint64_t d : ts[i].shape) put<uint64_t>(h, d);
        put<uint32_t>(h, ts[i].type);
        put<uint64_t>(h, offs[i]);
    }
    h.resize((h.size() + 31) / 32 * 32, 0);
    for (size_t i = 0; i < ts.size(); ++i) {
        h.insert(h.end(), ts[i].bytes.begin(), ts[i].bytes.end());
        h.resize((h.size() + 31) / 32 * 32, 0);
    }
    std::ofstream f(path, std::ios::binary);
    f.write((const char*) h.data(), (std::streamsize) h.size());
    return (bool) f;
}

// Q8_0 bytes for `rows` PLE rows, finite scales, every byte a deterministic function of (row, block, index):
// a wrong offset, a wrong stride or another tensor's bytes read as the table show up as a mismatch rather than as
// plausible data.
std::vector<uint8_t> q8_table_bytes(uint32_t rows, uint32_t seed) {
    std::vector<uint8_t> b((size_t) rows * k::PLE_ROW_BYTES_Q8_0);
    for (uint32_t r = 0; r < rows; ++r) {
        uint8_t* row = b.data() + (size_t) r * k::PLE_ROW_BYTES_Q8_0;
        for (int blk = 0; blk < k::PLE_HEAD_DIM / 32; ++blk) {
            const uint16_t scale = (uint16_t) (0x2400 + ((r * 7919u + blk * 104729u + seed) % 0x1800u));
            std::memcpy(row + blk * 34, &scale, 2);
            for (int i = 0; i < 32; ++i)
                row[blk * 34 + 2 + i] = (uint8_t) (r * 31u + blk * 17u + i * 7u + seed);
        }
    }
    return b;
}

// What `--ple-file` has to get right, and what it has to refuse.
int ple_file_check(const std::string& dir) {
    constexpr uint32_t ROWS = 4096;
    const std::vector<uint8_t> table = q8_table_bytes(ROWS, 3);
    const std::vector<uint8_t> decoy = q8_table_bytes(ROWS / 2, 99);
    const std::string alone = dir + "/ple_file_alone.gguf";
    const std::string shared = dir + "/ple_file_shared.gguf";
    CHECK(write_gguf(alone, {{"per_layer_token_embd.weight", 8, {k::PLE_HEAD_DIM, ROWS}, table}}), "write %s", alone.c_str());
    // The decoy comes FIRST: it is what a reader that assumed the table starts at the data section's beginning
    // would decode, and it is a different table, so the mistake is a mismatch and not a plausible row.
    CHECK(write_gguf(shared, {{"token_embd.weight", 8, {k::PLE_HEAD_DIM * 2, ROWS / 2}, decoy},
                              {"per_layer_token_embd.weight", 8, {k::PLE_HEAD_DIM, ROWS}, table}}),
          "write %s", shared.c_str());

    // The same bytes at a non-zero offset must give the same rows, in both read modes, and the values must be what
    // the shared dequantizer makes of them (an independent anchor, not another pass of the same reader).
    const size_t T = 8;                       // gather_rows takes 16 row ids per token
    std::vector<uint32_t> rows(T * k::PLE_N_HEADS);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = (uint32_t) ((i * 3571u + 17u) % ROWS);
    std::string err;
    std::vector<float> ref;
    for (const char* mode_name : {"mmap", "direct"}) {
        for (const std::string& path : {alone, shared}) {
            k::PleTable t;
            k::PleIoOptions io;
            io.mode = std::strcmp(mode_name, "mmap") == 0 ? k::PleIo::Mmap : k::PleIo::Direct;
            io.cache_rows = 0;    // every row from the file, so a wrong offset cannot hide behind a cache hit
            CHECK(t.open(path, err, io), "%s open %s: %s", mode_name, path.c_str(), err.c_str());
            CHECK(t.rows() == ROWS && std::strcmp(t.format(), "Q8_0") == 0,
                  "%s %s: %llu rows of %s", mode_name, path.c_str(), (unsigned long long) t.rows(), t.format());
            CHECK(t.source() == path, "%s %s: source() is '%s'", mode_name, path.c_str(), t.source().c_str());
            std::vector<float> got(T * k::NG_N_EMBD, 0.f);
            CHECK(t.gather_rows(rows.data(), T, got.data(), err), "%s gather %s: %s", mode_name, path.c_str(), err.c_str());
            if (ref.empty()) ref = got;
            else CHECK(ref == got, "%s: rows from %s differ from the other file's", mode_name, path.c_str());
            // 16 rows of one token through the block's own entry point, decoded independently.
            for (uint32_t h = 0; h < 4; ++h) {
                const uint32_t row = rows[h];
                std::vector<float> want(k::PLE_HEAD_DIM);
                const uint8_t* src = table.data() + (size_t) row * k::PLE_ROW_BYTES_Q8_0;
                for (int b = 0; b < k::PLE_HEAD_DIM / 32; ++b) basalt::dequantize_q8_0(src + b * 34, want.data() + b * 32);
                std::vector<float> one(k::PLE_HEAD_DIM);
                t.read_row(row, one.data());
                CHECK(want == one, "%s %s: read_row(%u) is not the Q8_0 decode of its bytes", mode_name, path.c_str(), row);
            }
            t.close();
            CHECK(t.source().empty() && !t.is_open(), "close() left the table open or its source set");
        }
    }

    // What it refuses, and that the reason names the file: with --ple-file the file asked for is not the one the
    // weights come from, and "unexpected shape" alone would not say which table was meant.
    struct Bad {
        const char* what;
        std::vector<SynthTensor> tensors;
        const char* expect;
    };
    const std::vector<uint8_t> short_table(table.begin(), table.begin() + (size_t) (ROWS - 64) * k::PLE_ROW_BYTES_Q8_0);
    const std::vector<Bad> bad = {
        // the table under a different name: the file holds a table-shaped tensor the PLE has no part in
        {"no table", {{"token_embd.weight", 8, {k::PLE_HEAD_DIM, ROWS}, table}}, "not in"},
        // a real Q8_0 shape that is not the PLE's: 192 values per row, and its payload sized to match, so the
        // refusal is the shape and not an artifact of a byte count
        {"wrong shape", {{"per_layer_token_embd.weight", 8, {192, ROWS / 2},
                          std::vector<uint8_t>((size_t) (ROWS / 2) * 6 * 34, 0x5A)}}, "shape [192, 2048]"},
        // a table of an encoding the reader does not have
        {"wrong type", {{"per_layer_token_embd.weight", 2, {k::PLE_HEAD_DIM, ROWS},
                         std::vector<uint8_t>((size_t) ROWS * 90, 0x5A)}}, "is Q4_0, not IQ4_NL"},
        // alone in its file but a row count that does not match the bytes present
        {"short, alone", {{"per_layer_token_embd.weight", 8, {k::PLE_HEAD_DIM, ROWS + 64}, short_table}},
         "size mismatch"},
        // and the same table in a file with others, where the fit is checked at its own offset
        {"short, shared", {{"token_embd.weight", 8, {k::PLE_HEAD_DIM * 2, ROWS / 2}, decoy},
                           {"per_layer_token_embd.weight", 8, {k::PLE_HEAD_DIM, ROWS + 64}, short_table}},
         "size mismatch"},
    };
    for (const Bad& b : bad) {
        const std::string path = dir + "/ple_file_bad.gguf";
        CHECK(write_gguf(path, b.tensors), "write %s (%s)", path.c_str(), b.what);
        for (const k::PleIo mode : {k::PleIo::Direct, k::PleIo::Mmap}) {
            k::PleTable t;
            k::PleIoOptions io;
            io.mode = mode;
            err.clear();
            const bool opened = t.open(path, err, io);
            CHECK(!opened, "%s: a %s table opened", b.what, mode == k::PleIo::Direct ? "direct" : "mapped");
            CHECK(err.find(b.expect) != std::string::npos,
                  "%s: the error was '%s', expected it to mention '%s'", b.what, err.c_str(), b.expect);
            CHECK(err.find(path) != std::string::npos, "%s: the error does not name the file: '%s'", b.what, err.c_str());
            t.close();
        }
        std::filesystem::remove(path);
    }
    for (const std::string& p : {alone, shared}) std::filesystem::remove(p);
    std::printf("--ple-file table cases: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

// The real files.  Each path is a .basalt pack (its `gguf` section registered as the view the loaders use, which
// is what the engine does) or a plain GGUF; the same rows are read from every one of them and compared.  Tables of
// one encoding have to agree bit for bit; a table of another encoding is a different quantization of the same
// source, so its error against the first file is what gets reported, and it has to stay small: reading the wrong
// bytes looks like noise, not like a plausible row.
int cross_check(const std::vector<std::string>& paths) {
    constexpr size_t T = 64;                      // tokens -> 16 row ids each
    std::vector<std::unique_ptr<k::PleTable>> tabs;
    std::vector<uint32_t> rows;
    for (const std::string& path : paths) {
        // A .basalt is not a GGUF: its tensor directory comes from its `meta` section, and the model's own loaders
        // register that view.  Do the same before opening the table out of it.
        basalt::core::ModelFile mf;
        std::string merr;
        if (mf.open(path, merr)) {
            if (!mf.load_meta(merr) || !mf.register_gguf_view(merr)) {
                std::fprintf(stderr, "%s: %s\n", path.c_str(), merr.c_str());
                return 1;
            }
        }
        auto t = std::make_unique<k::PleTable>();
        k::PleIoOptions io;                       // Direct, the production mode
        io.cache_rows = 0;                        // every row comes from the drive
        std::string err;
        if (!t->open(path, err, io)) {
            std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str());
            return 1;
        }
        std::printf("%s: per_layer_token_embd.weight %s, %llu rows, from %s\n", path.c_str(), t->format(),
                    (unsigned long long) t->rows(), t->source().c_str());
        if (tabs.empty()) {
            std::mt19937_64 rng(20261029);
            const uint64_t n = t->rows();
            rows.push_back(0);
            rows.push_back(1);
            rows.push_back((uint32_t) (n / 3));
            rows.push_back((uint32_t) (n - 2));
            rows.push_back((uint32_t) (n - 1));
            for (size_t i = 5; i < T * k::PLE_N_HEADS; ++i) rows.push_back((uint32_t) (rng() % n));
        } else if (tabs.back()->rows() != t->rows()) {
            std::fprintf(stderr, "%s: %llu rows, but the first file has %llu\n", path.c_str(),
                         (unsigned long long) t->rows(), (unsigned long long) tabs.back()->rows());
            return 1;
        }
        tabs.push_back(std::move(t));
    }
    if (tabs.size() < 2) { std::fprintf(stderr, "give two or more files\n"); return 2; }
    std::vector<std::vector<float>> got(tabs.size());
    for (size_t i = 0; i < tabs.size(); ++i) {
        std::string err;
        got[i].assign(T * k::NG_N_EMBD, 0.f);
        if (!tabs[i]->gather_rows(rows.data(), T, got[i].data(), err)) {
            std::fprintf(stderr, "%s: gather_rows: %s\n", paths[i].c_str(), err.c_str());
            return 1;
        }
    }
    bool failed = false;
    for (size_t i = 0; i < tabs.size(); ++i) {
        const size_t bad = (size_t) std::count_if(got[i].begin(), got[i].end(),
                                                  [](float v) { return !std::isfinite(v); });
        if (bad) { std::fprintf(stderr, "%s: %zu non-finite values\n", paths[i].c_str(), bad); failed = true; }
    }
    for (size_t i = 1; i < tabs.size(); ++i) {
        size_t exact = 0; double max_abs = 0, sum_abs = 0, num = 0, den = 0;
        for (size_t j = 0; j < got[i].size(); ++j) {
            const float a = got[0][j], b = got[i][j];
            if (a == b) ++exact;             // equal floats, so equal bits except for a signed zero
            const double d = std::fabs((double) a - (double) b);
            max_abs = std::max(max_abs, d);
            sum_abs += d;
            num += d * d;
            den += (double) a * (double) a;
        }
        const double rel = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1e9 : 0);
        const bool same_fmt = std::strcmp(tabs[0]->format(), tabs[i]->format()) == 0;
        // Two files holding the same encoding of the same table are byte-identical, so every value has to match.
        // Across encodings there is nothing to match bit for bit: a re-quantization of the same source table sits a
        // few percent away (the gap is the coarser encoding's own error), while reading the wrong bytes - a wrong
        // offset, a wrong row stride - is uncorrelated with the right ones, which puts the relative rms near sqrt(2).
        // 0.15 separates the two.
        const bool ok = same_fmt ? (exact == got[i].size()) : rel < 0.15;
        std::printf("%s vs %s (%s): %zu/%zu values exact, max|d| %.4g, mean|d| %.4g, rel rms %.4g -> %s (%s)\n",
                    tabs[0]->format(), tabs[i]->format(), same_fmt ? "same encoding" : "different encoding",
                    exact, got[i].size(), max_abs, sum_abs / (double) got[i].size(), rel,
                    ok ? "agrees" : "DISAGREES", same_fmt ? "bit-identical required" : "rel rms under 0.15");
        if (!ok) failed = true;
    }
    std::printf("PLE table cross-check: %s\n", failed ? "FAILED" : "OK");
    return failed ? 1 : 0;
}

// PleTable::warm_rows on a synthetic one-tensor GGUF (Q8_0 rows): rows read ahead come back byte-identical to the
// mapped table's, are served from the row cache (a read held back by injected delay is not waited for), and in
// Mmap mode the call does nothing.
int warm_rows_check(const std::string& dir) {
    constexpr uint32_t N = 20000;
    const std::string path = dir + "/ple_warm_rows_selftest.gguf";
    std::mt19937 rng(5);
    {
        std::vector<uint8_t> b;
        put<uint32_t>(b, 0x46554747u);   // "GGUF"
        put<uint32_t>(b, 3);
        put<uint64_t>(b, 1);             // tensors
        put<uint64_t>(b, 0);             // metadata
        const std::string name = "per_layer_token_embd.weight";
        put<uint64_t>(b, name.size());
        b.insert(b.end(), name.begin(), name.end());
        put<uint32_t>(b, 2);
        put<uint64_t>(b, k::PLE_HEAD_DIM);
        put<uint64_t>(b, N);
        put<uint32_t>(b, 8);             // Q8_0
        put<uint64_t>(b, 0);
        b.resize((b.size() + 31) / 32 * 32);
        for (uint32_t r = 0; r < N; ++r)
            for (int blk = 0; blk < k::PLE_HEAD_DIM / 32; ++blk) {
                put<uint16_t>(b, (uint16_t) (0x2800 + rng() % 0x800));   // a finite scale
                for (int i = 0; i < 32; ++i) put<uint8_t>(b, (uint8_t) rng());
            }
        std::ofstream(path, std::ios::binary).write((const char*) b.data(), (std::streamsize) b.size());
    }
    std::string err;
    k::PleTable mm, direct;
    k::PleIoOptions mo, dopt;
    mo.mode = k::PleIo::Mmap;
    dopt.cache_rows = 1u << 16;
    CHECK(mm.open(path, err, mo), "mmap open: %s", err.c_str());
    CHECK(direct.open(path, err, dopt), "direct open: %s", err.c_str());
    constexpr size_t T = 64;
    std::vector<uint32_t> warmed(T * k::PLE_N_HEADS), cold(T * k::PLE_N_HEADS);
    for (auto& r : warmed) r = rng() % (N / 2);
    for (auto& r : cold) r = N / 2 + rng() % (N / 2);
    std::vector<float> want(T * k::NG_N_EMBD), got(T * k::NG_N_EMBD);
    CHECK(direct.cache_rows() == dopt.cache_rows && mm.cache_rows() == 0, "cache_rows: %llu / %llu",
          (unsigned long long) direct.cache_rows(), (unsigned long long) mm.cache_rows());
    CHECK(mm.warm_rows(warmed.data(), T, err), "mmap warm_rows: %s", err.c_str());   // no-op
    CHECK(mm.gather_rows(warmed.data(), T, want.data(), err), "mmap gather: %s", err.c_str());
    CHECK(direct.warm_rows(warmed.data(), T, err), "warm_rows: %s", err.c_str());
    direct.set_injected_delay_us(100000);
    double t0 = now_us();
    CHECK(direct.gather_rows(warmed.data(), T, got.data(), err), "warmed gather: %s", err.c_str());
    const double t_warm = now_us() - t0;
    CHECK(!std::memcmp(want.data(), got.data(), want.size() * sizeof(float)), "warmed rows differ from the mapped table's");
    CHECK(t_warm < 50000, "warmed rows were not served from the cache (%.0f us)", t_warm);
    t0 = now_us();
    CHECK(direct.gather_rows(cold.data(), T, got.data(), err), "cold gather: %s", err.c_str());
    CHECK(now_us() - t0 >= 100000, "unwarmed rows were not read (%.0f us)", now_us() - t0);
    CHECK(mm.gather_rows(cold.data(), T, want.data(), err), "mmap gather: %s", err.c_str());
    CHECK(!std::memcmp(want.data(), got.data(), want.size() * sizeof(float)), "cold rows differ from the mapped table's");
    mm.close();
    direct.close();
    std::filesystem::remove(path);
    std::printf("PleTable::warm_rows: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, dir = std::filesystem::temp_directory_path().string(), tokens;
    std::vector<std::string> cross;
    int rows = 20000;
    uint32_t inflight = 64;
    bool direct_first = false;
    bool direct_only = false;
    bool self = false;
    double stress_secs = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") self = true;
        else if (a == "--cross-check" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string p;
            while (std::getline(ss, p, ','))
                if (!p.empty()) cross.push_back(p);
        }
        else if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) tokens = argv[++i];
        else if (a == "--inflight" && i + 1 < argc) inflight = (uint32_t) std::atoi(argv[++i]);
        else if (a == "--direct-first") direct_first = true;
        else if (a == "--direct-only") direct_only = true;
        else if (a == "--stress" && i + 1 < argc) stress_secs = std::atof(argv[++i]);
        else { std::fprintf(stderr, "usage: ple_reader_test --selftest [--dir D] | --gguf SHARD2 [--rows N] [--tokens F] [--stress SECS] | --cross-check A[,B,...]\n"); return 2; }
    }
    if (self) {
        // ng::ROW_BYTES (90, IQ4_NL, production default), 110 (Q5_0) and 170 (Q8_0) rows through the
        // same generic row_bytes path -- see the comment on selftest().
        const int r90 = selftest(dir, ng::ROW_BYTES);
        const int r110 = selftest(dir, k::PLE_ROW_BYTES_Q5_0);
        const int r170 = selftest(dir, k::PLE_ROW_BYTES_Q8_0);
        const int q8 = q8_decode_check();
        const int warm = warm_rows_check(dir);
        const int pf = ple_file_check(dir);
        return r90 != 0 ? r90 : r110 != 0 ? r110 : r170 != 0 ? r170 : q8 != 0 ? q8 : warm != 0 ? warm : pf;
    }
    if (!cross.empty()) return cross_check(cross);
    if (!gguf.empty()) return real(gguf, rows, tokens, inflight, direct_first, direct_only, stress_secs);
    std::fprintf(stderr, "nothing to do\n");
    return 2;
}
