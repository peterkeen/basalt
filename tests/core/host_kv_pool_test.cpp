// tests/core/host_kv_pool_test.cpp - unit test for HostKvPool (slab boundaries, COW refcounts, ram_reserve, sleep/wake).
#include "basalt/models/qwen4exp/decode/host_kv_pool.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace basalt::core;
using namespace basalt::kernels;

namespace {
int g_checks = 0;
void check(bool ok, const char* msg) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

void run_cpu_tests() {
    HostKvPool pool(/*register_cuda=*/false);
    check(pool.allocated_slabs() == 0 && pool.live_pages() == 0, "starts with 0 slabs and 0 pages");

    HostKvTable a, b;
    a.h_pages.assign(40000, -1);   // crosses the 32,768-page slab boundary
    b.h_pages.assign(40000, -1);

    std::string err;
    // 1. Cross the 32,768-page slab boundary -> 2 slabs allocated
    check(pool.ensure_writable(a, 0, 33000, err), "ensure_writable across 32,768-page slab boundary");
    check(pool.allocated_slabs() == 2 && pool.live_pages() == 33000, "two 132 MiB slabs allocated");
    check((a.h_pages[0] >> HostKvPool::kSlabShift) == 0 && (a.h_pages[32767] >> HostKvPool::kSlabShift) == 0 &&
          (a.h_pages[32768] >> HostKvPool::kSlabShift) == 1, "slab index boundary at page 32,768");

    // Stamp distinguishable patterns at boundary pages
    for (int64_t p : {0LL, 1023LL, 32767LL, 32768LL, 32999LL}) {
        HostKvPage* hp = pool.page_ptr(a.h_pages[(size_t) p]);
        check(hp != nullptr && ((uintptr_t) hp % 16) == 0, "16-byte aligned HostKvPage");
        std::memset(hp->k_q, (int) (p & 0x7f), sizeof(hp->k_q));
        hp->k_scale[0] = (uint16_t) (p & 0xffff);
    }

    // 2. Zero-copy clone_range -> refcount = 2 on [0, 32800), 0 new pages allocated
    check(pool.clone_range(b, a, 32800 * 4, err), "clone_range across slab boundary");
    check(pool.live_pages() == 33000 && pool.allocated_slabs() == 2, "clone_range allocates zero pages");
    check(pool.refcount(a.h_pages[0]) == 2 && pool.refcount(a.h_pages[32768]) == 2 &&
          pool.refcount(a.h_pages[32900]) == 1, "refcounts match shared prefix");
    check(b.h_pages[32768] == a.h_pages[32768] && b.h_pages[32800] == -1, "cloned table aliases source pages");

    // 3. COW split on overwrite: writing [32767, 32770) on `b` allocates 3 fresh pages, keeps `a`'s bytes intact
    const int32_t orig_32768 = a.h_pages[32768];
    check(pool.ensure_writable(b, 32767, 32770, err), "COW split on shared boundary pages");
    check(b.h_pages[32768] != orig_32768 && a.h_pages[32768] == orig_32768, "COW split allocates distinct page");
    check(pool.refcount(orig_32768) == 1 && pool.refcount(b.h_pages[32768]) == 1, "COW split drops refcounts to 1");
    check(pool.page_ptr(b.h_pages[32768])->k_scale[0] == (uint16_t) (32768 & 0xffff), "COW split copies previous page contents");
    pool.page_ptr(b.h_pages[32768])->k_scale[0] = 0xbeef;
    check(pool.page_ptr(a.h_pages[32768])->k_scale[0] == (uint16_t) (32768 & 0xffff), "mutating COW page leaves source untouched");

    // 4. ram_reserve refusal & atomic rollback
    const uint64_t saved_reserve = basalt::platform::ram_reserve();
    basalt::platform::set_ram_reserve(2048ull << 20);
    // Simulate MemAvailable = 2048 MiB + 64 MiB (< 132 MiB slab needed for a 3rd slab)
    pool.set_avail_fn([]() -> uint64_t { return (2048ull + 64ull) << 20; });
    HostKvTable c;
    c.h_pages.assign(40000, -1);
    const int64_t pages_before = pool.live_pages();
    check(!pool.ensure_writable(c, 0, 35000, err) && !err.empty(), "ram_reserve refuses slab allocation below floor");
    check(pool.live_pages() == pages_before && c.h_pages[0] == -1, "failed ensure_writable rolls back cleanly");
    pool.set_avail_fn(nullptr);
    basalt::platform::set_ram_reserve(saved_reserve);

    // 5. Releasing `a` leaves `b`'s shared pages alive; releasing `b` down to 100 pages keeps 1 spare slab; releasing all frees to 0 slabs
    pool.release_table(a);
    check(pool.live_pages() == 32800 && pool.page_ptr(b.h_pages[0])->k_scale[0] == 0, "shared pages survive source release");
    pool.release_range(b, 100, /*keep_spare=*/true);
    check(pool.live_pages() == 100 && pool.allocated_slabs() == 2, "keeps at most one spare empty slab while active");
    pool.release_table(b);
    check(pool.live_pages() == 0 && pool.allocated_slabs() == 0, "frees all slabs when all tables release");
}

void run_gpu_tests() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return;
    HostKvPool pool(/*register_cuda=*/true);
    HostKvTable t;
    t.h_pages.assign(1024, -1);
    check(cudaMalloc((void**) &t.d_pages, 1024 * sizeof(int32_t)) == cudaSuccess, "alloc device page table");
    std::string err;
    uint8_t* const* d_slabs = pool.device_slabs(0);
    check(d_slabs != nullptr, "device slab directory allocated");
    check(pool.ensure_writable(t, 0, 128, err), "GPU ensure_writable");
    check(pool.registered_slabs() == 1 && pool.pinned_bytes() == HostKvPool::kSlabBytes, "one slab pinned");

    int32_t dev_p0 = -1;
    check(cudaMemcpy(&dev_p0, t.d_pages, sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess && dev_p0 == t.h_pages[0],
          "device page table mirrors host");

    // SLEEP / WAKE cycle: unregister_all drops registration, register_all restores it at the same address
    pool.page_ptr(t.h_pages[0])->k_scale[0] = 0x1234;
    pool.unregister_all();
    check(pool.registered_slabs() == 0 && pool.allocated_slabs() == 1, "unregister_all drops pin while keeping slab bytes");
    check(pool.register_all(err) && pool.registered_slabs() == 1, "register_all restores pin");
    check(pool.page_ptr(t.h_pages[0])->k_scale[0] == 0x1234, "slab contents preserved across SLEEP/WAKE");

    cudaFree(t.d_pages);
    pool.release_table(t);
    check(pool.allocated_slabs() == 0 && pool.pinned_bytes() == 0, "0 B pinned after release_table");
}
}  // namespace

int main() {
    run_cpu_tests();
    run_gpu_tests();
    std::printf("host_kv_pool_test: %d checks passed\n", g_checks);
    return 0;
}
