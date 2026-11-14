// include/basalt/models/qwen4exp/decode/host_kv_pool.hpp - shared demand-allocated pinned INT8 KV slab pool.
//
// Streamed QSA sessions share one host pool of 132 MiB slabs (32,768 interleaved 4-cell INT8 pages of 4,224 B each =
// 66 x 2 MiB hugepages). Slabs are allocated on demand, registered as portable device-mapped pinned memory, reference
// counted per page for zero-copy prefix cloning (COW on overwrite), and returned to the OS when emptied.
#pragma once

#include "basalt/models/qwen4exp/kernels/kv_stream.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace basalt::core {

/// Per-layer page table mapping logical 4-cell blocks to physical pages in a `HostKvPool`.
struct HostKvTable {
    int32_t* d_pages = nullptr;       ///< device [n_pages], fixed address carved from the QsaState arena
    std::vector<int32_t> h_pages;     ///< host mirror [n_pages], -1 when unallocated
    int64_t writable_lo = 0;          ///< [writable_lo, writable_hi) pages are allocated with refcount == 1
    int64_t writable_hi = 0;
    int device = 0;
};

class HostKvPool {
public:
    static constexpr int kSlabShift = basalt::kernels::kHostKvSlabShift;
    static constexpr int64_t kSlabPages = basalt::kernels::kHostKvSlabPages;   ///< 32,768 pages
    static constexpr uint64_t kPageBytes = sizeof(basalt::kernels::HostKvPage); ///< 4,224 B
    static constexpr uint64_t kSlabBytes = (uint64_t) kSlabPages * kPageBytes;  ///< 132 MiB exact
    static constexpr int kMaxSlabs = 256;                                       ///< 33 GiB ceiling

    explicit HostKvPool(bool register_cuda = true);
    ~HostKvPool();
    HostKvPool(const HostKvPool&) = delete;
    HostKvPool& operator=(const HostKvPool&) = delete;

    /// Device pointer to the fixed `kMaxSlabs`-entry slab directory on `device` (< 0: current device).
    uint8_t* const* device_slabs(int device = -1);

    /// Host pointer to physical page `phys_page` (nullptr if out of range or unallocated slab).
    basalt::kernels::HostKvPage* page_ptr(int32_t phys_page);
    const basalt::kernels::HostKvPage* page_ptr(int32_t phys_page) const;

    /// Current reference count of physical page `phys_page` (0 when free or out of range).
    uint16_t refcount(int32_t phys_page) const;

    /// Ensures logical pages `[page_lo, page_hi)` in `tab` are allocated and uniquely owned (`refcount == 1`),
    /// COW-splitting any shared page (`refcount > 1`) and uploading dirtied entries to `tab.d_pages`.
    bool ensure_writable(HostKvTable& tab, int64_t page_lo, int64_t page_hi, std::string& err, void* stream = nullptr,
                         bool zero_fresh = true);

    /// Zero-copy prefix clone: `dst` shares pages `[0, (upto_cells + 3) / 4)` of `src` by bumping refcounts.
    bool clone_range(HostKvTable& dst, HostKvTable& src, int64_t upto_cells, std::string& err, void* stream = nullptr);

    /// Releases pages `[from_page, end)` in `tab`, freeing empty slabs (keeping at most one spare while any page lives).
    void release_range(HostKvTable& tab, int64_t from_page, bool keep_spare = true, void* stream = nullptr);

    /// Releases all pages owned by `tab` without touching device memory (safe during teardown).
    void release_table(HostKvTable& tab);

    /// Drops CUDA host registration on every slab (and frees any spare slab) before driver checkpointing (SLEEP).
    void unregister_all();
    /// Restores CUDA host registration on every active slab after wake (WAKE).
    bool register_all(std::string& err);

    /// Additional host RAM headroom kept free above `platform::ram_reserve()`.
    void set_extra_reserve(uint64_t bytes) { extra_reserve_ = bytes; }
    /// Test hook overriding `platform::available_memory()`.
    void set_avail_fn(std::function<uint64_t()> fn) { avail_fn_ = std::move(fn); }

    /// Telemetry / test inspection.
    int64_t allocated_slabs() const;
    int64_t registered_slabs() const;
    int64_t live_pages() const { return live_pages_; }
    uint64_t allocated_bytes() const { return (uint64_t) allocated_slabs() * kSlabBytes; }
    uint64_t pinned_bytes() const { return (uint64_t) registered_slabs() * kSlabBytes; }

private:
    struct Slab {
        uint8_t* host = nullptr;
        uint8_t* dev = nullptr;
        int32_t used_pages = 0;
        int32_t next_fresh = 0;
        bool registered = false;
        std::vector<int32_t> free_list;
    };

    int32_t alloc_page_locked(std::string& err);
    void dec_ref_locked(int32_t phys_page);
    bool open_slab_locked(int idx, std::string& err);
    void free_slab_locked(int idx);
    void reclaim_empty_locked(bool keep_one_spare);
    bool sync_dirs_locked(std::string& err);
    bool upload_slice_locked(const HostKvTable& tab, int64_t lo, int64_t hi, std::string& err, void* stream);

    bool register_cuda_ = true;
    mutable std::mutex mu_;
    std::vector<Slab> slabs_;
    std::vector<uint16_t> refcount_;
    uint8_t* h_slabs_[kMaxSlabs] = {};
    std::vector<uint8_t**> dev_dirs_;
    int64_t live_pages_ = 0;
    uint64_t extra_reserve_ = 0;
    std::function<uint64_t()> avail_fn_;
};

}  // namespace basalt::core
