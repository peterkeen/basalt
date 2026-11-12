// src/models/qwen4exp/decode/host_kv_pool.cpp - see include/basalt/models/qwen4exp/decode/host_kv_pool.hpp.
#include "basalt/models/qwen4exp/decode/host_kv_pool.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

namespace basalt::core {

HostKvPool::HostKvPool(bool register_cuda) : register_cuda_(register_cuda) {}

HostKvPool::~HostKvPool() {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < slabs_.size(); ++i)
        if (slabs_[i].host != nullptr) free_slab_locked((int) i);
    for (size_t d = 0; d < dev_dirs_.size(); ++d) {
        if (dev_dirs_[d] != nullptr) {
            const OnDevice on((int) d);
            cudaFree(dev_dirs_[d]);
            dev_dirs_[d] = nullptr;
        }
    }
}

uint8_t* const* HostKvPool::device_slabs(int device) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!register_cuda_) return h_slabs_;
    if (device < 0) cudaGetDevice(&device);
    if (device < 0) device = 0;
    if ((size_t) device >= dev_dirs_.size()) dev_dirs_.resize((size_t) device + 1, nullptr);
    if (dev_dirs_[(size_t) device] == nullptr) {
        const OnDevice on(device);
        uint8_t** d = nullptr;
        if (cudaMalloc((void**) &d, sizeof(h_slabs_)) != cudaSuccess) return nullptr;
        if (cudaMemcpy(d, h_slabs_, sizeof(h_slabs_), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d);
            return nullptr;
        }
        dev_dirs_[(size_t) device] = d;
    }
    return dev_dirs_[(size_t) device];
}

basalt::kernels::HostKvPage* HostKvPool::page_ptr(int32_t phys_page) {
    if (phys_page < 0) return nullptr;
    const int s = phys_page >> kSlabShift;
    const int64_t off = (phys_page & basalt::kernels::kHostKvSlabMask) * (int64_t) kPageBytes;
    if (s < 0 || (size_t) s >= slabs_.size() || slabs_[(size_t) s].host == nullptr) return nullptr;
    return reinterpret_cast<basalt::kernels::HostKvPage*>(slabs_[(size_t) s].host + off);
}

const basalt::kernels::HostKvPage* HostKvPool::page_ptr(int32_t phys_page) const {
    if (phys_page < 0) return nullptr;
    const int s = phys_page >> kSlabShift;
    const int64_t off = (phys_page & basalt::kernels::kHostKvSlabMask) * (int64_t) kPageBytes;
    if (s < 0 || (size_t) s >= slabs_.size() || slabs_[(size_t) s].host == nullptr) return nullptr;
    return reinterpret_cast<const basalt::kernels::HostKvPage*>(slabs_[(size_t) s].host + off);
}

uint16_t HostKvPool::refcount(int32_t phys_page) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (phys_page < 0 || (size_t) phys_page >= refcount_.size()) return 0;
    return refcount_[(size_t) phys_page];
}

int64_t HostKvPool::allocated_slabs() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const Slab& s : slabs_)
        if (s.host != nullptr) ++n;
    return n;
}

int64_t HostKvPool::registered_slabs() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const Slab& s : slabs_)
        if (s.host != nullptr && s.registered) ++n;
    return n;
}

bool HostKvPool::sync_dirs_locked(std::string& err) {
    if (!register_cuda_) return true;
    for (size_t d = 0; d < dev_dirs_.size(); ++d) {
        if (dev_dirs_[d] == nullptr) continue;
        const OnDevice on((int) d);
        if (cudaMemcpy(dev_dirs_[d], h_slabs_, sizeof(h_slabs_), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = std::string("HostKvPool: slab directory upload failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    return true;
}

bool HostKvPool::open_slab_locked(int idx, std::string& err) {
    if (idx < 0 || idx >= kMaxSlabs) {
        err = "HostKvPool: slab ceiling reached";
        return false;
    }
    const uint64_t avail = avail_fn_ ? avail_fn_() : basalt::platform::available_memory();
    const uint64_t floor = basalt::platform::ram_reserve() + extra_reserve_;
    if (avail > 0 && avail < floor + kSlabBytes) {
        err = "HostKvPool: free RAM (" + std::to_string(avail >> 20) + " MiB) cannot spare a 132 MiB KV slab above the " +
              std::to_string(floor >> 20) + " MiB floor";
        return false;
    }
    if ((size_t) idx >= slabs_.size()) {
        slabs_.resize((size_t) idx + 1);
        refcount_.resize(slabs_.size() * (size_t) kSlabPages, 0);
    }
    Slab& s = slabs_[(size_t) idx];
    basalt::platform::RegionBacking got = basalt::platform::RegionBacking::NormalPages;
    std::string note;
    void* map_base = nullptr;
    uint64_t map_bytes = 0;
    void* p = basalt::platform::alloc_region(kSlabBytes, got, note, "", 0, map_base, map_bytes);
    if (p == nullptr) {
        err = "HostKvPool: cannot allocate 132 MiB KV slab (" + note + ")";
        return false;
    }
    s.host = (uint8_t*) p;
    s.mapping_base = map_base ? map_base : p;
    s.mapping_bytes = map_bytes ? map_bytes : kSlabBytes;
    s.used_pages = 0;
    s.next_fresh = 0;
    s.free_list.clear();
    s.registered = false;
    s.dev = nullptr;
    if (register_cuda_) {
        // Register before touching pages so registration pins without walking already-faulted pages twice.
        const cudaError_t re = cudaHostRegister(s.host, (size_t) kSlabBytes,
                                                cudaHostRegisterMapped | cudaHostRegisterPortable);
        void* dev_ptr = nullptr;
        if (re != cudaSuccess || cudaHostGetDevicePointer(&dev_ptr, s.host, 0) != cudaSuccess) {
            const cudaError_t ce = re != cudaSuccess ? re : cudaGetLastError();
            if (re == cudaSuccess) cudaHostUnregister(s.host);
            basalt::platform::free_region(s.mapping_base, s.mapping_bytes);
            s = Slab{};
            err = std::string("HostKvPool: cudaHostRegister failed for 132 MiB slab: ") + cudaGetErrorString(ce);
            return false;
        }
        s.registered = true;
        s.dev = (uint8_t*) dev_ptr;
        h_slabs_[idx] = s.dev;
    } else {
        h_slabs_[idx] = s.host;
    }
    if (!sync_dirs_locked(err)) {
        free_slab_locked(idx);
        return false;
    }
    return true;
}

void HostKvPool::free_slab_locked(int idx) {
    if (idx < 0 || (size_t) idx >= slabs_.size()) return;
    Slab& s = slabs_[(size_t) idx];
    if (s.host == nullptr) return;
    h_slabs_[idx] = nullptr;
    std::string dummy;
    (void) sync_dirs_locked(dummy);
    if (s.registered) {
        (void) cudaHostUnregister(s.host);
        s.registered = false;
    }
    basalt::platform::free_region(s.mapping_base ? s.mapping_base : s.host,
                                  s.mapping_bytes ? s.mapping_bytes : kSlabBytes);
    s = Slab{};
}

void HostKvPool::reclaim_empty_locked(bool keep_one_spare) {
    bool kept_spare = false;
    // Scan from highest slab downward so kept spare sits at the lowest index.
    for (int i = 0; i < (int) slabs_.size(); ++i) {
        Slab& s = slabs_[(size_t) i];
        if (s.host == nullptr || s.used_pages > 0) continue;
        if (keep_one_spare && !kept_spare) {
            kept_spare = true;
            continue;
        }
        free_slab_locked(i);
    }
}

int32_t HostKvPool::alloc_page_locked(std::string& err) {
    int empty_slot = -1;
    for (int i = 0; i < (int) slabs_.size(); ++i) {
        Slab& s = slabs_[(size_t) i];
        if (s.host == nullptr) {
            if (empty_slot < 0) empty_slot = i;
            continue;
        }
        int32_t page = -1;
        if (!s.free_list.empty()) {
            page = s.free_list.back();
            s.free_list.pop_back();
        } else if (s.next_fresh < kSlabPages) {
            page = (i << kSlabShift) | s.next_fresh++;
        }
        if (page >= 0) {
            ++s.used_pages;
            ++live_pages_;
            refcount_[(size_t) page] = 1;
            return page;
        }
    }
    const int idx = empty_slot >= 0 ? empty_slot : (int) slabs_.size();
    if (!open_slab_locked(idx, err)) return -1;
    Slab& s = slabs_[(size_t) idx];
    const int32_t page = (idx << kSlabShift) | s.next_fresh++;
    ++s.used_pages;
    ++live_pages_;
    refcount_[(size_t) page] = 1;
    return page;
}

void HostKvPool::dec_ref_locked(int32_t phys_page) {
    if (phys_page < 0 || (size_t) phys_page >= refcount_.size()) return;
    uint16_t& rc = refcount_[(size_t) phys_page];
    if (rc == 0) return;
    if (--rc > 0) return;
    const int idx = phys_page >> kSlabShift;
    if (idx >= 0 && (size_t) idx < slabs_.size() && slabs_[(size_t) idx].host != nullptr) {
        Slab& s = slabs_[(size_t) idx];
        s.free_list.push_back(phys_page);
        --s.used_pages;
    }
    --live_pages_;
}

bool HostKvPool::upload_slice_locked(const HostKvTable& tab, int64_t lo, int64_t hi, std::string& err, void* stream) {
    if (!register_cuda_ || tab.d_pages == nullptr || hi <= lo) return true;
    const OnDevice on(tab.device);
    const size_t bytes = (size_t) (hi - lo) * sizeof(int32_t);
    const cudaError_t e = stream != nullptr
        ? cudaMemcpyAsync(tab.d_pages + lo, tab.h_pages.data() + lo, bytes, cudaMemcpyHostToDevice, (cudaStream_t) stream)
        : cudaMemcpy(tab.d_pages + lo, tab.h_pages.data() + lo, bytes, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) return true;
    err = std::string("HostKvPool: uploading host_page_table failed: ") + cudaGetErrorString(e);
    return false;
}

bool HostKvPool::ensure_writable(HostKvTable& tab, int64_t page_lo, int64_t page_hi, std::string& err, void* stream) {
    page_lo = std::max<int64_t>(0, page_lo);
    page_hi = std::min<int64_t>(page_hi, (int64_t) tab.h_pages.size());
    if (page_hi <= page_lo) return true;
    if (page_lo >= tab.writable_lo && page_hi <= tab.writable_hi) return true;

    std::lock_guard<std::mutex> lk(mu_);
    int64_t dirty_lo = page_hi, dirty_hi = page_lo;
    struct Undo { int64_t b; int32_t old_p, new_p; };
    std::vector<Undo> undo;
    for (int64_t b = page_lo; b < page_hi; ++b) {
        const int32_t old_p = tab.h_pages[(size_t) b];
        if (old_p >= 0 && refcount_[(size_t) old_p] == 1) continue;
        const int32_t new_p = alloc_page_locked(err);
        if (new_p < 0) {
            for (auto it = undo.rbegin(); it != undo.rend(); ++it) {
                if (it->old_p >= 0) ++refcount_[(size_t) it->old_p];
                tab.h_pages[(size_t) it->b] = it->old_p;
                dec_ref_locked(it->new_p);
            }
            reclaim_empty_locked(live_pages_ > 0);
            return false;
        }
        basalt::kernels::HostKvPage* dst = page_ptr(new_p);
        if (old_p >= 0) {
            std::memcpy(dst, page_ptr(old_p), sizeof(basalt::kernels::HostKvPage));
            dec_ref_locked(old_p);
        } else {
            std::memset(dst, 0, sizeof(basalt::kernels::HostKvPage));
        }
        tab.h_pages[(size_t) b] = new_p;
        undo.push_back({b, old_p, new_p});
        dirty_lo = std::min(dirty_lo, b);
        dirty_hi = std::max(dirty_hi, b + 1);
    }
    if (dirty_hi > dirty_lo && !upload_slice_locked(tab, dirty_lo, dirty_hi, err, stream)) return false;
    if (page_lo <= tab.writable_hi && page_hi >= tab.writable_lo) {
        tab.writable_lo = std::min(tab.writable_lo, page_lo);
        tab.writable_hi = std::max(tab.writable_hi, page_hi);
    } else {
        tab.writable_lo = page_lo;
        tab.writable_hi = page_hi;
    }
    reclaim_empty_locked(live_pages_ > 0);
    return true;
}

bool HostKvPool::clone_range(HostKvTable& dst, HostKvTable& src, int64_t upto_cells, std::string& err, void* stream) {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t cap = (int64_t) std::min(dst.h_pages.size(), src.h_pages.size());
    const int64_t pages = std::clamp<int64_t>((upto_cells + 3) / 4, 0, cap);
    for (size_t b = 0; b < dst.h_pages.size(); ++b) {
        if (dst.h_pages[b] >= 0) {
            dec_ref_locked(dst.h_pages[b]);
            dst.h_pages[b] = -1;
        }
    }
    for (int64_t b = 0; b < pages; ++b) {
        const int32_t p = src.h_pages[(size_t) b];
        if (p < 0) continue;
        if (refcount_[(size_t) p] == UINT16_MAX) {
            err = "HostKvPool: page refcount overflow";
            return false;
        }
        ++refcount_[(size_t) p];
        dst.h_pages[(size_t) b] = p;
    }
    src.writable_lo = std::max(src.writable_lo, pages);
    if (src.writable_lo > src.writable_hi) src.writable_lo = src.writable_hi = pages;
    dst.writable_lo = dst.writable_hi = pages;
    const bool ok = upload_slice_locked(dst, 0, (int64_t) dst.h_pages.size(), err, stream);
    reclaim_empty_locked(live_pages_ > 0);
    return ok;
}

void HostKvPool::release_range(HostKvTable& tab, int64_t from_page, bool keep_spare, void* stream) {
    std::lock_guard<std::mutex> lk(mu_);
    from_page = std::max<int64_t>(0, from_page);
    int64_t dirty_lo = (int64_t) tab.h_pages.size(), dirty_hi = from_page;
    for (int64_t b = from_page; b < (int64_t) tab.h_pages.size(); ++b) {
        const int32_t p = tab.h_pages[(size_t) b];
        if (p < 0) continue;
        dec_ref_locked(p);
        tab.h_pages[(size_t) b] = -1;
        dirty_lo = std::min(dirty_lo, b);
        dirty_hi = std::max(dirty_hi, b + 1);
    }
    tab.writable_hi = std::min(tab.writable_hi, from_page);
    tab.writable_lo = std::min(tab.writable_lo, tab.writable_hi);
    if (dirty_hi > dirty_lo) {
        std::string dummy;
        (void) upload_slice_locked(tab, dirty_lo, dirty_hi, dummy, stream);
    }
    reclaim_empty_locked(keep_spare && live_pages_ > 0);
}

void HostKvPool::release_table(HostKvTable& tab) {
    std::lock_guard<std::mutex> lk(mu_);
    for (int32_t& p : tab.h_pages) {
        if (p >= 0) {
            dec_ref_locked(p);
            p = -1;
        }
    }
    tab.writable_lo = tab.writable_hi = 0;
    tab.d_pages = nullptr;
    reclaim_empty_locked(live_pages_ > 0);
}

void HostKvPool::unregister_all() {
    std::lock_guard<std::mutex> lk(mu_);
    reclaim_empty_locked(false);
    if (!register_cuda_) return;
    for (size_t i = 0; i < slabs_.size(); ++i) {
        Slab& s = slabs_[i];
        if (s.host == nullptr || !s.registered) continue;
        (void) cudaHostUnregister(s.host);
        s.registered = false;
        s.dev = nullptr;
        h_slabs_[i] = nullptr;
    }
    std::string dummy;
    (void) sync_dirs_locked(dummy);
}

bool HostKvPool::register_all(std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!register_cuda_) return true;
    for (size_t i = 0; i < slabs_.size(); ++i) {
        Slab& s = slabs_[i];
        if (s.host == nullptr || s.registered) continue;
        const cudaError_t re = cudaHostRegister(s.host, (size_t) kSlabBytes,
                                                cudaHostRegisterMapped | cudaHostRegisterPortable);
        void* dev_ptr = nullptr;
        if (re != cudaSuccess || cudaHostGetDevicePointer(&dev_ptr, s.host, 0) != cudaSuccess) {
            const cudaError_t ce = re != cudaSuccess ? re : cudaGetLastError();
            if (re == cudaSuccess) cudaHostUnregister(s.host);
            err = std::string("HostKvPool: re-registering 132 MiB KV slab failed: ") + cudaGetErrorString(ce);
            return false;
        }
        s.registered = true;
        s.dev = (uint8_t*) dev_ptr;
        h_slabs_[i] = s.dev;
    }
    return sync_dirs_locked(err);
}

}  // namespace basalt::core
