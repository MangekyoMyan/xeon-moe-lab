#include "expert-host-register.h"
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <limits>

using xeon_expert_dma::page_ranges;
using xeon_expert_dma::range;
static std::vector<void *> acquired, released;
static bool mapping_alive = true;
static bool acquire(void * p, size_t) {
    if (p == reinterpret_cast<void *>(uintptr_t(3))) return false;
    acquired.push_back(p); return true;
}
static void release(void * p) {
    assert(mapping_alive); released.push_back(p);
}
static void * ptr(uintptr_t p) { return reinterpret_cast<void *>(p); }

int main() {
    const std::vector<range> a{{5000,6000},{100,500},{4000,4100},{17000,18000},{25000,25000}};
    const std::vector<range> want{{0,8192},{16384,20480}};
    assert(page_ranges(a,30000,4096)==want);
    assert((page_ranges({{8191,9001}},9001,4096)==std::vector<range>{{4096,9001}}));
    assert((page_ranges({{0,4096},{4096,8192}},8192,4096)==std::vector<range>{{0,8192}}));
    assert(page_ranges({},1,4096).empty());
    int errors=0;
    for (auto input : std::vector<std::vector<range>>{{{9001,9002}},{{8,7}}}) {
        try { page_ranges(input,9001,4096); } catch (const std::runtime_error &) { ++errors; }
    }
    try { page_ranges({},1,0); } catch (const std::runtime_error &) { ++errors; }
    assert(errors==3);
    const size_t max=std::numeric_limits<size_t>::max();
    assert(page_ranges({{max-7,max}},max,4096).back().second==max);
    {
        xeon_expert_dma::registrations owner;
        owner.reserve(3);
        assert(owner.add(ptr(1),4096,acquire,release));
        assert(owner.add(ptr(2),4096,acquire,release));
        assert(!owner.add(ptr(3),4096,acquire,release));
    }
    assert((acquired==std::vector<void *>{ptr(1),ptr(2)}));
    assert((released==std::vector<void *>{ptr(2),ptr(1)}));
    {
        xeon_expert_dma::registrations owner;
        assert(owner.add(ptr(4),4096,acquire,release));
        owner.clear();mapping_alive=false;
    }
    assert(released.back()==ptr(4));
    std::puts("registration range bounds, merging, failure cleanup and early release: PASS");
}
