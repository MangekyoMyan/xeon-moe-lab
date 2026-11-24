#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <sys/syscall.h>
#include <unistd.h>

struct AddressRange {
    uintptr_t end;
    std::string policy;
};

static std::map<uintptr_t, AddressRange> memory_policies() {
    std::map<uintptr_t, AddressRange> result;
    std::ifstream ranges("/proc/self/maps");
    std::string line;
    while (std::getline(ranges, line)) {
        unsigned long first, last;
        if (std::sscanf(line.c_str(), "%lx-%lx", &first, &last) == 2) {
            result[first] = {last, ""};
        }
    }
    std::ifstream policies("/proc/self/numa_maps");
    while (std::getline(policies, line)) {
        std::istringstream fields(line);
        uintptr_t address;
        std::string policy;
        if (fields >> std::hex >> address >> policy && result.count(address)) {
            result.at(address).policy = policy;
        }
    }
    return result;
}

static json compact_registration(const llama_model & model) {
    using Lookup = const void * const * (*)(const ggml_tensor *);
    auto lookup = reinterpret_cast<Lookup>(dlsym(RTLD_DEFAULT, "xeon_iq3_cache_lookup_experts_v2"));
    need(lookup != nullptr, "compact lookup symbol");
    auto policies = memory_policies();
    const long page_size = sysconf(_SC_PAGESIZE);
    need(page_size > 0, "page size");
    json inventory = json::object();
    std::vector<void *> pages;
    std::map<std::string, unsigned> sampled_policies;
    size_t selected_count = 0;
    size_t registered_tensors = 0;
    for (const auto & layer : model.layers) {
        for (auto * tensor : {layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
            const auto table = lookup(tensor);
            std::vector<int> selected;
            std::vector<uintptr_t> pointers;
            if (table) {
                ++registered_tensors;
                for (int expert = 0; expert < tensor->ne[2]; ++expert) {
                    if (table[expert]) {
                        selected.push_back(expert);
                        pointers.push_back(reinterpret_cast<uintptr_t>(table[expert]));
                    }
                }
                need(!selected.empty(), "registered empty tensor");
                selected_count += selected.size();
                std::sort(pointers.begin(), pointers.end());
                need(std::adjacent_find(pointers.begin(), pointers.end()) == pointers.end(), "aliased expert payloads");
                // Sample six populated pages in each of three slots per registered tensor.
                // The fixed model uses a 6,635,520-byte compact projection per expert.
                constexpr uintptr_t stride = 6635520;
                if (pointers.size() > 1) need(pointers[1]-pointers[0] == stride, "packed stride");
                for (size_t slot : {size_t(0), pointers.size()/2, pointers.size()-1}) {
                    for (uintptr_t offset : std::array<uintptr_t,6>{0,uintptr_t(page_size),stride/2-page_size,stride/2,stride/2+page_size,stride-page_size}) {
                        const auto address = pointers[slot] + offset;
                        auto iterator = policies.upper_bound(address);
                        need(iterator != policies.begin(), "unmapped compact page");
                        --iterator;
                        need(address < iterator->second.end && !iterator->second.policy.empty(), "numa_maps policy unavailable");
                        ++sampled_policies[iterator->second.policy];
                        pages.push_back(reinterpret_cast<void *>(address));
                    }
                }
            }
            inventory[tensor->name] = selected;
        }
    }
    std::vector<int> status(pages.size(), -1);
    errno = 0;
    const long rc = syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr, status.data(), 0);
    const int query_errno = rc < 0 ? errno : 0;
    std::map<std::string,unsigned> nodes;
    if (!query_errno) {
        for (int node : status) ++nodes[std::to_string(node)];
    }
    return {{"experts", inventory}, {"registered_tensors", registered_tensors},
            {"selected_projections", selected_count}, {"sampled_pages", pages.size()},
            {"sampled_vma_policies", sampled_policies}, {"page_location_errno", query_errno},
            {"sampled_page_nodes", nodes}, {"page_size", page_size}};
}

static json process_memory() {
    json fields = json::object();
    std::ifstream stream("/proc/self/status");
    std::string line;
    while (std::getline(stream, line)) {
        std::istringstream row(line);
        std::string key, unit;
        uint64_t value;
        if (row >> key >> value >> unit && unit == "kB" &&
            (key == "VmRSS:" || key == "VmLck:" || key == "VmHWM:" || key == "RssAnon:")) {
            fields[key.substr(0, key.size()-1)] = value;
        }
    }
    return fields;
}
