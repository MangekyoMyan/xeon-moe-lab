#pragma once
#include "ggml.h"
#include <charconv>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

namespace xeon_iq3 {
struct selection_plan {
    std::vector<std::vector<uint32_t>> experts;
    size_t selected = 0;
    size_t bytes = 0;
};

inline selection_plan parse_plan(const std::string & text, const std::vector<const ggml_tensor *> & tensors,
                                  size_t budget, size_t page) {
    if (text.size() > 1024*1024 || !page || page > 65536) throw std::runtime_error("IQ plan: size/page limit");
    std::istringstream input(text);
    auto token = [&]() {
        std::string value;
        if (!(input >> value)) throw std::runtime_error("IQ plan: incomplete file");
        return value;
    };
    auto number = [&]() -> size_t {
        const std::string value = token();
        size_t result = 0;
        const auto converted = std::from_chars(value.data(),value.data()+value.size(),result);
        if (converted.ec != std::errc() || converted.ptr != value.data()+value.size()) throw std::runtime_error("IQ plan: invalid unsigned integer");
        return result;
    };
    if (token() != "XEON_IQ_CACHE_PLAN_V2") throw std::runtime_error("IQ plan: wrong format");
    const size_t count = number(), selected = number(), bytes = number();
    if (count != tensors.size() || count > 1024 || bytes > budget) throw std::runtime_error("IQ plan: inventory/budget mismatch");
    std::unordered_map<std::string,size_t> positions;
    std::unordered_set<const void *> pointers;
    for (size_t i = 0; i < count; ++i) {
        const auto * t = tensors[i];
        if (!t || !positions.emplace(t->name,i).second || (t->data && !pointers.insert(t->data).second)) throw std::runtime_error("IQ plan: duplicate tensor or source alias");
    }
    selection_plan result;
    result.experts.resize(count);
    std::vector<bool> seen(count,false);
    const size_t maximum = std::numeric_limits<size_t>::max();
    for (size_t line = 0; line < count; ++line) {
        const auto found = positions.find(token());
        if (found == positions.end() || seen[found->second]) throw std::runtime_error("IQ plan: unknown or duplicate tensor");
        const size_t index = found->second;
        seen[index] = true;
        const auto * t = tensors[index];
        const std::string type = token();
        const size_t k = number(), m = number(), e = number(), n = number();
        if (t->ne[0] <= 0 || t->ne[0] > INT32_MAX || t->ne[1] <= 0 || t->ne[2] <= 0 ||
            type != ggml_type_name(t->type) || k != size_t(t->ne[0]) || m != size_t(t->ne[1]) || e != size_t(t->ne[2]) ||
            !k || k%256 || !m || !e || e > 65536 || n > e || t->ne[3] != 1 || !ggml_is_contiguous(t) ||
            (t->type != GGML_TYPE_IQ3_XXS && t->type != GGML_TYPE_IQ2_S && t->type != GGML_TYPE_IQ3_S))
            throw std::runtime_error("IQ plan: type/shape mismatch");
        auto & ids = result.experts[index];
        ids.reserve(n);
        for (size_t slot = 0; slot < n; ++slot) {
            const size_t id = number();
            if (id >= e || (slot && id <= ids.back())) throw std::runtime_error("IQ plan: expert IDs must be unique, sorted and in range");
            ids.push_back(uint32_t(id));
        }
        if (n > maximum-result.selected) throw std::runtime_error("IQ plan: selected count overflow");
        result.selected += n;
        if (!n) continue;
        if (k/256 > maximum/m || (k/256)*m > maximum/144) throw std::runtime_error("IQ plan: expert size overflow");
        const size_t expert_bytes = (k/256)*m*144;
        if (n > (maximum-page+1)/expert_bytes) throw std::runtime_error("IQ plan: allocation size overflow");
        const size_t allocation = (n*expert_bytes+page-1)/page*page;
        if (allocation > maximum-result.bytes) throw std::runtime_error("IQ plan: total size overflow");
        result.bytes += allocation;
    }
    std::string extra;
    if (input >> extra) throw std::runtime_error("IQ plan: trailing token");
    if (result.selected != selected || result.bytes != bytes || result.bytes > budget) throw std::runtime_error("IQ plan: declared totals mismatch");
    return result;
}

inline selection_plan read_plan(const char * path, const std::vector<const ggml_tensor *> & tensors, size_t budget) {
    const int fd = open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
    if (fd < 0) throw std::runtime_error("IQ plan: cannot open regular file");
    struct descriptor { int fd; ~descriptor() { close(fd); } } owned{fd};
    struct stat info{};
    if (fstat(fd,&info) || !S_ISREG(info.st_mode) || info.st_size < 0 || info.st_size > 1024*1024)
        throw std::runtime_error("IQ plan: regular file size limit");
    std::string text;
    text.reserve(size_t(info.st_size));
    char buffer[16384];
    for (;;) {
        const ssize_t count = ::read(fd,buffer,sizeof(buffer));
        if (count == 0) break;
        if (count < 0) { if (errno == EINTR) continue; throw std::runtime_error("IQ plan: file read error"); }
        if (size_t(count) > 1024*1024-text.size()) throw std::runtime_error("IQ plan: growing file size limit");
        text.append(buffer,size_t(count));
    }
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) throw std::runtime_error("IQ plan: page size unavailable");
    return parse_plan(text,tensors,budget,size_t(page));
}
}
