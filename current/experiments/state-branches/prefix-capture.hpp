// Private, single-slot Linux adapter. Only the manager that submitted this request
// has the random local endpoint. Full state comes from the running backend.
#pragma once
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <set>
#include <string>
#include <vector>

struct prefix_channel {
    int fd = -1;

    explicit prefix_channel(const std::string & name) {
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        const std::string label = "xeon-prefix-" + name;
        memcpy(address.sun_path + 1, label.data(), label.size());
        fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        timeval wait{120, 0};
        if (fd >= 0 && setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &wait, sizeof(wait)) == 0 &&
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &wait, sizeof(wait)) == 0 &&
            ::connect(fd, (sockaddr *)&address, offsetof(sockaddr_un, sun_path) + 1 + label.size()) == 0) {
            return;
        }
        if (fd >= 0) ::close(fd);
        throw std::runtime_error("prefix manager unavailable");
    }
    ~prefix_channel() { ::close(fd); }
    prefix_channel(const prefix_channel &) = delete;
    prefix_channel & operator=(const prefix_channel &) = delete;

    void put(const void * data, size_t length) {
        auto p = static_cast<const char *>(data);
        while (length) {
            const auto n = ::send(fd, p, length, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("prefix connection interrupted");
            p += n;
            length -= n;
        }
    }
    char answer() {
        char c;
        ssize_t n;
        do { n = ::recv(fd, &c, 1, 0); } while (n < 0 && errno == EINTR);
        if (n != 1) throw std::runtime_error("missing prefix manager acknowledgement");
        return c;
    }
};

struct prefix_plan {
    std::set<int> stops, captures;
    std::string channel;
    bool logits = false;

    void reset(const std::string & wire, int limit) {
        *this = {};
        if (wire.empty()) return;
        const auto plan = json::parse(wire);
        channel = plan.at("channel").get<std::string>();
        if (channel.size() != 32 || channel.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::runtime_error("invalid prefix channel");
        logits = plan.value("diagnostic_logits", false);
        for (const auto & v : plan.at("stops")) {
            const int pos = v.get<int>();
            if (pos < 1 || pos > limit) throw std::runtime_error("invalid prefix boundary");
            stops.insert(pos);
        }
        for (const auto & v : plan.at("captures")) {
            const int pos = v.get<int>();
            if (!stops.count(pos)) throw std::runtime_error("missing prefix boundary");
            captures.insert(pos);
        }
    }

    template<class Slot> void complete(Slot & slot, bool prefill) {
        const int pos = slot.prompt.n_tokens();
        if (!stops.erase(pos)) return;
        llama_synchronize(slot.ctx_tgt);
        if (llama_memory_seq_pos_max(llama_get_memory(slot.ctx_tgt), slot.id) != pos - 1)
            throw std::runtime_error("evaluated state and token log disagree");
        const bool capture = captures.erase(pos) != 0;
        // Size/data API has two private 32-bit header fields before the very same
        // full sequence payload written by llama_state_seq_save_file().
        const size_t state_n = capture ? llama_state_seq_get_size(slot.ctx_tgt, slot.id) : 0;
        if (capture && state_n <= 8) throw std::runtime_error("cannot size complete prefix state");
        const size_t packed_n = sizeof(llama_token) * (size_t(pos) + 4);
        const size_t total = capture ? 12 + packed_n + state_n - 8 : 0;
        prefix_channel ch(channel);
        const json record = {{"position",pos},{"phase",prefill ? "prefill" : "decode"},
            {"bytes",total},{"cached",slot.stats.n_prompt_cached},{"processed",slot.stats.n_prompt_processed}};
        const auto line = record.dump() + "\n";
        ch.put(line.data(),line.size());
        const char action = ch.answer();
        if (action == 's') return;
        if (!capture || action != 'w') throw std::runtime_error("invalid prefix decision");
        const auto packed = slot.prompt.tokens.serialize();
        if (packed.size() != packed_n) throw std::runtime_error("only text prefixes are supported");
        std::vector<uint8_t> state(state_n);
        if (llama_state_seq_get_data(slot.ctx_tgt, state.data(), state_n, slot.id) != state_n)
            throw std::runtime_error("full prefix state export failed");
        const uint32_t header[] = {LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION, uint32_t(packed_n / 4)};
        ch.put(header, sizeof(header));
        ch.put(packed.data(),packed.size());
        ch.put(state.data() + 8,state_n - 8);
        if (ch.answer() != 'k') throw std::runtime_error("prefix publication interrupted");
    }
};
