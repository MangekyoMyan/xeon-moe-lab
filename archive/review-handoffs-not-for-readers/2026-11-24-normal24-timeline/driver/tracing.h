#pragma once

// CUPTI must start before any CUDA graph is instantiated, including during
// argument processing and projector capability detection.
struct TraceHooks {
    using Simple = int (*)();
    using Call = int (*)(unsigned, unsigned);
    using Dump = int (*)(const char *);
    Simple initialize = reinterpret_cast<Simple>(dlsym(RTLD_DEFAULT, "s169_trace_initialize"));
    Simple window_begin = reinterpret_cast<Simple>(dlsym(RTLD_DEFAULT, "s169_trace_window_begin"));
    Call call_begin = reinterpret_cast<Call>(dlsym(RTLD_DEFAULT, "s169_trace_call_begin"));
    Simple call_end = reinterpret_cast<Simple>(dlsym(RTLD_DEFAULT, "s169_trace_call_end"));
    Simple window_end = reinterpret_cast<Simple>(dlsym(RTLD_DEFAULT, "s169_trace_window_end"));
    Dump dump = reinterpret_cast<Dump>(dlsym(RTLD_DEFAULT, "s169_trace_dump"));

    void start(bool enabled, const std::string &path) const {
        if (!enabled) {
            need(!initialize && !window_begin && !call_begin && !call_end && !window_end && !dump && !std::getenv("LD_PRELOAD"), "control isolation");
            return;
        }
        need(initialize && window_begin && call_begin && call_end && window_end && dump, "trace hooks");
        Dl_info provider{};
        const char *preload = std::getenv("LD_PRELOAD");
        need(preload && path == preload && dladdr(reinterpret_cast<void *>(initialize), &provider) &&
             provider.dli_fname && fs::canonical(provider.dli_fname) == fs::canonical(path), "trace provider");
        need(initialize() == 0, "early CUPTI initialization");
    }
};

// This handshake is after CUPTI's 4-call window. CPU PC samples target
// the remaining 60 calls and are never attached to the control process.
static void arm_pc_collector(const fs::path &out, int c) {
    const auto tag = "pc-" + std::to_string(c);
    save(out / (tag + "-ready.json"), {{"pid",getpid()},{"ns",ns()}});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    while (!fs::exists(out / (tag + "-armed.json")) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
