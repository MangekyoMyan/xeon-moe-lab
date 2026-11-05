#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <cstdlib>
#include <time.h>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}


static void perf_control(const char * command) {
    const char * ctl_path = std::getenv("IPC_CONTROL_FIFO");
    const char * ack_path = std::getenv("IPC_ACK_FIFO");
    check(ctl_path && ack_path, "perf control FIFO environment missing");
    static const int ctl = open(ctl_path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    static const int ack = open(ack_path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    check(ctl >= 0 && ack >= 0, "open perf control FIFOs");
    const std::string message = std::string(command) + "\n";
    check(write(ctl, message.data(), message.size()) == (ssize_t) message.size(), "perf command write");
    std::string reply;
    while (reply.empty() || reply.back() != '\n') {
        pollfd waitfd = {ack, POLLIN, 0};
        check(poll(&waitfd, 1, 5000) == 1, "perf acknowledgement timeout");
        char c; check(read(ack, &c, 1) == 1, "perf acknowledgement read");
        // perf 6.12 writes sizeof("ack\n"), including a trailing NUL.
        if (c == '\0' && reply.empty()) continue;
        reply += c; check(reply.size() < 32, "invalid perf acknowledgement length");
    }
    check(reply == "ack\n", "invalid perf acknowledgement");
}

static std::vector<int> int_list(const std::string & value) {
    std::vector<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        size_t end = 0;
        const int number = std::stoi(item, &end);
        check(end == item.size() && number >= 1 && number <= 4096, "invalid token count");
        result.push_back(number);
    }
    check(!result.empty(), "empty token list");
    return result;
}

static uint64_t fingerprint(const void * data, size_t bytes) {
    uint64_t value = UINT64_C(14695981039346656037);
    const auto * input = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        value = (value ^ input[i]) * UINT64_C(1099511628211);
    }
    return value;
}

int main(int argc, char ** argv) try {
    int threads = 2, experts = 4, used = 2, repeats = 3, warmups = 1, input_pad = 0;
    int64_t custom_k = 0, custom_m = 0;
    int requested_reference_rows = 0;
    bool quick = false, real_shapes = false, explicit_tokens = false, use_ref = false, strict_cpu = false;
    std::vector<int> token_counts = {1, 8, 16};
    std::string wanted_type = "all", mode = "both", routing = "cyclic", dump_prefix;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            check(i + 1 < argc, "missing argument value");
            return argv[++i];
        };
        if (a == "--threads") threads = std::stoi(value());
        else if (a == "--experts") experts = std::stoi(value());
        else if (a == "--topk") used = std::stoi(value());
        else if (a == "--tokens") { token_counts = int_list(value()); explicit_tokens = true; }
        else if (a == "--type") wanted_type = value();
        else if (a == "--mode") mode = value();
        else if (a == "--routing") routing = value();
        else if (a == "--k") custom_k = std::stoll(value());
        else if (a == "--m") custom_m = std::stoll(value());
        else if (a == "--repeats") repeats = std::stoi(value());
        else if (a == "--warmups") warmups = std::stoi(value());
        else if (a == "--reference-rows") requested_reference_rows = std::stoi(value());
        else if (a == "--input-pad") input_pad = std::stoi(value());
        else if (a == "--dump-prefix") dump_prefix = value();
        else if (a == "--use-ref") use_ref = true;
        else if (a == "--strict-cpu") strict_cpu = true;
        else if (a == "--quick") quick = true;
        else if (a == "--real-shapes") real_shapes = true;
        else throw std::runtime_error("usage: cpu-quant-probe [--threads N] [--quick] [--real-shapes] [--type all|mxfp4|iq3_xxs|iq3_s|iq2_s] [--tokens 1,2,3,7,8,9,16,17] [--experts N] [--topk N] [--routing cyclic|shared|spread] [--mode up|down|both] [--k N] [--m N] [--repeats N] [--warmups N] [--reference-rows N] [--input-pad N] [--use-ref] [--dump-prefix PATH]");
    }
    check(threads >= 1 && threads <= 64, "thread bound");
    check(experts >= 1 && experts <= 384 && used >= 1 && used <= experts && used <= 32, "expert/topk bound");
    check(custom_k >= 0 && custom_k <= 65536 && custom_m >= 0 && custom_m <= 65536, "shape bound");
    check(repeats >= 1 && repeats <= 101 && warmups >= 0 && warmups <= 10, "repeat bound");
    check(requested_reference_rows >= 0 && requested_reference_rows <= 65536, "reference row bound");
    check(input_pad >= 0 && input_pad <= 128, "input pad bound");
    check(mode == "up" || mode == "down" || mode == "both", "unknown mode");
    check(routing == "cyclic" || routing == "shared" || routing == "spread", "unknown routing");
    check(wanted_type == "all" || wanted_type == "mxfp4" || wanted_type == "iq3_xxs" || wanted_type == "iq3_s" || wanted_type == "iq2_s", "unknown type");
    if (quick && !explicit_tokens) token_counts = {1, 16};

    auto cpu = ggml_backend_cpu_init();
    check(cpu, "CPU backend");
    ggml_backend_cpu_set_use_ref(cpu, use_ref);
    ggml_backend_cpu_set_n_threads(cpu, threads);
    auto params = ggml_threadpool_params_default(threads);
    params.poll = 0;
    if (strict_cpu) {
        cpu_set_t selected; CPU_ZERO(&selected);
        check(sched_getaffinity(0, sizeof(selected), &selected) == 0, "get affinity");
        int count = 0;
        for (int id = 0; id < CPU_SETSIZE; ++id) if (CPU_ISSET(id, &selected)) {
            check(id < GGML_MAX_N_THREADS, "CPU id beyond ggml mask");
            params.cpumask[id] = true; ++count;
        }
        check(count == threads, "strict mode requires exactly threads CPUs in inherited mask");
        params.strict_cpu = true;
    }
    auto pool = ggml_threadpool_new(&params);
    check(pool, "threadpool");
    ggml_backend_cpu_set_threadpool(cpu, pool);
    int failures = 0, cases = 0;
    for (auto type : {GGML_TYPE_MXFP4, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S}) {
        const std::string type_name = ggml_type_name(type);
        if (wanted_type != "all" && wanted_type != type_name) continue;
        ggml_quantize_init(type);
        for (bool down : {false, true}) {
            if ((quick && down && mode == "both") || (mode == "up" && down) || (mode == "down" && !down)) continue;
            const int64_t k = custom_k ? custom_k : (real_shapes ? (down ? 2304 : 5120) : (down ? 256 : 512));
            const int64_t m = custom_m ? custom_m : (real_shapes ? (down ? 5120 : 2304) : (down ? 512 : 256));
            check(k % ggml_blck_size(type) == 0, "k must be divisible by the quantization block size");
            check(k * m <= (INT64_C(256) * 1024 * 1024) / experts, "synthetic FP32 weight allocation exceeds 1 GiB");
            std::mt19937 weight_rng(42);
            std::uniform_real_distribution<float> dist(-.5f, .5f);
            std::vector<float> weights(k * m * experts), importance(k, 1.f);
            for (float & x : weights) x = dist(weight_rng);
            std::vector<uint8_t> packed(ggml_row_size(type, k) * m * experts);
            const auto written = ggml_quantize_chunk(type, weights.data(), packed.data(), 0, m * experts, k, importance.data());
            check(written == packed.size(), "quantized size");
            weights.clear();
            weights.shrink_to_fit();
            const uint64_t weight_hash = fingerprint(packed.data(), packed.size());
            for (int tokens : token_counts) {
                check(k * (down ? used : 1) * tokens <= INT64_C(64) * 1024 * 1024, "input allocation bound");
                check(m * used * tokens <= INT64_C(64) * 1024 * 1024, "output allocation bound");
                auto ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
                check(ctx, "context");
                auto w = ggml_new_tensor_3d(ctx, type, k, m, experts);
                auto x_storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k + input_pad, down ? used : 1, tokens);
                auto x = input_pad ? ggml_view_3d(ctx, x_storage, k, down ? used : 1, tokens, x_storage->nb[1], x_storage->nb[2], 0) : x_storage;
                auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
                auto y = ggml_mul_mat_id(ctx, w, x, ids);
                auto graph = ggml_new_graph_custom(ctx, 32, false);
                ggml_build_forward_expand(graph, y);
                auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
                check(buffer, "tensor memory");
                std::vector<float> input(ggml_nelements(x)), storage(ggml_nelements(x_storage), 13.375f), result(ggml_nelements(y));
                std::mt19937 input_rng(1009u + 37u * tokens + (down ? 17u : 0u));
                for (float & v : input) v = dist(input_rng);
                for (int row = 0; row < tokens * (down ? used : 1); ++row) {
                    std::copy_n(input.data() + row * k, k, storage.data() + row * (k + input_pad));
                }
                std::vector<int32_t> route(used * tokens);
                std::vector<int> occupancy(experts, 0);
                for (int t = 0; t < tokens; ++t) for (int j = 0; j < used; ++j) {
                    const int e = routing == "shared" ? j : (routing == "spread" ? (t * used + j) % experts : (t + j) % experts);
                    route[t * used + j] = e;
                    ++occupancy[e];
                }
                ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
                ggml_backend_tensor_set(x_storage, storage.data(), 0, storage.size() * sizeof(float));
                ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
                auto compute = [&] {
                    check(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "graph compute");
                    ggml_backend_synchronize(cpu);
                };
                for (int warmup = 0; warmup < warmups; ++warmup) compute();
                if (strict_cpu && cases == 0 && warmups > 0) {
                    std::fprintf(stderr, "{\"strict_worker_affinities\":[");
                    bool first = true;
                    for (const auto & entry : std::filesystem::directory_iterator("/proc/self/task")) {
                        int tid = std::stoi(entry.path().filename().string()); cpu_set_t actual; CPU_ZERO(&actual);
                        if (sched_getaffinity(tid, sizeof(actual), &actual) != 0) continue;
                        std::fprintf(stderr, "%s{\"tid\":%d,\"cpus\":[", first ? "" : ",", tid); first = false;
                        bool first_cpu = true;
                        for (int id = 0; id < CPU_SETSIZE; ++id) if (CPU_ISSET(id, &actual)) {
                            std::fprintf(stderr, "%s%d", first_cpu ? "" : ",", id); first_cpu = false;
                        }
                        std::fprintf(stderr, "]}");
                    }
                    std::fprintf(stderr, "]}\n");
                }
                check(cases == 0 && token_counts.size() == 1 && wanted_type != "all" && mode != "both", "perf run must contain one case");
                std::vector<double> times;
                times.reserve(repeats);
                perf_control("enable");
                timespec cpu_start, cpu_end;
                check(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start) == 0, "process CPU clock");
                const auto roi_start = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    const auto start = std::chrono::steady_clock::now();
                    compute();
                    times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                }
                const double roi_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - roi_start).count();
                check(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end) == 0, "process CPU clock");
                const double process_cpu_ms = (cpu_end.tv_sec-cpu_start.tv_sec)*1000.0+(cpu_end.tv_nsec-cpu_start.tv_nsec)/1e6;
                perf_control("disable");
                std::fprintf(stderr, "{\"roi_process_cpu_ms\":%.9f}\n", process_cpu_ms);
                std::fprintf(stderr, "{\"perf_roi_wall_ms\":%.9f,\"perf_roi_repeats\":%d,\"perf_controls_acknowledged\":2}\n", roi_ms, repeats);
                auto sorted_times = times;
                std::sort(sorted_times.begin(), sorted_times.end());
                const double median = (sorted_times[(repeats - 1) / 2] + sorted_times[repeats / 2]) * .5;
                ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
                int nonfinite = 0;
                for (float v : result) nonfinite += !std::isfinite(v);
                double err = 0, norm = 0, maxabs = 0;
                std::vector<float> row(k);
                auto convert = ggml_get_type_traits(type)->to_float;
                check(convert, "dequantizer");
                const size_t stride = ggml_row_size(type, k);
                const int64_t reference_rows = std::min<int64_t>(m, requested_reference_rows ? requested_reference_rows : (k * m > 512 * 512 ? 8 : m));
                for (int t = 0; t < tokens; ++t) for (int j = 0; j < used; ++j) for (int64_t sample = 0; sample < reference_rows; ++sample) {
                    const int64_t ri = reference_rows == 1 ? 0 : (m - 1) * sample / (reference_rows - 1);
                    convert(packed.data() + (route[t * used + j] * m + ri) * stride, row.data(), k);
                    const auto * in = input.data() + (t * (down ? used : 1) + (down ? j : 0)) * k;
                    double ref = 0;
                    for (int64_t z = 0; z < k; ++z) ref += double(row[z]) * in[z];
                    const double delta = ref - result[(t * used + j) * m + ri];
                    err += delta * delta;
                    norm += ref * ref;
                    maxabs = std::max(maxabs, std::abs(delta));
                }
                const double nmse = err / std::max(norm, 1e-30);
                const bool pass = norm > 0 && nonfinite == 0 && nmse < 5e-4;
                failures += !pass;
                ++cases;
                if (!dump_prefix.empty()) {
                    const std::string path = dump_prefix + "-" + type_name + "-" + (down ? "down" : "up") + "-k" + std::to_string(k) + "-m" + std::to_string(m) + "-t" + std::to_string(tokens) + "-e" + std::to_string(experts) + "-u" + std::to_string(used) + "-" + routing + "-pad" + std::to_string(input_pad) + ".f32";
                    std::ofstream output(path, std::ios::binary);
                    output.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
                    check(output.good(), "output dump write");
                }
                std::printf("{\"type\":\"%s\",\"k\":%lld,\"m\":%lld,\"tokens\":%d,\"experts\":%d,\"topk\":%d,\"threads\":%d,\"down\":%s,\"routing\":\"%s\",\"input_pad\":%d,\"use_ref\":%s,\"median_ms\":%.9f,\"sample_reference_nmse\":%.9g,\"max_abs_error\":%.9g,\"reference_rows\":%lld,\"full_reference\":%s,\"nonfinite\":%d,\"weight_bytes\":%zu,\"input_bytes\":%zu,\"input_storage_bytes\":%zu,\"output_bytes\":%zu,\"weight_fnv1a64\":\"%016llx\",\"input_fnv1a64\":\"%016llx\",\"route_fnv1a64\":\"%016llx\",\"output_fnv1a64\":\"%016llx\",\"expert_rows\":[",
                    type_name.c_str(), (long long) k, (long long) m, tokens, experts, used, threads, down ? "true" : "false", routing.c_str(), input_pad, use_ref ? "true" : "false", median, nmse, maxabs, (long long) reference_rows, reference_rows == m ? "true" : "false", nonfinite, packed.size(), input.size() * sizeof(float), storage.size() * sizeof(float), result.size() * sizeof(float), (unsigned long long) weight_hash, (unsigned long long) fingerprint(storage.data(), storage.size() * sizeof(float)), (unsigned long long) fingerprint(route.data(), route.size() * sizeof(int32_t)), (unsigned long long) fingerprint(result.data(), result.size() * sizeof(float)));
                for (int e = 0; e < experts; ++e) std::printf("%s%d", e ? "," : "", occupancy[e]);
                std::printf("],\"times_ms\":[");
                for (int repeat = 0; repeat < repeats; ++repeat) std::printf("%s%.9f", repeat ? "," : "", times[repeat]);
                std::printf("],\"pass\":%s}\n", pass ? "true" : "false");
                std::fflush(stdout);
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
            }
        }
    }
    ggml_backend_cpu_set_threadpool(cpu, nullptr);
    ggml_threadpool_free(pool);
    ggml_backend_free(cpu);
    std::printf("{\"cases\":%d,\"failures\":%d,\"synthetic_weights\":true,\"model_downloaded\":false,\"dump_native_fp32\":%s}\n", cases, failures, dump_prefix.empty() ? "false" : "true");
    return failures ? 2 : 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
