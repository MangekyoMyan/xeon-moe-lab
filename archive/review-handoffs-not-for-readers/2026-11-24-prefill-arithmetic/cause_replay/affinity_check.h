#pragma once
#include <sched.h>
#include <linux/mempolicy.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <thread>

static std::vector<int> cpu_ids(pid_t tid) {
    cpu_set_t set;
    if (sched_getaffinity(tid,sizeof(set),&set)<0) return {};
    std::vector<int> ids;
    for (int cpu=0;cpu<CPU_SETSIZE;++cpu) if (CPU_ISSET(cpu,&set)) ids.push_back(cpu);
    return ids;
}

static std::vector<int> parsed_ids(const common_cpu_params &p) {
    std::vector<int> ids;
    for (int i=0;i<GGML_MAX_N_THREADS;++i) if(p.cpumask[i]) ids.push_back(i);
    return ids;
}

static json task_activity() {
    // Boundary observations use the kernel's per-TID scheduled runtime counter.
    // CPU time across a phase does not identify the team in an individual graph.
    json rows=json::array();
    const auto begin=ns();
    for (const auto &task:fs::directory_iterator("/proc/self/task")) {
        int tid=std::stoi(task.path().filename());
        std::ifstream stat(task.path()/"stat"),schedstat(task.path()/"schedstat");
        std::string line;
        uint64_t runtime=0,waiting=0,slices=0;
        if(!std::getline(stat,line) || !(schedstat>>runtime>>waiting>>slices)) continue;
        auto pos=line.rfind(')');
        if(pos==std::string::npos) continue;
        std::istringstream fields(line.substr(pos+2));
        std::vector<std::string> parts;
        for (std::string field;fields>>field;) parts.push_back(field);
        if(parts.size()<=36) continue;
        rows.push_back({{"tid",tid},{"start_ticks",std::stoull(parts[19])},{"runtime_ns",runtime},
                        {"runqueue_ns",waiting},{"slices",slices},{"last_cpu",std::stoi(parts[36])},
                        {"affinity",cpu_ids(tid)}});
    }
    return {{"begin_ns",begin},{"end_ns",ns()},{"tasks",rows}};
}

static json io_readings() {
    struct rusage usage{};
    need(getrusage(RUSAGE_SELF,&usage)==0,"getrusage");
    json result={{"monotonic_ns",ns()},{"minor_faults",usage.ru_minflt},{"major_faults",usage.ru_majflt}};
    std::ifstream io("/proc/self/io");
    std::string key;
    unsigned long long number;
    while(io>>key>>number) {
        if(key=="read_bytes:" || key=="write_bytes:") result[key.substr(0,key.size()-1)]=number;
    }
    return result;
}

static void await_quality_observer(const fs::path &out) {
    save(out/"quality-observer-ready.json",{{"pid",getpid()},{"monotonic_ns",ns()}});
    auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(12);
    while(!fs::exists(out/"quality-observer-armed.json") && std::chrono::steady_clock::now()<limit)
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
}
