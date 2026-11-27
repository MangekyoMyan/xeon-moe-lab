#pragma once
#include <fstream>
#include <map>
#include <sched.h>
#include <set>
#include <vector>
#include <linux/mempolicy.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

// Remember the launcher mask before initializing ggml. Later, the calling
// thread itself can be pinned to only the first physical core.
static std::vector<int> physical_ids;
static constexpr int private_threads = 24;

static json read_entry_topology() {
    cpu_set_t allowed;
    need(sched_getaffinity(0,sizeof(allowed),&allowed)==0,"initial operator mask");
    std::set<std::pair<int,int>> cores;
    std::map<int,int> sockets;
    json rows=json::array();
    for (int cpu=0;cpu<CPU_SETSIZE;++cpu) {
        if (!CPU_ISSET(cpu,&allowed)) continue;
        auto base=std::filesystem::path("/sys/devices/system/cpu")/("cpu"+std::to_string(cpu))/"topology";
        std::ifstream socket_file(base/"physical_package_id"), core_file(base/"core_id");
        int socket,core;
        need(bool(socket_file>>socket) && bool(core_file>>core),"read physical topology");
        need(cpu<GGML_MAX_N_THREADS && cores.emplace(socket,core).second,"one logical CPU per physical core");
        ++sockets[socket];
        physical_ids.push_back(cpu);
        rows.push_back({{"cpu",cpu},{"socket",socket},{"core",core}});
    }
    need(physical_ids.size()==private_threads && sockets.size()==2 && sockets.at(0)==12 && sockets.at(1)==12,"dual physical 12+12");
    int mode=-1;unsigned long nodes=0;
    need(syscall(SYS_get_mempolicy,&mode,&nodes,8*sizeof(nodes),nullptr,0)==0 && mode==MPOL_INTERLEAVE && nodes==3 &&
         prctl(PR_GET_THP_DISABLE,0,0,0,0)==1,"operator NUMA and THP");
    return {{"pid",getpid()},{"cores",rows},{"mempolicy_mode",mode},{"node_mask",nodes},{"thp_disabled",true}};
}

static ggml_threadpool_params private_pool_params() {
    auto params=ggml_threadpool_params_default(private_threads);
    params.poll=0;
    params.strict_cpu=true;
    for (int cpu:physical_ids) params.cpumask[cpu]=true;
    return params;
}
