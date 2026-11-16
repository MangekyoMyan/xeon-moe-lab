#include "census.h"
#include <cstring>
#include <thread>
#include <vector>
int main() {
    ggml_tensor weights{}, ids{};
    char dummy;
    weights.data = &dummy; weights.type = GGML_TYPE_IQ3_XXS;
    weights.ne[0] = 5120; weights.ne[1] = 2304; weights.ne[2] = 384;
    std::strcpy(weights.name, "blk.0.ffn_gate_exps.weight");
    ids.ne[0] = 6; ids.ne[1] = 1;
    int64_t counts[384]{}; counts[0] = 2; counts[17] = 1; counts[383] = 3;
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) threads.emplace_back([&] {
        for (int j = 0; j < 100; ++j) xeon_moe_census_v1(&weights, &ids, counts, 384);
    });
    for (auto & thread : threads) thread.join();
    ids.ne[1] = 16;
    xeon_moe_census_v1(&weights, &ids, counts, 384);
}
