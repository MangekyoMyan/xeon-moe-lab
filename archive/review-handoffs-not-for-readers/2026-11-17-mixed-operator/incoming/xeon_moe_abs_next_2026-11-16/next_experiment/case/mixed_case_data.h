#pragma once
// Saved synthetic inputs / route IDs only; no model weights.
// Cplan is a cache-membership diagnostic, not typical API traffic.
namespace xeon_mixed_case {
inline constexpr int K = 5120;
inline constexpr int M = 2304;
inline constexpr int EXPERTS = 384;
inline constexpr int USED = 6;
inline constexpr int GROUPS = 8;
inline constexpr int INPUTS = 1;
inline constexpr int routes[48] = {
    79, 176, 232, 279, 273, 173, 256, 363,
    47, 140, 154, 4, 299, 223, 276, 235,
    133, 196, 245, 112, 321, 101, 244, 183,
    300, 295, 224, 288, 206, 287, 226, 33,
    98, 120, 283, 124, 12, 338, 14, 191,
    270, 81, 310, 61, 107, 89, 352, 27,
};
inline constexpr int cache_ids[39] = {
    4, 12, 14, 33, 61, 81, 89, 98,
    101, 107, 112, 120, 124, 133, 140, 154,
    173, 176, 183, 191, 196, 223, 232, 245,
    256, 270, 273, 276, 279, 283, 287, 288,
    299, 300, 310, 321, 338, 352, 363,
};
inline constexpr int masks[8] = {
    62, 59, 55, 47, 41, 62, 63, 31,
};
inline constexpr const char* commit = "31b1292b2767618a28ef6e3f1c9c4450b1fc21be";
inline constexpr const char* tensor_name = "blk.0.ffn_up_exps.weight";
inline constexpr const char* inputs_sha256 = "9dc5705e04389d3d08d894ddf25ce675fa2ff3bed73eccc4ba9a156c8028ec93";
inline constexpr const char* plan_sha256 = "cf11b82e3e953bbb06a0d5df164dbb6d53f6b19923efaeb79c72ceeef068bee6";
} // namespace xeon_mixed_case
