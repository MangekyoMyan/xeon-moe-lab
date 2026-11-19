#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum iq08_quality_type_index {
    IQ08_QUALITY_IQ2_S = 0,
    IQ08_QUALITY_IQ3_XXS = 1,
    IQ08_QUALITY_IQ3_S = 2,
    IQ08_QUALITY_TYPE_COUNT = 3
};

// dso_path belongs to the loader for the lifetime of the loaded DSO. A null
// path denotes a function that has never been called in this process.
struct iq08_quality_binding_v1 {
    const char *dso_path;
    uintptr_t offset_from_dso_base;
};

struct iq08_quality_type_counts_v1 {
    uint64_t selector_calls;
    uint64_t dot_rows_nrc1;
    uint64_t other_nrc_calls;
    struct iq08_quality_binding_v1 underlying_dot;
};

struct iq08_quality_counts_v1 {
    uint32_t abi_version;
    uint32_t struct_bytes;
    uint64_t pack_calls;
    uint64_t pack_blocks;
    uint64_t empty_pack_calls;
    uint64_t other_selector_calls;
    struct iq08_quality_binding_v1 underlying_pack;
    struct iq08_quality_binding_v1 underlying_selector;
    struct iq08_quality_type_counts_v1 type[IQ08_QUALITY_TYPE_COUNT];
};

// Call between synchronous CPU graphs with the workers quiescent. Counts are
// monotonic. Subtract two snapshots to isolate a single graph or projection.
// Returns 1 on success and 0 for a size mismatch or an unknown provider.
int xeon_iq08_quality_snapshot_v1(struct iq08_quality_counts_v1 *out, size_t out_bytes);

#ifdef __cplusplus
}
#endif
