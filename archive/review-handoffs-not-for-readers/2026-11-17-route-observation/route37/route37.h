#pragma once
#include "ggml.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Diagnostic singleton: one context, one caller, one CPU producer (ith==0).
// Control/read APIs require no graph in flight. end_eval requires that all CPU
// workers have returned; complete_call additionally requires the caller's normal
// end-of-decode synchronization. No API here performs synchronization for you.
enum xeon_route37_phase { XR_UNKNOWN=0, XR_PREFILL=1, XR_DECODE=2, XR_WARMUP=3 };
enum xeon_route37_error {
    XR_MAPPING=1, XR_GEOMETRY=2, XR_IDS=4, XR_DUPLICATE=8, XR_COVERAGE=16,
    XR_OVERFLOW=32, XR_GRAPH_FAILED=64, XR_CALL_FAILED=128, XR_UNTAGGED=256,
    XR_UNSYNCHRONIZED=512, XR_NO_EVAL=1024
};
struct xeon_route37_record {
    uint64_t ordinal, attempt;
    uint32_t request; int32_t sequence;
    uint32_t decode_call, ubatch;
    uint64_t logical_token; int64_t position;
    uint64_t graph_generation;
    uint32_t context; uint16_t layer; uint8_t phase, projection;
    uint32_t n_tokens; int32_t ids[6]; uint32_t flags; uint8_t reserved[32];
};
struct xeon_route37_label {
    uint32_t request; int32_t sequence; uint32_t decode_call; uint8_t phase;
    uint8_t reserved[3]; uint64_t logical_token;
};
struct xeon_route37_attempt {
    struct xeon_route37_label label;
    uint64_t id, generation, first_record, records, layers;
    int64_t position; uint32_t context, ubatch, n_tokens, flags;
    int32_t graph_status; uint8_t observing, scope_t1, graph_returned, caller_completed;
};
struct xeon_route37_call {
    struct xeon_route37_label label;
    uint64_t first_attempt, attempts;
    uint32_t flags; int32_t status; uint8_t synchronized, completed; uint8_t reserved[6];
};
struct xeon_route37_stats {
    uint64_t records, attempts, calls, dropped, complete_decode_attempts;
    uint32_t flags; uint8_t enabled, call_open, eval_open, observing;
};
int xeon_route37_open_v1(const void * owner, uint32_t context_id, int observing,
                       uint32_t record_limit, uint32_t attempt_limit, uint32_t call_limit);
int xeon_route37_matches_v1(const void * owner);
int xeon_route37_label_v1(const void * owner, const struct xeon_route37_label * label);
int xeon_route37_begin_eval_v1(const void * owner, const struct ggml_tensor * const * gates,
    size_t count, int graph_rebuilt, uint32_t n_tokens, uint32_t n_positions,
    uint32_t sequence_count, int32_t sequence, int64_t position, int pipeline_parallel);
int xeon_route37_end_eval_v1(const void * owner, int32_t graph_status);
int xeon_route37_complete_call_v1(const void * owner, int32_t status, int synchronized);
int xeon_route37_close_v1(const void * owner);
// Worker-only hook. No allocation, lock, clock, file I/O, or added barrier.
void xeon_route37_record_v1(const struct ggml_tensor * weight, const struct ggml_tensor * ids);
int xeon_route37_stats_v1(struct xeon_route37_stats * out);
int xeon_route37_read_record_v1(size_t index, struct xeon_route37_record * out);
int xeon_route37_read_attempt_v1(size_t index, struct xeon_route37_attempt * out);
int xeon_route37_read_call_v1(size_t index, struct xeon_route37_call * out);
#ifdef __cplusplus
}
static_assert(sizeof(xeon_route37_record)==128,"route ABI must be128 bytes");
static_assert(offsetof(xeon_route37_record,ids)==68,"route ABI ids offset");
static_assert(offsetof(xeon_route37_record,reserved)==96,"route ABI reserved offset");
#endif
