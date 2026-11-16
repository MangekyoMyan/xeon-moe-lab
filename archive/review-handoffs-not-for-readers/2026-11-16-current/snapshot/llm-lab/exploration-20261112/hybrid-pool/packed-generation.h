#pragma once
#include <stdint.h>
#include <string.h>
// Preserve the existing private signed atomic transport without signed shifts.
static inline int xeon_next_packed_graph(int previous, int threads) {
#ifdef __cplusplus
    static_assert(sizeof(int) == sizeof(uint32_t));
#else
    _Static_assert(sizeof(int) == sizeof(uint32_t), "32-bit private transport required");
#endif
    uint32_t old = (uint32_t) previous;
    uint32_t packed = ((old + UINT32_C(0x10000)) & UINT32_C(0xffff0000)) |
                      ((uint32_t) threads & UINT32_C(0xffff));
    int result;
    memcpy(&result, &packed, sizeof(result));
    return result;
}
