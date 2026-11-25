#pragma once
#include <cstdint>
struct margin_row {
    int32_t sixth;
    int32_t seventh;
    float gap;
    int32_t near_boundary;
};
static_assert(sizeof(margin_row)==16,"receipt layout");
margin_row * xeon_margin_rows_for(const char *name, const char *prefix, int n, float *cut);
