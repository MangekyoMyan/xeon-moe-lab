#pragma once
#include "ggml.h"
#include <cstring>
// Ranked by measured single-input savings on this DeepSeek/Xeon configuration.
// Similar up/gate results are tied; stable model order avoids overfitting small timing differences.
inline int xeon_iq_cache_priority(enum ggml_type type, const char *name) {
    if (type == GGML_TYPE_IQ3_S) return 0;
    if (!std::strstr(name, "ffn_down_exps.weight")) return 1;
    return type == GGML_TYPE_IQ2_S ? 2 : 3;
}
