#!/usr/bin/env python3
"""Create an isolated CPU-only, decode-only paired MUL_MAT_ID experiment."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source):
        parser.error("Use a new output outside the source directory")
    output.mkdir(parents=True)
    shutil.copytree(source, output / "source")
    rel = Path("ggml/src/ggml-cpu/ggml-cpu.c")
    original = (source / rel).read_text()
    text = replace_once(original,
        "static void ggml_compute_forward_mul_mat_id(\n        const struct ggml_compute_params * params,\n              struct ggml_tensor * dst) {",
        "static void ggml_compute_forward_mul_mat_id(\n        const struct ggml_compute_params * params,\n              struct ggml_tensor * dst,\n              struct ggml_tensor * pair) {")
    text = replace_once(text,
        "if (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {",
        "if (!pair && ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {")
    text = replace_once(text,
        "                src0_cur, matrix_rows, row_size, src1_cont, wdata\n            );",
        """                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (pair) {
                const struct ggml_tensor * pair_weight = pair->src[0];
                ggml_compute_forward_mul_mat_id_one_chunk(
                    pair, pair_weight, src1, ids, cur_a,
                    ir0_start, ir0_end, ir1_start, ir1_end,
                    (const char *) pair_weight->data + cur_a * pair_weight->nb[2],
                    matrix_rows, row_size, src1_cont, wdata
                );
            }""")
    text = replace_once(text,
        "ggml_compute_forward_mul_mat_id(params, tensor);",
        "ggml_compute_forward_mul_mat_id(params, tensor, NULL);")
    anchor = "static bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards\n"
    text = replace_once(text, anchor, anchor + """
static bool ggml_cpu_moe_pair = true;
static bool ggml_cpu_moe_pair_trace = false;

static bool ggml_cpu_plain_tensor(const struct ggml_tensor * tensor) {
    if (tensor->extra != NULL) {
        return false;
    }
    if (tensor->buffer == NULL) {
        return true;
    }
    // CPU and CPU_Mapped have no format conversion and share the host predicate.
    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
    const ggml_backend_buffer_type_t plain = ggml_backend_cpu_buffer_type();
    return buft->iface.is_host == plain->iface.is_host &&
        buft->iface.alloc_buffer == plain->iface.alloc_buffer &&
        tensor->buffer->iface.init_tensor == NULL;
}

static bool ggml_cpu_can_pair_moe(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->op != GGML_OP_MUL_MAT_ID || b->op != GGML_OP_MUL_MAT_ID ||
        !(b->flags & GGML_TENSOR_FLAG_COMPUTE) || ggml_is_empty(a) || ggml_is_empty(b) ||
        a->src[1] != b->src[1] || a->src[2] != b->src[2] ||
        a->src[0]->type != GGML_TYPE_IQ2_XXS || b->src[0]->type != GGML_TYPE_IQ2_XXS ||
        a->src[1]->type != GGML_TYPE_F32 ||
        a->src[1]->ne[1] != 1 || a->src[1]->ne[2] != 1 || a->src[1]->ne[3] != 1 ||
        a->src[2]->ne[1] != 1 ||
        a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(a->src[0], b->src[0]) || !ggml_are_same_shape(a, b) ||
        !ggml_is_contiguous(a->src[0]) || !ggml_is_contiguous(b->src[0]) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(b) ||
        !ggml_cpu_plain_tensor(a) || !ggml_cpu_plain_tensor(b)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (!ggml_cpu_plain_tensor(a->src[i]) || !ggml_cpu_plain_tensor(b->src[i])) {
            return false;
        }
    }
    // Duplicate expert IDs can select a tiled kernel in the separate path.
    const struct ggml_tensor * ids = a->src[2];
    for (int64_t i = 0; i < ids->ne[0]; ++i) {
        const int32_t id = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0]);
        for (int64_t j = 0; j < i; ++j) {
            if (id == *(const int32_t *) ((const char *) ids->data + j * ids->nb[0])) {
                return false;
            }
        }
    }
    const uintptr_t ap = (uintptr_t) a->data;
    const uintptr_t bp = (uintptr_t) b->data;
    const size_t bytes = ggml_nbytes(a);
    return ap < bp ? bp - ap >= bytes : ap - bp >= bytes;
}
""")
    anchor = "    struct ggml_tensor * node = cgraph->nodes[node_n];\n\n    if (node->op == GGML_OP_RMS_NORM) {"
    text = replace_once(text, anchor, """    struct ggml_tensor * node = cgraph->nodes[node_n];

    if (ggml_cpu_moe_pair && node_n + 1 < cgraph->n_nodes &&
        ggml_cpu_can_pair_moe(node, cgraph->nodes[node_n + 1])) {
        struct ggml_tensor * pair = cgraph->nodes[node_n + 1];
        if (ggml_cpu_moe_pair_trace && params->ith == 0) {
            fprintf(stderr, "MOE_PAIR %s %s K=%" PRId64 " M=%" PRId64 " selected=%" PRId64 "\\n",
                node->name, pair->name, node->src[0]->ne[0], node->src[0]->ne[1], node->src[2]->ne[0]);
        }
        ggml_compute_forward_mul_mat_id(params, node, pair);
        return 1;
    }

    if (node->op == GGML_OP_RMS_NORM) {""")
    anchor = "            ggml_cpu_disable_fusion = (env != NULL && atoi(env) == 1);"
    text = replace_once(text, anchor, anchor + """
            env = getenv("GGML_CPU_MOE_PAIR");
            ggml_cpu_moe_pair = (env == NULL || strcmp(env, "0") != 0);
            env = getenv("GGML_CPU_TILED_MM_FORCE");
            if (env != NULL && atoi(env) != 0) {
                ggml_cpu_moe_pair = false;
            }
            env = getenv("GGML_CPU_MOE_PAIR_TRACE");
            ggml_cpu_moe_pair_trace = (env != NULL && strcmp(env, "1") == 0);""")
    (output / "source" / rel).write_text(text)
    patch = "".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                        fromfile="a/" + str(rel), tofile="b/" + str(rel)))
    (output / "moe-pair.patch").write_text(patch)
    (output / "prepare.json").write_text(json.dumps({
        "source": str(source), "changed_files": [str(rel)],
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scope": "CPU plain buffers, IQ2_XXS, single-token shared activation and IDs; existing clamp retained",
        "adopted": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
