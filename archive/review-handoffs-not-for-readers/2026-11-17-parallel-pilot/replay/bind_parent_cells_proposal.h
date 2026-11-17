#pragma once
#include "bounded_moe_capture_proposal.h"

// Binding example only. Exact tuples copied from ../parent/CAPTURE_CELLS.json.
// Only for case text_cache_budget_ja, prefix 272, original bank-free graph.
// Here call_key=decode_call is unique because this proposal replays ONE case.
// Model is the pinned llama_model with layers[].ffn_gate_exps/ffn_down_exps.
// Parameters/callback must already be installed before context creation.
// Call after model initialization, before warmup/prefill/teacher-forced decode.
// Bind the manifest separately: provider hashes, types, shape, GGUF, clamp,
// template+tokens, candidate SHA, original slot order and completed route ledger.
template <typename Model>
bool bind_parent_selected_cells(bounded_moe_capture::Capture & cap, const Model & model) {
    struct Entry {
        uint64_t call;
        int layer;
        std::array<int32_t, 6> ids;
    };
    constexpr std::array<Entry, 8> plan{{
        {0,  7, {152, 245, 161, 157,  38, 207}},
        {5,  7, { 38,  54, 313, 369,  85, 331}},
        {3,  7, { 10, 154,  19,   7, 313, 165}},
        {2,  7, { 19, 154,   7, 331, 157,  38}},
        {8, 35, { 78, 201, 225, 100, 306,  26}},
        {0, 13, {123, 377, 174, 213,   3, 113}},
        {2, 28, {175,  13, 332, 138, 296, 316}},
        {2, 20, {129, 262, 103, 273, 346,  19}},
    }};
    if (model.layers.size() != 40) return false;
    for (size_t i = 0; i < plan.size(); ++i) {
        auto & c = cap.cells[i];
        c.key = plan[i].call;
        c.layer = plan[i].layer;
        c.gate_weight = model.layers[c.layer].ffn_gate_exps;
        c.down_weight = model.layers[c.layer].ffn_down_exps;
        c.expected_ids = plan[i].ids;
    }
    return cap.prepare(plan.size());
}

// Runtime placement in a bounded caller, shown as pseudocode:
//
// auto cap = std::make_unique<bounded_moe_capture::Capture>();
// params.warmup = false;
// params.cb_eval = bounded_moe_capture::Capture::callback;
// params.cb_eval_user_data = cap.get();
// init_model_and_context(params);
// need(bind_parent_selected_cells(*cap, *model), "capture binding");
// replay_original_prefix_272();     // cap remains unarmed
// for (uint64_t i = 0; i < 9; ++i) {
//     set_original_token_and_position(272 + i);
//     cap->begin_call(i);
//     drive_existing_label_decode_normal_output_sync_finish();
//     need(cap->end_call(), "bounded capture failed");
// }
// validate_capture_finite_and_attach_same_attempt_labels();
// export_outside_decode();
//
// Save the final-prefix logits + first 4 T1 logits and compare those 5 rows
// against the EXISTING callback-free baseline. Also compare all 8 ordered-ID
// tuples against the frozen route records. Reuse this evidence first; a new
// 9-T1 callback-free baseline or repeated F/D suite is NOT an initial gate.
// These checks do NOT prove that call 5/8 intermediate values were bit-identical
// to the old callback-free run. Treat the actually captured F32 x as a fixed
// fixture. Original CPU operator (O) and private CPU-all6 (C) replay must BOTH
// be callback-free and use that identical x and joint6. Add a 9-T1 baseline only
// if observation differences are suspected, existing checks disagree, or a
// complete 9-call baseline bridge is specifically needed.
//
// This arm is never a speed measurement. The comparison/performance caller
// must be created with params.cb_eval = nullptr and cb_eval_user_data = nullptr.
