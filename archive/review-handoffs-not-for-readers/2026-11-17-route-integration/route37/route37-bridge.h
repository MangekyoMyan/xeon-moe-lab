#pragma once
struct llama_context;
struct llama_ubatch;
// Called at the actual process_ubatch execution boundary, not graph construction.
int xeon_route37_bridge_begin(llama_context *ctx, const llama_ubatch &ubatch,
                             bool rebuilt, bool pipeline_parallel);
