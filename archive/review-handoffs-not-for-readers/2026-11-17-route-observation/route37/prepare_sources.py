"""Copy frozen sources and add isolated route hooks; production sources untouched."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent;LAB=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cpu=LAB/'exploration-20261112/mx-prepack/release-candidate-a/sources/ggml-cpu.c'
ctx=LAB/'deepseek41-bringup-20261101/source/src/llama-context.cpp'
assert sha(cpu)=='e8fdf3d424463f9852fa774d9de6057dce35855a872936bda87d12d3d756e3a8'
old=cpu.read_text();new=old.replace('#include "mx-rhs.h"','#include "mx-rhs.h"\n#include "route37.h"',1)
anchor='''    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {'''
assert new.count(anchor)==1
new=new.replace(anchor,'''    // Diagnostic only: registered gates, one worker, before the existing barrier.
    if (ith == 0) xeon_route37_record_v1(src0, ids);

'''+anchor,1)
(R/'ggml-cpu.c').write_text(new)
(R/'cpu.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(cpu),tofile='diagnostic/ggml-cpu.c')))
old=ctx.read_text();new='#include "route37.h"\n#include "route37-bridge.h"\n'+old
start=new.index('llm_graph_result * llama_context::process_ubatch(');end=new.index('\nint llama_context::encode(',start)
piece=new[start:end]
anchor='    auto * res = get_gf_res_prev();';assert piece.count(anchor)==1
piece=piece.replace(anchor,'    bool route37_rebuilt = false;\n'+anchor)
anchor='    } else {\n        gf_res_prev_active = nullptr;';assert piece.count(anchor)==1
piece=piece.replace(anchor,'    } else {\n        route37_rebuilt = true;\n        gf_res_prev_active = nullptr;')
anchor='    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);';assert piece.count(anchor)==1
piece=piece.replace(anchor,'''    const int route37_ticket = xeon_route37_bridge_begin(this, ubatch, route37_rebuilt, cparams.pipeline_parallel);
'''+anchor+'''
    // CPU backend execution is synchronous within scheduler submission. This
    // closes CPU route production only; it does not claim GPU tail completion.
    // The diagnostic caller must later finalize the call after its normal sync.
    if (route37_ticket) xeon_route37_end_eval_v1(this, status);''')
new=new[:start]+piece+new[end:]
(R/'llama-context.cpp').write_text(new)
(R/'caller.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(ctx),tofile='diagnostic/llama-context.cpp')))
(R/'SOURCE.json').write_text(json.dumps({'base_cpu':{'path':str(cpu),'sha256':sha(cpu)},'base_context':{'path':str(ctx),'sha256':sha(ctx)},'generated':{n:sha(R/n) for n in ['ggml-cpu.c','llama-context.cpp','cpu.patch','caller.patch']},'production_modified':False},indent=2)+'\n')
print('Prepared isolated CPU and per-ubatch caller source diffs')
