"""Materialize the isolated scheduler source and a patch against the deployed base."""
import difflib
import hashlib
from pathlib import Path

here = Path(__file__).resolve().parent
lab = here.parents[2]
original = lab / "gpu-stream-20261103/ggml-backend.cpp"
raw = original.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "002749e65922a0ce87937df09064225940bca793e76032595ba274c87b982ddd"
s = raw.decode()
before = s

def change(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)

change("struct ggml_backend_sched {", "struct prefix169_state;\n\nstruct ggml_backend_sched {\n    prefix169_state * prefix169;")
change("static enum ggml_status ggml_backend_sched_compute_splits(",
       '#include "shared_prefix.inc"\n\nstatic enum ggml_status ggml_backend_sched_compute_splits(')
change("struct ggml_backend_sched_split * splits = sched->splits;\n\n    int prev_backend_id = -1;",
       "struct ggml_backend_sched_split * splits = sched->splits;\n"
       "    prefix169_state * prefix = prefix169_prepare(sched);\n"
       "    prefix169_pending pending;\n\n    int prev_backend_id = -1;")
change("enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);",
       "enum ggml_status ec = prefix ? prefix169_dispatch(*prefix, pending, split_id, split_backend, &split->graph)\n"
       "                                         : ggml_backend_graph_compute_async(split_backend, &split->graph);")
change("struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));",
       "struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));\n"
       "    sched->prefix169 = prefix169_new();")
change("void ggml_backend_sched_free(ggml_backend_sched_t sched) {\n    if (sched == NULL) {\n        return;\n    }",
       "void ggml_backend_sched_free(ggml_backend_sched_t sched) {\n    if (sched == NULL) {\n        return;\n    }\n"
       "    prefix169_free(sched->prefix169);")
(here / "ggml-backend.cpp").write_text(s)
diff = difflib.unified_diff(before.splitlines(True), s.splitlines(True), "a/ggml-backend.cpp", "b/ggml-backend.cpp")
extra = difflib.unified_diff([], (here/"shared_prefix.inc").read_text().splitlines(True), "/dev/null", "b/shared_prefix.inc")
(here / "shared-prefix.patch").write_text("".join(diff)+"".join(extra))
