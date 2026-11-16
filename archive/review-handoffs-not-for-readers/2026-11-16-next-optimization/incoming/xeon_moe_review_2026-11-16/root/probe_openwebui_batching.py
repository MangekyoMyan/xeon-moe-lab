#!/usr/bin/env python3
"""Replay the four pinned Open WebUI v0.11.1 delta queue helpers with fake time.

No Open WebUI server, database, HTTP request, browser or inference is started.
Input deltas are synthetic text fragments, not model tokens.
"""
import argparse
import ast
import asyncio
import copy
import hashlib
import json
from pathlib import Path

COMMIT = "d3e8bf3405e848cfba377814d0aa7ba7290e414d"
BLOB_SHA = "cc5f57e0c320c2b75e106af409e67bea791ee898"
NAMES = {"get_response_delta_key", "get_response_data_with_full_output_index",
         "flush_pending_delta_data", "queue_pending_delta_data"}


def load_helpers(source):
    raw = source.read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if actual != BLOB_SHA:
        raise ValueError("Expected exact middleware.py from the pinned v0.11.1 commit")
    tree = ast.parse(raw, filename=str(source))
    selected = [node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in NAMES]
    if len(selected) != len(NAMES) or {node.name for node in selected} != NAMES:
        raise ValueError("Ambiguous source helper selection")
    wrapper = ast.parse("""
def build_batcher(delta_chunk_size, event_emitter, save_current_response_stream):
    delta_count = 0
    last_delta_data = None
    last_delta_type = None
    last_delta_key = None
    prior_output = []
""").body[0]
    wrapper.body += sorted(selected, key=lambda node: node.lineno)
    wrapper.body += ast.parse("return queue_pending_delta_data, flush_pending_delta_data").body
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["build_batcher"], {node.name: [node.lineno, node.end_lineno] for node in selected}


async def run_case(factory, environment_size, metadata_size, inputs, finish_ms):
    emitted = []
    now = 0
    saves = []

    async def emitter(event):
        emitted.append({"at_ms": now, "event": copy.deepcopy(event)})

    async def save():
        saves.append(now)

    effective = max(environment_size, int(metadata_size or 1))
    queue, flush = factory(effective, emitter, save)
    for at_ms, text, kind in inputs:
        now = at_ms
        payload = {"type": kind, "item_id": "synthetic-item", "output_index": 0,
                   "content_index": 0, "delta": text}
        await queue(payload, kind)
    before_finish = copy.deepcopy(emitted)
    now = finish_ms
    await flush()
    return {"environment_size": environment_size, "metadata_size": metadata_size,
            "effective_size": effective, "input_fragments": len(inputs), "emissions": emitted,
            "emissions_before_finish": before_finish, "save_calls": saves,
            "joined_text": "".join(row["event"]["data"]["delta"] for row in emitted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new file")
    factory, lines = load_helpers(args.source)
    kind = "response.output_text.delta"
    inputs = [((index + 1) * 130, char, kind) for index, char in enumerate("ABCDEF")]
    cases = {}
    for env_size, metadata_size in [(1, None), (1, 2), (2, 1)]:
        cases[f"env{env_size}_metadata{metadata_size}"] = asyncio.run(
            run_case(factory, env_size, metadata_size, inputs, 781))
    cases["type_change_flush"] = asyncio.run(run_case(factory, 1, 2,
        [(130, "A", kind), (260, "B", "response.reasoning_text.delta")], 261))
    cases["one_fragment_waits_until_finish"] = asyncio.run(run_case(factory, 1, 2,
        [(130, "A", kind)], 5000))
    checks = {
        "size1_emits_each_fragment": [x["at_ms"] for x in cases["env1_metadataNone"]["emissions"]] == [130, 260, 390, 520, 650, 780],
        "size2_emits_pairs": [x["at_ms"] for x in cases["env1_metadata2"]["emissions"]] == [260, 520, 780],
        "environment_can_prevent_metadata_size1": [x["at_ms"] for x in cases["env2_metadata1"]["emissions"]] == [260, 520, 780],
        "all_regular_text_equal": all(cases[name]["joined_text"] == "ABCDEF" for name in ["env1_metadataNone", "env1_metadata2", "env2_metadata1"]),
        "type_change_flushes_pending": [x["at_ms"] for x in cases["type_change_flush"]["emissions"]] == [260, 261],
        "final_flush_preserves_odd_fragment": cases["one_fragment_waits_until_finish"]["joined_text"] == "A",
        "no_emission_before_end_without_next_delta": not cases["one_fragment_waits_until_finish"]["emissions_before_finish"],
    }
    output = {"upstream_repository": "https://github.com/open-webui/open-webui", "commit": COMMIT,
              "version": "v0.11.1", "middleware_blob_sha": BLOB_SHA,
              "middleware_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "helper_source_lines": lines, "cases": cases, "checks": checks,
              "all_passed": all(checks.values()),
              "scope": "Exact AST helper definitions with fake emitter/save and virtual timestamps; no server, network, rendering or model test.",
              "limits": ["This does not identify the user's effective chunk setting.",
                         "Fragments are not model tokens, SSE events, or browser frames.",
                         "Only the helper batching layer is exercised; other buffering remains unmeasured."]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"all_passed": output["all_passed"], "checks": len(checks), "output": str(args.output)}))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
