"""Serialize native work and publish logical heads only after confirmed completion."""

import json
import time
from native import Error, Uncertain, encode, pack, unpack
from storage import name_ok, uid


def check_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise Error("messages", "provide a nonempty list of text messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in (
            "system",
            "user",
            "assistant",
            "tool",
        ):
            raise Error("messages", "unsupported message role")
        if not isinstance(message.get("content"), str) and not (
            message.get("role") == "assistant"
            and message.get("tool_calls")
            and message.get("content") is None
        ):
            raise Error("messages", "this adapter accepts text and recorded tool results only")
        if set(message) - {
            "role",
            "content",
            "name",
            "tool_call_id",
            "tool_calls",
            "reasoning_content",
        }:
            raise Error("messages", "unsupported fields in text conversation")
    return messages


class Manager:
    def __init__(self, store):
        self.store = store
        self.native = store.native
        self.changed = False

    def place(self, branch):
        """Install a compatible ancestor, then evaluate precisely the missing tokens."""
        store = self.store
        target_tokens = branch["tokens"]
        chosen, skipped = store.choose(branch["head"] or branch["base"], target_tokens)
        self.changed = True
        if chosen and chosen["sha"]:
            self.native.restore(chosen["sha"], chosen["tokens"], chosen["bytes"])
            store.db.execute("UPDATE blobs SET used=? WHERE sha=?", (time.time(), chosen["sha"]))
            at = len(chosen["tokens"]) // 4
        else:
            self.native.erase()
            at = 0
        count = 0
        if len(target_tokens) // 4 > at:
            count = self.native.replay(target_tokens, at > 0)
        return {
            "from_snapshot": chosen["id"] if chosen else None,
            "from_position": at,
            "processed": len(target_tokens) // 4,
            "replayed": count,
            "skipped": skipped,
            "exact_head": bool(
                chosen and chosen["id"] == branch["head"] and chosen["tokens"] == target_tokens
            ),
        }

    def prepare_turn(self, branch, ids):
        if ids.startswith(branch["tokens"]):
            return {"previous_boundary_matches": True, "restored_prefix": None, "skipped": []}
        chosen, skipped = self.store.choose(branch["base"], ids)
        self.changed = True
        if chosen and chosen["sha"]:
            self.native.restore(chosen["sha"], chosen["tokens"], chosen["bytes"])
            self.store.db.execute(
                "UPDATE blobs SET used=? WHERE sha=?", (time.time(), chosen["sha"])
            )
        else:
            self.native.erase()
        return {
            "previous_boundary_matches": False,
            "restored_prefix": chosen["id"] if chosen else None,
            "skipped": skipped,
        }

    def turn(self, request, branch, emit):
        if "messages" in request:
            messages = check_messages(request["messages"])
        else:
            if not isinstance(request.get("text"), str):
                raise Error("messages", "a text turn or complete message list is required")
            messages = check_messages(
                json.loads(branch["messages"]) + [{"role": "user", "content": request["text"]}]
            )
        maximum = request.get("max_tokens", 64)
        if type(maximum) is not int or not 1 <= maximum < self.native.limit:
            raise Error("max_tokens", "invalid generation length")
        ids = self.native.render(messages)
        if (
            not ids
            or len(ids) + maximum > self.native.limit
            or any(type(i) != int or not 0 <= i < self.native.vocab for i in ids)
        ):
            raise Error(
                "context",
                "normalized input plus requested output exceeds the slot or has invalid token IDs",
            )
        data = pack(ids)
        old = unpack(branch["tokens"])
        common = next(
            (i for i, (x, y) in enumerate(zip(old, ids)) if x != y), min(len(old), len(ids))
        )
        positioning = self.prepare_turn(branch, data)
        self.changed = True
        result = self.native.generate(ids, maximum, emit)
        samples = result.pop("sampled")
        tokens = data + pack(samples[:-1])
        history = messages + [{"role": "assistant", "content": result["content"]}]
        response = {
            **result,
            **positioning,
            "common_prefix": common,
            "processed": len(tokens) // 4,
            "pending_sampled_ids": samples[-1:],
            "normalized_input_count": len(ids),
            "previous_pending_matches_next_input": bool(
                branch["pending"]
                and len(old) < len(ids)
                and common == len(old)
                and pack([ids[len(old)]]) == branch["pending"]
            ),
        }
        with self.store.tx():
            self.store.db.execute(
                "UPDATE branches SET tokens=?,pending=?,messages=?,head=NULL,gen=gen+1,used=?,replay=? WHERE id=?",
                (
                    tokens,
                    pack(samples[-1:]),
                    encode(history),
                    time.time(),
                    (
                        result["timings"]["prompt_n"]
                        if not positioning["previous_boundary_matches"]
                        else 0
                    ),
                    branch["id"],
                ),
            )
            self.finish()
        return response

    def finish(self):
        self.store.db.execute("UPDATE control SET rev=rev+1")
        self.store.state("ready")

    def operation(self, op, request, branch, emit):
        store = self.store
        if op == "turn":
            return self.turn(request, branch, emit)
        if op == "checkout":
            target = store.branch(request["name"])
            if target["id"] == branch["id"]:
                with store.tx():
                    self.finish()
                return {"already_active": True, "replayed": 0}
            # Reserve the target's referenced file while committing A. Verify B after A is durable.
            target_point = target["head"] or target["base"]
            outgoing = store.save_active([store.snap(target_point)["sha"]] if target_point else [])
            detail = self.place(target)
            with store.tx():
                store.db.execute(
                    "UPDATE branches SET head=?,used=?,replay=? WHERE id=?",
                    (
                        target["head"] if detail["exact_head"] else None,
                        time.time(),
                        detail["replayed"],
                        target["id"],
                    ),
                )
                store.db.execute("UPDATE control SET active=?", (target["id"],))
                self.finish()
            return {"outgoing_snapshot": outgoing["id"], **detail}
        if op == "recover":
            detail = self.place(branch)
            with store.tx():
                store.db.execute(
                    "UPDATE branches SET head=?,used=?,replay=? WHERE id=?",
                    (
                        branch["head"] if detail["exact_head"] else None,
                        time.time(),
                        detail["replayed"],
                        branch["id"],
                    ),
                )
                self.finish()
            store.clean_files()
            return detail
        if op == "mark":
            label = name_ok(request["name"])
            old = store.db.execute("SELECT sid FROM marks WHERE name=?", (label,)).fetchone()
            if old and old[0] != branch["head"]:
                raise Error("mark_exists", "a different point already owns this mark")
            point = store.save_active()
            if old and old[0] != point["id"]:
                raise Error("mark_exists", "the named snapshot is immutable")
            with store.tx():
                store.db.execute(
                    "INSERT OR IGNORE INTO marks VALUES(?,?,?)",
                    (label, point["id"], bool(request.get("pin"))),
                )
                if request.get("pin"):
                    store.db.execute("UPDATE marks SET pin=1 WHERE name=?", (label,))
                self.finish()
            return {"mark": label, "snapshot": store.point_info(point)}
        if op == "fork":
            label = name_ok(request["name"])
            if store.db.execute(
                "SELECT 1 FROM branches WHERE name=? OR id=?", (label, label)
            ).fetchone():
                raise Error("branch_exists", "branch name is already in use")
            point = store.resolve(request["from"]) if request.get("from") else store.save_active()
            if request.get("pin") and store.valid(point) not in ("available", "empty"):
                raise Error("missing_image", "the requested pin has no usable image")
            branch_id = uid()
            with store.tx():
                store.db.execute(
                    "INSERT INTO branches VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        branch_id,
                        label,
                        point["id"],
                        point["id"],
                        point["id"],
                        point["tokens"],
                        point["pending"],
                        point["messages"],
                        0,
                        bool(request.get("pin")),
                        time.time(),
                        0,
                    ),
                )
                self.finish()
            return {"branch": store.describe(store.branch(branch_id))}
        if op in ("pin", "unpin", "release"):
            kind = request["kind"]
            if kind not in ("mark", "branch"):
                raise Error("kind", "choose branch or mark")
            if kind == "branch":
                item = store.branch(request["name"])
                key = item["id"]
                table, col = "branches", "id"
                if op == "release" and key == branch["id"]:
                    raise Error("active_branch", "switch away before releasing the active branch")
                if op == "pin" and key == branch["id"] and not item["head"]:
                    store.save_active()
                    item = store.branch(key)
                snapshot_id = item["head"]
            else:
                item = store.db.execute(
                    "SELECT * FROM marks WHERE name=?", (request["name"],)
                ).fetchone()
                if not item:
                    raise Error("mark_missing", "mark does not exist")
                table, col, key, snapshot_id = "marks", "name", item["name"], item["sid"]
            if op == "pin" and (
                not snapshot_id
                or store.valid(store.snap(snapshot_id)) not in ("available", "empty")
            ):
                raise Error("missing_image", "materialize the branch head before pinning")
            if op == "release" and item["pin"]:
                raise Error("pinned", "unpin this reference before release")
            with store.tx():
                if op == "release":
                    store.db.execute(f"DELETE FROM {table} WHERE {col}=?", (key,))
                    store.collect()
                else:
                    store.db.execute(f"UPDATE {table} SET pin=? WHERE {col}=?", (op == "pin", key))
                self.finish()
            store.clean_files()
            return {"operation": op, "kind": kind, "name": request["name"]}
        if op == "budget":
            with store.tx():
                victims = store.trim(request["bytes"])
                store.db.execute("UPDATE control SET budget=?", (request["bytes"],))
                self.finish()
            store.clean_files()
            return {"budget_bytes": request["bytes"], "evicted_images": victims}
        raise Error("operation", "unknown operation")

    def handle(self, request, emit=lambda _: None):
        op = request["op"]
        store = self.store
        if op == "ticket":
            return {"ticket": store.stamp()}
        if op in ("list", "status"):
            return store.listing()
        if op == "inspect":
            if request.get("kind") == "snapshot":
                return store.point_info(store.resolve(request["name"]), True)
            branch = store.branch(request.get("name"))
            response = store.describe(branch)
            if branch["head"]:
                response["snapshot"] = store.point_info(store.snap(branch["head"]), True)
            else:
                response["snapshot"] = None
            return response
        if request.get("ticket") != store.stamp():
            raise Error(
                "stale", "session, active branch or generation changed", current=store.stamp()
            )
        control = store.control()
        if control["mode"] == "uncertain":
            raise Error(
                "native_outcome_unknown",
                "restart the private inference backend and reopen its binding before recovery",
            )
        if control["mode"] != "ready" and op != "recover":
            raise Error("invalid_slot", "recover the active branch before using this slot")
        branch = store.branch()
        self.changed = False
        store.state("mutating", op)
        try:
            answer = self.operation(op, request, branch, emit)
            return {**answer, "ticket": store.stamp()}
        except Uncertain as exc:
            store.state("uncertain", str(exc))
            raise
        except Exception as exc:
            code = exc.code if isinstance(exc, Error) else "operation_failed"
            more = dict(exc.details) if isinstance(exc, Error) else {}
            if store.control()["mode"] == "ready":
                raise Error("committed", str(exc), ticket=store.stamp()) from exc
            if op == "recover":
                store.state("invalid", str(exc))
            elif self.changed:
                try:
                    source = store.branch(branch["id"])
                    back = self.place(source)
                    with store.tx():
                        store.db.execute(
                            "UPDATE branches SET head=?,replay=? WHERE id=?",
                            (
                                source["head"] if back["exact_head"] else None,
                                back["replayed"],
                                branch["id"],
                            ),
                        )
                        self.finish()
                    more["rollback"] = back
                except Exception as secondary:
                    mode = "uncertain" if isinstance(secondary, Uncertain) else "invalid"
                    store.state(mode, str(secondary))
                    raise Error(
                        "rollback_failed",
                        "logical active head retained; physical slot needs recovery",
                        cause=code,
                        rollback_error=str(secondary),
                        mode=mode,
                        ticket=store.stamp(),
                    ) from secondary
            else:
                store.state("ready")
            raise Error(code, str(exc), **more) from exc
