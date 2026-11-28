"""Bind the prefix planner to a single native slot and its durable catalogue."""

import contextlib
import hashlib
import time

from native import Error, Uncertain, digest, image_tokens, pack, unpack
from planner import Config, Profile, plan, resume_route
from prefix_io import boundaries, exchange, read_image
from storage import uid


class Checkpoints:
    def __init__(self, store, config):
        self.s = store
        self.n = store.native
        self.cfg = config
        self.anchors = store.anchors
        self.model = Profile(
            self.n.compat, getattr(self.n, "identity", {}).get("offload_thresholds", ())
        )
        self.memory = {}
        self.protect = ()
        self.last = None
        self.n.prefill_observer = None

    def preview(
        self,
        branch,
        history=None,
        *,
        start=None,
        prefix=None,
        config=None,
        return_position=None,
        empty_suffix=False,
    ):
        data = branch["tokens"] if prefix is None else prefix
        progress = len(data) // 4 if start is None else start
        h = progress if history is None else history
        if type(h) is not int or h < progress or h > self.n.limit:
            raise Error(
                "history", "history must be within the context and cover the starting position"
            )
        rows = self.anchors.inventory(branch, data, self.protect)
        v = plan(
            config or self.cfg,
            branch["id"],
            progress,
            h,
            rows,
            self.s.control()["budget"],
            self.model,
            self.memory.get(branch["id"]),
        )
        v["resource_accounting"] = {
            "disk_images": {
                "retained_increment_bytes": v["capacity"]["retained_increment_bytes"],
                "peak_total_bytes_observed_since_open": self.anchors.observed_image_peak,
                "scope": "image payload bytes; shared SHA counted once",
            },
            "disk_metadata": {
                "occupied_bytes": sum(
                    p.stat().st_size for p in self.s.root.glob("branches.sqlite3*")
                ),
                "retained_increment_bytes": None,
                "peak_increment_bytes": None,
            },
            "ram": {"retained_increment_bytes": None, "peak_increment_bytes": None},
            "vram": {"retained_increment_bytes": None, "peak_increment_bytes": None},
            "fixed_pool": "unsupported",
        }
        v["branch_generation"] = branch["gen"]
        v["catalog_revision"] = self.s.control()["rev"]
        v["plan_id"] = hashlib.sha256(
            f"{v['plan_id']}:{branch['gen']}:{v['catalog_revision']}".encode()
        ).hexdigest()[:20]
        v["capabilities"] = dict(
            exact_prefix=True,
            phases=["prefill", "decode"],
            full_state=True,
            resume_random_generation=False,
            empty_suffix="earlier full anchor and measured replay",
            hard_image_budget=self.cfg.mode != "off",
            storage="local_disk",
        )
        if return_position is not None:
            if type(return_position) is not int or not 0 <= return_position <= len(data) // 4:
                raise Error("return_position", "preview needs a position in the recorded branch")
            stride = v["return_interval"]
            c, ms, reason = resume_route(
                return_position,
                v["currently_available"],
                self.model,
                strict=empty_suffix,
                capture_interval=(
                    stride
                    if (config or self.cfg).mode != "off" and v["capacity"]["new_save_possible"]
                    else 0
                ),
            )
            v["return_route"] = dict(
                p=return_position,
                c=c,
                estimated_ms=ms,
                strategy=reason,
                replay_tokens=return_position - c,
                empty_suffix=empty_suffix,
            )
        return v

    def select(self, branch, data, goal, *, strict=False):
        begin = time.monotonic_ns()
        candidates = [
            r
            for r in self.anchors.visible(branch)
            if len(r["tokens"]) <= goal * 4
            and data.startswith(r["tokens"])
            and not (strict and len(r["tokens"]) == goal * 4)
        ]
        skipped = []
        available = []
        for r in candidates:
            status = self.s.valid(r, False)
            if status == "available":
                available.append(r)
            else:
                skipped.append(dict(snapshot=r["id"], reason=status))
        budget = self.s.control()["budget"]
        rows = self.anchors.inventory(branch, data, self.protect)
        can_write = sum(item["bytes"] for item in rows.values() if item["protected"]) < budget
        while True:
            positions = {len(r["tokens"]) // 4 for r in available}
            stride = self.memory.get(branch["id"], {}).get("interval", self.cfg.interval)
            c, estimated, source = resume_route(
                goal,
                positions,
                self.model,
                strict=strict,
                capture_interval=stride if self.cfg.mode != "off" and can_write else 0,
            )
            r = next((r for r in available if len(r["tokens"]) == 4 * c), None)
            chosen_ns = time.monotonic_ns()
            if r is None:
                self.n.erase()
                checked_ns = chosen_ns
                break
            try:
                status = self.s.valid(r, True)
                checked_ns = time.monotonic_ns()
                if status == "available":
                    self.n.restore(r["sha"], r["tokens"], r["bytes"])
                    break
            except Uncertain:
                raise
            except (Error, OSError) as exc:
                status = getattr(exc, "code", "read_error")
            skipped.append(dict(snapshot=r["id"], reason=status))
            available = [v for v in available if v["id"] != r["id"]]
        ready = time.monotonic_ns()
        if not skipped:
            self.model.add(
                "restore" if c else "empty", c, 0, (ready - begin) / 1e6, r["bytes"] if r else 0
            )
        return r, dict(
            from_snapshot=r["id"] if r else None,
            from_position=c,
            skipped=skipped,
            strategy=source,
            estimated_ms=estimated,
            return_interval=stride,
            route_start_ns=begin,
            route_ready_ns=ready,
            selection_ms=(chosen_ns - begin) / 1e6,
            verify_ms=(checked_ns - chosen_ns) / 1e6,
            restore_ms=(ready - checked_ns) / 1e6,
            route_ms=(ready - begin) / 1e6,
        )

    def receive(self, state, event, conn, stream):
        pos = event["position"]
        size = event["bytes"]
        n = self.n
        data = state["data"]
        previous = state.get("evaluated", b"")
        now = event["arrived_ns"]
        prior = state["last_position"]
        if event["phase"] == "prefill" and prior < pos:
            state["samples"].append((prior, pos - prior, (now - state["last_done_ns"]) / 1e6))
        event["status"] = "probe"
        path = n.images / (".writing-" + uid() + ".slot")
        admitted = False
        try:
            if size:
                if type(size) is not int or size < 4 * pos + 28:
                    raise Error("invalid_size", "native reported an invalid full state size")
                # Native exact size is known before the first byte is permitted onto disk.
                room = self.anchors.reserve(
                    state["branch"],
                    data if len(data) >= len(previous) else previous,
                    size,
                    state["domain"],
                    incoming=pos,
                    keep=self.protect,
                )
                event.update(room)
                conn.sendall(b"w")
                admitted = True
                read_image(stream, path, size)
                self.anchors.observed_image_peak = max(
                    self.anchors.observed_image_peak or 0, room["peak_bytes"]
                )
                ids = image_tokens(path, n.limit)
                if (
                    len(ids) != 4 * pos
                    or ids[: len(data)] != data[: len(ids)]
                    or not ids.startswith(previous)
                ):
                    raise Error(
                        "prefix_mismatch", "native image does not match the evaluated input"
                    )
                sha = digest(path)
                sid = self.anchors.publish(state["branch"], ids, path, sha, size)
                state["written"].append((sid, ids))
                state["evaluated"] = ids
                event.update(status="saved", snapshot_id=sid, sha=sha)
            else:
                conn.sendall(b"s")
        except (Error, OSError) as exc:
            if admitted and isinstance(exc, Error):
                raise
            event.update(status="skipped", reason=getattr(exc, "code", "storage_error"))
            if not admitted:
                conn.sendall(b"s")
        finally:
            path.unlink(missing_ok=True)
        if admitted:
            conn.sendall(b"k")
        end = time.monotonic_ns()
        event.update(done_ns=end, pause_ms=(end - now) / 1e6)
        if event["status"] == "saved":
            self.model.add("save", pos, 0, event["pause_ms"], size)
        state["events"].append(event)
        state["last_position"], state["last_done_ns"] = pos, end

    @contextlib.contextmanager
    def processing(self, branch, data, start, horizon, common, route):
        # A forecast uses only reachable future points. A reverse edit starts a new traversal at c.
        profile = self.preview(branch, horizon, start=start, prefix=data)
        self.memory[branch["id"]] = profile["hysteresis_after"]
        # An already resident image of the exact input path can be reused. Beyond model input
        # the actual generated tokens are unknown until evaluated by the backend.
        known = {len(r["tokens"]) // 4 for r in self.anchors.matching(branch, data)}
        interval = profile["chosen_interval"]
        return_stride = route.get("return_interval", interval)
        replay_grid = range((start // return_stride + 1) * return_stride, common + 1, return_stride)
        forward_grid = range((common // interval + 1) * interval, horizon + 1, interval)
        captures = [p for grid in (replay_grid, forward_grid) for p in grid if p not in known]
        denied = [] if profile["capacity"]["new_save_possible"] else captures
        if denied:
            captures = []
        stops = set(captures)
        if common > start:
            stops.add(common)
        prompt_len = len(data) // 4
        if prompt_len > start:
            stops.add(prompt_len)
        # If capacity prevented a forecast point it may still fit at its exact native size.
        # Actual admission never assumes an unmeasured size is zero.
        state = dict(
            branch=branch,
            data=data,
            domain=self.cfg.domain(horizon),
            events=[],
            samples=[],
            written=[],
            last_position=start,
            last_done_ns=route["route_ready_ns"],
        )
        with boundaries(self.n, stops, captures, lambda e, c, s: self.receive(state, e, c, s)):
            yield state
        result = state["result"]
        actual = state["actual"]
        tm = result["timings"]
        if tm["cache_n"] != start:
            raise Error(
                "prefix_mismatch",
                "the native backend replayed a different boundary",
                expected=start,
                reported=tm["cache_n"],
            )
        for sid, ids in state["written"]:
            if not actual.startswith(ids):
                self.s.db.execute(
                    "DELETE FROM anchor_refs WHERE branch=? AND anchor=?", (branch["id"], sid)
                )
                raise Error(
                    "prefix_mismatch",
                    "published prefix differs from final native receipt",
                    snapshot=sid,
                )
        reached = [e["position"] for e in state["events"]]
        if reached != sorted(q for q in stops if start < q <= len(actual) // 4):
            raise Error("missing_boundary", "backend omitted a planned processed boundary")
        for p in denied:
            if p <= len(actual) // 4:
                state["events"].append(
                    dict(
                        position=p,
                        phase="prefill" if p <= prompt_len else "decode",
                        bytes=0,
                        status="skipped",
                        reason="protected_capacity",
                        recorded_from_final_receipt=True,
                        pause_ms=0,
                    )
                )
        model = self.model
        for at, length, ms in state["samples"]:
            model.add("prefill", at, length, ms)
        known_ready = (
            route["route_ready_ns"]
            if common == start
            else next((e["done_ns"] for e in state["events"] if e["position"] == common), None)
        )
        if known_ready is None:
            raise Error("receipt", "common prefix boundary was not observed")
        result["resume"] = dict(
            route,
            common_prefix=common,
            old_replayed=common - start,
            new_input=prompt_len - common,
            boundary_extra_evaluations=0,
            prefix_ready_ns=known_ready,
            replay_ms=(known_ready - route["route_ready_ns"]) / 1e6,
            total_ms=(known_ready - route["route_start_ns"]) / 1e6,
        )
        result["checkpoint"] = dict(
            mode=self.cfg.mode,
            plan_id=profile["plan_id"],
            interval_tokens=interval,
            attempt_positions=captures,
            preflight_refusals=denied,
            return_interval=return_stride,
            status="complete",
            events=state["events"],
        )
        self.last = dict(
            plan_id=profile["plan_id"],
            interval=interval,
            events=[
                {k: v for k, v in e.items() if k not in ("arrived_ns", "done_ns")}
                for e in state["events"]
            ],
        )

    def generate(self, branch, ids, maximum, common, emit):
        data = pack(ids)
        old = branch["tokens"]
        if data.startswith(old) and len(data) > len(old):
            now = time.monotonic_ns()
            r = None
            c = len(old) // 4
            route = dict(
                from_snapshot=None,
                from_position=c,
                skipped=[],
                strategy="live",
                route_start_ns=now,
                route_ready_ns=now,
                selection_ms=0,
                verify_ms=0,
                restore_ms=0,
                route_ms=0,
            )
        else:
            r, route = self.select(branch, data, common, strict=common == len(ids))
            c = route["from_position"]
        with self.processing(branch, data, c, len(ids) + maximum - 1, common, route) as state:
            res = self.n.generate(ids, maximum, emit)
            state["actual"] = data + pack(res["sampled"][:-1])
            state["result"] = res
        return res, dict(
            previous_boundary_matches=(route["strategy"] == "live"),
            restored_prefix=r["id"] if r else None,
            skipped=route["skipped"],
        )

    def place(self, branch):
        data = branch["tokens"]
        goal = len(data) // 4
        r, route = self.select(branch, data, goal)
        c = route["from_position"]
        if goal > c:
            with self.processing(branch, data, c, goal, goal, route) as state:
                count = self.n.replay(data, c > 0)
                state["actual"] = data
                state["result"] = {"timings": {"prompt_n": count, "cache_n": goal - count}}
            detail = state["result"]
        else:
            count = 0
            detail = {
                "resume": dict(
                    route,
                    common_prefix=goal,
                    old_replayed=0,
                    new_input=0,
                    boundary_extra_evaluations=0,
                    replay_ms=0,
                    total_ms=route["route_ms"],
                )
            }
        return {
            **route,
            **detail,
            "processed": goal,
            "replayed": count,
            "exact_head": bool(r and r["id"] == branch["head"] and r["tokens"] == data),
        }

    def mandatory(self, expected, keep):
        # The idle slot uses the same bounded transport as a moving cursor. Store.save_active
        # retains responsibility for the ordinary mark/head transaction and immutable file.
        path = self.n.images / (".writing-" + uid() + ".slot")
        n = len(expected) // 4
        result = {}

        def take(e, conn, stream):
            try:
                room = self.anchors.reserve(
                    self.s.branch(), expected, e["bytes"], self.cfg.domain(n), keep=keep
                )
            except Error as exc:
                result["error"] = exc
                conn.sendall(b"s")
                return
            conn.sendall(b"w")
            read_image(stream, path, e["bytes"])
            self.anchors.observed_image_peak = max(
                self.anchors.observed_image_peak or 0, room["peak_bytes"]
            )
            if image_tokens(path, self.n.limit) != expected:
                raise Error("boundary_mismatch", "idle export differs from committed processed IDs")
            result.update(sha=digest(path), size=e["bytes"])
            conn.sendall(b"k")

        try:
            request = ([n], [n], take, False)
            v = exchange(
                self.n,
                request,
                lambda spec: self.n._http(
                    "POST",
                    "/slots/0?action=save",
                    {"filename": "managed.slot", "branch_prefix": spec},
                ),
            )
            if "error" in result:
                raise result["error"]
            if v["n_saved"] != n:
                raise Error("boundary_mismatch", "incorrect idle slot size")
            return path, result["sha"], result["size"]
        except BaseException:
            path.unlink(missing_ok=True)
            raise
