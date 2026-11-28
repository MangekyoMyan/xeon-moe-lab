"""Detect logical-head corruption, inaccurate boundaries, optimism about coverage and over-budget staging."""

import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from native import Error, Uncertain, image_tokens, pack, unpack
from manager import Manager
from storage import Store
from test_branches import SmallSlot
from planner import Config, Profile, evaluate, eviction, plan, resume_route
from prefix_io import read_image


class Linear:
    def restore(self, c):
        return 250

    def prefill(self, c, n):
        return 1000 * n / 1024

    def save(self, c):
        return 50

    def image_size(self, p):
        return 32 * 1024**2

    def basis(self):
        return {"observations": {"prefill": 4}, "samples": 12}


class Planning(unittest.TestCase):
    def test_readme_fixture_all_integer_positions(self):
        positions = list(range(4096, 8193, 256))
        size = 32 * 1024**2
        v = evaluate((4096, 8192), positions, Linear(), 500)
        self.assertEqual(v["max_replay_tokens"], 255)
        self.assertEqual(v["estimated_max_resume"], dict(p=4351, c=4096, ms=499.0234375))
        minimum = v["estimated_min_equivalent"]
        self.assertEqual((minimum["p"], minimum["c"]), (4351, 4096))
        self.assertAlmostEqual(minimum["tokens_per_second"], 1000 * 4351 / 499.0234375)
        self.assertEqual(len(positions) * size, 544 * 1024**2)
        self.assertEqual((len(positions) + 1) * size, 576 * 1024**2)

    def test_cost_can_favour_more_distant_point_and_empty_suffix(self):
        class Slow(Linear):
            def restore(self, c):
                return {0: 2500, 8: 50, 16: 900}[c]

            def prefill(self, c, n):
                return n * n if n else 0

        self.assertEqual(resume_route(19, [8, 16], Slow())[:2], (8, 171))
        self.assertEqual(resume_route(8, [8], Slow(), strict=True)[0], 0)
        self.assertEqual(evaluate((8, 8), [8], Slow(), 30)["target_met"], "no")

    def test_budget_and_time_are_independent(self):
        rows = {"pin": dict(bytes=200, protected=True, positions=[8])}
        v = plan(Config("fixed", 8, 2000, "recent", 2), "b", 8, 10, rows, 200, Linear())
        self.assertEqual(v["requested_range"], (8, 10))
        self.assertFalse(v["capacity"]["new_save_possible"])
        self.assertEqual(v["current"]["target_met"], "yes")
        cold = plan(Config("auto", 8), "b", 24, 33, rows, 200, Profile("x"))
        self.assertIsNone(cold["current"]["estimated_max_resume"])
        self.assertEqual(cold["current"]["target_met"], "unknown")
        self.assertNotIn(16, cold["after_plan_forecast"]["expected_retained"])
        self.assertEqual(cold["after_plan_forecast"]["attempt_positions"], [32])

    def test_each_retention_policy_keeps_its_own_range_and_pin(self):
        rows = {
            str(p): dict(bytes=10, protected=(p == 16), positions=[p]) for p in (16, 32, 48, 64)
        }
        a = eviction(rows, 10, 40, (0, 80), 80)
        r = eviction(rows, 10, 40, (64, 80), 80)
        self.assertIsNotNone(r)
        self.assertNotIn("16", r)
        self.assertNotIn("64", r)
        self.assertNotEqual(a, r)
        self.assertIsNone(eviction(rows, 41, 40, (64, 80), 80))

    def test_zero_partial_and_preview_hysteresis(self):
        class Zero(Linear):
            def restore(self, c):
                return 0

            def prefill(self, c, n):
                return 0

        self.assertIsNone(evaluate((0, 1), [], Zero(), 5)["estimated_min_equivalent"])
        data = dict(
            config=Config("auto", 8, target_ms=265),
            branch="b",
            start=0,
            history=32,
            rows={},
            budget=300 * 1024**2,
            costs=Linear(),
            hysteresis={"interval": 8},
        )
        a = plan(**data)
        b = plan(**data)
        self.assertEqual(a, b)
        self.assertEqual(data["hysteresis"], {"interval": 8})
        second = plan(**{**data, "hysteresis": a["hysteresis_after"]})
        self.assertEqual(second["chosen_interval"], second["hysteresis_after"]["pending"])
        cold = Profile("x")
        cold.add("prefill", 1, 7, float("nan"))
        self.assertEqual(cold.rows, [])
        json.dumps(a, allow_nan=False)

    def test_partial_capacity_and_discarded_public_image_peak(self):
        class UnknownSize(Linear):
            def image_size(self, p):
                return None

        rows = {"baseline": dict(bytes=32 * 1024**2, protected=False, positions=[], baseline=True)}
        data = plan(Config("fixed", 8), "b", 0, 8, rows, 32 * 1024**2, Linear())
        proposed = data["after_plan_forecast"]
        self.assertEqual(proposed["retained_increment_bytes"], 32 * 1024**2)
        self.assertEqual(proposed["peak_increment_bytes"], 32 * 1024**2)
        missing = plan(Config("fixed", 8), "b", 0, 8, rows, 64 * 1024**2, UnknownSize())
        self.assertEqual(missing["after_plan_forecast"]["forecast"]["target_met"], "unknown")
        c = Profile("x")
        for value in (None, True, float("inf"), 10**600):
            c.add("prefill", 1, 4, value)
        self.assertFalse(c.rows)

    def test_sampled_long_horizon_can_select_an_interval_without_claiming_global_success(self):
        cfg = Config("auto", 1024, target_ms=100000)
        a = plan(cfg, "b", 0, 12000, {}, 2**30, Linear())
        b = plan(cfg, "b", 0, 12000, {}, 2**30, Linear(), a["hysteresis_after"])
        self.assertGreater(b["chosen_interval"], 1024)
        self.assertEqual(b["selection_evidence"], "sample")
        self.assertEqual(b["after_plan_forecast"]["forecast"]["target_met"], "unknown")


class Replies:
    def __init__(self):
        self.ack = []

    def sendall(self, v):
        self.ack.append(v)


class Slot(SmallSlot):
    def forward(self, ids, prompt_n, maximum, emit):
        start = len(self.live) // 4
        values = ids + [65] * (maximum - 1)
        plan = self.boundary_request
        for pos in plan[0] if plan else []:
            if not start < pos <= len(values):
                continue
            self.live = pack(values[:pos])
            blob = b""
            if pos in plan[1]:
                p, _, _ = self.save(self.live)
                blob = p.read_bytes()
                p.unlink()
            conn = Replies()
            plan[2](
                dict(
                    position=pos,
                    bytes=len(blob),
                    phase="prefill" if pos <= prompt_n else "decode",
                    arrived_ns=time.monotonic_ns(),
                ),
                conn,
                io.BytesIO(blob),
            )
            if getattr(self, "break_after", None) == pos:
                raise Uncertain("cancelled", "fixture interrupt")
        self.live = pack(values)
        emit("A")
        return dict(
            content="A",
            sampled=[65] * (maximum - 1) + [1],
            timings={"cache_n": start, "prompt_n": len(ids) - start, "predicted_n": maximum},
        )

    def generate(self, ids, maximum, emit):
        return self.forward(ids, len(ids), maximum, emit)

    def replay(self, tokens, cache):
        start = len(self.live) // 4
        self.forward(unpack(tokens), len(tokens) // 4, 1, lambda _: None)
        return len(tokens) // 4 - start


class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = Path(self.tmp.name)
        self.n = Slot(p)
        self.s = Store(p, self.n, 2000)
        self.addCleanup(self.s.db.close)
        self.m = Manager(self.s, "fixed", 4)
        # Separate integration tests on the real native slot cover bounded named saves.
        self.s.bounded_save = None
        self.do("recover")

    def do(self, op, **kw):
        return self.m.handle(dict(op=op, ticket=self.s.stamp(), **kw))

    def test_decode_tail_absolute_grid_and_preview_read_only(self):
        ticket = self.s.stamp()
        q = dict(op="checkpoint", history=13)
        a = self.m.handle(q)
        self.assertEqual(a, self.m.handle(q))
        self.assertEqual(self.s.stamp(), ticket)
        r = self.do("turn", text="abcdefgh", max_tokens=5)
        self.assertEqual(r["checkpoint"]["plan_id"], a["plan_id"])
        points = [
            (e["position"], e["phase"]) for e in r["checkpoint"]["events"] if e["status"] == "saved"
        ]
        self.assertEqual(points, [(4, "prefill"), (8, "prefill"), (12, "decode")])
        self.assertEqual(r["processed"], 13)
        self.assertIsNone(self.s.branch()["head"])
        self.assertEqual(self.s.branch()["gen"], 1)
        self.do("mark", name="not-on-grid")
        r = self.do(
            "turn", messages=[dict(role="user", content="abcdefghAAAAxyzxyz")], max_tokens=1
        )
        self.assertIn(16, [e["position"] for e in r["checkpoint"]["events"]])
        self.assertNotIn(
            17, [e["position"] for e in r["checkpoint"]["events"] if e["status"] == "saved"]
        )

    def test_before_on_after_edits_and_zero_suffix(self):
        for p in (7, 8, 9):
            self.do("turn", messages=[dict(role="user", content="abcdefghijkl")], max_tokens=1)
            text = "abcdefghijkl"[: p - 1] + "ZZ"
            r = self.do("turn", messages=[dict(role="user", content=text)], max_tokens=1)["resume"]
            self.assertEqual(
                (r["common_prefix"], r["from_position"], r["old_replayed"]),
                (p, 4 * (p // 4), p % 4),
            )
        self.do("turn", messages=[dict(role="user", content="abcdefghijkl")], max_tokens=1)
        r = self.do("turn", messages=[dict(role="user", content="abcdefg")], max_tokens=1)["resume"]
        self.assertEqual(
            (r["common_prefix"], r["from_position"], r["old_replayed"], r["new_input"]),
            (8, 4, 4, 0),
        )

    def test_damage_ancestors_and_cancel_do_not_move_head(self):
        self.do("turn", text="abcdefghijkl", max_tokens=1)
        row = self.s.db.execute("SELECT * FROM anchors WHERE length(tokens)=32").fetchone()
        path = self.n.images / (row["sha"] + ".slot")
        raw = path.read_bytes()
        path.chmod(0o600)
        path.write_bytes(raw[:-1] + b"!")
        r = self.do("turn", messages=[dict(role="user", content="abcdefgZ")], max_tokens=1)[
            "resume"
        ]
        self.assertEqual(r["from_position"], 4)
        self.assertTrue(any(x["reason"] == "damaged" for x in r["skipped"]))
        old = dict(self.s.branch())
        self.n.break_after = 12
        with self.assertRaises(Uncertain):
            self.do(
                "turn", messages=[dict(role="user", content="different abcdefghijkl")], max_tokens=1
            )
        self.assertEqual(dict(self.s.branch()), old)
        self.assertEqual(self.s.control()["mode"], "uncertain")
        with self.assertRaises(Error):
            self.do("recover")

    def test_prewrite_cap_and_missing_and_shared_references(self):
        r = self.do("turn", text="abcdefghijkl", max_tokens=1)
        self.do("mark", name="parent", pin=True)
        self.do("fork", name="sibling", **{"from": "parent"})
        total = self.s.listing()["image_bytes"]
        self.assertEqual(total, sum(p.stat().st_size for p in self.n.images.glob("*.slot")))
        self.assertEqual(self.s.db.execute("SELECT COUNT(*) FROM anchor_refs").fetchone()[0], 6)
        with self.assertRaises(Error):
            self.do("budget", bytes=1)
        self.assertEqual(self.s.control()["budget"], 2000)
        self.do("budget", bytes=total)
        r = self.do("turn", text="x" * 16, max_tokens=1)
        self.assertTrue(any(e["status"] == "skipped" for e in r["checkpoint"]["events"]))
        self.assertLessEqual(sum(p.stat().st_size for p in self.n.images.glob("*.slot")), total)
        self.do("release", kind="branch", name="sibling")
        self.assertEqual(self.s.db.execute("SELECT COUNT(*) FROM anchor_refs").fetchone()[0], 3)

    def test_failed_stage_keeps_the_preceding_turn_committable(self):
        # read_image's disk failure contract drains the full admitted export.
        def reject(stream, path, size):
            self.assertEqual(len(stream.read()), size)
            raise OSError("synthetic full disk")

        with patch("checkpoints.read_image", side_effect=reject):
            result = self.do("turn", text="abcdefghi", max_tokens=1)
        self.assertTrue(result["turn_committed"])
        self.assertEqual(self.s.branch()["gen"], 1)
        self.assertEqual(self.s.listing()["image_bytes"], 0)
        self.assertTrue(
            all(
                e["reason"] == "storage_error" for e in result["checkpoint"]["events"] if e["bytes"]
            )
        )
        self.assertFalse(list(self.n.images.glob(".writing-*")))

    def test_missing_file_and_foreign_compatibility_fall_back(self):
        self.do("turn", text="abcdefghijkl", max_tokens=1)
        row = self.s.db.execute("SELECT * FROM anchors WHERE length(tokens)=32").fetchone()
        (self.n.images / (row["sha"] + ".slot")).unlink()
        self.s.db.execute("UPDATE anchors SET compat='foreign' WHERE length(tokens)=16")
        r = self.do("turn", messages=[dict(role="user", content="abcdefgZ")], max_tokens=1)[
            "resume"
        ]
        self.assertEqual(r["from_position"], 0)
        self.assertTrue(any(v["reason"] == "missing" for v in r["skipped"]))

    def test_shared_sha_survives_until_last_reference(self):
        self.do("turn", text="abcdefg", max_tokens=1)
        self.do("mark", name="root")
        self.do("fork", name="fork", **{"from": "root"})
        point = self.s.db.execute("SELECT * FROM anchors WHERE length(tokens)=16").fetchone()
        p = self.n.images / (point["sha"] + ".slot")
        self.assertTrue(p.exists())
        self.do("release", kind="branch", name="fork")
        self.assertTrue(p.exists())
        with self.s.tx():
            self.s.anchors.release(self.s.branch()["id"])
            self.s.collect()
        self.s.clean_files()
        self.assertFalse(p.exists())


if __name__ == "__main__":
    unittest.main()
