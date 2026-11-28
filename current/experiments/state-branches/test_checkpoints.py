"""Placement and failure boundaries, without a model or GPU."""

import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from checkpoint_policy import CheckpointPolicy
from manager import Manager
from native import Error, Native, Uncertain, pack
from storage import Store
from test_branches import SmallSlot


class PolicyTests(unittest.TestCase):
    def test_only_validated_native_replay_feeds_prompt_cost(self):
        p = CheckpointPolicy("auto", 1)

        class Engine:
            prefill_observer = p.observe_prefill
            result = {"tokens_cached": 4, "timings": {"cache_n": 1, "prompt_n": 3, "prompt_ms": 5}}

            def call(self, *args):
                return self.result

        e = Engine()
        self.assertEqual(Native.replay(e, pack([1, 2, 3, 4]), True), 3)
        self.assertEqual(p.observed["prefill"], 1)
        e.result["timings"].pop("prompt_ms")
        self.assertEqual(Native.replay(e, pack([1, 2, 3, 4]), True), 3)
        self.assertEqual(p.observed["prefill"], 1)
        e.result["tokens_cached"] = 2
        with self.assertRaises(Error):
            Native.replay(e, pack([1, 2, 3, 4]), True)

    def test_fixed_boundary_and_one_return_cost_model(self):
        fixed = CheckpointPolicy("fixed", 8)
        for d, status in [(7, "skipped"), (8, "attempt"), (38, "attempt"), (0, "skipped")]:
            self.assertEqual(fixed.decide(None, 0, d)["status"], status)
        p = CheckpointPolicy("auto", 8, runtime="one", thresholds=(128, 64))
        for n, ms in [(4, 8), (8, 8), (16, 8)]:
            p.observe_prefill(
                {"prompt_ms": ms, "prompt_n": n, "cache_n": 10000, "predicted_ms": 100000}
            )
        for _ in range(3):
            p.observe_save(16, 900, 4)
            p.observe_restore("place", 3, 300, 8)
            p.observe_restore("place", 16, 900, 10)
        # Position ranges overlap only where explicitly allowed.
        r = p.decide("p", 3, 16)
        self.assertEqual(r["basis"], "cold_fallback")  # only 8 and 16 are near distance 13
        p.observe_prefill({"prompt_n": 13, "prompt_ms": 26})
        r = p.decide("p", 3, 16)
        self.assertEqual(r["basis"], "cost_model")
        self.assertAlmostEqual(r["estimate_ms"]["replay"], 13 * (8 + 8 + 26) / (8 + 16 + 13))
        self.assertEqual(
            (r["estimate_ms"]["restore_anchor"], r["estimate_ms"]["restore_candidate"]), (8, 10)
        )
        self.assertTrue(r["model_accept"])
        self.assertEqual(p.decide(None, 0, 600)["basis"], "cold_fallback")
        p.observe_restore("prepare_turn", 16, 900, 9)
        self.assertEqual(p.decide("p", 3, 16)["basis"], "cold_fallback")
        self.assertEqual(
            CheckpointPolicy("auto", 8, runtime="two").decide("p", 3, 16)["samples"]["prefill"], 0
        )

    def test_equal_cost_missing_samples_and_sliding_windows(self):
        p = CheckpointPolicy("auto", 1024, thresholds=(128, 128))
        for bad in (0, -1, None, True, float("inf"), float("nan")):
            p.observe_prefill({"prompt_n": 32, "prompt_ms": bad})
        p.observe_prefill({"prompt_n": 0, "prompt_ms": 3})
        p.observe_prefill({"prompt_n": 32})
        self.assertFalse(p.observed)
        for _ in range(3):
            p.observe_prefill({"prompt_n": 32, "prompt_ms": 15})
            p.observe_save(232, 888, 10)
            p.observe_restore("place", 200, 777, 3)
        r = p.decide("p", 200, 232)
        self.assertEqual(r["estimate_ms"]["gain"], 15)
        self.assertFalse(r["model_accept"])
        p.observe_prefill({"prompt_n": 32, "prompt_ms": 150})
        self.assertTrue(p.decide("p", 200, 232)["model_accept"])
        for _ in range(8):
            p.observe_prefill({"prompt_n": 32, "prompt_ms": 15})
        self.assertFalse(p.decide("p", 200, 232)["model_accept"])
        for _ in range(4):
            p.observe_prefill({"prompt_n": 128, "prompt_ms": 100})
        self.assertEqual(p.decide("p", 200, 327)["samples"]["prefill"], 0)
        self.assertEqual(p.decide("p", 200, 328)["samples"]["prefill"], 4)
        q = p.decide(None, 0, 232)
        self.assertTrue(q["empty_lower_bound"])
        self.assertEqual(q["estimate_ms"]["restore_anchor"], 0.0)
        p.budget_failed("A", 130, (20, 10, ()))
        self.assertTrue(p.blocked("A", 142, (20, 10, ())))
        for args in [("A", 129, (20, 10, ())), ("B", 142, (20, 10, ())), ("A", 142, (100, 10, ()))]:
            self.assertFalse(p.blocked(*args))


class TextSlot(SmallSlot):
    def render(self, messages):
        text = "".join(("|" if m["role"] == "user" else "") + m["content"] for m in messages)
        return [0, *text.encode()]


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.n = TextSlot(self.root)
        self.s = Store(self.root, self.n, 10_000)
        self.addCleanup(self.s.db.close)
        self.m = Manager(self.s, "fixed", 1)
        self.do("recover")

    def do(self, op, **kw):
        return self.m.handle({"op": op, "ticket": self.s.stamp(), **kw})

    def test_off_and_optional_publication_tickets_and_edit_ancestry(self):
        self.m = Manager(self.s)
        first = self.s.stamp()
        with patch.object(
            self.n, "save", side_effect=AssertionError("off wrote an image")
        ), patch.object(
            self.s, "checkpoint_anchor", side_effect=AssertionError("off scanned ancestors")
        ):
            x = self.do("turn", text="introduction")
        self.assertEqual(x["checkpoint"]["status"], "disabled")
        self.assertEqual(x["ticket"]["revision"], first["revision"] + 1)
        self.m = Manager(self.s, "fixed", 1)
        a = self.do("turn", text="P")
        p = a["checkpoint"]["snapshot_id"]
        head = self.s.snap(p)
        self.assertEqual(head["tokens"], self.n.live)
        self.assertEqual(head["pending"], pack([1]))
        self.assertEqual(a["ticket"]["generation"], first["generation"] + 2)
        self.assertEqual(a["ticket"]["revision"], x["ticket"]["revision"] + 2)
        with self.assertRaisesRegex(Error, "generation changed"):
            self.m.handle({"op": "mark", "name": "late", "ticket": x["ticket"]})
        self.assertEqual(a["ticket"], self.do("status")["ticket"])
        self.assertEqual(self.s.db.execute("SELECT count(*) FROM marks").fetchone()[0], 0)
        self.assertEqual(self.do("mark", name="pin-me", pin=True)["snapshot"]["id"], p)
        b = self.do("turn", text="number 731")
        q = b["checkpoint"]["snapshot_id"]
        later = self.do("turn", text="after B")["checkpoint"]["snapshot_id"]
        messages = json.loads(head["messages"]) + [{"role": "user", "content": "number 284"}]
        edit = self.do("turn", messages=messages)
        self.assertEqual(edit["restored_prefix"], p)
        self.assertEqual(edit["checkpoint"]["anchor_id"], p)
        self.assertEqual(self.s.snap(edit["checkpoint"]["snapshot_id"])["parent"], later)
        self.assertNotEqual(q, edit["checkpoint"]["anchor_id"])
        self.assertEqual(self.s.control()["mode"], "ready")

    def test_anchor_hint_residency_and_longer_ancestor_after_edit(self):
        p = self.do("turn", text="root")["checkpoint"]["snapshot_id"]
        longer = self.do("turn", text="a longer suffix")["checkpoint"]["snapshot_id"]
        self.m = Manager(self.s, "fixed", 1000)
        old = self.s.snap(p)
        z = self.do(
            "turn", messages=json.loads(old["messages"]) + [{"role": "user", "content": "x"}]
        )
        self.assertEqual(z["checkpoint"]["anchor_id"], p)
        with patch(
            "storage.digest", side_effect=AssertionError("checkpoint decision read whole image")
        ):
            z = self.do("turn", text="more")
        self.assertEqual(z["checkpoint"]["anchor_id"], p)
        self.do("budget", bytes=1)
        z = self.do("turn", text="again")
        self.assertIsNone(z["checkpoint"]["anchor_id"])
        self.assertEqual(z["checkpoint"]["distance_tokens"], z["processed"])
        self.assertEqual(self.s.resolve(longer)["id"], longer)

    def test_committed_turn_survives_known_failure_and_budget_retries(self):
        self.do("budget", bytes=112)
        calls = []
        original = self.n.save

        def larger(ids):
            calls.append(1)
            path, _, _ = original(ids)
            with path.open("ab") as f:
                f.write(bytes(1000))
            from native import digest

            return path, digest(path), path.stat().st_size

        with patch.object(self.n, "save", side_effect=larger):
            t = self.s.stamp()
            a = self.do("turn", text="abcdefgh")
            b = self.do("turn", text="x")
            self.assertEqual(a["checkpoint"]["status"], "failed")
            self.assertEqual(a["checkpoint"]["reason"], "budget")
            self.assertEqual(b["checkpoint"]["reason"], "budget_retry_suppressed")
            self.assertEqual(len(calls), 1)
            self.do("budget", bytes=5000)
            c = self.do("turn", text="y")
            self.assertEqual(c["checkpoint"]["status"], "saved")
        self.assertEqual(a["ticket"]["revision"], t["revision"] + 1)
        self.assertEqual(self.s.branch()["gen"], 3)
        self.assertEqual(len(json.loads(self.s.branch()["messages"])), 6)
        self.assertEqual(self.s.branch()["tokens"], self.n.live)

    def test_abort_before_commit_and_warning_after_commit(self):
        start = self.s.stamp()
        self.s.db.execute(
            """CREATE TEMP TRIGGER cancel_autosave BEFORE INSERT ON snapshots
               BEGIN SELECT RAISE(ABORT,'injected snapshot failure'); END"""
        )
        x = self.do("turn", text="a")
        self.assertTrue(x["turn_committed"])
        self.assertEqual(x["checkpoint"]["status"], "failed")
        self.assertEqual(x["ticket"]["revision"], start["revision"] + 1)
        self.assertIsNone(self.s.branch()["head"])
        self.assertEqual(self.s.control()["mode"], "ready")
        self.s.db.execute("DROP TRIGGER cancel_autosave")
        with patch.object(self.s, "clean_files", side_effect=OSError("synthetic cleanup")):
            y = self.do("turn", text="b")
        c = y["checkpoint"]
        self.assertEqual(c["status"], "saved_cleanup_warning")
        self.assertEqual(c["snapshot_id"], self.s.branch()["head"])
        self.assertTrue(c["snapshot_committed"])
        self.assertEqual(y["ticket"]["revision"], x["ticket"]["revision"] + 2)
        self.assertEqual(self.s.snap(c["snapshot_id"])["tokens"], self.n.live)

    def test_ambiguous_save_and_bad_boundary_do_not_regenerate(self):
        with patch.object(self.n, "save", side_effect=Uncertain("native_outcome_unknown", "lost")):
            x = self.do("turn", text="P")
        self.assertTrue(x["turn_committed"])
        self.assertEqual(x["checkpoint"]["status"], "unknown")
        self.assertEqual(self.s.control()["mode"], "uncertain")
        with self.assertRaises(Error):
            self.do("recover")
        expected = self.s.branch()["tokens"]
        self.n.epoch = "fresh"
        s = Store(self.root, self.n, 10_000)
        self.addCleanup(s.db.close)
        self.s = s
        self.m = Manager(s, "fixed", 1)
        with patch.object(
            self.n, "generate", side_effect=AssertionError("generated during recovery")
        ):
            self.do("recover")
        self.assertEqual(self.n.live, expected)
        self.assertEqual(self.s.branch()["gen"], 1)
        with patch.object(
            self.n, "save", side_effect=Error("boundary_mismatch", "synthetic boundary")
        ):
            out = self.do("turn", text="B")
        self.assertEqual(out["checkpoint"]["status"], "failed")
        self.assertEqual(self.s.control()["mode"], "invalid")
        self.assertEqual(self.s.branch()["gen"], 2)
        self.assertEqual(self.s.branch()["pending"], pack([1]))

    def test_required_checkout_keeps_mutating_until_target_commit(self):
        self.m = Manager(self.s, "fixed", 1000)
        self.do("turn", text="parent")
        self.do("mark", name="parent", pin=True)
        self.do("fork", name="B")
        self.do("turn", text="dirty")
        stamp = self.s.stamp()
        original = self.n.restore

        def see(sha, tokens, size):
            self.assertEqual(self.s.control()["mode"], "mutating")
            self.assertEqual(self.s.stamp(), stamp)
            self.assertIsNotNone(self.s.branch()["head"])
            return original(sha, tokens, size)

        with patch.object(self.n, "restore", side_effect=see):
            switched = self.do("checkout", name="B")
        self.assertTrue(switched["exact_head"])
        self.assertEqual(self.s.stamp()["revision"], stamp["revision"] + 1)

    def test_reconciliation_error_preserves_streamed_turn_and_fences(self):
        with patch.object(self.s, "clean_files", side_effect=OSError("cleanup")), patch.object(
            self.s, "snap", side_effect=sqlite3.OperationalError("catalog unavailable")
        ):
            x = self.do("turn", text="completed")
        self.assertTrue(x["turn_committed"])
        self.assertEqual(x["content"], "A")
        self.assertEqual(x["checkpoint"]["status"], "unknown")
        self.assertEqual(self.do("status")["mode"], "uncertain")
        with self.assertRaises(Error):
            self.do("recover")
        self.assertEqual(self.s.branch()["gen"], 1)

    def test_manager_interruption_in_optional_phase_keeps_committed_turn(self):
        pid = os.fork()
        if pid == 0:
            try:
                self.s.db.close()
                s = Store(self.root, self.n, 10000)
                m = Manager(s, "fixed", 1)
                self.n.save = lambda _: os._exit(74)
                m.handle({"op": "turn", "text": "finished", "max_tokens": 2, "ticket": s.stamp()})
            finally:
                os._exit(75)
        self.assertEqual(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]), 74)
        self.assertEqual(self.s.control()["mode"], "mutating")
        self.assertEqual(self.s.branch()["gen"], 1)
        tokens = self.s.branch()["tokens"]
        self.assertEqual(self.s.branch()["pending"], pack([1]))
        reopened = Store(self.root, self.n, 10000)
        self.addCleanup(reopened.db.close)
        self.assertEqual(reopened.control()["mode"], "uncertain")
        self.n.epoch = "new epoch"
        fresh = Store(self.root, self.n, 10000)
        self.addCleanup(fresh.db.close)
        with patch.object(self.n, "generate", side_effect=AssertionError("resampled")):
            Manager(fresh, "fixed", 1).handle({"op": "recover", "ticket": fresh.stamp()})
        self.assertEqual(self.n.live, tokens)
        self.assertEqual(fresh.branch()["gen"], 1)


if __name__ == "__main__":
    unittest.main()
