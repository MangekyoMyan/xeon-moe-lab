"""Offline protocol regressions; the separate x9ram report contains the hardware trials."""

import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch
from manager import Manager
from native import Error, Native, digest, pack
from storage import Store, uid


class SmallSlot:
    def __init__(self, p):
        self.images = p / "images"
        self.images.mkdir()
        self.compat = "fixture-codec"
        self.epoch = "boot:1:3"
        self.limit = 1000
        self.vocab = 256
        self.live = b""
        self.fail = 0

    def erase(self):
        self.live = b""

    def render(self, messages):
        return [0, *("|".join(m["content"] for m in messages).encode())]

    def generate(self, ids, maximum, emit):
        cache = len(self.live) // 4 if pack(ids).startswith(self.live) else 0
        self.live = pack(ids + [65])
        emit("A")
        return {
            "content": "A",
            "sampled": [65, 1],
            "timings": {"cache_n": cache, "prompt_n": len(ids) - cache, "predicted_n": 2},
        }

    def save(self, expected):
        if expected != self.live:
            raise Error("boundary_mismatch", "mismatched tokens")
        n = len(expected) // 4
        p = self.images / (".writing-" + uid() + ".slot")
        p.write_bytes(
            struct.pack("<IIIiII", 0x67677371, 4, n + 4, -1, 1, n) + expected + bytes(4) + b"state"
        )
        return p, digest(p), p.stat().st_size

    def restore(self, sha, tokens, size):
        if self.fail:
            self.live = b""
            self.fail -= 1
            raise Error("native_error", "private slot was cleared")
        self.live = tokens

    def replay(self, tokens, cache):
        reused = len(self.live) // 4 if cache and tokens.startswith(self.live) else 0
        self.live = tokens
        return len(tokens) // 4 - reused


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.n = SmallSlot(self.root)
        self.s = Store(self.root, self.n, 4096)
        self.m = Manager(self.s)
        self.do("recover")

    def tearDown(self):
        self.s.db.close()
        self.tmp.cleanup()

    def do(self, op, **args):
        return self.m.handle({"op": op, "ticket": self.s.stamp(), **args})

    def error(self, code, op, **args):
        with self.assertRaises(Error) as caught:
            self.do(op, **args)
        self.assertEqual(caught.exception.code, code)

    def test_reference_lifetime_and_dirty_pinned_rotation(self):
        self.do("turn", text="P")
        parent = self.do("mark", name="P", pin=True)["snapshot"]
        self.do("fork", name="A", **{"from": "P"})
        self.do("checkout", name="A")
        self.do("turn", text="A")
        a = self.do("mark", name="A0")["snapshot"]
        self.do("pin", kind="branch", name="A")
        self.do("unpin", kind="mark", name="P")
        self.do("release", kind="mark", name="P")
        self.do("budget", bytes=2 * a["image_bytes"])
        self.do("turn", text="B")
        self.do("mark", name="A1")
        self.do("checkout", name="main")
        self.error("pinned", "release", kind="branch", name="A")
        self.do("unpin", kind="branch", name="A")
        self.do("release", kind="branch", name="A")
        self.do("release", kind="mark", name="A0")
        self.do("release", kind="mark", name="A1")
        self.assertFalse((self.n.images / (a["image_sha256"] + ".slot")).exists())
        self.assertEqual(self.s.branch()["base"], parent["id"])

    def test_rollback_double_failure_and_restart_fence(self):
        self.do("turn", text="P")
        self.do("mark", name="P")
        self.do("fork", name="B")
        self.do("turn", text="A")
        expected = self.s.branch()["tokens"]
        branch = self.s.branch()["id"]
        self.n.fail = 1
        self.error("native_error", "checkout", name="B")
        self.assertEqual(self.n.live, expected)
        self.assertEqual(self.s.stamp()["branch"], branch)
        self.n.fail = 2
        self.error("rollback_failed", "checkout", name="B")
        self.assertEqual(self.s.control()["mode"], "invalid")
        self.assertEqual(self.s.branch()["tokens"], expected)
        self.do("recover")
        self.assertEqual(self.n.live, expected)
        ticket = self.s.stamp()
        self.do("checkout", name="B")
        self.error("stale", "checkout", name="main", ticket=ticket)
        self.do("checkout", name="main")
        self.assertEqual(self.s.stamp()["branch"], ticket["branch"])
        self.assertEqual(self.s.stamp()["generation"], ticket["generation"])
        self.error("stale", "mark", name="late", ticket=ticket)
        self.s.state("mutating")
        self.s.db.close()
        self.s = Store(self.root, self.n, 4096)
        self.m = Manager(self.s)
        self.error("native_outcome_unknown", "recover")
        self.s.db.close()
        self.n.epoch = "boot:2:4"
        self.s = Store(self.root, self.n, 4096)
        self.m = Manager(self.s)
        self.do("recover")
        self.assertEqual(self.s.control()["mode"], "ready")

    def test_checksum_fallback_and_no_display_string_matching(self):
        self.do("turn", text="P")
        self.do("mark", name="P")
        self.do("fork", name="B")
        self.do("checkout", name="B")
        self.do("turn", text="X")
        head = self.do("mark", name="B1")["snapshot"]
        expected = self.s.branch()["tokens"]
        self.do("checkout", name="main")
        p = self.n.images / (head["image_sha256"] + ".slot")
        p.chmod(0o600)
        p.write_bytes(b"z" * p.stat().st_size)
        result = self.do("checkout", name="B")
        self.assertEqual(result["skipped"][0]["reason"], "damaged")
        self.assertGreater(result["replayed"], 0)
        self.assertEqual(self.n.live, expected)

    def test_repeated_mark_preserves_immutable_id_and_has_no_side_effect_on_rejection(self):
        self.do("turn", text="P")
        original = self.do("mark", name="P")["snapshot"]
        self.assertEqual(self.do("mark", name="P")["snapshot"]["id"], original["id"])

        def catalog():
            tables = ("control", "branches", "snapshots", "marks", "blobs")
            return {
                table: [
                    tuple(row) for row in self.s.db.execute(f"SELECT * FROM {table} ORDER BY rowid")
                ]
                for table in tables
            }

        self.do("budget", bytes=1)
        self.do("budget", bytes=4096)
        ticket, before = self.s.stamp(), catalog()
        self.assertEqual(list(self.n.images.glob("*.slot")), [])
        for pin in (False, True):
            self.error("missing_image", "mark", name="P", pin=pin)
            self.assertEqual(catalog(), before)
            self.assertEqual(list(self.n.images.glob("*.slot")), [])
        self.assertEqual(self.s.stamp(), ticket)
        fork = self.do("fork", name="reader", ticket=ticket, **{"from": "P"})
        self.assertEqual(fork["branch"]["head"], original["id"])
        other = self.do("mark", name="replacement")["snapshot"]
        self.assertNotEqual(other["id"], original["id"])
        self.assertEqual(self.s.resolve("P")["id"], original["id"])

    def test_repeat_mark_rejects_damaged_image_or_a_different_head(self):
        self.do("turn", text="P")
        old = self.do("mark", name="P")["snapshot"]
        file = self.n.images / (old["image_sha256"] + ".slot")
        file.chmod(0o600)
        damaged = b"x" * file.stat().st_size
        file.write_bytes(damaged)
        before = dict(self.s.branch())
        self.error("missing_image", "mark", name="P", pin=True)
        self.assertEqual(self.s.db.execute("SELECT count(*) FROM snapshots").fetchone()[0], 1)
        self.assertEqual(dict(self.s.branch()), before)
        self.assertEqual(file.read_bytes(), damaged)
        self.do("turn", text="another input")
        self.error("mark_exists", "mark", name="P")
        self.assertEqual(self.s.resolve("P")["id"], old["id"])

    def test_restored_image_use_becomes_visible_with_the_active_branch(self):
        self.do("turn", text="P")
        target = self.do("mark", name="P")["snapshot"]
        other = self.do("fork", name="B")["branch"]["id"]
        self.do("turn", text="A")
        self.do("mark", name="A")
        self.s.db.execute("UPDATE blobs SET used=1 WHERE sha=?", (target["image_sha256"],))
        original = self.s.stamp()["branch"]
        reader = sqlite3.connect(self.root / "branches.sqlite3")
        self.addCleanup(reader.close)

        def visible():
            return reader.execute(
                "SELECT c.active,c.mode,b.used FROM control c,blobs b WHERE b.sha=?",
                (target["image_sha256"],),
            ).fetchone()

        place = self.m.place

        def after_native_restore(branch):
            receipt = place(branch)
            self.assertEqual(visible(), (original, "mutating", 1))
            return receipt

        with patch.object(self.m, "place", after_native_restore):
            self.do("checkout", name="B")
        active, mode, used = visible()
        self.assertEqual((active, mode), (other, "ready"))
        self.assertGreater(used, 1)

    def test_aborted_final_commit_does_not_touch_the_rejected_image(self):
        self.do("turn", text="P")
        target = self.do("mark", name="P")["snapshot"]
        self.do("fork", name="B")
        self.do("turn", text="A")
        source = self.do("mark", name="A")["snapshot"]
        origin = self.s.stamp()["branch"]
        self.s.db.execute("UPDATE blobs SET used=1")
        # Rev is written after active and last-use, inside the same final transaction.
        self.s.db.execute(
            """CREATE TEMP TRIGGER reject_target BEFORE UPDATE OF rev ON control
               WHEN (SELECT name FROM branches WHERE id=NEW.active)='B'
               BEGIN SELECT RAISE(ABORT,'injected final commit failure'); END"""
        )
        self.error("operation_failed", "checkout", name="B")
        self.assertEqual(self.s.stamp()["branch"], origin)
        self.assertEqual(self.s.control()["mode"], "ready")
        used = dict(self.s.db.execute("SELECT sha,used FROM blobs"))
        self.assertEqual(used[target["image_sha256"]], 1)
        self.assertGreater(used[source["image_sha256"]], 1)
        self.assertEqual(self.s.branch()["tokens"], self.n.live)

    def test_stream_receipt_includes_tokens_omitted_from_partial_utf8_events(self):
        class Receipt:
            def call(self, method, path, body, consume):
                samples = [10, 11, 12, 1]
                partial = {"stop": False, "content": "日", "tokens": [12]}
                final = {
                    "stop": True,
                    "branch_sampled_ids": samples,
                    "tokens_cached": 5,
                    "stop_type": "eos",
                    "timings": {"predicted_n": 4, "cache_n": 0, "prompt_n": 2},
                }
                return consume(
                    [b"data: " + json.dumps(v).encode() + b"\n" for v in (partial, final)]
                )

        out = Native.generate(Receipt(), [4, 5], 4, lambda _: None)
        self.assertEqual(out["sampled"], [10, 11, 12, 1])
        self.assertEqual(out["content"], "日")


if __name__ == "__main__":
    unittest.main()
