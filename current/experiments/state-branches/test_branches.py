"""Offline protocol regressions; the separate x9ram report contains the hardware trials."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
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
