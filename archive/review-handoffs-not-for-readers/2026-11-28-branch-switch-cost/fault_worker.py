"""Test-only wrapper. Exercise actual native failure on an absent private slot file."""

import json
import os
from pathlib import Path
import sqlite3
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "checkout/current/experiments/state-branches")
)
import branchctl
from manager import Manager
from native import Native


class FaultNative(Native):
    def __init__(self, binding):
        super().__init__(binding)
        self.plan = Path(binding).parent / "test-fault.json"

    def restore(self, sha, tokens, size):
        if self.plan.exists():
            spec = json.loads(self.plan.read_bytes())
            if spec["count"] > 0 and spec["mode"] in ("crash-after-confirmed-restore", "absent-native-image"):
                spec["count"] -= 1
                tmp = self.plan.with_suffix(".next")
                tmp.write_text(json.dumps(spec))
                tmp.replace(self.plan)
                if spec["mode"] == "crash-after-confirmed-restore":
                    super().restore(sha, tokens, size)
                    os._exit(73)
                if spec["mode"] == "absent-native-image":
                    self.call(
                        "POST",
                        "/slots/0?action=restore",
                        {"filename": "absent-synthetic-slot.slot"},
                    )
                    raise RuntimeError("missing native fixture unexpectedly loaded")
        return super().restore(sha, tokens, size)


original = branchctl.serve
finish = Manager.finish


def abort_final_transaction(self, placement=None):
    finish(self, placement=placement)
    plan = self.native.plan
    if placement is not None and plan.exists():
        spec = json.loads(plan.read_bytes())
        if spec["mode"] == "abort-final-tx" and spec["count"] > 0:
            spec["count"] -= 1
            tmp = plan.with_suffix(".next")
            tmp.write_text(json.dumps(spec))
            tmp.replace(plan)
            raise sqlite3.OperationalError("injected cancellation before final SQLite COMMIT")


Manager.finish = abort_final_transaction
branchctl.serve = lambda root, binding, budget: original(
    root, binding, budget, native_factory=FaultNative
)
if __name__ == "__main__":
    branchctl.cli()
