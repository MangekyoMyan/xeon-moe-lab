"""Durable branch metadata, immutable native images and an image-byte budget."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid
from native import Error, digest, encode, image_tokens, sync_dir


def uid():
    return uuid.uuid4().hex


def name_ok(name):
    if (
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 96
        or any(ord(control) < 32 or control == "/" for control in name)
    ):
        raise Error(
            "name",
            "choose a nonempty name of at most 96 characters without slashes or control characters",
        )
    if re.fullmatch(r"[0-9a-f]{32}", name):
        raise Error("name", "32 hexadecimal digits are reserved for immutable IDs")
    return name


class Store:
    def __init__(self, root, native, budget):
        self.root, self.native = Path(root), native
        self.db = sqlite3.connect(self.root / "branches.sqlite3", isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
          CREATE TABLE IF NOT EXISTS control(single INTEGER PRIMARY KEY CHECK(single=1), session TEXT, active TEXT,
             rev INTEGER, budget INTEGER, compat TEXT, mode TEXT, epoch TEXT, detail TEXT);
          CREATE TABLE IF NOT EXISTS branches(id TEXT PRIMARY KEY, name TEXT UNIQUE, origin TEXT, base TEXT, head TEXT,
             tokens BLOB NOT NULL, pending BLOB NOT NULL, messages TEXT NOT NULL, gen INTEGER, pin INTEGER, used REAL, replay INTEGER);
          CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY, parent TEXT, compat TEXT, tokens BLOB, pending BLOB,
             messages TEXT, sha TEXT, bytes INTEGER, created REAL);
          CREATE TABLE IF NOT EXISTS marks(name TEXT PRIMARY KEY, sid TEXT NOT NULL, pin INTEGER);
          CREATE TABLE IF NOT EXISTS blobs(sha TEXT PRIMARY KEY, bytes INTEGER NOT NULL, resident INTEGER, used REAL);
        """
        )
        from anchors import Anchors

        self.anchors = Anchors(self)
        self.bounded_save = None
        control = self.control()
        if not control:
            if type(budget) is not int or budget <= 0:
                raise Error("budget", "a positive image budget is required")
            branch_id = uid()
            with self.tx():
                self.db.execute(
                    "INSERT INTO branches VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (branch_id, "main", None, None, None, b"", b"", "[]", 0, 0, time.time(), 0),
                )
                self.db.execute(
                    "INSERT INTO control VALUES(1,?,?,0,?,?, 'invalid',?,'new store')",
                    (uid(), branch_id, budget, native.compat, native.epoch),
                )
        elif control["compat"] != native.compat:
            self.db.close()
            raise Error(
                "incompatible", "store belongs to a different model, engine configuration or codec"
            )
        elif control["epoch"] != native.epoch:
            self.state("invalid", "new backend; restore the recorded active head")
        elif control["mode"] == "mutating":
            self.state(
                "uncertain",
                "manager stopped during native work; restart the private backend before recovery",
            )

    @contextmanager
    def tx(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def control(self):
        return self.db.execute("SELECT * FROM control WHERE single=1").fetchone()

    def state(self, mode, detail=""):
        self.db.execute(
            "UPDATE control SET mode=?,epoch=?,detail=?", (mode, self.native.epoch, detail)
        )

    def branch(self, key=None):
        key = self.control()["active"] if key is None else key
        row = self.db.execute("SELECT * FROM branches WHERE id=? OR name=?", (key, key)).fetchone()
        if row is None:
            raise Error("branch_missing", "branch does not exist")
        return row

    def stamp(self):
        control = self.control()
        return {
            "session": control["session"],
            "branch": control["active"],
            "generation": self.branch()["gen"],
            "revision": control["rev"],
        }

    def snap(self, snapshot_id):
        row = self.db.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if row is None:
            raise Error("snapshot_missing", "snapshot does not exist")
        return row

    def resolve(self, label):
        m = self.db.execute("SELECT sid FROM marks WHERE name=?", (label,)).fetchone()
        return self.snap(m["sid"] if m else label)

    def chain(self, start):
        seen = set()
        while start:
            if start in seen:
                raise Error("metadata", "cycle in snapshot ancestry")
            seen.add(start)
            node = self.snap(start)
            yield node
            start = node["parent"]

    def valid(self, snapshot, verify=True):
        if snapshot["compat"] != self.native.compat:
            raise Error(
                "incompatible",
                "saved point belongs to a different model or codec",
                snapshot=snapshot["id"],
            )
        if snapshot["sha"] is None:
            return "empty" if not snapshot["tokens"] else "missing"
        if not re.fullmatch(r"[0-9a-f]{64}", snapshot["sha"]):
            raise Error("metadata", "invalid image identity")
        blob = self.db.execute(
            "SELECT resident FROM blobs WHERE sha=?", (snapshot["sha"],)
        ).fetchone()
        if not blob or not blob[0]:
            return "evicted"
        path = self.native.images / (snapshot["sha"] + ".slot")
        if not path.is_file():
            return "missing"
        if path.stat().st_size != snapshot["bytes"]:
            return "damaged"
        if verify and (
            digest(path) != snapshot["sha"]
            or image_tokens(path, self.native.limit) != snapshot["tokens"]
        ):
            return "damaged"
        return "available"

    def trim(self, budget, keep=(), coverage=None):
        if type(budget) is not int or budget < 1:
            raise Error("budget", "budget must be positive bytes")
        if coverage:
            from planner import eviction

            cp, branch = coverage
            rows = self.anchors.inventory(branch, branch["tokens"], keep)
            victims = eviction(rows, 0, budget, cp.cfg.domain(len(branch["tokens"]) // 4))
            if victims is None:
                raise Error("budget", "protected image bytes exceed the requested budget")
            self.db.executemany("UPDATE blobs SET resident=0 WHERE sha=?", [(s,) for s in victims])
            return victims
        pinned = set(keep)
        for row in self.db.execute(
            """SELECT s.sha FROM snapshots s WHERE s.id IN
            (SELECT sid FROM marks WHERE pin=1 UNION SELECT COALESCE(head,base) FROM branches WHERE pin=1)"""
        ):
            pinned.add(row[0])
        rows = list(self.db.execute("SELECT * FROM blobs WHERE resident=1 ORDER BY used,sha"))
        size = sum(r["bytes"] for r in rows)
        victims = []
        for row in rows:
            if size <= budget:
                break
            if row["sha"] not in pinned:
                victims.append(row["sha"])
                size -= row["bytes"]
        if size > budget:
            raise Error(
                "budget",
                "pinned images and the required working states exceed the image budget",
                required_bytes=size,
                budget_bytes=budget,
            )
        self.db.executemany("UPDATE blobs SET resident=0 WHERE sha=?", [(v,) for v in victims])
        return victims

    def clean_files(self):
        resident = {
            r[0] + ".slot" for r in self.db.execute("SELECT sha FROM blobs WHERE resident=1")
        }
        changed = False
        for path in self.native.images.iterdir():
            if path.name not in resident and re.fullmatch(
                r"(?:[0-9a-f]{64}|\.writing-[0-9a-f]{32})\.slot", path.name
            ):
                path.unlink(missing_ok=True)
                changed = True
        if changed:
            sync_dir(self.native.images)

    def collect(self):
        # A retained child still owns its complete ancestor chain and token log.
        keep = set()
        roots = [
            r[0]
            for r in self.db.execute(
                """SELECT sid FROM marks UNION SELECT head FROM branches
                                               UNION SELECT base FROM branches UNION SELECT origin FROM branches"""
            )
        ]
        for root in roots:
            for point in self.chain(root):
                if point["id"] in keep:
                    break
                keep.add(point["id"])
        unused = [(r[0],) for r in self.db.execute("SELECT id FROM snapshots") if r[0] not in keep]
        self.db.executemany("DELETE FROM snapshots WHERE id=?", unused)
        self.db.execute(
            "DELETE FROM blobs WHERE sha NOT IN (SELECT sha FROM snapshots WHERE sha IS NOT NULL UNION SELECT sha FROM anchors)"
        )

    def save_active(self, keep=(), *, on_publish=None, receipt=None):
        info = receipt if receipt is not None else {}
        info.update(
            stage="lookup", native_called=False, committed=False, written_bytes=0, evicted=[]
        )
        branch = self.branch()
        if branch["head"]:
            point = self.snap(branch["head"])
            if self.valid(point) in ("available", "empty"):
                info.update(stage="reused", snapshot_id=point["id"])
                return point
        path = None
        sha = None
        size = 0
        try:
            if branch["tokens"]:
                info.update(stage="native_save", native_called=True, written_bytes=None)
                if self.bounded_save:
                    path, sha, size = self.bounded_save(branch["tokens"], keep)
                else:
                    path, sha, size = self.native.save(branch["tokens"])
                info["written_bytes"] = size
            snapshot_id = uid()
            info.update(stage="publish", snapshot_id=snapshot_id)
            victims = []
            with self.tx():
                self.db.execute(
                    "INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        branch["base"],
                        self.native.compat,
                        branch["tokens"],
                        branch["pending"],
                        branch["messages"],
                        sha,
                        size,
                        time.time(),
                    ),
                )
                self.db.execute(
                    "UPDATE branches SET head=?,base=? WHERE id=?",
                    (snapshot_id, snapshot_id, branch["id"]),
                )
                if path:
                    dest = self.native.images / (sha + ".slot")
                    self.db.execute(
                        """INSERT INTO blobs VALUES(?,?,1,?) ON CONFLICT(sha) DO UPDATE SET
                                      resident=1,used=excluded.used""",
                        (sha, size, time.time()),
                    )
                    victims = self.trim(self.control()["budget"], [*keep, sha])
                    info["planned_evictions"] = victims
                    if dest.exists() and digest(dest) == sha:
                        info["existing_blob"] = True
                        path.unlink()
                    else:
                        info["existing_blob"] = False
                        path.chmod(0o400)
                        os.replace(path, dest)
                    sync_dir(self.native.images)
                if on_publish is not None:
                    on_publish()
            info.update(stage="cleanup", committed=True, evicted=victims)
            self.clean_files()
            info["stage"] = "done"
            return self.snap(snapshot_id)
        finally:
            if path:
                path.unlink(missing_ok=True)

    def choose(self, base, tokens):
        skipped = []
        candidates = sorted(
            (snapshot for snapshot in self.chain(base) if tokens.startswith(snapshot["tokens"])),
            key=lambda snapshot: len(snapshot["tokens"]),
            reverse=True,
        )
        for snapshot in candidates:
            status = self.valid(snapshot)
            if status in ("available", "empty"):
                return snapshot, skipped
            skipped.append({"snapshot": snapshot["id"], "reason": status})
        return None, skipped

    def describe(self, branch):
        image = self.snap(branch["head"]) if branch["head"] else None
        status = "unsaved"
        if image:
            try:
                status = self.valid(image, False)
            except Error as exc:
                status = exc.code
        return {
            "id": branch["id"],
            "name": branch["name"],
            "parent_snapshot": branch["origin"],
            "base": branch["base"],
            "head": branch["head"],
            "generation": branch["gen"],
            "processed": len(branch["tokens"]) // 4,
            "pending_sampled": len(branch["pending"]) // 4,
            "image_bytes": image["bytes"] if image else 0,
            "image_status": "present" if status == "available" else status,
            "pinned": bool(branch["pin"]),
            "last_used": branch["used"],
            "last_replay_tokens": branch["replay"],
        }

    def point_info(self, snapshot, verify=False):
        row = self.db.execute("SELECT used FROM blobs WHERE sha=?", (snapshot["sha"],)).fetchone()
        return {
            "id": snapshot["id"],
            "parent": snapshot["parent"],
            "processed": len(snapshot["tokens"]) // 4,
            "pending_sampled": len(snapshot["pending"]) // 4,
            "image_sha256": snapshot["sha"],
            "image_bytes": snapshot["bytes"],
            "image_status": self.valid(snapshot, verify),
            "created": snapshot["created"],
            "last_used": row[0] if row else snapshot["created"],
        }

    def listing(self):
        control = self.control()
        blobs = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM blobs WHERE resident=1"
        ).fetchone()
        return {
            "ticket": self.stamp(),
            "mode": control["mode"],
            "detail": control["detail"],
            "budget_bytes": control["budget"],
            "resident_images": blobs[0],
            "image_bytes": blobs[1],
            "metadata_bytes": sum(
                path.stat().st_size for path in self.root.glob("branches.sqlite3*")
            ),
            "branches": [
                self.describe(branch)
                for branch in self.db.execute("SELECT * FROM branches ORDER BY name")
            ],
            "marks": [
                dict(m) for m in self.db.execute("SELECT name,sid,pin FROM marks ORDER BY name")
            ],
        }
