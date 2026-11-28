"""Internal full states. Branch heads and named snapshots keep their original meaning."""

import hashlib
import os
import time

from native import Error, digest, image_tokens, sync_dir
from planner import eviction
from storage import uid


class Anchors:
    def __init__(self, store):
        self.s = store
        self.cleanup_needed = False
        store.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS anchors(
                id TEXT PRIMARY KEY, compat TEXT NOT NULL, tokens BLOB NOT NULL,
                sha TEXT NOT NULL, bytes INTEGER NOT NULL, created REAL NOT NULL,
                prefix_hash TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS anchor_hash ON anchors(compat,prefix_hash);
            CREATE TABLE IF NOT EXISTS anchor_refs(
                branch TEXT NOT NULL, anchor TEXT NOT NULL, PRIMARY KEY(branch,anchor));
        """
        )

    def visible(self, branch):
        rows = list(
            self.s.db.execute(
                """SELECT a.* FROM anchors a JOIN anchor_refs r ON r.anchor=a.id
                   WHERE r.branch=? AND a.compat=?""",
                (branch["id"], self.s.native.compat),
            )
        )
        rows.extend(self.s.chain(branch["head"] or branch["base"]))
        return [r for r in rows if r["compat"] == self.s.native.compat and r["sha"]]

    def matching(self, branch, data, verify=False):
        return [
            r
            for r in self.visible(branch)
            if data.startswith(r["tokens"]) and self.s.valid(r, verify) == "available"
        ]

    def inventory(self, branch, prefix, keep=()):
        s = self.s
        protected = set(keep)
        queries = [
            (
                """SELECT DISTINCT s.sha FROM snapshots s WHERE s.id IN (SELECT sid FROM marks WHERE pin=1
                UNION SELECT COALESCE(head,base) FROM branches WHERE pin=1)""",
                (),
            ),
            (
                """SELECT DISTINCT a.sha FROM anchors a JOIN anchor_refs r ON a.id=r.anchor
                WHERE r.branch<>?""",
                (branch["id"],),
            ),
        ]
        for q, args in queries:
            protected.update(r[0] for r in s.db.execute(q, args))
        # Other branches' available public ancestors also consume the shared store.
        for b in s.db.execute(
            "SELECT COALESCE(head,base) FROM branches WHERE id<>?", (branch["id"],)
        ):
            protected.update(r["sha"] for r in s.chain(b[0]))
        rows = {
            r["sha"]: dict(bytes=r["bytes"], positions=[], protected=r["sha"] in protected)
            for r in s.db.execute("SELECT * FROM blobs WHERE resident=1")
        }
        baseline = {
            r[0] for r in s.db.execute("SELECT DISTINCT sha FROM snapshots WHERE sha IS NOT NULL")
        }
        for sha, v in rows.items():
            v["baseline"] = sha in baseline
        for r in self.visible(branch):
            if (
                r["sha"] in rows
                and prefix.startswith(r["tokens"])
                and s.valid(r, False) == "available"
            ):
                pos = len(r["tokens"]) // 4
                if pos not in rows[r["sha"]]["positions"]:
                    rows[r["sha"]]["positions"].append(pos)
        return rows

    def reserve(self, branch, prefix, size, domain, *, incoming=None, keep=(), budget=None):
        s = self.s
        budget = s.control()["budget"] if budget is None else budget
        if self.cleanup_needed:
            s.clean_files()
            self.cleanup_needed = False
        rows = self.inventory(branch, prefix, keep)
        victims = eviction(rows, size, budget, domain, incoming)
        if victims is None:
            raise Error(
                "capacity_or_coverage",
                "protected bytes or existing coverage prevent this write",
                requested_bytes=size,
                budget_bytes=budget,
            )
        if victims:
            with s.tx():
                s.db.executemany("UPDATE blobs SET resident=0 WHERE sha=?", [(x,) for x in victims])
            self.cleanup_needed = True
            for sha in victims:
                (s.native.images / (sha + ".slot")).unlink(missing_ok=True)
            sync_dir(s.native.images)
            self.cleanup_needed = False
        retained = sum(r["bytes"] for sha, r in rows.items() if sha not in victims)
        return dict(evicted=victims, retained_before_stage=retained, peak_bytes=retained + size)

    def publish(self, branch, data, path, sha, size):
        s = self.s
        prefix_hash = hashlib.sha256(data).hexdigest()
        dest = s.native.images / (sha + ".slot")
        sid = uid()
        with s.tx():
            if dest.exists() and digest(dest) == sha:
                path.unlink()
            else:
                path.chmod(0o400)
                os.replace(path, dest)
            sync_dir(s.native.images)
            s.db.execute(
                """INSERT INTO blobs VALUES(?,?,1,?) ON CONFLICT(sha) DO UPDATE SET
                         resident=1,used=excluded.used""",
                (sha, size, time.time()),
            )
            row = s.db.execute(
                """SELECT id FROM anchors WHERE compat=? AND prefix_hash=? AND sha=?
                                AND tokens=?""",
                (s.native.compat, prefix_hash, sha, data),
            ).fetchone()
            if row:
                sid = row[0]
            else:
                s.db.execute(
                    "INSERT INTO anchors VALUES(?,?,?,?,?,?,?)",
                    (sid, s.native.compat, data, sha, size, time.time(), prefix_hash),
                )
            s.db.execute("INSERT OR IGNORE INTO anchor_refs VALUES(?,?)", (branch["id"], sid))
        return sid

    def fork(self, source, dest, data):
        # The named fork point may belong to a branch other than the active one.
        self.s.db.execute(
            """INSERT OR IGNORE INTO anchor_refs SELECT ?,a.id FROM anchors a
            WHERE a.compat=? AND a.tokens=substr(?,1,length(a.tokens)) AND length(a.tokens)<=?
            AND EXISTS (SELECT 1 FROM anchor_refs r WHERE r.anchor=a.id)""",
            (dest, self.s.native.compat, data, len(data)),
        )

    def release(self, branch):
        self.s.db.execute("DELETE FROM anchor_refs WHERE branch=?", (branch,))
        self.s.db.execute("DELETE FROM anchors WHERE id NOT IN (SELECT anchor FROM anchor_refs)")
