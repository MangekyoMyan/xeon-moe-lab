"""Prefix placement, coverage and return costs. All positions are counts of evaluated IDs."""

import bisect
import functools
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass


def valid_ms(value):
    if type(value) not in (int, float) or value <= 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class Config:
    mode: str = "off"
    interval: int = 1024
    target_ms: float = 5000.0
    coverage: str = "all_history"
    recent: int = 4096

    def __post_init__(self):
        if self.mode not in ("off", "fixed", "auto") or self.coverage not in (
            "all_history",
            "recent",
        ):
            raise ValueError("unsupported checkpoint mode or coverage policy")
        if any(type(x) is not int or x < 1 for x in (self.interval, self.recent)):
            raise ValueError("positive token counts required")
        if not valid_ms(self.target_ms):
            raise ValueError("finite positive resume target required")

    def domain(self, history):
        return (max(0, history - self.recent) if self.coverage == "recent" else 0, history)


class Profile:
    """One build/configuration and local warm disk. Invalid or failed events are never learned."""

    def __init__(self, runtime, thresholds=(), rows=()):
        self.runtime = runtime
        self.cuts = sorted(set(x for x in thresholds if type(x) is int and x > 0))
        self.rows = list(rows)[-192:]
        self._cost = functools.lru_cache(32768)(self._estimate)

    def add(self, kind, position, length, ms, size=0):
        if (
            kind not in ("empty", "restore", "prefill", "save")
            or not valid_ms(ms)
            or any(type(x) is not int or x < 0 for x in (position, length, size))
            or (kind == "prefill" and not length)
        ):
            return
        self.rows.append((kind, position, length, ms, size, time.time()))
        self.rows = self.rows[-192:]
        self._cost.cache_clear()

    def _estimate(self, kind, position, length):
        samples = [r for r in self.rows if r[0] == kind]
        if kind == "prefill":
            band = bisect.bisect_right(self.cuts, length)
            samples = [r for r in samples if bisect.bisect_right(self.cuts, r[2]) == band]
        # Bounded local interpolation/extrapolation. Prefer nearby context and length.
        samples = [
            r
            for r in samples
            if max(r[1] + 256, position + 256) <= 4 * min(r[1] + 256, position + 256)
        ]
        if kind == "prefill":
            samples = [r for r in samples if max(length, r[2]) <= 4 * min(length, r[2])]
        samples.sort(
            key=lambda r: (
                abs(math.log2((r[1] + 256) / (position + 256)))
                + (abs(math.log2(length / r[2])) if kind == "prefill" else 0),
                -r[5],
            )
        )
        if len(samples) < 2:
            return None
        selected = samples[:6]
        values = [r[3] * length / r[2] if kind == "prefill" else r[3] for r in selected]
        value = statistics.median(values)
        return value if math.isfinite(value) else None

    def restore(self, c):
        return self._cost("restore" if c else "empty", c, 0)

    def prefill(self, c, length):
        return self._cost("prefill", c, length) if length else 0.0

    def save(self, position):
        return self._cost("save", position, 0)

    def image_size(self, position):
        samples = [
            r
            for r in self.rows
            if r[4]
            and r[0] in ("restore", "save")
            and max(position + 256, r[1] + 256) <= 4 * min(position + 256, r[1] + 256)
        ]
        if not samples:
            return None
        nearest = sorted(samples, key=lambda r: abs(r[1] - position))[:3]
        return max(r[4] for r in nearest)

    def basis(self):
        return {
            "runtime": self.runtime,
            "samples": len(self.rows),
            "observations": {
                kind: sum(r[0] == kind for r in self.rows)
                for kind in ("empty", "restore", "prefill", "save")
            },
            "last_observed_unix": self.rows[-1][5] if self.rows else None,
            "context_range": (
                [min(r[1] for r in self.rows), max(r[1] for r in self.rows)] if self.rows else None
            ),
            "prefill_length_range": (
                [
                    min(r[2] for r in self.rows if r[0] == "prefill"),
                    max(r[2] for r in self.rows if r[0] == "prefill"),
                ]
                if any(r[0] == "prefill" for r in self.rows)
                else None
            ),
            "image_forecast": "largest of the nearest three image observations within the context band; independent future blobs",
            "route_cuts": self.cuts,
            "basis": "median of 2–6 nearby ordinary warm observations; context and replay length within factor 4, same dispatch band; locally linear in replay length",
        }


def resume_route(
    p, available, costs, *, strict=False, boundaries=(), missing=(), capture_interval=0
):
    """Use this same function for actual restores and the two forecast scenarios."""
    candidates = [c for c in available if 0 < c <= p and not (strict and c == p)]
    candidates = sorted(set([0, *candidates]), reverse=True)
    if p == 0:
        return (0, costs.restore(0), "empty")

    def total(c):
        value = costs.restore(c)
        if value is None:
            return None
        at = c
        extra = (
            set(
                q
                for q in range(
                    (c // capture_interval + 1) * capture_interval, p + 1, capture_interval
                )
                if q not in available
            )
            if capture_interval
            else set()
        )
        cuts = sorted(q for q in set(boundaries) | extra if c < q < p)
        for end in (*cuts, p):
            part = costs.prefill(at, end - at)
            if part is None:
                return None
            value += part
            if (end in missing or end in extra) and end > c:
                save = costs.save(end)
                if save is None:
                    return None
                value += save
            at = end
        return value if math.isfinite(value) else None

    ranked = [(c, total(c)) for c in candidates]
    if all(ms is not None for _, ms in ranked):
        c, ms = min(ranked, key=lambda v: (v[1], -v[0]))
        return c, ms, "estimated_total"
    c, ms = ranked[0]
    return c, ms, "nearest_valid_prefix"


def least_harm(positions, domain):
    """Lexicographic minimax replay gaps. A retained predecessor can cover the left edge."""
    lo, hi = domain
    positions = set(positions)
    before = max([0, *(p for p in positions if p <= lo)])
    edges = [before, *sorted(p for p in positions if lo < p <= hi), hi + 1]
    return tuple(sorted((b - a - 1 for a, b in zip(edges, edges[1:])), reverse=True))


def eviction(rows, need, budget, domain, incoming=None):
    # sha -> physical bytes, all branch-visible positions, protected. Metadata never gets evicted here.
    retained = dict(rows)
    size = sum(v["bytes"] for v in retained.values())
    removed = []
    while size + need > budget:
        possible = [k for k, v in retained.items() if not v["protected"]]
        if not possible:
            return None

        def score(k):
            others = [p for key, v in retained.items() if key != k for p in v["positions"]]
            if incoming is not None:
                others.append(incoming)
            return least_harm(others, domain), -retained[k]["bytes"], k

        victim = min(possible, key=score)
        size -= retained.pop(victim)["bytes"]
        removed.append(victim)
    # When the new point itself is least helpful, keep the current coverage.
    if removed and incoming is not None:
        old = [p for r in rows.values() for p in r["positions"]]
        new = [p for r in retained.values() for p in r["positions"]] + [incoming]
        if least_harm(new, domain) >= least_harm(old, domain):
            return None
    return removed


def evaluate(domain, available, costs, target, interval=0, *, capture_allowed=False):
    lo, hi = domain
    points = sorted(set(available))
    count = hi - lo + 1
    # Exact enumeration when affordable, otherwise explicitly partial coverage.
    exact = count * max(1, len(points)) <= 100000 and count <= 10000
    if exact:
        sample = range(lo, hi + 1)
    else:
        stride = max(1, math.ceil(count / 200))
        sample = sorted(
            {
                lo,
                hi,
                *range(lo, hi + 1, stride),
                *(q for c in points for q in (c - 1, c, c + 1) if lo <= q <= hi),
            }
        )
        if len(sample) > 800:
            sample = sorted({lo, hi, *sample[:: math.ceil(len(sample) / 796)]})
    max_time = min_rate = None
    max_replay = 0
    unknown = 0
    failed = []
    meets = []
    for p in sample:
        # Valid files already covering the lattice need no fresh save. New captures, if any,
        # split the re-prefill and their synchronous publication blocks prefix availability.
        c, ms, _ = resume_route(
            p, points, costs, capture_interval=interval if capture_allowed else 0
        )
        max_replay = max(max_replay, p - c)
        if ms is None:
            unknown += 1
            continue
        if max_time is None or ms > max_time["ms"]:
            max_time = dict(p=p, c=c, ms=ms)
        if ms > 0 and p:
            rate = 1000 * p / ms
            if math.isfinite(rate) and (min_rate is None or rate < min_rate["tokens_per_second"]):
                min_rate = dict(p=p, c=c, tokens_per_second=rate)
        (meets if ms <= target else failed).append(p)
    full = exact and not unknown

    def ranges(values):
        out = []
        for p in values:
            if out and p == out[-1][1] + 1:
                out[-1][1] = p
            else:
                out.append([p, p])
        return out

    return dict(
        evaluated_positions=len(sample),
        integer_positions=count,
        enumeration="all" if exact else "sample",
        unknown_positions=unknown,
        estimated_max_resume=max_time if full else None,
        worst_observed_in_model=max_time,
        estimated_min_equivalent=min_rate if full else None,
        min_equivalent_at_evaluated_positions=min_rate,
        max_replay_tokens=max_replay if exact else None,
        sampled_max_replay_tokens=max_replay,
        target_met="no" if failed else ("yes" if full else "unknown"),
        target_met_ranges=ranges(meets) if exact else None,
        miss_ranges=ranges(failed) if exact else None,
        rate_null_reason=(
            "p=0, zero-time or incomplete coverage" if min_rate is None or not full else None
        ),
    )


def plan(config, branch, start, history, rows, budget, costs, hysteresis=None, *, available=None):
    domain = config.domain(history)
    present = sorted(
        set(
            available
            if available is not None
            else (p for r in rows.values() for p in r["positions"])
        )
    )
    total = sum(r["bytes"] for r in rows.values())
    baseline = sum(r["bytes"] for r in rows.values() if r.get("baseline"))
    pinned = sum(r["bytes"] for r in rows.values() if r["protected"])
    hysteresis = hysteresis or {}
    intervals = {config.interval}
    if config.mode == "auto" and costs.basis()["observations"].get("prefill", 0) >= 2:
        intervals.update(max(1, config.interval * 2**power // 4) for power in range(0, 7))

    def project(interval):
        kept = dict(rows)
        positions = []
        attempts = []
        peak = total
        optional_peak = total - baseline
        known = True
        room = True
        # These points are reachable on this *future traversal*, including replay.
        grid = (
            range((start // interval + 1) * interval, history + 1, interval)
            if config.mode != "off"
            else ()
        )
        # Keep preview bounded even for a one-token lattice over a six-figure horizon.
        # The exact execution lattice still goes to the native cursor. The dense capacity
        # forecast is explicitly unknown instead of simulating hundreds of thousands of evictions.
        dense = len(grid) > 2048
        for p in grid:
            if p in present:
                continue
            attempts.append(p)
            if dense:
                known = False
                continue
            estimate = costs.image_size(p)
            if estimate is None:
                known = False
                if pinned >= budget:
                    room = False
                continue
            gone = eviction(kept, estimate, budget, domain, p)
            if gone is None:
                room = False
                continue
            for k in gone:
                kept.pop(k)
            peak = max(peak, sum(v["bytes"] for v in kept.values()) + estimate)
            optional_peak = max(
                optional_peak,
                sum(v["bytes"] for v in kept.values() if not v.get("baseline")) + estimate,
            )
            kept["plan-" + str(p)] = {"positions": [p], "bytes": estimate, "protected": False}
            positions.append(p)
        after = sorted(set(q for v in kept.values() for q in v["positions"]))
        forecast = evaluate(
            domain,
            after,
            costs,
            config.target_ms,
            interval,
            capture_allowed=config.mode != "off" and pinned < budget and not dense,
        )
        if not known and not dense:
            forecast.update(
                estimated_max_resume=None,
                estimated_min_equivalent=None,
                target_met="unknown",
                target_met_ranges=None,
                miss_ranges=None,
                reason="unmeasured_image_size",
            )
        if dense:
            forecast.update(
                estimated_max_resume=None,
                estimated_min_equivalent=None,
                target_met="unknown",
                target_met_ranges=None,
                miss_ranges=None,
                enumeration="dense_plan_not_simulated",
                max_replay_tokens=None,
            )
        return dict(
            interval=interval,
            attempt_positions=attempts,
            expected_saved=positions,
            expected_retained=after,
            retained_bytes=sum(v["bytes"] for v in kept.values()) if known else None,
            peak_bytes=peak if known else None,
            retained_increment_bytes=(
                sum(v["bytes"] for v in kept.values() if not v.get("baseline")) if known else None
            ),
            peak_increment_bytes=optional_peak if known else None,
            size_known=known,
            budget_feasible=(room if known or not room else None),
            forecast=forecast,
        )

    choices = [project(i) for i in sorted(intervals)]

    def score(v):
        forecast = v["forecast"]
        pred = forecast["estimated_max_resume"]
        if (
            pred is None
            and forecast["enumeration"] == "sample"
            and not forecast["unknown_positions"]
        ):
            pred = forecast["worst_observed_in_model"]
        if pred is None or not v["size_known"]:
            return (2, abs(v["interval"] - config.interval))
        if pred["ms"] <= config.target_ms:
            return (0, v["retained_bytes"], len(v["expected_saved"]), -v["interval"])
        return (1, pred["ms"], v["retained_bytes"], -v["interval"])

    winner = min(choices, key=score)
    preferred = winner["interval"]
    prev = hysteresis.get("interval", config.interval)
    old = next((v for v in choices if v["interval"] == prev), None)
    count = hysteresis.get("count", 0) + 1 if hysteresis.get("pending") == preferred else 1
    # Require two independent decisions before a routine change. Never delay a capacity refusal.
    if (
        config.mode == "auto"
        and old
        and old is not winner
        and old["budget_feasible"] is not False
        and count < 2
    ):
        winner = old
    next_state = dict(interval=winner["interval"], pending=preferred, count=min(count, 2))
    return_interval = (
        hysteresis.get("interval", config.interval) if config.mode == "auto" else config.interval
    )
    dense_current = config.mode != "off" and history // return_interval > 2048
    current = evaluate(
        domain,
        present,
        costs,
        config.target_ms,
        return_interval,
        capture_allowed=config.mode != "off" and pinned < budget and not dense_current,
    )
    if dense_current:
        current.update(
            estimated_max_resume=None,
            estimated_min_equivalent=None,
            target_met="unknown",
            target_met_ranges=None,
            miss_ranges=None,
            enumeration="dense_plan_not_simulated",
            max_replay_tokens=None,
        )
    body = dict(
        schema_version=2,
        branch=branch,
        config=asdict(config),
        requested_range=domain,
        start=start,
        assumed_history=history,
        return_interval=return_interval,
        currently_available=present,
        current=current,
        after_plan_forecast=winner,
        chosen_interval=winner["interval"],
        next_position=next(iter(winner["attempt_positions"]), None),
        candidate_intervals=sorted(intervals),
        selection_evidence=(
            ("fixed_fallback" if score(winner)[0] == 2 else winner["forecast"]["enumeration"])
            if config.mode == "auto"
            else config.mode
        ),
        assumptions=[
            "model already loaded; single warm local image store; no queue or model startup",
            "future independent images sized from nearby observations; exact native bytes admitted at each write",
            "replay through absent grid points includes estimated split and synchronous save when unpinned capacity exists",
        ],
        hysteresis_after=next_state,
        capacity={
            "resource": "disk native images",
            "budget_bytes": budget,
            "retained_bytes": total,
            "protected_bytes": pinned,
            "retained_increment_bytes": total - baseline,
            "baseline_public_image_bytes": baseline,
            "peak_increment_bytes": None,
            "budget_applied": config.mode != "off",
            "new_save_possible": pinned < budget,
            "excluded": ["SQLite and WAL", "RAM capture buffer and OS cache", "VRAM"],
        },
        profile=costs.basis(),
        reasons=[],
    )
    if score(winner)[0] == 2 and config.mode == "auto":
        body["reasons"].append("insufficient_cost_or_size_observations_fixed_fallback")
    if pinned >= budget:
        body["reasons"].append("protected_capacity")
    if dense_current:
        body["reasons"].append("dense_grid_time_estimate_unknown")
    if all((v := costs.restore(c)) is not None and v > config.target_ms for c in [0, *present]):
        body["reasons"].append("restore_alone_over_target")
    if winner["budget_feasible"] is False:
        body["reasons"].append("capacity_or_coverage_eviction")
    if current["target_met"] == "no":
        body["reasons"].append("present_routes_exceed_target")
    if current["target_met"] == "unknown":
        body["reasons"].append("incomplete_cost_coverage")
    key = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body["plan_id"] = hashlib.sha256(key).hexdigest()[:20]
    return body
