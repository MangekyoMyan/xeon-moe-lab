"""RAM-only decision for one possible future return to a completed text boundary."""

from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict, deque
from math import isfinite
from statistics import median
from time import perf_counter_ns


MODES = ("off", "fixed", "auto")
DEFAULT_INTERVAL = 1024
SAFETY = 1.5
WINDOW = 8
MIN_SAMPLES = 3


def positive_ms(value):
    if type(value) not in (int, float) or value <= 0:
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


def nearby(observations, position):
    return [item for item in observations if position <= item[0] * 2 and item[0] <= position * 2]


class CheckpointPolicy:
    def __init__(self, mode="off", interval=DEFAULT_INTERVAL, *, runtime="", thresholds=()):
        if mode not in MODES or type(interval) is not int or interval <= 0:
            raise ValueError("choose off, fixed or auto and a positive checkpoint interval")
        self.mode = mode
        self.interval = interval
        self.runtime = runtime
        # The MX gate is an additional requirement on top of the general offload gate.
        valid = [n for n in thresholds if type(n) is int and n > 0]
        self.cuts = tuple(sorted({valid[0], max(valid)})) if valid else (128,)
        self.prefills = defaultdict(lambda: deque(maxlen=WINDOW))
        self.saves = deque(maxlen=WINDOW)
        self.restores = defaultdict(lambda: deque(maxlen=WINDOW))
        self.restore_route = None
        self.observed = Counter()
        self.outcomes = Counter()
        self.bases = Counter()
        self.stats_ns = 0
        self.decision_ms = 0.0
        self.failed_capacity = OrderedDict()

    def observe_prefill(self, timings):
        start = perf_counter_ns()
        n, ms = timings.get("prompt_n"), timings.get("prompt_ms")
        if type(n) is int and n > 0 and positive_ms(ms):
            self.prefills[bisect_right(self.cuts, n)].append((n, ms))
            self.observed["prefill"] += 1
        self.stats_ns += perf_counter_ns() - start

    def observe_save(self, position, size, ms):
        start = perf_counter_ns()
        if position > 0 and size > 0 and positive_ms(ms):
            self.saves.append((position, ms, size))
            self.observed["optional_save"] += 1
        self.stats_ns += perf_counter_ns() - start

    def observe_restore(self, route, position, size, ms):
        start = perf_counter_ns()
        if positive_ms(ms) and ((position == size == 0) or (position > 0 and size > 0)):
            kind = "image" if size else "empty"
            self.restores[route, kind].append((position, ms, size))
            self.observed[route + "_" + kind] += 1
            self.restore_route = route
        self.stats_ns += perf_counter_ns() - start

    def decide(self, anchor_id, anchor, candidate, progressed=True):
        distance = candidate - anchor
        if distance < 0:
            raise ValueError("checkpoint ancestor is past the candidate")
        r = {
            "mode": self.mode,
            "interval_tokens": self.interval,
            "runtime": self.runtime,
            "anchor_id": anchor_id,
            "anchor_processed": anchor,
            "candidate_processed": candidate,
            "distance_tokens": distance,
            "status": "skipped",
            "basis": self.mode,
            "estimate_ms": {},
            "samples": {},
            "restore_route": None,
        }
        if self.mode == "off":
            return {**r, "status": "disabled", "reason": "disabled"}
        if distance == 0 or not progressed:
            return {
                **r,
                "status": "skipped",
                "reason": "covered_boundary" if distance == 0 and anchor_id else "no_progress",
            }
        if self.mode == "auto":
            band = bisect_right(self.cuts, distance)
            route = self.restore_route
            pp = nearby(self.prefills[band], distance)
            ss = nearby(self.saves, candidate)
            images = self.restores.get((route, "image"), ())
            ra = nearby(images, anchor) if anchor else self.restores.get((route, "empty"), ())
            rb = nearby(images, candidate)
            counts = {
                "prefill": len(pp),
                "save": len(ss),
                "restore_anchor": len(ra),
                "restore_candidate": len(rb),
            }

            def cost(samples):
                return median(s[1] for s in samples) if len(samples) >= MIN_SAMPLES else None

            p = (
                distance * (sum(s[1] for s in pp) / sum(s[0] for s in pp))
                if len(pp) >= MIN_SAMPLES
                else None
            )
            if p is not None and not isfinite(p):
                p = None
            a = cost(ra)
            empty_bound = anchor == 0 and a is None
            if empty_bound:
                a = 0.0
            b, save = cost(rb), cost(ss)
            r.update(
                samples=counts,
                restore_route=route,
                prefill_band=band,
                empty_lower_bound=empty_bound,
                estimate_ms={
                    "replay": p,
                    "save": save,
                    "restore_anchor": a,
                    "restore_candidate": b,
                    "gain": None,
                },
            )
            if all(v is not None and isfinite(v) for v in (p, a, b, save)) and isfinite(p + a - b):
                gain = p + a - b
                r["estimate_ms"]["gain"] = gain
                take = gain > SAFETY * save
                return {
                    **r,
                    "basis": "cost_model",
                    "model_accept": take,
                    "status": "attempt" if take else "skipped",
                    "reason": "worthwhile" if take else "cost_not_worthwhile",
                }
            r.update(
                basis="cold_fallback",
                fallback_reason="insufficient_comparable_data",
                missing=[
                    k
                    for k, v in r["estimate_ms"].items()
                    if k != "gain" and (v is None or not isfinite(v))
                ],
            )
        take = distance >= self.interval
        return {
            **r,
            "status": "attempt" if take else "skipped",
            "reason": "interval_reached" if take else "interval_not_reached",
        }

    def blocked(self, branch, n, capacity):
        item = self.failed_capacity.get(branch)
        return bool(
            item and item[0] == capacity and item[1] <= n and item[1].bit_length() == n.bit_length()
        )

    def budget_failed(self, branch, n, capacity):
        self.failed_capacity[branch] = (capacity, n)
        self.failed_capacity.move_to_end(branch)
        if len(self.failed_capacity) > 32:
            self.failed_capacity.popitem(last=False)

    def record(self, r):
        self.outcomes[r["status"]] += 1
        self.bases[r["basis"]] += 1
        self.decision_ms += r.get("decision_ms", 0)
        if r["basis"] == "cost_model":
            self.outcomes[
                "cost_model_selected" if r["model_accept"] else "cost_model_declined"
            ] += 1

    def status(self):
        return {
            "mode": self.mode,
            "interval_tokens": self.interval,
            "runtime": self.runtime,
            "prefill_cutpoints": self.cuts,
            "observations": dict(self.observed),
            "outcomes": dict(self.outcomes),
            "decision_bases": dict(self.bases),
            "stats_ms": self.stats_ns / 1e6,
            "decision_ms": self.decision_ms,
        }
