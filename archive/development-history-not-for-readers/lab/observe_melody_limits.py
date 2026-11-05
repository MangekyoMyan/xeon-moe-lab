#!/usr/bin/env python3
"""Use existing observation machinery with two programmable stall events."""
import observe_full_replica as observer

# Keep the supplemental occupancy event out to avoid multiplexing on SMT.
observer.EVENTS["stalls"] = [
    "cycle_activity.stalls_total:u",
    "cycle_activity.stalls_l2_pending:u",
]

if __name__ == "__main__":
    observer.main()
