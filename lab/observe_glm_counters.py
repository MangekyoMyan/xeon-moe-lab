import observe_full_replica as observer
from pathlib import Path
observer.HELPER = Path(__file__).resolve().parent / "observe_glm_phase.py"
observer.EVENTS["stalls"] = ["cycle_activity.stalls_total:u", "cycle_activity.stalls_l2_pending:u"]
if __name__ == "__main__":
    observer.main()
