import sys
from pathlib import Path
sys.path.insert(0,'/home/codex/projects/llm-lab')
import observe_glm_counters_light as observer
observer.HELPER=Path('/home/codex/projects/llm-lab/mxfp4-opt-20261023/observe_model_phase.py')
observer.main()
