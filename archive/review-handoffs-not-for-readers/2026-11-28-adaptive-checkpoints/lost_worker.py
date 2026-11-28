"""Test-only adapter: drop a completed native save's receipt before the catalog can publish."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"checkout/current/experiments/state-branches"))
import branchctl
from native import Native,Uncertain

class LostSave(Native):
    def __init__(self,binding):
        super().__init__(binding);self.trigger=Path(binding).parent/"lose-next-receipt"
    def save(self,expected):
        result=super().save(expected)
        if self.trigger.exists():
            self.trigger.unlink()
            raise Uncertain("native_outcome_unknown","test discarded the native save receipt")
        return result

serve=branchctl.serve
branchctl.serve=lambda root,binding,budget,**kwargs:serve(root,binding,budget,native_factory=LostSave,**kwargs)
if __name__=="__main__":branchctl.cli()
