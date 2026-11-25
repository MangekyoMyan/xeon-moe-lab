"""An exploratory predictor using activation quantization residual energy on the same two inputs."""
import array
import collections
import math
import statistics
import struct
from prepare import ROOT,PREV,load,save,f32,blob

def half(x):return struct.unpack("<e",struct.pack("<e",x))[0]
def round_nearest_away(v):return math.floor(v+.5) if v>=0 else math.ceil(v-.5)

def main():
    loc=load(ROOT/"LOCAL_RESULT.json")["cases"]; cases={}
    for tag in ("JA96","PY96"):
        d=PREV/"runs/precision-64-64"/(tag+"-early")
        x=blob(d/"ffn_moe_gate-0-input.bin","f");ids=blob(d/"ffn_moe_gate-0-ids.bin","i")
        assert (d/"ffn_moe_up-0-input.bin").read_bytes()==(d/"ffn_moe_gate-0-input.bin").read_bytes()
        hist=collections.Counter(ids);scores=[]
        for t in range(96):
            a=x[5120*t:5120*(t+1)];en=[0.,0.,0.]
            for j in range(0,5120,32):
                row=a[j:j+32];m=max(map(abs,row));d0=f32(m/127);h=half(d0);iv=f32(127/m) if m else 0;d1=f32(1/iv) if iv else 0
                for v in row:
                    a0=round_nearest_away(f32(v/d0))*h if m else 0
                    a1=round_nearest_away(f32(v*iv))*d1 if m else 0
                    en[0]+=(v-a0)**2;en[1]+=(v-a1)**2;en[2]+=(v-half(v))**2
            counts=[hist[e] for e in ids[6*t:6*t+6]]
            scores.append(sum(en[0 if n<=8 else 1 if n<64 else 2] for n in counts))
        cases[tag]=scores
    threshold=sorted(cases["JA96"],reverse=True);threshold=(threshold[9]+threshold[10])/2
    out={}
    for tag,scores in cases.items():
        vals=[x["SSE"] for x in loc[tag]["per_token"]];N=loc[tag]["marked"]
        order=sorted(range(96),key=lambda t:-scores[t]);selected=[i for i,s in enumerate(scores) if s>=threshold]
        out[tag]={"online_JA_cut_selected":len(selected),"online_JA_cut_SSE_share":sum(vals[i] for i in selected)/sum(vals),
                  "same_budget_selected":N,"same_budget_SSE_share":sum(vals[i] for i in order[:N])/sum(vals),
                  "input_residual_estimates":scores}
    save(ROOT/"INPUT_SIGNAL.json",{"calibrated_on":"JA96 only","cut":threshold,"cases":out,
        "scope":"Offline hypothesis: per-token sum of squared activation rounding residuals across the six real expert group sizes, not an implemented or timed trigger. FP16 also rounds weights."})
    print([(k,v["same_budget_selected"],round(v["same_budget_SSE_share"],4),v["online_JA_cut_selected"]) for k,v in out.items()])
if __name__=="__main__":main()
