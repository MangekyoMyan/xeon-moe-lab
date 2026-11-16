"""Apply the prospective mixed-run gate, preserving the analyzer's raw statistics."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
def main():
    spec_path=R/'incoming/xeon_moe_abs_next_2026-11-16/NEXT_RUN_SPEC.json'
    spec=json.loads(spec_path.read_text())['proposed_forward_screen']
    a=json.loads((R/'MIXED_ANALYSIS.json').read_text());assert a['passed'] and a['analysis_complete']
    rows=[]
    for label in ['overlay_within_process','cross_library']:
        for data in a['descriptive_summaries'][label]:
            if label=='cross_library' and data['overlay_logical_mode']!='S':continue
            case=data['case'];total_key='pooled_equal_work_total_ratio' if label=='overlay_within_process' else 'pooled_normalized_total_ratio'
            minimum=spec['E_C6_min_ratio_both_pooled_block_median_and_equal_work_total'] if case=='C6' else 1/(1+spec['E_C0_Cplan_max_candidate_time_increase_both_metrics'])
            median=data['pooled_block_paired_median'];total=data[total_key]
            rows.append({'comparison':label,'case':case,'minimum_ratio':minimum,'median_ratio':median,'total_time_ratio':total,'median_time_change_percent':100*(1/median-1),'total_time_change_percent':100*(1/total-1),'median_pass':median>=minimum,'total_pass':total>=minimum})
    c6=next(x for x in a['descriptive_summaries']['overlay_within_process'] if x['case']=='C6')
    wins=sum(x>1 for x in c6['process_total_time_ratios'])
    result={'spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),'source':'Prospective NEXT_RUN_SPEC gate accepted in POLICY_AND_PLAN_JA before timing','rows':rows,'overlay_C6_process_total_ratios':c6['process_total_time_ratios'],'overlay_C6_improving_processes':wins,'numeric_screen_passed':all(x['median_pass'] and x['total_pass'] for x in rows) and wins>=3,'production_adoption':False,'note':'A numerical pass permits considering a limited IQ2 gate/up whole-model A/B only; inspect process/placebo scatter before deciding. No automatic rerun or adoption.'}
    p=R/'SCREEN_DECISION.json'
    with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
