"""Record parent-reviewed answer evidence and independent constraint checks."""
from pathlib import Path
import hashlib,json
from check_schedule import check_schedule
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    responses=json.loads((R/'ANSWER_VERIFICATION.json').read_text())['responses'];out=[]
    for a in responses:
        p=R/a['rendered_file'];text=p.read_text();ci=a['case_id'];m=a['mode']
        c=dict(case_id=ci,mode=m,answer_file=a['rendered_file'],sha256=sha(p),generated_count=a['generated_count'],stop_reason=a['stop_reason'],response_complete=not a['truncated'],parent_read_entire_answer=True)
        if ci.startswith('text_plan'):
            # All four tables personally read; transcribed independently of the
            # reference. A different feasible optimal schedule is also allowed.
            times={'S':['09:00','09:20'],'I':['09:20','09:50'],'E':['09:50','10:30'],'R':['09:50','10:15'],'F':['10:30','10:45']}
            tables=[]
            for task,ts in times.items():
                matches=[dict(line=i+1,text=line) for i,line in enumerate(text.splitlines()) if line.startswith('| '+task+' ') and all(t in line for t in ts)]
                assert len(matches)==1,(ci,m,task,matches)
                tables+=matches
            c.update(schedule=check_schedule(times),transcribed_times=times,table_evidence=tables,
                proof_present=True,proof_reading='Precedence chain S20 -> I30 -> E40 -> F15 yields earliest10:45; R25 overlaps E without personnel conflicts. This does not require literal105 in the answer.',
                conflicting_schedule_found=False,semantic_requirements_observed_met=True,
                observed_response_language='Chinese' if ci.endswith('_ja') else 'English',
                language_note='JA prompt is Japanese but has no explicit language command. Both modes answered Chinese; usability caveat, not a new P-only defect.' if ci.endswith('_ja') else None)
            if a['truncated']:
                c['qualification']='Original1024-ID response retained, final repeated explanation cuts mid-sentence. Complete valid table and optimality argument precede cut. Do not call this a completed response or assume missing text.'
            else:c['qualification']='Observed task requirements met in a completed answer; no general-quality inference.'
        else:
            code=p.with_suffix('.py');behavior=json.loads(p.with_name(m+'-behavior.json').read_text())
            assert behavior['passed'] and behavior['source_sha256']==sha(code)
            c.update(code_file=str(code.relative_to(R)),code_sha256=sha(code),behavior=behavior,
                structural_review=dict(fixed_capacity_array=True,oldest_index_and_count=True,no_element_shifting=True,full_failure_before_mutation=True,empty_failure_before_mutation=True,capacity_range_checked=True),
                explanation_review=dict(wraparound_explained=True,given_example_correct=True,failed_operation_state_unchanged_explained=True),
                semantic_requirements_observed_met=True,qualification='Manually reviewed original code, executed unchanged class in separate non-root resource-bounded process. Only this task/these answers; not a general code-generation benchmark.')
        out.append(c)
    result=dict(responses=out,independent_review='ANSWER_REVIEW_TERRA.txt',scope='Three validation variants, two task families; one free answer per case/mode. Not six independent tasks.',general_quality_proven=False,P_only_clear_task_regression_observed=False,adopted=False)
    dest=R/'ANSWER_GRADING.json';assert not dest.exists();dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(case=c['case_id'],mode=c['mode'],requirements_met=c['semantic_requirements_observed_met'],complete=c['response_complete']) for c in out]))
if __name__=='__main__':main()
