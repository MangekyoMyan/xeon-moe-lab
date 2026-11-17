#!/usr/bin/env python3
"""Validate original corpus and its eight reference programs, not inference code.

Runs only Python code and C++ code authored in this review's corpus. It never
executes supplied repository scripts, model libraries, an API or a GPU program.
"""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

BASE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()


def main():
    corpus=json.loads((BASE/'PILOT_CASES_JA_EN_CODE.json').read_text())
    manifest=json.loads((BASE/'PILOT_STAGE_MANIFEST.json').read_text())
    assert sha((BASE/'PILOT_CASES_JA_EN_CODE.json').read_bytes())==manifest['corpus_sha256']
    families={f['family_id']:f for f in corpus['families']}
    cases={v['case_id']:v for f in families.values() for v in f['variants']}
    assert len(families)==12 and len(cases)==20
    splits=Counter(f['split'] for f in families.values() for v in f['variants'])
    assert dict(splits)=={'train':10,'validation':5,'holdout':5}
    assert not corpus['calibration_cases_reused']
    assert not any(x.startswith('screen_') or x=='wiring_fixture' for x in cases)
    for f in families.values():
        assert f['split_frozen_before_routes']
        assert f['translations_ports_windows_turns_must_remain_in_this_split']
        for v in f['variants']:
            assert not v['tokenization_performed']
            assert v['prefix_token_ids'] is None and v['continuation_token_ids'] is None
            for field in ('prompt','reference'):
                b=v[field+'_utf8'].encode()
                assert sha(b)==v[field+'_sha256']
                assert len(b)==v[field+'_utf8_bytes']
                assert len(v[field+'_utf8'])==v[field+'_codepoints']
                assert v[field+'_utf8'].endswith('\n')
            if 'reference_code_utf8' in v:
                assert sha(v['reference_code_utf8'].encode())==v['reference_code_sha256']
                assert v['reference_starts_with_code']
                language=v['reference_code_language']
                assert v['reference_utf8'].startswith('```'+language+'\n'+v['reference_code_utf8'])
    staged=[x for stage in manifest['stages'] for x in stage['cases']]
    assert len(staged)==len(set(staged))==20 and set(staged)==set(cases)
    stages={s['stage']:s for s in manifest['stages']}
    for name in ('train_first5','train_remaining5','validation5','holdout5'):
        assert len(stages[name]['cases'])==5
    split_by_case={v['case_id']:f['split'] for f in families.values() for v in f['variants']}
    assert all(split_by_case[x]=='train' for name in ('train_first5','train_remaining5') for x in stages[name]['cases'])
    assert all(split_by_case[x]=='validation' for x in stages['validation5']['cases'])
    assert all(split_by_case[x]=='holdout' for x in stages['holdout5']['cases'])

    # Independent arithmetic and constraints for the prose references.
    facts=families['text_cache_budget']['common_semantic_contract']
    counts=Counter(e for req in facts['requests'] for e in req)
    for name,bank in facts['candidates'].items():
        size=sum(facts['sizes_MiB'][e] for e in bank)
        hits=sum(counts[e] for e in bank)
        assert size==facts['expected_sizes_MiB'][name]==40
        assert hits==facts['expected_hits'][name]
        assert sum(counts.values())-hits==facts['expected_misses'][name]
    volume=Fraction(12,10)*10+Fraction(8,10)*5+Fraction(15,10)*8
    assert volume==28 and 10+5+8==23
    assert volume*1000==28000 and 23*60==1380
    assert (volume-Fraction(276,10))/volume*100==Fraction(10,7)
    schedule=families['text_plan_constraints']['common_semantic_contract']
    def minutes(s):
        h,m=map(int,s.split(':'));return 60*h+m
    windows={k:tuple(map(minutes,v)) for k,v in schedule['expected_schedule'].items()}
    for task,detail in schedule['tasks'].items():
        start,end=windows[task]
        assert end-start==detail['minutes']
        assert minutes('09:00')<=start<end<=minutes('11:00')
        assert all(windows[p][1]<=start for p in detail['after'])
    for person in ('Aya','Ben','Chen'):
        assigned=sorted(windows[t] for t,d in schedule['tasks'].items() if person in d['people'])
        assert all(a[1]<=b[0] for a,b in zip(assigned,assigned[1:]))
    assert sum(schedule['tasks'][t]['minutes'] for t in schedule['critical_path'])==105
    assert windows['F'][1]-minutes('09:00')==105
    museum=families['text_document_summary']['common_semantic_contract']
    assert museum['sessions']*museum['seats_per_session']==museum['reservations']==72
    assert museum['attendees']+museum['no_shows']==72
    assert museum['reservations']+museum['unreserved_requests']==museum['distinct_requests']==90
    assert museum['survey_good']+museum['survey_neutral']==museum['survey_responses']==44
    assert museum['attendees']-museum['survey_responses']==20
    assert museum['sessions']*museum['staff_extra_minutes_per_session']==180
    config=families['mixed_config_change']['common_semantic_contract']
    assert len(config['backoff_ms'])==config['retry_count']==3
    assert config['maximum_attempts']==1+config['retry_count']==4
    assert config['timeout_ms']*4+sum(config['backoff_ms'])==config['all_timeout_budget_ms']==6700
    table=families['mixed_table_reconciliation']['common_semantic_contract']
    net={k:v-table['accepted_unique_returns'][k] for k,v in table['shipped_by_sku'].items()}
    assert net==table['expected_net_by_sku']=={'A':15,'B':6} and sum(net.values())==21
    assert table['reported_net_by_sku']['A']==17-2-2

    py_results=[]
    for fid in ('code_stable_grouping','code_interval_merge','code_bounded_queue','code_utf8_line_reader'):
        v=next(v for v in families[fid]['variants'] if v['reference_code_language']=='python')
        code=v['reference_code_utf8'];ast.parse(code)
        env={};exec(compile(code,v['case_id'],'exec'),env)
        checks=0
        if fid=='code_stable_grouping':
            examples=[([],[]),([('north',4),('',9),('north',-2),('east',5),('north',4)],
                               [('north',[4,-2,4]),('',[9]),('east',[5])]),
                      ([('東京',0),('x',-2),('東京',0)],[('東京',[0,0]),('x',[-2])]),
                      ([('a',1),('A',2),('a',3)],[('a',[1,3]),('A',[2])])]
            for rows,expected in examples:
                before=rows.copy();assert env['ordered_groups'](rows)==expected and rows==before;checks+=1
        elif fid=='code_interval_merge':
            examples=[([],[]),([(2,2)],[]),([(5,7),(1,3),(3,5),(10,10)],[(1,7)]),
                      ([(1,10),(2,3),(1,10),(12,15)],[(1,10),(12,15)]),
                      ([(-5,-3),(-2,0)], [(-5,-3),(-2,0)]),
                      ([(-(2**63),0),(0,2**63-1)],[(-(2**63),2**63-1)])]
            for rows,expected in examples:
                before=rows.copy();assert env['merge_intervals'](rows)==expected and rows==before;checks+=1
            try: env['merge_intervals']([(3,2)])
            except ValueError: checks+=1
            else: raise AssertionError('reversed interval accepted')
        elif fid=='code_bounded_queue':
            for bad in (0,1025,-1):
                try: env['RingQueue'](bad)
                except ValueError: checks+=1
                else: raise AssertionError('invalid capacity accepted')
            q=env['RingQueue'](2)
            assert q.try_pop() is None and q.size()==0;checks+=1
            assert q.try_push(4) and q.try_push(9) and not q.try_push(12) and q.size()==2;checks+=1
            assert q.try_pop()==4 and q.try_push(-3) and q.try_pop()==9 and q.try_pop()==-3;checks+=1
            assert q.try_pop() is None and q.size()==0;checks+=1
            q=env['RingQueue'](1)
            for value in (0,-9,0,2**63-1,-(2**63)):
                assert q.try_push(value) and not q.try_push(99) and q.try_pop()==value and q.try_pop() is None;checks+=1
        else:
            b=env['LineBuffer']();assert b.feed(b'')==[] and b.finish() is None;checks+=1
            assert b.feed(b'one\n\ntwo\r\n')==[b'one',b'',b'two\r'];checks+=1
            assert b.finish() is None;checks+=1
            assert b.feed(b'last')==[] and b.finish()==b'last' and b.finish() is None;checks+=1
            assert b.feed(b'\xe6\x9d')==[] and b.feed(b'\xb1\xe4')==[] and b.feed(b'\xba\xac\nx\n')==['東京'.encode(),b'x'];checks+=1
            assert b.feed(b'a\x00b\n')==[b'a\x00b'];checks+=1
        py_results.append(dict(case=v['case_id'],code_sha256=v['reference_code_sha256'],passed=True,semantic_checks=checks))

    cpp_tests={
    'code_stable_grouping':r'''
#include <cassert>
int main() {
    assert(ordered_groups({}).empty());
    std::vector<Row> rows={{"north",4},{"",9},{"north",-2},{"east",5},{"north",4}};
    const auto before=rows;
    assert((ordered_groups(rows)==std::vector<Group>{{"north",{4,-2,4}},{"",{9}},{"east",{5}}}));
    assert(rows==before);
    assert((ordered_groups({{u8"東京",0},{"x",-2},{u8"東京",0}})==std::vector<Group>{{u8"東京",{0,0}},{"x",{-2}}}));
    assert((ordered_groups({{"a",1},{"A",2},{"a",3}})==std::vector<Group>{{"a",{1,3}},{"A",{2}}}));
}
''',
    'code_interval_merge':r'''
#include <cassert>
#include <limits>
int main() {
    assert(merge_intervals({}).empty());
    assert(merge_intervals({{2,2}}).empty());
    std::vector<Interval> rows={{5,7},{1,3},{3,5},{10,10}};
    const auto before=rows;
    assert((merge_intervals(rows)==std::vector<Interval>{{1,7}}));
    assert(rows==before);
    assert((merge_intervals({{1,10},{2,3},{1,10},{12,15}})==std::vector<Interval>{{1,10},{12,15}}));
    assert((merge_intervals({{-5,-3},{-2,0}})==std::vector<Interval>{{-5,-3},{-2,0}}));
    const auto lo=std::numeric_limits<std::int64_t>::min(),hi=std::numeric_limits<std::int64_t>::max();
    assert((merge_intervals({{lo,0},{0,hi}})==std::vector<Interval>{{lo,hi}}));
    bool caught=false;try {merge_intervals({{3,2}});}catch(const std::invalid_argument&){caught=true;}assert(caught);
}
''',
    'code_bounded_queue':r'''
#include <cassert>
#include <limits>
int main() {
    for(std::int64_t bad:{std::int64_t(0),std::int64_t(1025),std::int64_t(-1)}){
        bool caught=false;try{RingQueue q(bad);}catch(const std::invalid_argument&){caught=true;}assert(caught);
    }
    RingQueue q(2);assert(!q.try_pop() && q.size()==0);
    assert(q.try_push(4) && q.try_push(9) && !q.try_push(12) && q.size()==2);
    assert(q.try_pop()==4 && q.try_push(-3) && q.try_pop()==9 && q.try_pop()==-3);
    assert(!q.try_pop() && q.size()==0);
    RingQueue one(1);
    const std::vector<std::int64_t> values={0,-9,0,std::numeric_limits<std::int64_t>::max(),std::numeric_limits<std::int64_t>::min()};
    for(auto value:values){assert(one.try_push(value) && !one.try_push(99) && one.try_pop()==value && !one.try_pop());}
}
''',
    'code_utf8_line_reader':r'''
#include <cassert>
int main() {
    LineBuffer b;assert(b.feed("").empty() && !b.finish());
    assert((b.feed("one\n\ntwo\r\n")==std::vector<std::string>{"one","","two\r"}));
    assert(!b.finish());
    assert(b.feed("last").empty() && b.finish()=="last" && !b.finish());
    assert(b.feed(std::string("\xE6\x9D",2)).empty());
    assert(b.feed(std::string("\xB1\xE4",2)).empty());
    assert((b.feed(std::string("\xBA\xAC\nx\n",5))==std::vector<std::string>{u8"東京","x"}));
    assert((b.feed(std::string("a\0b\n",4))==std::vector<std::string>{std::string("a\0b",3)}));
}
'''}
    cpp_results=[]
    with tempfile.TemporaryDirectory(prefix='pilot_reference_checks_') as td:
        temp=Path(td)
        for fid,test in cpp_tests.items():
            v=next(v for v in families[fid]['variants'] if v['reference_code_language']=='cpp')
            src=temp/(fid+'.cpp');exe=temp/fid
            src.write_text(v['reference_code_utf8']+'\n'+test)
            command=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic',str(src),'-o',str(exe)]
            build=subprocess.run(command,text=True,capture_output=True,timeout=30)
            assert build.returncode==0,build.stderr
            run=subprocess.run([str(exe)],text=True,capture_output=True,timeout=5)
            assert run.returncode==0,(run.returncode,run.stderr)
            cpp_results.append(dict(case=v['case_id'],code_sha256=v['reference_code_sha256'],
                compiled_standard='C++17',compiler='g++',build_exit=build.returncode,run_exit=run.returncode,
                warnings_as_errors=True,build_stdout=build.stdout,build_stderr=build.stderr,
                validation_scope='handwritten expected-output/boundary assertions; not an engine test'))
    result=dict(schema=1,corpus_sha256=manifest['corpus_sha256'],
        cases=20,families=12,splits=dict(splits),hashes_and_stage_partition_passed=True,
        prose_numeric_and_schedule_checks_passed=True,
        translation_semantic_review='Author checked paired facts/numbers/tasks manually; not an automated translation-quality proof',
        code_references_start_with_code=True,
        queue_input_scope='Valid integer types (bool excluded) and signed64-representable values are preconditions; capacity range is validated',
        python_reference_checks=py_results,cpp_reference_checks=cpp_results,
        model_tokenizer_used=False,model_token_counts_verified=False,
        inference_engine_or_gpu_or_api_executed=False,
        supplied_repository_script_executed=False,
        tested_programs='Only the eight original reference programs in this new corpus',
        minimum_prompt_codepoints=min(c['prompt_codepoints'] for c in cases.values()),
        minimum_reference_codepoints=min(c['reference_codepoints'] for c in cases.values()),
        limitation='Character/byte lengths do not certify 64 model tokens. Actual tokenizer and assistant boundary must be verified before any pilot capture.')
    (BASE/'PILOT_CORPUS_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
