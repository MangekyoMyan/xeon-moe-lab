"""Check times manually transcribed from a saved answer; no LLM grading call."""
def check_schedule(times):
    people={'S':{'Aya'},'I':{'Aya','Ben'},'E':{'Ben','Chen'},'R':{'Aya'},'F':{'Aya','Ben','Chen'}}
    durations={'S':20,'I':30,'E':40,'R':25,'F':15};edges=[('S','I'),('I','E'),('I','R'),('E','F'),('R','F')]
    assert set(times)==set(people)
    def minute(t):
        h,m=map(int,t.split(':'));assert 0<=h<24 and 0<=m<60;return 60*h+m
    tt={k:tuple(map(minute,v)) for k,v in times.items()};checks=[]
    for k,(a,b) in tt.items():checks.append(dict(test='duration_and_availability_'+k,passed=b-a==durations[k] and 540<=a<b<=660))
    for a,b in edges:checks.append(dict(test=a+'_before_'+b,passed=tt[a][1]<=tt[b][0]))
    names=list(people)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if people[a]&people[b]:checks.append(dict(test='no_shared_person_overlap_'+a+'_'+b,passed=max(tt[a][0],tt[b][0])>=min(tt[a][1],tt[b][1])))
    checks.append(dict(test='earliest_finish_10_45',passed=max(b for a,b in tt.values())==645))
    return dict(passed=all(c['passed'] for c in checks),checks=checks,critical_path=['S','I','E','F'],lower_bound_minutes=105,
        scope='Feasibility/optimal finish of transcribed intervals. Parent separately checks explanation, omissions, contradictions and truncation. Assigned people are inherited from the prompt; spelling their names again is not required.')
