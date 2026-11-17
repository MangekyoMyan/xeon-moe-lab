"""Run a parent-reviewed RingQueue class in a separate, bounded Python process.

Not a general Python sandbox. Only a manually reviewed, hash-pinned class is
accepted; AST restrictions and minimal builtins are additional containment.
No provided reference code is executed. Public task semantics, not reference
wording, determine the behavior checks. Success-push return values are ignored.
"""
import argparse,ast,builtins,collections,hashlib,json,random,resource,sys,typing,__future__
from pathlib import Path

def reviewed_class(source):
    tree=ast.parse(source);classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RingQueue'];assert len(classes)==1
    cls=classes[0];assert not cls.bases and not cls.keywords and not cls.decorator_list,'review unsupported class construction separately'
    allowed_names={'__init__','__len__','__repr__','__slots__','__name__'}
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            assert isinstance(n,ast.ImportFrom) and n.level==0 and n.module in ('typing','__future__')
            allowed={'Optional','List','Tuple','Union','Any'} if n.module=='typing' else {'annotations'}
            assert all(a.name in allowed and not a.asname for a in n.names)
    for n in ast.walk(cls):
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.AsyncFunctionDef,ast.Await)):raise ValueError('unsupported class AST')
        if isinstance(n,ast.Name) and n.id.startswith('__'):assert n.id in allowed_names
        if isinstance(n,ast.Attribute):
            assert not n.attr.startswith('__'),'no introspection'
            root=n.value
            while isinstance(root,(ast.Attribute,ast.Subscript)):root=root.value
            assert isinstance(root,ast.Name) and root.id=='self','only instance attributes in reviewed class'
        if isinstance(n,ast.FunctionDef):
            assert not n.decorator_list,'review decorators separately'
            if n.name.startswith('__'):assert n.name in allowed_names
    # Import/type annotations are supplied explicitly. Omit surrounding examples;
    # the generated class AST itself is compiled unchanged.
    future=any(isinstance(n,ast.ImportFrom) and n.module=='__future__' and any(x.name=='annotations' for x in n.names) for n in tree.body)
    code=compile(ast.Module(body=[cls],type_ignores=[]),'<reviewed-model-class>','exec',flags=__future__.annotations.compiler_flag if future else 0)
    names=['__build_class__','object','int','bool','list','tuple','dict','len','range','enumerate','isinstance','type','min','max','abs','ValueError','IndexError','TypeError','RuntimeError','Exception']
    ns={'__builtins__':{n:getattr(builtins,n) for n in names},'__name__':'model_answer',
        'Optional':typing.Optional,'List':typing.List,'Tuple':typing.Tuple,'Union':typing.Union,'Any':typing.Any}
    exec(code,ns)
    return ns['RingQueue'],ast.dump(cls,include_attributes=False)

def behavior(cls):
    counts=collections.Counter();failures=[]
    def fail(where,why):failures.append(dict(where=where,error=str(why)))
    for cap in [0,-1,1025,-(2**63),2**63-1]:
        try:q=cls(cap)
        except Exception as e:counts['invalid_capacity_rejected']+=1;continue
        fail('capacity '+str(cap),'invalid capacity did not raise; inspect any documented explicit rejection separately')
    # None is the conventional Optional[int] empty representation. Alternative
    # documented representations require a separate, explicit adapter, never a
    # silent change to the generated class.
    cases=[(1,2000),(2,4000),(3,4000),(1024,6000)];values=[0,-3,4,9,12,-(2**63),2**63-1,-2,-2]
    for cap,repeats in cases:
        try:
            q=cls(cap);oracle=collections.deque();assert q.size()==0
            def push(value):
                old=len(oracle);result=q.try_push(value)
                if old==cap:assert result is False;counts['full_rejection']+=1
                else:oracle.append(value);counts['successful_push']+=1
                assert q.size()==len(oracle)
            def pop():
                result=q.try_pop()
                if oracle:
                    expected=oracle.popleft();assert isinstance(result,int) and not isinstance(result,bool) and result==expected;counts['successful_pop']+=1
                else:assert result is None;counts['empty_optional']+=1
                assert q.size()==len(oracle)
            pop()
            for i in range(cap):push(values[i%len(values)])
            for i in range(3):push(777)
            for i in range(cap+2):pop()
            rng=random.Random(9401+cap)
            for i in range(repeats):
                if rng.randrange(5)<3:push(values[rng.randrange(len(values))])
                else:pop()
            while oracle:pop()
            pop();counts['valid_capacity_cases']+=1
        except Exception as e:fail('capacity '+str(cap),type(e).__name__+': '+str(e))
    try:
        q=cls(2);q.try_push(4);q.try_push(9);assert q.try_push(12) is False and q.size()==2
        assert q.try_pop()==4;q.try_push(-3);assert q.try_pop()==9 and q.try_pop()==-3 and q.try_pop() is None and q.size()==0;counts['given_example']=1
    except Exception as e:fail('given_example',type(e).__name__+': '+str(e))
    return dict(passed=not failures,behavior_only=True,counts=dict(counts),failures=failures,
        success_push_return_checked=False,ring_direction_prescribed=False,
        structural_ring_and_explanation='Parent must assess generated source and answer separately',empty_adapter='None; alternative documented Optional representation requires separate adapter')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('--expected-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(8,8));resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2));resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    data=a.source.read_bytes();assert hashlib.sha256(data).hexdigest()==a.expected_sha256 and not a.output.exists()
    cls,dump=reviewed_class(data.decode('utf-8'));result=behavior(cls);result['source_sha256']=a.expected_sha256;result['class_AST_sha256']=hashlib.sha256(dump.encode()).hexdigest();result['parent_review_required']=True
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
