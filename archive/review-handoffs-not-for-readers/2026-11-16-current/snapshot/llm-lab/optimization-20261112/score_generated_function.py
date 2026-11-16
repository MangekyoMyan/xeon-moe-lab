"""Bounded, unprivileged scoring of allowlisted pure-list generated functions.

Input and output are synthetic. No imports, arbitrary attributes, indirect calls,
or module-level expressions are permitted in the generated program.
"""
import ast
import json
import resource
import sys

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))

ALLOWED = {
    'Module', 'FunctionDef', 'arguments', 'arg', 'Return', 'Assign', 'AugAssign',
    'AnnAssign', 'Expr', 'If', 'While', 'For', 'Break', 'Continue', 'Pass',
    'Name', 'Load', 'Store', 'Constant', 'List', 'Tuple', 'Dict', 'Subscript',
    'Slice', 'BinOp', 'UnaryOp', 'BoolOp', 'Compare', 'Call', 'keyword',
    'Attribute', 'IfExp', 'ListComp', 'comprehension', 'Add', 'Sub', 'Mult',
    'FloorDiv', 'Mod', 'USub', 'UAdd', 'Not', 'And', 'Or', 'Eq', 'NotEq',
    'Lt', 'LtE', 'Gt', 'GtE', 'In', 'NotIn', 'Is', 'IsNot',
}
BUILTINS = {'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
            'min': min, 'max': max, 'list': list, 'int': int, 'tuple': tuple,
            'sorted': sorted, 'sum': sum, 'abs': abs}


def score(data):
    name = data['name']
    code = data['content'].strip()
    if code.startswith('```python\n') and code.endswith('```'):
        code = code[len('```python\n'):-3].strip()
    elif code.startswith('```\n') and code.endswith('```'):
        code = code[4:-3].strip()
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Expected exactly one function')
    fn = tree.body[0]
    if fn.name != name or fn.decorator_list or fn.args.defaults or fn.args.kw_defaults:
        raise ValueError('Unexpected function definition')
    for node in ast.walk(tree):
        if type(node).__name__ not in ALLOWED:
            raise ValueError('Unsupported syntax: ' + type(node).__name__)
        if isinstance(node, (ast.Name, ast.arg)):
            symbol = node.id if isinstance(node, ast.Name) else node.arg
            if symbol.startswith('_'):
                raise ValueError('Private names are forbidden')
        if isinstance(node, ast.Attribute) and node.attr not in {'append', 'extend', 'copy', 'count'}:
            raise ValueError('Unsupported attribute')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in BUILTINS:
                    raise ValueError('Unsupported or recursive call')
                if name == 'merge_sorted' and node.func.id == 'sorted':
                    raise ValueError('Merge task requires linear merging')
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError('Indirect calls are forbidden')
    namespace = {'__builtins__': BUILTINS}
    exec(compile(tree, '<synthetic-response>', 'exec'), namespace)
    func = namespace[name]
    if name == 'merge_sorted':
        cases = [([], []), ([1], []), ([], [-2]), ([1, 1, 3], [1, 2, 3]),
                 ([-9, -2, 0], [-8, -2, 7]), ([1, 2], [3, 4]), ([4, 5], [1, 2, 3])]
        for seed in range(16):
            cases.append((sorted(((i * 7 + seed) % 19) - 9 for i in range(seed)),
                          sorted(((i * 11 + seed) % 23) - 11 for i in range(19 - seed))))
        for a, b in cases:
            aa, bb = a.copy(), b.copy()
            if func(aa, bb) != sorted(a + b) or aa != a or bb != b:
                return {'correct': False, 'reason': 'Output or input-preservation failure', 'cases': len(cases)}
    elif name == 'run_lengths':
        cases = [[], [1], [1, 1, 2, 2, 1], [-3, -3, 0, 2, 2], [0, 1, 0, 1], [5] * 20]
        cases += [[((i * i + seed) % 5) - 2 for i in range(31)] for seed in range(12)]
        for a in cases:
            expected = []
            for value in a:
                if expected and expected[-1][0] == value:
                    expected[-1][1] += 1
                else:
                    expected.append([value, 1])
            copy = a.copy()
            if func(copy) != expected or copy != a:
                return {'correct': False, 'reason': 'Output or input-preservation failure', 'cases': len(cases)}
    else:
        raise ValueError('Unknown task')
    return {'correct': True, 'cases': len(cases)}


try:
    print(json.dumps(score(json.load(sys.stdin))))
except Exception as exc:
    print(json.dumps({'correct': False, 'reason': type(exc).__name__ + ': ' + str(exc)}))
