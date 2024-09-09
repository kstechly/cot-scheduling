from collections import Counter
from sympy import simplify #type: ignore
from functools import cache

ops = ["+", "-", "/", "*"]
digits = list(range(1,100))
prev = {k: 1 for k in digits}

num_runs = 100

def one_run(prev, digits, ops):
    output = {k: 0 for k in digits}
    for a in prev.keys():
        for b in digits:
            for op in ops:
                s = simpl(a,op,b)
                if s in digits:
                    output[s]+=prev[a]
    return output     

@cache
def simpl(a,op,b):
    eq = f"{a}{op}{b}"
    return simplify(eq)

print("Run 1")
output = one_run(prev,digits, ops)
normalized = {k: output[k]/sum(output.values()) for k in output.keys()}
print(normalized)

for n in range(1,num_runs):
    print(f"Run {n+1}")
    output = one_run(output,digits, ops)
    normalized = {k: output[k]/sum(output.values()) for k in output.keys()}
    print(normalized)

