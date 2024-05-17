from fire import Fire #type: ignore
import random
import re
from sympy import simplify #type: ignore
from functools import cache
from itertools import product

from domain_utils import domain
import utils

#TODO stop telling the name in a billion places... And all this nonsense with __init__ hardwiring needs to go
DOMAIN_NAME = "pemdas"

SEED = "13"
RANDOM_FILE_LOC = f"random3"

INFIX = ["full"]


### SCRIPT FOR GENERATING INSTANCES ###

def generate_instances(num=0, overwrite_previous="False", num_steps=2, token_length=1, instance_type="instances"):
    if instance_type not in ["instances", "examples"]: raise ValueError("What are you trying to generate? I can only do instances and examples.")
    if overwrite_previous:
        random.seed(a=SEED)
    else: 
        try: utils.load_pickle(RANDOM_FILE_LOC)
        except: raise FileNotFoundError(f"Could not find pickled random state. If you want to start anew, pass --overwrite_previous=True")

    instances = utils.read_json(DOMAIN_NAME, overwrite_previous, instance_type, verbose=True)
    if overwrite_previous: instances = {}
    prev_num = len(instances.keys())
    for instance_num in range(prev_num,prev_num+num):
        raw_eq = generate_raw_eq(token_length, num_steps)
        instances[instance_num] = {"raw_instance": raw_eq, "number_of_digits": token_length, "steps_to_solve":num_steps}
    
    print(f'Writing {num} {instance_type} to json. (num steps: {num_steps}, token_length: {token_length})')
    # TODO This could be faster if it were only done every so often
    utils.write_json(DOMAIN_NAME, instances, instance_type)

def generate_raw_eq(digit_num, num_steps):
    # We exclude 0, as it is an absorbing state. Multiple compositions of multiplication etc will lead to ~50% of all answers being zero. 
    ans = random.randint(1,10**digit_num-1)
    eq = [[ans, "1"]]
    for _ in range(1, num_steps):
        ans, next_num, next_op = generate_legal_next_step(digit_num, ans)
        eq.append([next_num, next_op])
    print(raw_eq_to_str(eq))
    return eq
def generate_legal_next_step(digit_num, ans):
    allowed_nums = list(range(1,10**digit_num))
    ops = ["+","-","*","/"]
    cartesian = list(product(allowed_nums, ops))
    random.shuffle(cartesian)
    for proposed_num, proposed_op in cartesian:
        proposed_ans, legal_flag = simpl(proposed_num,proposed_op,ans,digit_num)
        if legal_flag: break
    utils.save_pickle(random.getstate(), RANDOM_FILE_LOC)
    return proposed_ans, proposed_num, proposed_op
@cache
def simpl(a,op,b,digit_num):
    allowed_nums = list(range(1,10**digit_num))
    all_s = map(lambda x: simplify(x),allowed_nums)
    ans = simplify(f"{a}{op}{b}")
    return ans, ans in all_s

### REQUIRED FUNCTIONS ###

def generate(*args, **kwargs):
    return domain.generator(DOMAIN_NAME, generate_instructions, generate_query, generate_thoughts, generate_correct_evaluation)(*args, **kwargs)

def evaluate(response,**kwargs):
    # TODO factor this out, as it's reused
    if not utils.includes_sub_dict(response, kwargs): return {}
    if response["relaxation"] in INFIX:
        if response["cot"] == "":
            try: llm_claim = response["response"].split("[Answer]")[1].strip().lower()
            except: llm_claim = response["response"].strip().lower()
            return evaluate_full_raw(response, llm_claim, response["relaxation"])
        else: raise NotImplementedError(f"CoT '{response['cot']}' does not have evaluation implemented!")
    else: raise NotImplementedError(f"Relaxation {response['relaxation']} does not have evaluation implemented!")

### HELPER FUNCTIONS ###

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    if problem_relaxation == "full":
        return "After the [Answer] tag, you may only respond with a single number representing the final value of the calculation. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    else: raise NotImplementedError

def generate_query(instance, relaxation):
    instance_data = instance["raw_instance"]
    query = "[Question]\nSimplify the following expression into a single number: "
    if relaxation in INFIX:
        query+= raw_eq_to_str(instance_data)
    print(query)
    return query

def raw_eq_to_str(eq):
    streq = ""
    for n,op in eq:
        if op == "1": streq = n
        else:         streq = f"{n} {op} ({streq})"
    return streq

## EVALUATION UTILITIES ##

def evaluate_full_raw(response, llm_claim, relaxation):
    evaluation = {}
    evaluation["ground_truth"] = generate_correct_evaluation(response, relaxation)
    evaluation["input_length"] = domain.token_l(response["prompt"])
    evaluation["output_length"] = domain.token_l(response["response"])
    llm_claim_cleaned = llm_claim.strip()
    if re.search(r"\s", llm_claim):
        try: 
            llm_claim_cleaned = llm_claim_cleaned.split("[answer]")[1].strip()
        except:
            evaluation["well_formed_response"] = False
            print(f"Ill-formed response! Can't parse:")
            print(response["response"])
            llm_claim_cleaned = "".join(llm_claim_cleaned.split()) #last ditch, but will still print about it
    else: evaluation["well_formed_response"] = True
    
    evaluation["correct"] = llm_claim_cleaned == evaluation["ground_truth"]
    return evaluation

## COT PROMPT UTILITIES ##
def generate_thoughts(instance, cot_type):
    if not cot_type: return ""
    else: raise NotImplementedError

def generate_correct_evaluation(instance, problem_relaxation):
    raw_eq = instance["raw_instance"]
    if problem_relaxation in INFIX:
        return str(simplify(raw_eq_to_str(raw_eq)))
    else: raise NotImplementedError

## SPECIFIC COT UTILITIES ##