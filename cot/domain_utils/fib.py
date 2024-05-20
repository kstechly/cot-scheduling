from fire import Fire #type: ignore
import random
import re
from itertools import product

from domain_utils import domain
import utils

#TODO stop telling the name in a billion places... And all this nonsense with __init__ hardwiring needs to go
DOMAIN_NAME = "fib"

FULL_RELAXATIONS = ["full"]
TOOL_RELAXATIONS = ["python","tool"]

### SCRIPT FOR GENERATING INSTANCES ###

def generate_instances(num=0, overwrite_previous="False", depth=1, modulo=64, instance_type="instances", **kwargs):
    if instance_type not in ["instances", "examples"]: raise ValueError("What are you trying to generate? I can only do instances and examples.")
    allowed_numbers = list(range(1,modulo))
    cartesian = list(product(allowed_numbers, allowed_numbers))
    random.shuffle(cartesian)

    instances = utils.read_json(DOMAIN_NAME, overwrite_previous, instance_type, verbose=True)
    if overwrite_previous: instances = {}
    prev_num = len(instances.keys())
    for instance_num in range(prev_num,prev_num+num):
        input_pair = []
        pairs = [instance["raw_instance"] for instance in instances.values() if instance["mod"] == modulo and instance["depth"] == depth]
        for pair in cartesian:
            if pair not in pairs:
                input_pair = pair
                break
        print(f"{input_pair}, mod: {modulo}, depth: {depth}")
        if input_pair: instances[instance_num] = {"raw_instance": input_pair, "mod": modulo, "depth":depth}
        else: 
            print("out of options")
            break
    pairs = [instance["raw_instance"] for instance in instances.values() if instance["mod"] == modulo and instance["depth"] == depth]
    assert len(pairs) == len(set(pairs))
    
    print(f'Writing {num} {instance_type} to json. (depth: {depth}, modulo: {modulo})')
    utils.write_json(DOMAIN_NAME, instances, instance_type)

### REQUIRED FUNCTIONS ###

def generate(*args, **kwargs):
    return domain.generator(DOMAIN_NAME, generate_instructions, generate_query, generate_thoughts, generate_correct_evaluation)(*args, **kwargs)

def evaluate(response,**kwargs):
    # TODO factor this out, as it's reused
    evaluation = {}
    if not utils.includes_sub_dict(response, kwargs): return {}
    if response["relaxation"] in FULL_RELAXATIONS:
        if response["cot"] == "":
            try: llm_claim = response["response"].split("[Answer]")[1].strip().lower()
            except: llm_claim = response["response"].strip().lower()
            return evaluate_full_raw(response, llm_claim, response["relaxation"])
        else: raise NotImplementedError(f"CoT '{response['cot']}' does not have evaluation implemented!")
    if response["relaxation"] in TOOL_RELAXATIONS:
        #TODO this may be bad bc of case sensitivity...
        # try: llm_claim = response["response"].split("[Answer]")[1].strip().lower()
        # except:
        llm_claim = response["response"].strip().lower()
        return evaluate_tool_use(response, llm_claim, response["relaxation"])
    else: raise NotImplementedError(f"Relaxation {response['relaxation']} does not have evaluation implemented!")

### HELPER FUNCTIONS ###

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    if problem_relaxation in FULL_RELAXATIONS:
        return "The fibonacci function takes two inputs x_n and x_{n+1} and outputs x_{n+2}, where x_{n+2} := x_n + x_{n+1}. When applied recursively, it continues this pattern, e.g. x_{n+3} := x_{n+1} + x_{n+2} = x_{n+1} + (x_n + x_{n+1}). You will use this function in your calculations.\nAfter the [Answer] tag, you may only respond with a single number. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    elif problem_relaxation in ["python"]:
        return "The fibonacci function takes two inputs x_n and x_{n+1} and outputs x_{n+2}, where x_{n+2} := x_n + x_{n+1}. When applied recursively, it continues this pattern, e.g. x_{n+3} := x_{n+1} + x_{n+2} = x_{n+1} + (x_n + x_{n+1}). You will need to know this function to solve these problems.\nAfter the [Answer] tag, put a new line and then only respond with a Python script. Do not include anything other than the program after that tag. The [Answer] tag must precede the final answer. Ensure that the script can be run by the standard Python interpreter. An external program will run eval() on your script, so make sure that it outputs the answer to the query!"
    elif problem_relaxation in ["tool"]:
        return "The fibonacci function takes two inputs x_n and x_{n+1} and outputs x_{n+2}, where x_{n+2} := x_n + x_{n+1}. When applied recursively, it continues this pattern, e.g. x_{n+3} := x_{n+1} + x_{n+2} = x_{n+1} + (x_n + x_{n+1}). You will need to know this function to solve these problems.\nWe introduce the FIB tool. To use it, put \{FIB(a,b,d)\} anywhere in your output. Any FIB calls will be replaced downstream by d recursive calls to a fibbonaci function tool given starting inputs of a and b.\nAfter the [Answer] tag, you may only respond with your final answer, which can be either the final number or a tool call. The [Answer] tag must precede the final answer."
    else: raise NotImplementedError(f"Relaxation \"{problem_relaxation}\" is not implemented!")

def generate_query(instance, problem_relaxation):
    instance_data = instance["raw_instance"]
    depth = instance["depth"]
    modulo = instance["mod"]
    query  = f'[Question]\n'
    query += f"What is the result of applying the fibonacci function starting from input {instance_data[0]} and {instance_data[1]} recursively {depth} times? Return an answer mod {modulo}."
    if problem_relaxation in ["python"]:
        query+= f"\n\nDo not solve the above problem directly! Instead, output a Python program which, when run, will solve it on its own."
    return query

## EVALUATION UTILITIES ##

def evaluate_full_raw(response, llm_claim, problem_relaxation):
    evaluation = {}
    evaluation["ground_truth"] = generate_correct_evaluation(response, problem_relaxation)
    evaluation["input_length"] = domain.token_l(response["prompt"])
    evaluation["output_length"] = domain.token_l(response["response"])
    llm_claim_cleaned = llm_claim.strip().lower()
    if re.search(r"\s", llm_claim):
        try: 
            llm_claim_cleaned = llm_claim_cleaned.split("[answer]")[1].strip()
        except:
            try: llm_claim_cleaned = llm_claim_cleaned.split("answer")[1].strip()
            except: evaluation["well_formed_response"] = False
            # print(f"Ill-formed response! Can't parse:")
            # print(response["response"])
            llm_claim_cleaned = "".join(llm_claim_cleaned.split())
            llm_claim_cleaned = "".join(llm_claim_cleaned.split("-"))
    else: evaluation["well_formed_response"] = True
    # if llm_claim_cleaned != evaluation["ground_truth"] and problem_relaxation == "no_space": print(f"claimed: {llm_claim_cleaned}, truth: {evaluation['ground_truth']}")
    # if response["relaxation"] == "tool":
        # print("---------------")
        # print(response["raw_instance"],response["depth"])
        # print(fib(response["raw_instance"][0],response["raw_instance"][1],response["depth"]))
        # print(f'===RESPONSE===\n{response["response"]}\n')
        # print("===END RESPONSE===")
        # print(llm_claim_cleaned)
        # print(f'ground truth: {evaluation["ground_truth"]}')
    evaluation["correct"] = str(llm_claim_cleaned) == str(evaluation["ground_truth"])
    return evaluation

def evaluate_tool_use(response, llm_claim, problem_relaxation):
    # first clean the llm claim
    #TODO
    llm_program = llm_claim
    if problem_relaxation == "python": raise NotImplementedError("Implement Python evaluation in a safe way.")#program_output = eval(llm_program)
    elif problem_relaxation == "tool":
        # print(llm_program)
        llm_program = "".join(llm_program.split("\\"))
        try: 
            a,b,d = llm_program.split('{fib(')[1].split(')}')[0].split(',')
            # print("curly")
            program_output = fib(int(a),int(b),int(d))%response["mod"]
        except: 
            try:
                a,b,d = llm_program.split('[fib(')[1].split(')]')[0].split(',')
                # print("square")
                # print(a,b,d)
                program_output = fib(int(a),int(b),int(d))%response["mod"]
                # print("output")
                # print(program_output)
            except: program_output = llm_program
    return evaluate_full_raw(response, str(program_output), problem_relaxation)

## COT PROMPT UTILITIES ##
def generate_thoughts(example_instance, cot_type):
    if not cot_type: return ""
    else: raise NotImplementedError

def generate_correct_evaluation(example_instance, problem_relaxation):
    nums = example_instance["raw_instance"]
    depth = example_instance["depth"]
    modulo = example_instance["mod"]
    if problem_relaxation in FULL_RELAXATIONS+TOOL_RELAXATIONS:
        return fib(nums[0], nums[1], depth)%modulo

    else: raise NotImplementedError

def fib(a,b,depth):
    # print(f"fib {a},{b},{depth}")
    if depth == 0: return b
    else: 
        # print(f"a: {a}, b: {b}, s: {s}, d: {depth}")
        return fib(b,a+b,depth-1)
    raise NotImplementedError

## SPECIFIC COT UTILITIES ##