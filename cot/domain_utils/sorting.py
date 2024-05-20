from fire import Fire #type: ignore
import random
import re

from domain_utils import domain
import utils

#TODO stop telling the name in a billion places... And all this nonsense with __init__ hardwiring needs to go
DOMAIN_NAME = "sorting"

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

FULL_RELAXATIONS = ["full", "no_space"]
TOOL_RELAXATIONS = ["tool"]

### SCRIPT FOR GENERATING INSTANCES ###

def generate_instances(num=0, overwrite_previous="False", str_len=2, vocab_size=2, instance_type="instances", **kwargs):
    if instance_type not in ["instances", "examples"]: raise ValueError("What are you trying to generate? I can only do instances and examples.")

    # Note: this can easily break in all sorts of ways, esp with (total so far + num) > vocab_size**str_len
    # TODO fix pleaseeee
    instances = utils.read_json(DOMAIN_NAME, overwrite_previous, instance_type, verbose=True)
    if overwrite_previous: instances = {}
    prev_num = len(instances.keys())
    for instance_num in range(prev_num,prev_num+num):
        ordered_chars = []
        flattened_instances = flatten(instances)
        while not_new(ordered_chars, flattened_instances):
            ordered_chars = []
            for _ in range(0, str_len):
                ordered_chars.append(generate_character(vocab_size))
        print(ordered_chars)
        instances[instance_num] = {"raw_instance": ordered_chars, "vocab_size": vocab_size, "string_length":str_len}
    print(flattened_instances)
    
    print(f'Writing {num} {instance_type} to json. (string length: {str_len}, vocab size: {vocab_size})')
    assert len(flattened_instances) == len(set(flattened_instances))
    # TODO This could be faster if it were only done every so often
    utils.write_json(DOMAIN_NAME, instances, instance_type)

def flatten(instances):
    return ["".join(val["raw_instance"]) for val in instances.values()]

def not_new(ordered_chars, flattened_instances):
    if not ordered_chars: return True 
    return "".join(ordered_chars) in flattened_instances

def generate_character(vocab_size):
    if vocab_size > 26: raise ValueError("Can't have a vocab size bigger than the English alphabet, silly. That's impossible!")
    vocabulary = ALPHABET[:vocab_size]
    character = random.choice(vocabulary)
    return character


### REQUIRED FUNCTIONS ###

def generate(*args, **kwargs):
    return domain.generator(DOMAIN_NAME, generate_instructions, generate_query, generate_thoughts, generate_correct_evaluation)(*args, **kwargs)

def evaluate(response,**kwargs):
    # TODO factor this out, as it's reused
    evaluation = {}
    if not utils.includes_sub_dict(response, kwargs): return {}
    if response["relaxation"] in ["lucas"] + FULL_RELAXATIONS:
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
        return "After the [Answer] tag, you may only respond with a single sorted string. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    elif problem_relaxation == "lucas":
        return ""
    elif problem_relaxation in ["tool"]:
        return "We introduce the SORT tool. To use it, put \{SORT(s)\} anywhere in your output. Any SORT calls will be replaced downstream by a sorted version of the string s. For example, \{Sort(cabd)\} would be replaced with abcd.\nAfter the [Answer] tag, you may only respond with your final answer, which can be either the final sorted string or a tool call. The [Answer] tag must precede the final answer."
    else: raise NotImplementedError

def generate_query(instance, problem_relaxation):
    instance_data = instance["raw_instance"]
    query  = f'[Question]\n'
    query += f"Please sort the string  \""
    query += "".join(instance_data) if problem_relaxation in ["no_space","tool"] else " ".join(instance_data)
    query += "\"."
    return query

## EVALUATION UTILITIES ##

def evaluate_full_raw(response, llm_claim, problem_relaxation):
    evaluation = {}
    evaluation["ground_truth"] = generate_correct_evaluation(response, problem_relaxation)
    evaluation["input_length"] = domain.token_l(response["prompt"])
    evaluation["output_length"] = domain.token_l(response["response"])
    llm_claim_cleaned = llm_claim.strip()
    if re.search(r"\s", llm_claim):
        try: 
            llm_claim_cleaned = llm_claim_cleaned.split("[answer]")[1].strip()
        except:
            evaluation["well_formed_response"] = False
            # print(f"Ill-formed response! Can't parse:")
            # print(response["response"])
            llm_claim_cleaned = "".join(llm_claim_cleaned.split()) #last ditch, but will still print about it
            llm_claim_cleaned = "".join(llm_claim_cleaned.split(","))
            llm_claim_cleaned = "".join(llm_claim_cleaned.split("-"))
    else: evaluation["well_formed_response"] = True
    
    if llm_claim_cleaned != evaluation["ground_truth"] and problem_relaxation in ["no_space","tool"]: print(f"claimed: {llm_claim_cleaned}, truth: {evaluation['ground_truth']}")

    evaluation["set_correct"]        = set(llm_claim_cleaned)    == set(evaluation["ground_truth"])
    evaluation["bag_correct"]        = sorted(llm_claim_cleaned) == sorted(evaluation["ground_truth"])
    evaluation["correct"]            = llm_claim_cleaned         == evaluation["ground_truth"]
    return evaluation

def evaluate_tool_use(response, llm_claim, problem_relaxation):
    # first clean the llm claim
    #TODO
    llm_program = llm_claim
    if problem_relaxation == "python": raise NotImplementedError("Implement Python evaluation in a safe way.")#program_output = eval(llm_program)
    elif problem_relaxation == "tool":
        # print(llm_program)
        try: 
            x,y,op = llm_program.split('{calc(')[1].split(')}')[0].split(',')
            # print("curly")
            program_output = sort_function(int(x),int(y),int(op))%response["mod"]
        except: 
            try:
                x,y,op = llm_program.split('[calc(')[1].split(')]')[0].split(',')
                # print("square")
                # print(a,b,d)
                program_output = sort_function(int(x),int(y),int(op))%response["mod"]
                # print("output")
                # print(program_output)
            except: program_output = llm_program
    return evaluate_full_raw(response, str(program_output), problem_relaxation)

## COT PROMPT UTILITIES ##
def generate_thoughts(example_instance, cot_type):
    if not cot_type: return ""
    else: raise NotImplementedError

def generate_correct_evaluation(example_instance, problem_relaxation):
    if problem_relaxation in ["full", "lucas"]:
        return " ".join(sorted(example_instance["raw_instance"]))
    elif problem_relaxation in ["no_space", "tool"]:
        return "".join(sorted(example_instance["raw_instance"]))
    else: raise NotImplementedError

def sort_function(s):
    return sorted(s)


## SPECIFIC COT UTILITIES ##