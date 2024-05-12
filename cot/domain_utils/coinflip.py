import json
from fire import Fire #type: ignore
import tiktoken # type: ignore
import random
from functools import cache

from domain_utils import domain
import utils

#TODO stop telling the name in a billion places... And all this nonsense with __init__ hardwiring needs to go
DOMAIN_NAME = "coinflip"

SEED = "13"
RANDOM_FILE_LOC = f"random"

### SCRIPT FOR GENERATING INSTANCES ###

def generate_instances(num=0, overwrite_previous="False", num_steps=2, token_length=1):
    if overwrite_previous:
        random.seed(a=SEED)
    else: 
        try: utils.load_pickle(RANDOM_FILE_LOC)
        except: raise FileNotFoundError(f"Could not find pickled random state. If you want to start anew, pass --overwrite_previous=True")
    
    instances = utils.read_json(DOMAIN_NAME, overwrite_previous, "instances", verbose=True)
    if overwrite_previous: instances = {}
    prev_num = len(instances.keys())
    for instance_num in range(prev_num,prev_num+num):
        inst_names = []
        for _ in range(0, num_steps):
            flip = random.choice([True, False])
            inst_names.append((generate_names(token_length),flip))
        # print(f'{instance_num}: {inst_names}')
        instances[instance_num] = {"raw_instance": inst_names, "uniform_token_length": token_length, "steps_to_solve":num_steps}
    
    print(f'Writing {num} instances to json. (num steps: {num_steps}, token_length: {token_length})')
    # TODO This could be faster if it were only done every so often
    utils.write_json(DOMAIN_NAME, instances, "instances")

def generate_names(token_length):
    allowed_names = get_allowed_names(token_length)
    if not len(allowed_names): raise ValueError(f"There are no names of token length {token_length} in the list.")

    name = random.choice(allowed_names)
    utils.save_pickle(random.getstate(), RANDOM_FILE_LOC)
    return name

@cache
def get_allowed_names(token_length):
    names = load_all_names()
    return [n for n in names if token_l(n)==token_length]
@cache
def token_l(x):
    enc = get_encoding()
    return len(enc.encode(x))
@cache
def get_encoding():
    return tiktoken.get_encoding("cl100k_base")

def load_all_names():
    return utils.read_json(DOMAIN_NAME, False, "instances", strange_subloc="names/ssa_names_data.json")
    # return ["John", "Josh", "Alice", "Anne", "Margaret", "Zagathar", "Grzegorz", "Kolmogorov", "Neumann", "Fong", "Sipser", "Polya", "Nagel", "Godel", "Bach", "Yogi", "Adam", "Lars"]

### REQUIRED FUNCTIONS ###

def generate(*args, **kwargs):
    return domain.generator(DOMAIN_NAME, generate_instructions, generate_query, generate_thoughts, generate_correct_evaluation)(*args, **kwargs)

def evaluate(response,**kwargs):
    evaluation = {}
    if not utils.includes_sub_dict(response, kwargs): return {}
    if response["relaxation"] == "full":
        legal_answers = ["yes", "no"]
        if response["cot"] == "":
            raw = response["raw_instance"]
            sum_flips = [int(x[1]) for x in raw]
            heads_ground_truth = bool((sum(sum_flips)+1)%2)
            evaluation["ground_truth"] = heads_ground_truth
            evaluation["input_length"] = token_l(response["prompt"])
            evaluation["output_length"] = token_l(response["response"])

            llm_claim = response["response"].strip().lower()
            if llm_claim in legal_answers:
                llm_claim_bool = llm_claim == "yes"
                evaluation["llm_claim"] = llm_claim_bool
                evaluation["well_formed_response"] = True
                evaluation["correct"] = heads_ground_truth if llm_claim_bool else not heads_ground_truth
            else: 
                evaluation["well_formed_response"] = False
                print(f"Ill-formed response! Can't parse {response}")
            return evaluation
        else: raise NotImplementedError(f"CoT {response['cot']} does not have evaluation implemented!")
    else: raise NotImplementedError(f"Relaxation {response['relaxation']} does not have evaluation implemented!")

### HELPER FUNCTIONS ###

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    if problem_relaxation == "full":
        return "Respond only with 'yes' or 'no'. Do not include anything else in your response."
    else: raise NotImplementedError

def generate_query(instance_data):
    query  = f'[QUESTION]\n'
    query += f"A coin is heads up. "
    for name in instance_data:
        query += name[0]
        query += " flips the coin. " if name[1] else " does not flip the coin. "
    query += "Is the coin still heads up?"
    print(query)
    return query

## COT PROMPT UTILITIES ##
def generate_thoughts(example_instance, cot_type):
    if not cot_type: return ""
    elif cot_type == "global": return generate_thoughts_global(example_instance)
    else: raise NotImplementedError

def generate_correct_evaluation(example_instance, extraction_label, problem_relaxation):
    if problem_relaxation == "full":
        flips = [x[1] for x in example_instance]
        #TODO check this
        print(sum(flips))
        heads = bool(sum(flips)%2)
        if heads: return "yes"
        else: return "no"
    else: raise NotImplementedError

## SPECIFIC COT UTILITIES ##
#TODO 

def generate_thoughts_global(example_instance):
    raise NotImplementedError