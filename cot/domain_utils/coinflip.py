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
        inst_names = []
        for _ in range(0, num_steps):
            flip = random.choice([True, False])
            inst_names.append((generate_names(token_length),flip))
        # print(f'{instance_num}: {inst_names}')
        instances[instance_num] = {"raw_instance": inst_names, "uniform_token_length": token_length, "steps_to_solve":num_steps}
    
    print(f'Writing {num} {instance_type} to json. (num steps: {num_steps}, token_length: {token_length})')
    # TODO This could be faster if it were only done every so often
    utils.write_json(DOMAIN_NAME, instances, instance_type)

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
    if response["relaxation"] in ["full", "explained", "turn"]:
        if response["cot"] == "":
            llm_claim = response["response"].strip().lower()
            return evaluate_full_raw(response, llm_claim)
        elif response["cot"] == "wei" or response["cot"] == "wei_incorrect":
            #TODO implement computational graph evaluation
            try: llm_claim = response["response"].split("[Answer]")[1].strip().lower()
            except: 
                if response["response"] in ["yes", "no"]:
                    # Sometimes the LLM just outputs the answer and refuses to use the thought format.
                    llm_claim = response["response"]
                    evaluation = {"refused_to_think": True}
                    evaluation.update(evaluate_full_raw(response, llm_claim))
                    return evaluation
                else: llm_claim = None
            return evaluate_full_raw(response, llm_claim)
        else: raise NotImplementedError(f"CoT {response['cot']} does not have evaluation implemented!")
    else: raise NotImplementedError(f"Relaxation {response['relaxation']} does not have evaluation implemented!")

### HELPER FUNCTIONS ###

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    if problem_relaxation in ["full", "turn"] :
        return "After the [Answer] tag, you may respond only with 'yes' or 'no'. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    elif problem_relaxation == "explained":
        instructions = "The coin flipping problem is a test of ability to reason about parity. A coin begins heads up, and a sequence of (potentially repeated) people flip or do not flip it. If someone flips a coin that is heads up, they flip it to tails. In this problem, contrary to the commonsense meaning of the phrase, \"flipping\" a coin does not involve tossing it into the air and catching it, but instead means turning it over onto its other side. The puzzle is to figure out if the coin is back to heads up after some number of flips.\n"
        instructions+= "After the [Answer] tag, you may respond only with 'yes' or 'no'. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
        return instructions
    else: raise NotImplementedError

def generate_query(instance, problem_relaxation):
    flips = "turns the coin over" if problem_relaxation == "turn" else "flips the coin"
    flip  = "turn the coin over"  if problem_relaxation == "turn" else "flip the coin"
    instance_data = instance["raw_instance"]
    query  = f'[QUESTION]\n'
    query += f"A coin is heads up. "
    for name in instance_data:
        query += name[0]
        query += " {flips}. " if name[1] else " does not {flip}. "
    query += "Is the coin still heads up?"
    return query

## EVALUATION UTILITIES ##

def evaluate_full_raw(response, llm_claim):
    evaluation = {}
    legal_answers = ["yes", "no"]
    raw = response["raw_instance"]
    sum_flips = [int(x[1]) for x in raw]
    heads_ground_truth = bool((sum(sum_flips)+1)%2)
    evaluation["ground_truth"] = heads_ground_truth
    evaluation["input_length"] = token_l(response["prompt"])
    evaluation["output_length"] = token_l(response["response"])

    if llm_claim in legal_answers:
        llm_claim_bool = llm_claim == "yes"
        evaluation["llm_claim"] = llm_claim_bool
        evaluation["well_formed_response"] = True
        evaluation["correct"] = heads_ground_truth if llm_claim_bool else not heads_ground_truth
    else: 
        evaluation["well_formed_response"] = False
        print(f"Ill-formed response! Can't parse:")
        print(response["response"])
    return evaluation

## COT PROMPT UTILITIES ##
def generate_thoughts(example_instance, cot_type):
    if not cot_type: return ""
    elif cot_type == "wei": return generate_thoughts_wei(example_instance)
    elif cot_type == "wei_incorrect": return generate_thoughts_wei_incorrect(example_instance)
    else: raise NotImplementedError

def generate_correct_evaluation(example_instance, problem_relaxation):
    if problem_relaxation in ["full", "explained", "turn"]:
        flips = [x[1] for x in example_instance["raw_instance"]]
        heads = bool(sum(flips)%2)
        if heads: return "yes"
        else: return "no"
    else: raise NotImplementedError

## SPECIFIC COT UTILITIES ##
#TODO 

def generate_thoughts_wei(example_instance):
    # Replicated from Wei, et. al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    #
    # Some original examples from the paper (Table 23 on Page 37):
        # Q: Q: A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?
        # A: The coin was flipped by Ka and Sherrie. So the coin was flipped 2 times, which is an even number. The coin
        # started heads up, so after an even number of flips, it will still be heads up. So the answer is yes.
        #
        # Q: A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up?
        # A: The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started
        # heads up, so after an odd number of flips, it will be tails up. So the answer is no.
        #
        # Q: A coin is heads up. Inga does not flip the coin. Elanor does not flip the coin. Is the coin still heads up?
        # A: The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, and it was not
        # flipped, so it is still heads up. So the answer is yes.
    raw = example_instance["raw_instance"]
    flipper_list = [x[0] for x in raw if x[1]==1]
    flippers = " and ".join(flipper_list) if flipper_list else "no one"
    flip_times = len(flipper_list)
    heads = not bool(flip_times%2)
    zero_case = "and it was not flipped, so it is still heads up"
    reasoning = f"so after an {'even' if heads else 'odd'} number of flips, it will {'still be heads up' if heads else 'be tails up'}" if flip_times else zero_case
    answer = 'yes' if heads else 'no'
    cot = f"The coin was flipped by {flippers}. So the coin was flipped {flip_times} times. The coin started heads up, {reasoning}. So the answer is {answer}."
    return cot

def generate_thoughts_wei_incorrect(example_instance):
    # An inccorrect, less informative version of the original
    return "The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, and it was not flipped, so it is still heads up. So the answer is yes."