import json
from fire import Fire #type: ignore
import tiktoken # type: ignore
import random

from domain_utils import domain
import utils

#TODO stop telling the name in a billion places... And all this nonsense with __init__ hardwiring needs to go
DOMAIN_NAME = "coinflip"
EXAMPLE_DIRECTORY = f"data/examples/{DOMAIN_NAME}/"

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
    print(instances)
    prev_num = len(instances.keys())
    for instance_num in range(prev_num,prev_num+num):
        inst_names = []
        for _ in range(0, num_steps):
            flip = random.choice([True, False])
            inst_names.append((generate_names(overwrite_previous, token_length),flip))
        print(f'{instance_num}: {inst_names}')
        instances[instance_num] = {"raw_instance": inst_names}
    
    print(f'Writing to json.')
    utils.write_json(DOMAIN_NAME, instances, "instances")

def generate_names(overwrite_previous, token_length):
    names = load_all_names()

    enc = tiktoken.get_encoding("cl100k_base")
    nl  = lambda x: len(enc.encode(x))
    allowed_names = [n for n in names if nl(n)==token_length]
    if not len(allowed_names): raise ValueError(f"There are no names of token length {token_length} in the list.")

    name = random.choice(allowed_names)
    utils.save_pickle(random.getstate(), RANDOM_FILE_LOC)
    print(name)
    return name

def load_all_names():
    #TODO get a list of names
    return ["John", "Josh", "Alice", "Anne", "Margaret", "Zagathar", "Grzegorz", "Kolmogorov", "Neumann", "Fong", "Sipser", "Polya", "Nagel", "Godel", "Bach", "Yogi", "Adam", "Lars"]

### REQUIRED FUNCTIONS ###

def file_ending():
    return ".json"
def extraction_labels():
    return ['']

def generate(*args, **kwargs):
    return domain.generator(generate_instructions, generate_query, generate_thoughts, generate_correct_evaluation, EXAMPLE_DIRECTORY)(*args, **kwargs)

def evaluate(instance_text, response_trace, extraction_label="", backprompt_type="", cot_type=""):
    #TODO
    raise NotImplementedError

### HELPER FUNCTIONS ###

## DATA UTILITIES ##

def check_instance_info(instance_text, extraction_label):
    #TODO 
    raise NotImplementedError

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    if problem_relaxation == "full":
        return "Respond only with 'yes' or 'no'. Do not include anything else in your response."
    else: raise NotImplementedError

def generate_query(instance_data, extraction_label):
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
    # elif cot_type == "global": return generate_thoughts_global(example_instance)
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