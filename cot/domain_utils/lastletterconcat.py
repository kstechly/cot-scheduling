from fire import Fire #type: ignore
import random
import re

from domain_utils import domain
import utils

#TODO stop telling the name in a billion places... And all this nonsense with __init__ hardwiring needs to go
DOMAIN_NAME = "lastletterconcat"

SEED = "13"
RANDOM_FILE_LOC = f"random2"
WORDS_LOCATION = "names/ssa_names_data.json" # TODO: while this follows some available databases that only do names, it may be worth extending

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
            inst_names.append(generate_word(token_length, WORDS_LOCATION))
        # print(f'{instance_num}: {inst_names}')
        instances[instance_num] = {"raw_instance": inst_names, "uniform_token_length": token_length, "steps_to_solve":num_steps}
    
    print(f'Writing {num} {instance_type} to json. (num steps: {num_steps}, token_length: {token_length})')
    # TODO This could be faster if it were only done every so often
    utils.write_json(DOMAIN_NAME, instances, instance_type)

def generate_word(token_length, words_location):
    # Note that this is "coinflip" because it uses data in that directory. Not ideal! #TODO
    allowed_words = domain.get_allowed_words("coinflip", token_length, words_location)
    if not len(allowed_words): raise ValueError(f"There are no names of token length {token_length} in the list.")

    name = random.choice(allowed_words)
    utils.save_pickle(random.getstate(), RANDOM_FILE_LOC)
    return name

### REQUIRED FUNCTIONS ###

def generate(*args, **kwargs):
    return domain.generator(DOMAIN_NAME, generate_instructions, generate_query, generate_thoughts, generate_correct_evaluation)(*args, **kwargs)

def evaluate(response,**kwargs):
    # TODO factor this out, as it's reused
    evaluation = {}
    if not utils.includes_sub_dict(response, kwargs): return {}
    if response["relaxation"] == "full":
        if response["cot"] == "":
            llm_claim = response["response"].strip().lower()
            return evaluate_full_raw(response, llm_claim)
        else: raise NotImplementedError(f"CoT '{response['cot']}' does not have evaluation implemented!")
    else: raise NotImplementedError(f"Relaxation {response['relaxation']} does not have evaluation implemented!")

### HELPER FUNCTIONS ###

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    if problem_relaxation == "full":
        return "After the [Answer] tag, you may respond only a lowercase string of concatenated characters. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    else: raise NotImplementedError

def generate_query(instance):
    # Mirroring instances at https://huggingface.co/datasets/ChilleD/LastLetterConcat/viewer/default/train except with spelling and grammar errors cleaned up.
    instance_data = instance["raw_instance"]
    query  = f'[Question]\n'
    query += f"Take the last letters of each word in \""
    query += " ".join(instance_data)
    query += " and concatenate them."
    return query

## EVALUATION UTILITIES ##

def evaluate_full_raw(response, llm_claim):
    evaluation = {}
    evaluation["ground_truth"] = generate_correct_evaluation(response, "full")
    evaluation["input_length"] = domain.token_l(response["prompt"])
    evaluation["output_length"] = domain.token_l(response["response"])
    evaluation["correct"] = llm_claim == evaluation["ground_truth"]

    if re.search(r"\s", llm_claim):
        evaluation["well_formed_response"] = False
        print(f"Ill-formed response! Can't parse:")
        print(response["response"])
    else: evaluation["well_formed_response"] = True
    return evaluation

## COT PROMPT UTILITIES ##
def generate_thoughts(example_instance, cot_type):
    if not cot_type: return ""
    #TODO 
    else: raise NotImplementedError

def generate_correct_evaluation(example_instance, problem_relaxation):
    if problem_relaxation == "full":
        return "".join([x[-1] for x in example_instance["raw_instance"]]).lower()
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
    raw = [['Abe',False]]
    flipper_list = [x[0] for x in raw if x[1]==1]
    flippers = " and ".join(flipper_list) if flipper_list else "no one"
    flip_times = len(flipper_list)
    heads = not bool(flip_times%2)
    zero_case = "and it was not flipped, so it is still heads up"
    reasoning = f"so after an {'even' if heads else 'odd'} number of flips, it will {'still be heads up' if heads else 'be tails up'}" if flip_times else zero_case
    answer = 'yes' if heads else 'no'
    cot = f"The coin was flipped by {flippers}. So the coin was flipped {flip_times} times. The coin started heads up, {reasoning}. So the answer is {answer}."
    return cot