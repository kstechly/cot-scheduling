from fire import Fire #type: ignore
import random
import re
from Levenshtein import distance #type: ignore
from textdistance import levenshtein

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
        if response["cot"] in ["wei", "wei_incorrect", "first_letter_incorrect"] or response["magic"] in ["", " ", "Let's think step by step."]:
            try: llm_claim = response["response"].split("[Answer]")[1].strip().lower()
            except: llm_claim = response["response"].strip().lower()
            return evaluate_full_raw(response, llm_claim)
        else: raise NotImplementedError(f"CoT '{response['cot']}' does not have evaluation implemented!")
    else: raise NotImplementedError(f"Relaxation {response['relaxation']} does not have evaluation implemented!")

### HELPER FUNCTIONS ###

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    if problem_relaxation == "full":
        return "After the [Answer] tag, you may only respond with a lowercase string of concatenated characters. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    else: raise NotImplementedError

def generate_query(instance, problem_relaxation):
    # Mirroring instances at https://huggingface.co/datasets/ChilleD/LastLetterConcat/viewer/default/train except with spelling and grammar errors cleaned up.
    instance_data = instance["raw_instance"]
    query  = f'[Question]\n'
    query += f"Take the last letters of each word in \""
    query += " ".join(instance_data)
    query += "\" and concatenate them."
    return query

## EVALUATION UTILITIES ##

def evaluate_full_raw(response, llm_claim):
    evaluation = {}
    ground_truth = generate_correct_evaluation(response, "full")
    evaluation["ground_truth"] = ground_truth
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
            print(f'I\'m marking this as {llm_claim_cleaned == evaluation["ground_truth"]}')
    else: evaluation["well_formed_response"] = True
    
    # TODO put the raw and do the division and filtering in pandas later
    evaluation["levenshtein_distance_normalized"] = distance(ground_truth, llm_claim_cleaned, score_cutoff=len(ground_truth)-1)/len(ground_truth)
    evaluation["token_distance_normalized"] = token_distance(ground_truth, llm_claim_cleaned, domain.token_l(ground_truth))/domain.token_l(ground_truth)
    # if distance(ground_truth, llm_claim_cleaned)>1: print(f'Way too big! {llm_claim_cleaned}')
    evaluation["correct"] = llm_claim_cleaned == evaluation["ground_truth"]
    return evaluation

def token_distance(a, b, m=0):
    ta = domain.get_tokens(a)
    tb = domain.get_tokens(b)
    if m: return min(levenshtein.distance(ta,tb),m)
    else: return levenshtein.distance(ta, tb)

## COT PROMPT UTILITIES ##
def generate_thoughts(example_instance, cot_type):
    if not cot_type: return ""
    elif cot_type == "wei": return generate_thoughts_wei(example_instance)
    elif cot_type == "wei_incorrect": return generate_thoughts_wei_incorrect(example_instance)
    elif cot_type == "first_letter_incorrect": return generate_thoughts_first_letter_incorrect(example_instance)
    else: raise NotImplementedError

def generate_correct_evaluation(example_instance, problem_relaxation):
    if problem_relaxation == "full":
        return "".join([x[-1] for x in example_instance["raw_instance"]]).lower()
    else: raise NotImplementedError

## SPECIFIC COT UTILITIES ##

def generate_thoughts_wei(example_instance):
    # Replicated from Wei, et. al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    #
    # Original examples from the paper (Table 22 on Page 36):
        # Q: Take the last letters of the words in "Elon Musk" and concatenate them.
        # A: The last letter of "Elon" is "n". The last letter of "Musk" is "k". Concatenating them is "nk". The answer is nk.
        #
        # Q: Take the last letters of the words in "Larry Page" and concatenate them.
        # A: The last letter of "Larry" is "y". The last letter of "Page" is "e". Concatenating them is "ye". The answer is ye.
        #
        # Q: Take the last letters of the words in "Sergey Brin" and concatenate them.
        # A: The last letter of "Sergey" is "y". The last letter of "Brin" is "n". Concatenating them is "yn". The answer is yn.
        # 
        # Q: Take the last letters of the words in "Bill Gates" and concatenate them.
        # A: The last letter of "Bill" is "l". The last letter of "Gates" is "s". Concatenating them is "ls". The answer is ls.
    word_list = example_instance["raw_instance"]
    answer = "".join([word[-1] for word in word_list])
    per_word = [f'The last letter of \"{word}\" is {word[-1]}.' for word in word_list]
    
    cot = " ".join(per_word)
    cot+= f' Concatenating them is \"{answer}\". The answer is {answer}.' 
    return cot

def generate_thoughts_wei_incorrect(example_instance):
    return "The last letter of \"Bill\" is l. Concatenating them is \"l\". The answer is l."

def generate_thoughts_first_letter_incorrect(example_instance):
    return "The last letter of \"Bill\" is B. The last letter of \"Gates\" is G. Concatenating them is \"bg\". The answer is bg."