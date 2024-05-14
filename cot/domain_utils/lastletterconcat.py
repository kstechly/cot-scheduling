from fire import Fire #type: ignore
import random
import re
from Levenshtein import distance #type: ignore
from textdistance import levenshtein #type: ignore

from domain_utils import domain
import utils

#TODO stop telling the name in a billion places... And all this nonsense with __init__ hardwiring needs to go
DOMAIN_NAME = "lastletterconcat"

SEED = "13"
RANDOM_FILE_LOC = f"random2"
WORDS_LOCATION = "names/ssa_names_data.json" # TODO: while this follows some available databases that only do names, it may be worth extending

FULL_RELAXATIONS = ["full", "info_dump", "dont_think"]
FOOM_RELAXATIONS = ["foom", "foom_clearer"]
VOWEL_RELAXATIONS = ["vowel"]

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
    try: llm_claim = response["response"].split("[Answer]")[1].strip().lower()
    except: llm_claim = response["response"].strip().lower()
    if response["relaxation"] in FULL_RELAXATIONS+FOOM_RELAXATIONS+VOWEL_RELAXATIONS:
        if response["cot"] in ["", "wei", "wei_incorrect", "first_letter_incorrect"] or response["magic"] in ["", " ", "Let's think step by step."]:
            return evaluate_raw(response, llm_claim, response["relaxation"])
        else: raise NotImplementedError(f"CoT '{response['cot']}' does not have evaluation implemented!")
    else: raise NotImplementedError(f"Relaxation {response['relaxation']} does not have evaluation implemented!")

### HELPER FUNCTIONS ###

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    basic_instructions = "After the [Answer] tag, you may only respond with a lowercase string of concatenated characters. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    if problem_relaxation == "info_dump":
        info_dump = "You are about to be presented with a last letter concatenation problem. A last letter concatenation problem is one in which you take an ordered list of words and extract the final letters of each and then concatenate them into a single string.\n"
        info_dump+= "Each word will be a sequence of characters. Each character has a position within the word. The word starts with the first letter, then continues with a second letter, and so forth. We can also index letters in a word from the other end. There is a last letter, a second-to-last letter, and so forth. For these problems, you only need to consider the last letter of each word.\n"
        info_dump+= "Once the requisite characters are extracted, a last letter concatenation problem requires that they are concatenated. Concatenation of characters is the operation of joining character strings end-to-end. It can be applied to strings of arbitary length, but in this problem, all that is required is to concatenate strings of length one. A string of length one is a single character. Once strings are concatenated, they result in a final string.\n"
        info_dump+= "The correct answer to a last letter concatenation problem is this final string.\n"
        return info_dump + basic_instructions
    elif problem_relaxation == "dont_think":
        return "While the examples show how to work through the problem, you should only output your answer. Do NOT output any thoughts. " + basic_instructions
    elif problem_relaxation in ["full"] + FOOM_RELAXATIONS + VOWEL_RELAXATIONS:
        return "For the purposes of these problems, a vowel is any one of the letters \"a\",\"e\",\"i\",\"o\", or \"u\", but NOT \"y\". "  + basic_instructions
    else: raise NotImplementedError

def generate_query(instance, problem_relaxation):
    # Mirroring instances at https://huggingface.co/datasets/ChilleD/LastLetterConcat/viewer/default/train except with spelling and grammar errors cleaned up.
    instance_data = instance["raw_instance"]
    query  = f'[Question]\n'
    if problem_relaxation in ['full', 'info_dump', 'dont_think'] + VOWEL_RELAXATIONS:
        query += f"Take the last {'vowel' if problem_relaxation in VOWEL_RELAXATIONS else 'letters'} of each word in \""
        query += " ".join(instance_data)
        query += "\" and concatenate them."
        return query
    elif problem_relaxation in FOOM_RELAXATIONS:
        query += f"Take the following words and produce a new string using them: \""
        query += " ".join(instance_data)
        query += "\". The first letter of your string should be the first letter of the first word, the second letter of your string should be the second letter of the second word, and so forth. "
        if problem_relaxation == "foom_clearer": query+="Only use one letter from each word. "
        query += "If you need the nth letter of the nth word, but that word is less than n letters long, then insert a 0 character instead."
        return query
    else: raise NotImplementedError(f"There is no {problem_relaxation} relaxation implemented!")

## EVALUATION UTILITIES ##

def evaluate_raw(response, llm_claim, relaxation="full"):
    evaluation = {}
    ground_truth = generate_correct_evaluation(response, relaxation)
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
    evaluation["set_correct"]        = set(llm_claim_cleaned)    == set(evaluation["ground_truth"])
    evaluation["bag_correct"]        = sorted(llm_claim_cleaned) == sorted(evaluation["ground_truth"])
    evaluation["correct"]            = llm_claim_cleaned         == evaluation["ground_truth"]
    zero_free_llm_claim    = "".join(llm_claim_cleaned.split("0"))
    zero_free_ground_truth = "".join(evaluation["ground_truth"].split("0"))
    evaluation["0_free_set_correct"] = set(zero_free_llm_claim)  == set(zero_free_ground_truth)
    evaluation["0_free_correct"]     = zero_free_llm_claim       == zero_free_ground_truth
    return evaluation

def token_distance(a, b, m=0):
    ta = domain.get_tokens(a)
    tb = domain.get_tokens(b)
    if m: return min(levenshtein.distance(ta,tb),m)
    else: return levenshtein.distance(ta, tb)

## COT PROMPT UTILITIES ##
def generate_thoughts(example_instance, cot_type, problem_relaxation):
    if not cot_type: return ""
    elif cot_type == "wei": return generate_thoughts_wei(example_instance, problem_relaxation)
    elif cot_type == "wei_incorrect": return generate_thoughts_wei_incorrect(example_instance)
    elif cot_type == "first_letter_incorrect": return generate_thoughts_first_letter_incorrect(example_instance)
    else: raise NotImplementedError

def generate_correct_evaluation(instance, problem_relaxation):
    word_list = instance["raw_instance"]
    if problem_relaxation in FULL_RELAXATIONS:
        return "".join([x[-1] for x in word_list]).lower()
    if problem_relaxation in FOOM_RELAXATIONS:
        return "".join([word_list[n].ljust(n+1,'0')[n] for n in range(0,len(word_list))]).lower()
    if problem_relaxation in VOWEL_RELAXATIONS:
        return vowel_answer(word_list)
    else: raise NotImplementedError(f"I don't know how to evaluate problems in relaxation class {problem_relaxation}")

def vowel_answer(word_list):
    return "".join([lastvowel(x) for x in word_list]).lower()
def lastvowel(word):
    vowels = "aeiou"
    for c in reversed(word.lower()):
        if c in vowels: return c
    print(f"{word} has no vowels from the set {vowels} by the way.")
    return "" # It's just empty 
    raise ValueError(f"There are no vowels in the word {word}! So I can't find the last vowel. I think vowels are {vowels}.")


## SPECIFIC COT UTILITIES ##

def generate_thoughts_wei(example_instance, problem_relaxation):
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
    if problem_relaxation in FULL_RELAXATIONS:
        answer = "".join([word[-1] for word in word_list])
        per_word = [f'The last letter of \"{word}\" is {word[-1]}.' for word in word_list]
    elif problem_relaxation in FOOM_RELAXATIONS:
        nth_list = [word_list[n].ljust(n+1,'0')[n] for n in range(0,len(word_list))]
        answer = "".join(nth_list).lower()
        per_word = [nth_gen(n, word_list[n], nth_list[n]) for n in range(0,len(word_list))]
    elif problem_relaxation in VOWEL_RELAXATIONS:
        answer = vowel_answer(word_list)
        per_word = [f'The last vowel of \"{word}\" is {lastvowel(word)}.' for word in word_list]
    cot = " ".join(per_word)
    cot+= f' Concatenating them is \"{answer}\". The answer is {answer}.' 
    return cot

def nth(n):
    if   n==1: return "1st"
    elif n==2: return "2nd"
    elif n==3: return "3rd"
    elif n>20: raise ValueError("I don't know how to write 21st, so I'm going to throw a fit. Update lastletterconcat.py") #TODO
    else:      return f"{n}th" #Technically fails for 21
def nth_gen(n, word, ans):
    if ans == "0":
        return f"\"{word}\" does not have an {nth(n)} letter, as it is too short, so we replace it with \"0\"."
    else: return f"The {nth(n+1)} letter of \"{word}\" is {ans}."

def generate_thoughts_wei_incorrect(example_instance):
    return "The last letter of \"Bill\" is l. Concatenating them is \"l\". The answer is l."

def generate_thoughts_first_letter_incorrect(example_instance):
    return "The last letter of \"Bill\" is B. The last letter of \"Gates\" is G. Concatenating them is \"bg\". The answer is bg."