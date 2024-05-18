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

# Relaxations
INFIX = ["full","chain","chain_int"]
# CoTs
STANDARD = ["", "basic", "intermediate"]


### SCRIPT FOR GENERATING INSTANCES ###

def generate_instances(num=0, overwrite_previous="False", num_steps=2, token_length=1, instance_type="instances"):
    if instance_type not in ["instances", "examples"]: raise ValueError("What are you trying to generate? I can only do \"instances\" and \"examples\".")
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

def evaluate(response_data,**kwargs):
    # TODO factor this out, as it's reused
    if not utils.includes_sub_dict(response_data, kwargs): return {}
    if response_data["relaxation"] in INFIX:
        if response_data["cot"] in STANDARD:
            try: llm_claim = response_data["response"].split("[Answer]")[1].strip().lower()
            except: llm_claim = response_data["response"].strip().lower()
            return evaluate_full_raw(response_data, llm_claim, response_data["relaxation"])
        else: raise NotImplementedError(f"CoT '{response_data['cot']}' does not have evaluation implemented!")
    else: raise NotImplementedError(f"Relaxation {response_data['relaxation']} does not have evaluation implemented!")

### HELPER FUNCTIONS ###

## BASIC PROMPT UTILITIES ##
def generate_instructions(problem_relaxation):
    if problem_relaxation == "full":
        return "After the [Answer] tag, you may only respond with a single number representing the final value of the calculation. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    elif problem_relaxation == "chain":
        return "After each thought, provide an intermediate answer in the form of a single number, labeled by the [Intermediate Answer n] tag, where n is replaced with the number of the intermediate answer. Do not put anything other than the intermediate answer number between the intermediate answer tag and the next thought tag (e.g. [Thought n+1]). When you are done thinking and have outputted all the requisite intermediate answers, put the [Answer] tag. After the [Answer] tag, you may only respond with a single number representing the final value of the calculation. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    elif problem_relaxation == "chain_int":
        return "After each thought, provide the intermediate expression which that thought evaluates, labeled by the [Intermediate Expression n] tag, where n is replaced with the number of the intermediate expression. Remember, an intermediate expression may only contain two numbers. YOU MUST PROVIDE ALL INTERMEDIATE EXPRESSIONS. DO NOT SKIP ANY. After each intermediate expression, provide the intermediate answer it simplifies to in the form of a single number, labeled by the [Intermediate Answer n] tag, where n is replaced with the number of the intermediate answer. Do not put anything other than the intermediate answer between the intermediate answer tag and the next thought tag (e.g. [Thought n+1]). When you are done thinking and have outputted all the requisite intermediate answers and expressions, put the [Answer] tag. After the [Answer] tag, you may only respond with a single number representing the final value of the calculation. Do not include anything else after that tag. The [Answer] tag must precede the final answer."
    else: raise NotImplementedError

def generate_query(instance, relaxation):
    instance_data = instance["raw_instance"]
    query = "[Question]\nSimplify the following expression into a single number: "
    if relaxation in INFIX:
        query+= raw_eq_to_str(instance_data)
    return query

def raw_eq_to_str(eq):
    streq = ""
    for n,op in eq:
        if op == "1" or streq == "": streq = n
        else: streq = f"{n} {op} ({streq})"
    return streq

## EVALUATION UTILITIES ##

def evaluate_full_raw(response_data, llm_claim, relaxation):
    evaluation = {}
    evaluation["ground_truth"] = generate_correct_evaluation(response_data, relaxation)
    evaluation["input_length"] = domain.token_l(response_data["prompt"])
    evaluation["output_length"] = domain.token_l(response_data["response"])
    llm_claim_cleaned = llm_claim.strip()
    if re.search(r"\s", llm_claim):
        try: 
            llm_claim_cleaned = llm_claim_cleaned.split("[answer]")[1].strip()
        except:
            evaluation["well_formed_response"] = False
            print(f"Ill-formed response! Can't parse:")
            print(response_data["response"])
            llm_claim_cleaned = "".join(llm_claim_cleaned.split()) #last ditch, but will still print about it
    else: evaluation["well_formed_response"] = True

    if relaxation == "chain" or "chain_int":
        evaluation["ground_truth_chain"] = generate_correct_chain(response_data["raw_instance"])
        assert list(evaluation["ground_truth_chain"].values())[-1] == evaluation["ground_truth"]
        chain_claim_answers = parse_intermediates(response_data["response"])
        evaluation["chain_length"] = len(chain_claim_answers)
        evaluation["normalized_chain_length"] = len(chain_claim_answers)/(response_data["steps_to_solve"]-1)
        if evaluation["chain_length"] != len(chain_claim_answers): evaluation["chain_correct"] = False
        else: 
            evaluation["chain_errors"] = check_chain_errors(evaluation["ground_truth_chain"], chain_claim_answers)
            evaluation["chain_correct"] = evaluation["chain_errors"] == 0
            # off-by-one error fixing: num of steps is actually num of numbers, not number of operations
            evaluation["smooth_chain_correct"] = 1-min(1,evaluation["chain_errors"]/(response_data["steps_to_solve"]-1))
    if relaxation == "chain_int":
        # extract the expressions the llm gave
        chain_claim_expressions = parse_intermediates(response_data["response"],"expression")
        # check if 1) these are correctly chosen -- output the number of expression errors and normalized
        # TODO
        # evaluation["expression_errors"] = check_expression_errors(chain_claim_expressions, chain_claim_answers, response_data["raw_instance"])
        # evaluation["expression_correct"] = evaluation["expression_errors"] == 0
        # evaluation["smooth_expression_correct"] = 1-min(1,evaluation["expression_errors"]/(response_data["steps_to_solve"]-1))
        #          2) these evaluated correctly  -- output the number of eval errors and normalized
        if check_eval_errors(chain_claim_answers, chain_claim_expressions) == "BARHRST": 
            print(f"cca: {chain_claim_answers}\ncce: {chain_claim_expressions}")
            raise ValueError(response_data["response"])
        evaluation["eval_errors"] = check_eval_errors(chain_claim_answers, chain_claim_expressions)
        evaluation["eval_correct"] = evaluation["eval_errors"] == 0
        evaluation["smooth_eval_correct"] = 1-min(1,evaluation["eval_errors"]/evaluation["chain_length"])
        evaluation["digit_eval_errors"] = check_digit_eval_errors(chain_claim_answers, chain_claim_expressions, response_data["number_of_digits"])
        evaluation["digit_eval_correct"] = evaluation["digit_eval_errors"] == 0
        # evaluation["smooth_digit_eval_correct"] = 1-evaluation["digit_eval_errors"]/evaluation["chain_length"]
    evaluation["correct"] = llm_claim_cleaned == evaluation["ground_truth"]
    return evaluation

def check_eval_errors(chain_claim_answers, chain_claim_expressions):
    eval_errors = 0
    for k in chain_claim_expressions.keys():
        # print(chain_claim_expressions.keys())
        # print(chain_claim_answers.keys())
        if k not in chain_claim_answers.keys():
            eval_errors+=1
            continue
        if "undefined" in chain_claim_answers[k].lower() or "division by zero" in chain_claim_answers[k].lower():
            eval_errors+=1
            continue
        chain_claim_expression = chain_claim_expressions[k].split("=")[0]
        # print(f'{chain_claim_expression.count("(")} vs {chain_claim_expression.count(")")}')
        if chain_claim_expression.count("(") > chain_claim_expression.count(")"):
            print("ADDING PARENTHESES")
            chain_claim_expression += ")"*(chain_claim_expression.count("(")-chain_claim_expression.count(")"))
            print(chain_claim_expression)
        try: chain_claim_expression = simplify(chain_claim_expression)
        except:
            print("expression error")
            print(k)
            print(chain_claim_expressions[k])
            print(chain_claim_expression)
            return "BARHRST"
        try: chain_claim_answer = simplify(chain_claim_answers[k].split("=")[-1])
        except:
            print("answer error")
            print(k)
            print(chain_claim_answers[k]) 
            return "BARHRST"
        eval_errors+= simplify(chain_claim_expression) != simplify(chain_claim_answer)
    return eval_errors
def check_digit_eval_errors(chain_claim_answers, chain_claim_expressions, num_digits):
    allowed_numbers = list(range(1,10**num_digits))
    eval_errors = 0
    for k in chain_claim_expressions.keys():
        if chain_claim_expressions[k].count("(") != chain_claim_expressions[k].count(")"):
            continue
        try: int(chain_claim_answers[k])
        except: continue
        eval_errors+= simplify(chain_claim_expressions[k].split("=")[0]) != simplify(chain_claim_answers[k].split("=")[-1]) if int(chain_claim_answers[k]) in allowed_numbers and simplify(int(chain_claim_answers[k])) == simplify(chain_claim_answers[k]) else 0
    return eval_errors
def check_expression_errors(chain_claim_expressions, chain_claim_answers, raw_instance):
    # for each expression, it takes some portion of the instance, and some portion of the previous answers
    # The output should be the number of errors made (where one error counts per basic (aka 2-number) expression,
    #        and so an error including three numbers will count for 2 errors)
    print(raw_instance)
    print(chain_claim_expressions)
    print(chain_claim_answers)

    # for each portion, we want to see if it is 
    #       1) using the previous answer in the new expression
    #       2) using some next portion of the full expression in the next answer
    #           - a few ways that can go wrong: skips parts, ??? TODO check the actual data for ones that didn't work
    #           - skip part, do some operation twice
    
    raise NotImplementedError("Implement expression error checking!!")

def check_chain_errors(ground_truth_chain, chain_claim):
    # generates a correct chain, then checks whether at each step it's the same
    lg = len(ground_truth_chain)
    lc = len(chain_claim)
    min_l = min(lg, lc)
    num_errors = abs(lg-lc)
    for n in range(1, min_l+1):
        num_errors+= int(ground_truth_chain[str(n)]!=chain_claim[str(n)])
    return num_errors
def generate_correct_chain(raw_eq):
    prev = raw_eq[0]
    chain = {}
    for n in range(0,len(raw_eq)-1):
        inner = raw_eq_to_str([prev,raw_eq[n+1]])
        prev = [str(simplify(inner)),'1']
        chain[str(n+1)] = str(prev[0]).strip()
    return chain
def parse_intermediates(response_text, key = "answer"):
    response_text = response_text.lower()
    answers_processing = response_text.split(f"\n[intermediate {key} ")
    intermediate_answers = {}
    for n in range(1,len(answers_processing)):
        answer_num = answers_processing[n].split("]")[0]
        intermediate_answers[answer_num] = answers_processing[n].split(']')[1].split('[')[0].strip()
    return intermediate_answers

## COT PROMPT UTILITIES ##
def generate_thoughts(instance, cot_type, relaxation):
    if not cot_type: return ""
    if cot_type == "basic": return generate_thoughts_basic(instance)
    if cot_type == "intermediate": return generate_thoughts_intermediate(instance)
    else: raise NotImplementedError

def generate_correct_evaluation(instance, problem_relaxation):
    raw_eq = instance["raw_instance"]
    if problem_relaxation in INFIX:
        return str(simplify(raw_eq_to_str(raw_eq)))
    else: raise NotImplementedError

## SPECIFIC COT UTILITIES ##

def generate_thoughts_basic(instance):
    raw_eq = instance["raw_instance"]
    cot = "We simplify one set of parentheses at a time, starting from the inside.\n"
    #TODO finish this
    current_eq = raw_eq_to_str(raw_eq)
    prev = raw_eq[0]
    for n in range(0,len(raw_eq)-1):
        inner = raw_eq_to_str([prev,raw_eq[n+1]])
        prev = [str(simplify(inner)),'1']
        cot+= f"[Thought {n+1}]\n"
        if n == 0:
            cot+= f"The current form of the expression is {current_eq}.\n"
        else:
            cot+= f"We plug in the previous intermediate answer into the previous expression to simplify it by one step. "
            cot+= f"This gives the expression {current_eq}.\n"
        cot+= f"The innermost expression is {inner}, which simplifies to {prev[0]}.\n"
        if n == len(raw_eq)-2: cot+= "The expression cannot be simplified further, so this will also be the final answer.\n"
        cot+= f"[Intermediate Answer {n+1}]\n"
        cot+= f"{prev[0]}\n"
        current_eq = raw_eq_to_str([prev]+raw_eq[n+2:])
    return cot

def generate_thoughts_intermediate(instance):
    raw_eq = instance["raw_instance"]
    cot = "We simplify one set of parentheses at a time, starting from the inside.\n"
    #TODO finish this
    current_eq = raw_eq_to_str(raw_eq)
    prev = raw_eq[0]
    for n in range(0,len(raw_eq)-1):
        inner = raw_eq_to_str([prev,raw_eq[n+1]])
        prev = [str(simplify(inner)),'1']
        cot+= f"[Thought {n+1}]\n"
        if n == 0:
            cot+= f"The current form of the expression is {current_eq}.\n"
        else:
            cot+= f"We plug in the previous intermediate answer into the previous expression to simplify it by one step. "
            cot+= f"This gives the expression {current_eq}.\n"
        cot+= f"The innermost expression is {inner}, which simplifies to {prev[0]}.\n"
        if n == len(raw_eq)-2: cot+= "The expression cannot be simplified further, so this will also be the final answer.\n"
        cot+= f"[Intermediate Expression {n+1}]\n"
        cot+= f"{inner}\n"
        cot+= f"[Intermediate Answer {n+1}]\n"
        cot+= f"{prev[0]}\n"
        current_eq = raw_eq_to_str([prev]+raw_eq[n+2:])
    return cot