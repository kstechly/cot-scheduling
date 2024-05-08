import os
import argparse
import time
import json
import time
from tqdm import tqdm
import openai
import tiktoken

import utils
import domain_utils
from domain_utils import *

MAX_GPT_RESPONSE_LENGTH = 8000
STOP_STATEMENT = "[ANSWER END]" # what we look for to end LLM response generation

SYSTEM_PROMPT = "You are a system that solves reasoning problems presented in text form."

input_costs_per_million = {"gpt-4": 30, "gpt-4-turbo-2024-04-09": 10, "gpt-3.5-turbo-0125": 0.5}
output_costs_per_million = {"gpt-4": 60, "gpt-4-turbo-2024-04-09": 30, "gpt-3.5-turbo-0125": 1.5}

def get_responses(llm, domain_name, specified_instances = [], overwrite_previous=False, verbose=False, temp=0, num_trials=0, **prompt_specification):
    domain = domain_utils.domains[domain_name]

    # Cost calculation setup
    input_cost = 0
    output_cost = 0
    million = 1000000
    if llm in input_costs_per_million.keys():
        input_cost = input_costs_per_million[llm]/million
        output_cost = input_costs_per_million[llm]/million
    cost = 0.0
    enc = tiktoken.get_encoding("cl100k_base")

    # Load prompts and possibly filter for specified_instances
    instances = utils.read_json(domain_name, overwrite_previous, "prompts", verbose)
    if specified_instances: working_instances = {num: instances[num] for num in instances.key() if num in specified_instances}
    else: working_instances = instances

    # Load previously done work
    previous = utils.read_json(domain_name, overwrite_previous, "responses", verbose=verbose)

    failed_instances = []
    for instance in tqdm(working_instances):
        if instance not in previous.keys(): previous[instance] = []
        previous_instance_output = previous[instance]         
        for prompt in working_instances[instance]:
            assert prompt is dict
            if not utils.includes_dict(prompt, prompt_specification,()): continue
            for trial_id in range(0, num_trials):
                trial_specification = {"trial_id": trial_id, "llm": llm, "temp": temp}
                trial_specification.update(prompt)
                if utils.includes_dict(previous_instance_output, trial_specification): continue
                prompt_text = prompt["prompt"]
                cost += len(enc.encode(prompt_text))*input_cost
                if verbose:
                    print(f"==Sending prompt {len(prompt['prompt'])} of character length {len(prompt['prompt'][-1])} to LLM for trial {trial_id} (instance {instance})==")
                    print(prompt_text)
                trial_output = trial_specification
                trial_output.update(prompt)

                llm_response = send_query(prompt_text, llm, temp=temp)
                if not llm_response:
                    failed_instances.append(instance)
                    print(f"==Failed instance: {instance}==")
                    continue
                cost += len(enc.encode(llm_response))*output_cost
                if verbose:     
                    print(f'==LLM Response==')
                    print(llm_response)
                    print(f"***Current cost: {cost:.2f}***")

                trial_output.update({"response": llm_response})
                previous[instance].append(trial_output)
                utils.write_json(domain_name, previous, "responses")
                
    # Print any failed instances
    print(f"Failed instances: {failed_instances}")
    print(f"Total Cost: {cost:.2f}")

def send_query(query_text, llm, temp=0, max_tokens=MAX_GPT_RESPONSE_LENGTH, stop_statement=STOP_STATEMENT):
    if 'gpt' in llm:
        # TODO do this in a principled way (grab from openai api and check etc)
        messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query_text}
        ]
        try:
            response = openai.ChatCompletion.create(model=llm, messages=messages, temperature=temp, stop=stop_statement, max_tokens=max_tokens)
        except Exception as e:
            print("[-]: Failed GPT query execution: {}".format(e))
            return ""
        text_response = response['choices'][0]['message']['content']
        return text_response.strip()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine', type=str, required=True, help='Engine to use \
                        \n gpt-4_chat = GPT-4 \
                        \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
                        ')
    parser.add_argument('-d', '--domain', type=str, required=True, help='Problem domain to query for')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-r', '--run_till_completion', type=str, default="False", help='Run till completion')
    parser.add_argument('-s', '--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('-i', '--ignore_existing', action='store_true', help='Ignore existing output')
    parser.add_argument('-n', '--end_number', type=int, default=0, help='For running from instance m to n')
    parser.add_argument('-m', '--start_number', type=int, default=1, help='For running from instance m to n. You must specify -n for this to work')
    parser.add_argument('-p', '--problem', type=str, default='', help='If doing a domain subproblem, specify it here')
    parser.add_argument('-t', '--temperature', type=float, default=0, help='Temperature from 0.0 to 2.0')
    parser.add_argument('-T', '--trial', type=int, default=1, help='Number of trials to run for this.')
    args = parser.parse_args()
    engine = args.engine
    domain_name = args.domain
    if domain_name not in domain_utils.domains:
        raise ValueError(f"Domain name must be an element of {list(domain_utils.domains)}.")
    specified_instances = args.specific_instances
    verbose = args.verbose
    backprompt = args.backprompt
    backprompt_num = args.backprompt_num
    if not backprompt: backprompt_num = 1
    run_till_completion = eval(args.run_till_completion)
    ignore_existing = args.ignore_existing
    end_number = args.end_number
    start_number = args.start_number
    problem_type = args.problem
    temperature = args.temperature
    trial_num = args.trial
    if end_number>0 and specified_instances:
        print("You can't use both -s and -n")
    elif end_number>0:
        specified_instances = list(range(start_number,end_number+1))
        print(f"Running instances from {start_number} to {end_number}")
    print(f"Engine: {engine}, Domain: {domain_name}, Verbose: {verbose}, Multiprompt Type: {backprompt}, Problem Type: {problem_type}, Trial ID: {trial_num}")

    get_responses(engine, domain_name, specified_instances, ignore_existing, verbose, backprompt, problem_type, multiprompt_num=backprompt_num, temp=temperature, num_trials=x)