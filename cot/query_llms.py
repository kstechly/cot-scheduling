import openai # type: ignore
import tiktoken # type: ignore
from fire import Fire # type: ignore
from functools import cache
import time
from groq import Groq #type: ignore
import os
import requests
import boto3 #type: ignore

import utils

STOP_STATEMENT = "[ANSWER END]" # what we look for to end LLM response generation

SYSTEM_PROMPT  = "You are a system that solves reasoning problems presented in text form."

input_costs_per_million  = {"gpt-4": 30, "gpt-4-turbo":10, "gpt-4-turbo-2024-04-09": 10, "gpt-3.5-turbo-0125": 0.5, "llama3-8b-8192": 0.0, "gpt-4o-2024-05-13": 5}
output_costs_per_million = {"gpt-4": 60, "gpt-4-turbo":30, "gpt-4-turbo-2024-04-09": 30, "gpt-3.5-turbo-0125": 1.5, "llama3-8b-8192": 0.0, "gpt-4o-2024-05-13": 15}

def get_responses(llm, domain_name, specified_instances = [], print_models=False, overwrite_previous=False, verbose=False, temp=0, num_trials=1, **prompt_specification):
    if print_models:
        #TODO refactor this
        client = openai.OpenAI()
        openai_models = client.models.list().data
        print([x.id for x in openai_models])

        api_key = os.environ.get("GROQ_API_KEY")
        url = "https://api.groq.com/openai/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers)
        d = response.json()['data']
        groq_models = [x['id'] for x in d]
        print(groq_models)
        exit()

    if llm != "gpt-4-turbo-2024-04-09":
        print("This isn't the right model, is it??? " + llm)
        # exit()
    # Cost calculation setup
    input_cost  = 0.0
    output_cost = 0.0
    million = 1000000
    if llm in input_costs_per_million.keys():
        input_cost  = input_costs_per_million[llm]/million
        output_cost = output_costs_per_million[llm]/million
    else: raise ValueError(f"Can't calculate cost for llm {llm}. Change query_llms to fix this.")
    current_sesh_cost = 0.0
    enc  = tiktoken.get_encoding("cl100k_base")

    #TODO janky, plz fix
    try: total_cost = utils.get_total_cost(domain_name)
    except: total_cost = 0

    # Load prompts and possibly filter for specified_instances
    instances = utils.read_json(domain_name, False, "prompts", verbose)
    if specified_instances: working_instances = {num: instances[num] for num in instances.key() if num in specified_instances}
    else: working_instances = instances

    # Load previously done work
    previous = utils.read_json(domain_name, overwrite_previous, "responses", verbose=verbose)

    prog = utils.progress_bar()
    instance_task = None

    failed_instances = []
    with prog as p:
        for instance in p.track(working_instances, description=f"{llm}, {domain_name}"):
            instance_task = utils.replace_task(p, instance_task, description=f'Instance {instance} (Cost so far: ${total_cost:.02f})', total=len(working_instances[instance]))
            if instance not in previous.keys(): previous[instance] = []
            previous_instance_output = previous[instance]         
            for prompt in working_instances[instance]:
                if not utils.includes_dict([prompt], prompt_specification):
                    p.update(instance_task, advance=1)
                    continue
                for trial_id in range(0, num_trials):
                    trial_specification = {"trial_id": trial_id, "llm": llm, "temp": temp}
                    trial_specification.update(prompt)

                    if utils.includes_dict(previous_instance_output, trial_specification) and not overwrite_previous: continue
                    ind = utils.dict_index(previous_instance_output, trial_specification)

                    prompt_text = prompt["prompt"]
                    token_length = len(enc.encode(prompt_text))
                    trial_cost = token_length*input_cost
                    if verbose:
                        print(f"==Instance: {instance}, Tokens: {token_length}==")
                        info_dict = {x: trial_specification[x] for x in trial_specification.keys() if x != "prompt"}
                        print(f'=={info_dict}==')
                        print(prompt_text)
                    trial_output = trial_specification
                    trial_output.update(prompt)

                    llm_response = send_query(prompt_text, llm, temp=temp)
                    if not llm_response:
                        failed_instances.append(instance)
                        print(f"==Failed instance: {instance}==")
                        continue
                    trial_cost += len(enc.encode(llm_response))*output_cost
                    current_sesh_cost += trial_cost
                    total_cost += trial_cost

                    trial_output.update({"response": llm_response, "timestamp": time.time(), "estimated_cost": trial_cost})
                    # print(f'Trial output: {trial_output}')
                    if ind == -1: previous[instance].append(trial_output)
                    else: previous[instance][ind] = trial_output
                    utils.write_json(domain_name, previous, "responses")
                    if verbose:     
                        print(f'==LLM Response==')
                        print(llm_response)
                        # print(f"***Current cost: {current_sesh_cost:.4f}***")
                p.update(instance_task, advance=1)
                    
    # Print any failed instances
    print(f"Failed instances: {failed_instances}")
    print(f"Total Session Cost: {current_sesh_cost:.2f}")
    print(f"Total Cost: {total_cost:.2f}")
    

@cache
def is_openai_model(llm):
    client = openai.OpenAI()
    openai_models = client.models.list().data
    return llm in [x.id for x in openai_models]
@cache
def is_bedrock_model(llm):

    #TODO
    pass
@cache
def is_groq_model(llm):
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    d = response.json()['data']
    groq_models = [x['id'] for x in d]
    return llm in groq_models

def send_query(query_text, llm, temp=0, stop_statement=STOP_STATEMENT):
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query_text}
        ]
    if is_openai_model(llm):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(model=llm, messages=messages, temperature=temp, stop=stop_statement)
        except Exception as e:
            print("[-]: Failed GPT query execution: {}".format(e))
            return ""
        text_response = response.choices[0].message.content
        return text_response.strip()
    elif is_groq_model(llm):
        #TODO add some way to slow it down if it goes too fast!
        try: 
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            response = client.chat.completions.create(model=llm, messages=messages, temperature=temp, stop=stop_statement)
        except Exception as e:
            print("[-]: Failed Groq query execution: {}".format(e))
            return ""
        text_response = response.choices[0].message.content
        return text_response.strip()
    elif is_bedrock_model(llm):
        raise NotImplementedError("Haven't implemented bedrock prompting yet!")
    else: raise NotImplementedError(f"Evaluating on \"{llm}\" is not implemented yet!")

if __name__=="__main__":
    Fire(get_responses)