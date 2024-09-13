import openai # type: ignore
import tiktoken # type: ignore
from fire import Fire # type: ignore
from functools import cache
import time
from groq import Groq #type: ignore
import os
import requests
import boto3 #type: ignore
import concurrent.futures
import time

import utils

STOP_STATEMENT = "[ANSWER END]" # what we look for to end LLM response generation

SYSTEM_PROMPT  = "You are a system that solves reasoning problems presented in text form."

WRITING_DELAY  = 10

input_costs_per_million  = {
    "gpt-4": 30, 
    "gpt-4-turbo":10, 
    "gpt-4-turbo-2024-04-09": 10, 
    "gpt-3.5-turbo-0125": 0.5, 
    "llama3-8b-8192": 0.0, 
    "gpt-4o-2024-05-13": 5, 
    "gpt-4o-mini-2024-07-18":0.15,
    "o1-preview":15}
output_costs_per_million = {
    "gpt-4": 60, 
    "gpt-4-turbo":30, 
    "gpt-4-turbo-2024-04-09": 30, 
    "gpt-3.5-turbo-0125": 1.5, 
    "llama3-8b-8192": 0.0, 
    "gpt-4o-2024-05-13": 15, 
    "gpt-4o-mini-2024-07-18": 0.6,
    "o1-preview":60}

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

    if llm != "gpt-4o-2024-05-13":
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
    if specified_instances: working_instances = {num: instances[num] for num in instances.keys() if num in specified_instances}
    else: working_instances = instances

    # Load previously done work
    previous = utils.read_json(domain_name, overwrite_previous, "responses", verbose=verbose)

    prog = utils.progress_bar()
    instance_task = None

    failed_instances = []
    write_counter = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        with prog as p:
            def process_item(instance):
                print(f'processing {instance}')
                #instance_task = utils.replace_task(p, instance_task, description=f'Instance {instance} (Cost so far: ${total_cost:.02f})', total=len(working_instances[instance]))
                if instance not in previous.keys(): previous[instance] = []
                previous_instance_output = previous[instance]         
                for prompt in working_instances[instance]:
                    print(f'processing prompt {prompt} of instance {instance}')
                    if not utils.includes_dict([prompt], prompt_specification):
                        if "examples_all" in prompt_specification.values():
                            if "examples_" not in prompt["example_type"]:
                                p.update(instance_task, advance=1)
                                continue
                        else: 
                            p.update(instance_task, advance=1)
                            print(f'rejecting prompt {prompt} of {instance} bc {prompt_specification}')
                            continue
                    for trial_id in range(0, num_trials):
                        print(trial_id)
                        trial_specification = {"trial_id": trial_id, "llm": llm, "temp": temp}
                        trial_specification.update(prompt)
                        if utils.includes_dict(previous_instance_output, trial_specification): continue
                        trial_specification.pop('prompt',None)

                        if utils.includes_dict(previous_instance_output, trial_specification) and not overwrite_previous: continue
                        ind = utils.dict_index(previous_instance_output, trial_specification)

                        prompt_text = prompt["prompt"]
                        if llm != "gpt-4o-mini-2024-07-18": # TODO get rid of all the lines that start with this. It's just a hack to speed things up right now!!! Also go back and calc costs for the ones I'm skipping
                            token_length = len(enc.encode(prompt_text))
                            trial_cost = token_length*input_cost
                        if verbose:
                            if llm != "gpt-4o-mini-2024-07-18": print(f"==Instance: {instance}, Tokens: {token_length}==")
                            else: print(f"==Instance: {instance}")
                            info_dict = {x: trial_specification[x] for x in trial_specification.keys()}
                            print(f'=={info_dict}==')
                            print(prompt_text)
                        trial_output = trial_specification
                        trial_output.update(prompt)

                        llm_response = send_query(prompt_text, llm, temp=temp)
                        if not llm_response:
                            failed_instances.append(instance)
                            print(f"==Failed instance: {instance}==")
                            continue
                        if llm != "gpt-4o-mini-2024-07-18":
                            trial_cost += len(enc.encode(llm_response))*output_cost
                            current_sesh_cost += trial_cost
                            total_cost += trial_cost

                        if llm != "gpt-4o-mini-2024-07-18": trial_output.update({"response": llm_response, "timestamp": time.time(), "estimated_cost": trial_cost})
                        else: trial_output.update({"response": llm_response, "timestamp": time.time()})
                        # print(f'Trial output: {trial_output}')
                        if ind == -1: previous[instance].append(trial_output)
                        else: previous[instance][ind] = trial_output
                        if write_counter >= WRITING_DELAY:
                            utils.write_json(domain_name, previous, "responses")
                            write_counter = 0
                        else: write_counter+=1
                        if verbose:     
                            print(f'==LLM Response==')
                            print(llm_response)
                            # print(f"***Current cost: {current_sesh_cost:.4f}***")
                    #p.update(instance_task, advance=1)
            print('e time')
            executor.map(process_item, working_instances)
            print('e over')

    # a final write, if using writing_delay
    utils.write_json(domain_name, previous, "responses")                    
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
        #{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query_text}
        ]
    if is_openai_model(llm):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(model=llm, messages=messages)#, temperature=temp, stop=stop_statement)
        except Exception as e:
            print("[-]: Failed GPT query execution: {}".format(e))
            time.sleep(3000)
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
