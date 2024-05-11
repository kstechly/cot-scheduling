from tqdm import tqdm # type: ignore
import openai # type: ignore
import tiktoken # type: ignore
from fire import Fire # type: ignore
from functools import cache

import utils

STOP_STATEMENT = "[ANSWER END]" # what we look for to end LLM response generation

SYSTEM_PROMPT  = "You are a system that solves reasoning problems presented in text form."

input_costs_per_million  = {"gpt-4": 30, "gpt-4-turbo":10, "gpt-4-turbo-2024-04-09": 10, "gpt-3.5-turbo-0125": 0.5}
output_costs_per_million = {"gpt-4": 60, "gpt-4-turbo":30, "gpt-4-turbo-2024-04-09": 30, "gpt-3.5-turbo-0125": 1.5}

def get_responses(llm, domain_name, specified_instances = [], overwrite_previous=False, verbose=False, temp=0, num_trials=1, **prompt_specification):


    # Cost calculation setup
    input_cost  = 0.0
    output_cost = 0.0
    million = 1000000
    if llm in input_costs_per_million.keys():
        input_cost  = input_costs_per_million[llm]/million
        output_cost = output_costs_per_million[llm]/million
    cost = 0.0
    enc  = tiktoken.get_encoding("cl100k_base")

    # Load prompts and possibly filter for specified_instances
    instances = utils.read_json(domain_name, False, "prompts", verbose)
    if specified_instances: working_instances = {num: instances[num] for num in instances.key() if num in specified_instances}
    else: working_instances = instances

    # Load previously done work
    previous = utils.read_json(domain_name, overwrite_previous, "responses", verbose=verbose)

    failed_instances = []
    for instance in tqdm(working_instances):
        if instance not in previous.keys(): previous[instance] = []
        previous_instance_output = previous[instance]         
        for prompt in working_instances[instance]:
            if not utils.includes_dict(prompt, prompt_specification): continue
            for trial_id in range(0, num_trials):
                trial_specification = {"trial_id": trial_id, "llm": llm, "temp": temp}
                trial_specification.update(prompt)

                if utils.includes_dict(previous_instance_output, trial_specification) and not overwrite_previous: continue
                ind = utils.dict_index(previous_instance_output, trial_specification)

                prompt_text = prompt["prompt"]
                token_length = len(enc.encode(prompt_text))
                cost += token_length*input_cost
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
                cost += len(enc.encode(llm_response))*output_cost
                if verbose:     
                    print(f'==LLM Response==')
                    print(llm_response)
                    print(f"***Current cost: {cost:.4f}***")

                trial_output.update({"response": llm_response})
                if ind == -1: previous[instance].append(trial_output)
                else: previous[instance][ind] = trial_output
                utils.write_json(domain_name, previous, "responses")
                
    # Print any failed instances
    print(f"Failed instances: {failed_instances}")
    print(f"Total Cost: {cost:.2f}")

@cache
def is_openai_model(llm):
    client = openai.OpenAI()
    openai_models = client.models.list().data
    model_names = [x.id for x in openai_models]
    return llm in model_names

def send_query(query_text, llm, temp=0, stop_statement=STOP_STATEMENT):
    if is_openai_model(llm):
        # TODO do this in a principled way (grab from openai api and check etc)
        messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query_text}
        ]
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(model=llm, messages=messages, temperature=temp, stop=stop_statement)
        except Exception as e:
            print("[-]: Failed GPT query execution: {}".format(e))
            return ""
        text_response = response.choices[0].message.content
        return text_response.strip()

if __name__=="__main__":
    Fire(get_responses)