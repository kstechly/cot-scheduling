import argparse
from fire import Fire # type: ignore

import utils
import domain_utils

def generate_prompts(domain_name, n_examples = 0, example_type ='basic', cot='', magic='', relaxation='full', overwrite_previous=False):
    if domain_name not in domain_utils.domains:
        raise ValueError(f"Domain name must be an element of {list(domain_utils.domains)}.")
    domain = domain_utils.domains[domain_name]

    prompts = utils.read_json(domain_name, overwrite_previous, "prompts")
    instances = utils.read_json(domain_name, False, "instances")

    # Just so that we don't double store 0-example CoT prompts on accident:
    if n_examples == 0: cot = ""
    else: cot = cot
    
    # TODO implement subprompt generation. Want to check exactly the distribution that the models get.
    #      Both the expected, correct one, and the actual ones they end up seeing...
    #       (Relevant for Paradox of Learning reasons)
    #      Specifically, maybe let the domain decide how many prompts to output? **kwargs

    for instance in instances:
        # TODO clean this up
        prompt = domain.generate(instances[instance], problem_relaxation=relaxation, cot_type=cot, n_examples=n_examples, magic=magic)
        # computation_graph = domain.generate_graph(instance, cot_type=cot_type, extraction_label=label)
        full_prompt_info = {"relaxation":relaxation, "cot":cot, "n_examples": n_examples, "magic": magic, "example_type": example_type} #, "computation_graph": computation_graph}
        # TODO this could be done better
        if instance in prompts:
            old_prompt_num = utils.dict_index(prompts[instance], full_prompt_info)
        else: 
            old_prompt_num = -1
            prompts[instance] = []
        full_prompt_info.update({"prompt": prompt})
        full_prompt_info.update(instances[instance])
        if old_prompt_num>-1: 
            if overwrite_previous: prompts[instance][old_prompt_num] = full_prompt_info
            continue
        prompts[instance].append(full_prompt_info)
    
    utils.write_json(domain_name, prompts, "prompts")

if __name__=="__main__":
    Fire(generate_prompts)
