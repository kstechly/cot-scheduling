import json
import os
import domain_utils
from domain_utils import *
import argparse
import time

START = 1
END = 100

def loc(domain_name):
    return f"prompts/{domain_name}/prompts.json"

# TODO factor these out into a utils file
def write_json(domain_name,text_to_write):
    os.makedirs(f"prompts/{domain_name}", exist_ok=True)
    location = loc(domain_name)
    with open(f'{location}.tmp',"w") as fp:
        json.dump(text_to_write, fp, indent = 4)
    os.replace(f'{location}.tmp', location)

def read_json(domain_name, overwrite_previous):
    if os.path.exists(loc(domain_name)):
        with open(loc(domain_name), 'r') as file:
            previous = json.load(file)
        if overwrite_previous:
            stamp = str(time.time())
            with open(f"prompts/{domain_name}/prompts-{stamp}.json","w") as file:
                json.dump(previous, file, indent=4)
        return previous
    else: return {}

def includes_dict(l, b, ignore_keys):
    for a in l:
        ka = set(a).difference(ignore_keys)
        kb = set(b).difference(ignore_keys)
        if ka == kb and all(a[k] == b[k] for k in ka): return True
    return False

def read_instance(domain_name,number_of_instance,file_ending):
    try:
        with open(f"data/{domain_name}/instance-{number_of_instance}{file_ending}") as fp:
            return fp.read()
    except FileNotFoundError:
        print(f"data/{domain_name}/instance-{number_of_instance} not found.")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--domain', type=str, required="True", help='Problem domain to generate for')
    # TODO make this just auto-process everything
    parser.add_argument('-s','--start', type=int, default=START, help='Start number of instances to process')
    parser.add_argument('-e','--end', type=int, default=END, help='End number of instances to process')
    parser.add_argument('-c','--cot', type=str, default='', help='Chain of Thought type. Leave empty for no CoT. Only works if non-zero number of examples.')
    parser.add_argument('-r','--relaxation', type=str, default='full', help='Relaxation of the problem. Changes initial instructions, example answers, and final evaluation, but not thoughts.')
    parser.add_argument('-m','--magic', type=str, default='', help='Magic phrase to use. Leave empty for none.')
    parser.add_argument('-n','--n_examples', type=int, default=0, help='Number of examples to provide.')
    parser.add_argument('-p','--example_prefix', type=str, default='example_basic', help='The prefix at which to look for examples in the example folder.')
    parser.add_argument('-o','--overwrite_previous', type=bool, default=False, help='Overwrite previously generated prompts. Automatically creates a backup.')
    args = parser.parse_args()
    domain_name = args.domain
    if domain_name not in domain_utils.domains:
        raise ValueError(f"Domain name must be an element of {list(domain_utils.domains)}.")
    domain = domain_utils.domains[domain_name]

    labels = domain.extraction_labels()
    prompts = read_json(domain_name, args.overwrite_previous)

    # Just so that we don't double store 0-example CoT prompts on accident:
    if args.n_examples == 0: cot_type = ""
    else: cot_type = args.cot

    for x in range(args.start,args.end+1):
        instance = read_instance(domain_name,x,domain.file_ending())
        if instance:
            for label in labels:
                prompt_name = f'{x}{label}'
                prompt = domain.generate(instance, problem_relaxation=args.relaxation, cot_type=cot_type, n_examples=args.n_examples, magic=args.magic, example_prefix=args.example_prefix, extraction_label=label)
                full_prompt_info = {"prompt": prompt, "extraction_label":label, "relaxation":args.relaxation, "cot":cot_type, "n_examples": args.n_examples, "magic": args.magic, "example_prefix": args.example_prefix}
                if prompt_name in prompts and includes_dict(prompts[prompt_name], full_prompt_info, ("prompt")): continue
                if prompt_name not in prompts: prompts[prompt_name] = []
                prompts[prompt_name].append(full_prompt_info)
    write_json(domain_name, prompts)