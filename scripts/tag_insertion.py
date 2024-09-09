import os
import json
import time

#TODO before committing this shit, make running scripts work right

def write_json(domain_name,text_to_write,data_type):
    directory = f"data/{data_type}/{domain_name}"
    os.makedirs(directory, exist_ok=True)
    location = f"{directory}/{data_type}.json"
    with open(f'{location}.tmp',"w") as fp:
        json.dump(text_to_write, fp, indent = 4)
    os.replace(f'{location}.tmp', location)

def read_json(domain_name, overwrite_previous, data_type, verbose=False, strange_subloc=""):
    location = f"data/{data_type}/{domain_name}/{data_type}.json"
    if strange_subloc:
        location = f"data/{data_type}/{domain_name}/{strange_subloc}"
    if os.path.exists(location):
        with open(location, 'r') as file:
            previous = json.load(file)
        if overwrite_previous:
            stamp = str(time.time())
            with open(f"data/{data_type}/{domain_name}/{data_type}-{stamp}.json.old","w") as file:
                json.dump(previous, file, indent=4)
        return previous
    else:
        if verbose: print(f"{location} does not exist. Returning empty dictionary.")
        return {}

d = "lastletterconcat"
k = "random_word"
instances   = read_json(d, True, "instances")
prompts     = read_json(d, True, "prompts")
responses   = read_json(d, True, "responses")
evaluations = read_json(d, True, "evaluations")

def insert_key(dictionary, key):
    for x in dictionary.keys():
        for y in range(0, len(dictionary[x])):
            if key not in dictionary[x][y]:
                dictionary[x][y][key] = False
    return dictionary
def insert_key_instances(instances, key):
    for x in instances.keys():
        if key not in instances[x]:
            instances[x][key] = False
    return instances

write_json(d, insert_key_instances(instances, k), "instances")
write_json(d, insert_key(prompts, k), "prompts")
write_json(d, insert_key(responses, k), "responses")
write_json(d, insert_key(evaluations, k), "evaluations")
