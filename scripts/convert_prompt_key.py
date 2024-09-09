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

d = "coinflip"
k = "cot"
v = "wei-incorrect"
nv = "wei_incorrect"
prompts = read_json(d, True, "prompts")
responses = read_json(d, True, "responses")
evaluations = read_json(d, True, "evaluations")

def change_wei(dictionary, key, value, new_value):
	for x in dictionary.keys():
		for y in range(0, len(dictionary[x])):
			if dictionary[x][y][key] == value:
				dictionary[x][y][key] = new_value
	return dictionary

write_json(d, change_wei(prompts, k, v, nv), "prompts")
write_json(d, change_wei(responses, k, v, nv), "responses")
write_json(d, change_wei(evaluations, k, v, nv), "evaluations")
