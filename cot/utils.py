import json
import os


### json utils

def write_json(domain_name,text_to_write,data_type):
    directory = f"data/{data_type}/{domain_name}"
    os.makedirs(directory, exist_ok=True)
    location = f"{directory}/prompts.json"
    with open(f'{location}.tmp',"w") as fp:
        json.dump(text_to_write, fp, indent = 4)
    os.replace(f'{location}.tmp', location)

def read_json(domain_name, overwrite_previous, data_type):
    location = f"data/{data_type}/{domain_name}/prompts.json"
    if os.path.exists(location):
        with open(location, 'r') as file:
            previous = json.load(file)
        if overwrite_previous:
            stamp = str(time.time())
            with open(f"data/{data_type}/{domain_name}/prompts-{stamp}.json","w") as file:
                json.dump(previous, file, indent=4)
        return previous
    else: return {}

### other utils

def includes_dict(l, b, ignore_keys):
    for a in l:
        ka = set(a).difference(ignore_keys)
        kb = set(b).difference(ignore_keys)
        if ka == kb and all(a[k] == b[k] for k in ka): return True
    return False