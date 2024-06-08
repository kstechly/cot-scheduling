from fire import Fire #type: ignore
from random import choice
from pprint import pprint
import utils

def sample_evaluation(domain, which="all",**kwargs):
    evals = utils.read_json(domain, False, 'evaluations', verbose=True)
    filtered = []
    for k in evals:
        for x in evals[k]:
            next = {f'instance_num':int(k)}
            next.update(x)
            if utils.includes_dict([next], kwargs):
                filtered.append(next)

    if not len(filtered): 
        print("No matching evals found.")
        return None
    elif which == "all": pprint(filtered)
    elif which == "random": pprint(choice(filtered))
    else: raise NotImplementedError(f"Can't grab '{which}' instances.")
    print(f'{len(filtered)} matching evals found.')

if __name__ == "__main__":
    Fire(sample_evaluation)