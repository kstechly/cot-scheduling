import os
import argparse
import json
from tqdm import tqdm #type: ignore
import domain_utils
from domain_utils import *
from fire import Fire #type: ignore
from itertools import chain
import pandas as pd #type: ignore


import utils

def evaluate_responses(domain_name, specified_instances=[], overwrite_previous=False, verbose=False, **kwargs):
    domain = domain_utils.domains[domain_name]

    # Load response data
    responses = utils.read_json(domain_name, False, "responses", verbose)
    if specified_instances: working_instances = {num: responses[num] for num in responses.key() if num in specified_instances}
    else: working_instances = responses
    
    # Load previously done work
    previous = utils.read_json(domain_name, overwrite_previous, "evaluations", verbose=verbose)
    
    # for each instance, pass through the eval function, then update the dictionary with the eval
    for instance in tqdm(working_instances):
        if instance not in previous.keys(): previous[instance] = []
        for response in working_instances[instance]:
            ind = utils.dict_index(previous[instance], response)
            if not overwrite_previous and ind > -1: continue
            response.update(domain.evaluate(response, **kwargs))

            if ind == -1: previous[instance].append(response)
            else: previous[instance][ind] = response
            previous[instance].append(response)
    utils.write_json(domain_name, previous, "evaluations")
    flat_results = flatten(previous)
    df = pd.DataFrame(flat_results)
    print(df.describe())

def flatten(dict):
    return list(chain(*dict.values()))

if __name__=="__main__":
    Fire(evaluate_responses)