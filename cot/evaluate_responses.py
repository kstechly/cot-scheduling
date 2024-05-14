import os
import argparse
import json
from tqdm import tqdm #type: ignore
import domain_utils
from domain_utils import *
from fire import Fire #type: ignore
import pandas as pd #type: ignore
import seaborn as sns #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore

import utils

def evaluate_responses(domain_name, specified_instances=[], overwrite_previous=False, verbose=False, graph_it= '', x = 'steps_to_solve', y = 'correct', values='', columns='', h='', s='', idict={}, **kwargs):
    domain = domain_utils.domains[domain_name]

    # Load response data
    responses = utils.read_json(domain_name, False, "responses", verbose)
    if specified_instances: working_instances = {num: responses[num] for num in responses.keys() if num in specified_instances}
    else: working_instances = responses
    
    # Load previously done work
    previous = utils.read_json(domain_name, overwrite_previous, "evaluations", verbose=verbose)
    
    # for each instance, pass through the eval function, then update the dictionary with the eval
    for instance in tqdm(working_instances):
        if instance not in previous.keys(): previous[instance] = []
        for response in working_instances[instance]:
            ind = utils.dict_index(previous[instance], response)
            if not overwrite_previous and ind > -1: continue
            response.update(domain.evaluate(response))

            if ind == -1: previous[instance].append(response)
            else: previous[instance][ind] = response
    utils.write_json(domain_name, previous, "evaluations")

    # TODO factor this into a better place 
    flat_results = utils.flatten(previous)
    df = pd.DataFrame(flat_results)
    df = df.drop(df[df['llm'] != 'gpt-4-turbo-2024-04-09'].index)
    df.loc[(df.magic =="Let's think step by step.") & (df.cot==""), 'cot'] = 'kojima_magic'
    df.loc[(df.magic ==" ") & (df.cot==""), 'cot'] = 'thought_tag_only'
    df.replace({"cot":{"":"direct"}}, inplace=True)
    boolean_table = df.select_dtypes(np.bool_).value_counts(normalize=True).mul(100).astype(str)
    print(boolean_table+"%")
    # print(df.pivot_table(columns='uniform_token_length', values='correct'))
    # df['binned'] = pd.cut(df['input_length'],5)
    # print(df.pivot_table(index='steps_to_solve',columns='binned', values='correct'))
    print(f"\n${df['estimated_cost'].sum():.02f} estimated total spend.\n")
    df2 = df.drop(columns=['trial_id', 'temp', 'n_examples', 'output_length', 'well_formed_response', 'timestamp', 'estimated_cost'])
    print(df2.corr(numeric_only=True))
    # steps_pivot = df.pivot_table(columns='steps_to_solve', values='correct')
    # print(steps_pivot.head())
    # print(steps_pivot.columns)
    # x = 'uniform_token_length'
    subdf = df
    print(idict)
    for k in idict.keys():
        subdf = subdf[subdf[k].isin(idict[k])]
    # subdf = subdf[subdf.steps_to_solve>1]
    if values and columns:
        print("\n")
        print(subdf.pivot_table(columns=columns, values=values))
    if graph_it:
        sns.color_palette("colorblind")
        sns.set_theme(style="whitegrid")
        if   graph_it == "line":
            if h and s: sns.lineplot(x=x, y=y, hue=h, style=s, data=subdf, palette="deep")
            elif h: sns.lineplot(x=x, y=y, hue=h, data=subdf, palette="deep")
            else: sns.lineplot(x=x, y=y, data=subdf)
        elif graph_it == "corr":
            ssubdf = df[df.cot.isin(['wei','direct'])]
            ssubdf = ssubdf[ssubdf.bag_correct==True]
            subdf = pd.melt(ssubdf[[x,'correct', 'bag_correct','set_correct','cot']],ssubdf[[x,'cot']])
            sns.lineplot(x=x, y='value', hue='variable',style='cot', data=subdf, palette="deep")
            # sns.lineplot(data=df[['correct', 'set_correct', 'bag_correct']])
        elif graph_it == "scatter":
            if h and s: sns.scatterplot(x=x, y=y, hue=h, style=s, data=subdf, palette="deep")
            elif h: sns.scatterplot(x=x, y=y, hue=h, data=subdf, palette="deep")
            else: sns.scatterplot(x=x, y=y, data=subdf)
        else: raise ValueError(f"Can't plot something of type {graph_it}")
        sns.despine(offset=10, trim=True)
        if domain_name == "coinflip": plt.plot([subdf.min()[x], subdf.max()[x]], [0.5, 0.5])
        plt.show()


if __name__=="__main__":
    Fire(evaluate_responses)