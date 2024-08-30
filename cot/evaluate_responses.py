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
import matplotlib.ticker as ticker #type: ignore
from matplotlib.font_manager import fontManager, FontProperties #type: ignore


import utils

def evaluate_responses(domain_name, llm=None, specified_instances=[], overwrite_previous=False, verbose=False, graph_it= '', x = 'steps_to_solve', y = 'correct', values='', columns='', aggfunc='mean', h='', s='', idict={}, **kwargs):
    domain = domain_utils.domains[domain_name]

    # Load response data
    responses = utils.read_json(domain_name, False, "responses", verbose)
    if specified_instances: working_instances = {num: responses[num] for num in responses.keys() if num in specified_instances}
    else: working_instances = responses
    
    ####### SUPER UGLY #######
    # graph setup

    # path = "/home/kaya/yochan/cot/Poppins-SemiBold.ttf"
    # fontManager.addfont(path)
    # prop = FontProperties(fname=path)
    # FONTSCALE = 2
    # FONTSIZE = 35
    # LEGENDFONT = 25
    # FIGSIZE = (30, 10)
    # print("t")
    # fig, axs = plt.subplots(1, 3, figsize=FIGSIZE)
    # print("s")
    # #font size
    # sns.set_context("poster", font_scale=FONTSCALE)
    # sns.set_theme(style="darkgrid", font=prop.get_name())

    # coinflip = utils.read_json("coinflip", False, "evaluations", verbose=verbose)
    # cf_axis_num = 0
    # flat_cf = utils.flatten(coinflip)
    # cf_df = pd.DataFrame(flat_cf)
    # if llm: cf_df = cf_df.drop(cf_df[cf_df['llm'] != llm].index)
    # cf_df.loc[(cf_df.magic =="Let's think step by step.") & (cf_df.cot==""), 'cot'] = 'Zero-Shot CoT'
    # cf_df.loc[(cf_df.magic ==" ") & (cf_df.cot==""), 'cot'] = 'Thought Tag Only'
    # cf_df.replace({"cot":{"":"direct"}}, inplace=True)
    # cf_subdf = cf_df
    # cf_idict = {"cot":["direct","wei"],"relaxation":["full"]}
    # for k in cf_idict.keys():
    #     cf_subdf = cf_subdf[cf_subdf[k].isin(cf_idict[k])]

    # cf_subdf['correct'] = cf_subdf['correct'].map(lambda x: 100*x).fillna(cf_df['correct'])
    # # cf_subdf['cot'] = cf_subdf['cot'].map(lambda x: x.capitalize())
    # cf_subdf['cot'] = cf_subdf['cot'].map({"direct":"Direct","wei":"CoT"}).fillna(cf_subdf["cot"])
    # # cf_subdf['relaxation'] = cf_subdf['relaxation'].map({"chain":"Arbitrary", "full":"Arbitrary", "told":"Single-Digit"}).fillna(cf_df['relaxation'])
    # # cf_subdf['relaxation'] = cf_subdf['relaxation'].map(lambda x: x.capitalize())
    # cf_subdf.rename(columns={"cot": "CoT"}, inplace=True)
    # cf_subdf = cf_subdf[cf_subdf.steps_to_solve<30]
    # # subd
    # sns.lineplot(x=x, y=y, style="CoT", data=cf_subdf, palette="deep", err_style=None, style_order=["CoT", "Direct"], ax=axs[cf_axis_num])
    # axs[cf_axis_num].set_title(f'CoinFlip', fontsize=FONTSIZE)
    # axs[cf_axis_num].set_xlabel('# of People', fontsize=FONTSIZE)
    # axs[cf_axis_num].set_ylabel('% of Instances correct', fontsize=FONTSIZE)
    # axs[cf_axis_num].legend(fontsize=LEGENDFONT)
    # axs[cf_axis_num].set_xticks(range(2, 29, 3))
    # axs[cf_axis_num].set_yticks(range(0, 101, 20))
    # axs[cf_axis_num].set_ylim(bottom=0)
    # axs[cf_axis_num].set_xlim(left=1, right=29)
    # axs[cf_axis_num].grid(True)
    # axs[cf_axis_num].tick_params(axis='both', which='major', labelsize=FONTSIZE)
    # axs[cf_axis_num].tick_params(axis='both', which='minor', labelsize=FONTSIZE)
    # # axs[cf_axis_num].plot([cf_subdf.min()[x], cf_subdf.max()[x]], [0.5, 0.5])
    # axs[cf_axis_num].axhline(y=50, linewidth=2, color='orange', ls=':')


    # ########## LLC
    # llc = utils.read_json("lastletterconcat", False, "evaluations", verbose=verbose)
    # llc_axis_num = 1
    # flat_llc = utils.flatten(llc)
    # llc_df = pd.DataFrame(flat_llc)
    # if llm: llc_df = llc_df.drop(llc_df[llc_df['llm'] != llm].index)
    # # llc_df.loc[(llc_df.magic =="Let's think step by step.") & (llc_df.cot==""), 'cot'] = 'Zero-Shot CoT'
    # # llc_df.loc[(llc_df.magic ==" ") & (llc_df.cot==""), 'cot'] = 'Thought Tag Only'
    # llc_df.replace({"cot":{"":"direct"}}, inplace=True)
    # llc_subdf = llc_df
    # llc_idict = {"cot":["wei","direct"],"relaxation":["full","vowel","foom_clearer"]}
    # for k in llc_idict.keys():
    #     llc_subdf = llc_subdf[llc_subdf[k].isin(llc_idict[k])]

    # llc_subdf['correct'] = llc_subdf['correct'].map(lambda x: 100*x).fillna(llc_df['correct'])
    # llc_subdf['cot'] = llc_subdf['cot'].map({"direct":"Direct","wei":"Basic"}).fillna(llc_subdf["cot"])
    # # llc_subdf['cot'] = llc_subdf['cot'].map(lambda x: x.capitalize())
    # # llc_subdf['relaxation'] = llc_subdf['relaxation'].map({"chain":"Arbitrary", "full":"Arbitrary", "told":"Single-Digit"}).fillna(llc_df['relaxation'])
    # # llc_subdf['relaxation'] = llc_subdf['relaxation'].map(lambda x: x.capitalize())
    # llc_subdf.rename(columns={"cot": "CoT", "relaxation": "Variant"}, inplace=True)
    # # subd
    # sns.lineplot(x=x, y=y, hue="Variant", style="CoT", data=llc_subdf, palette="deep", err_style=None, style_order=["Basic", "Direct"], ax=axs[llc_axis_num])
    # axs[llc_axis_num].set_title(f'LastLetterConcatenation', fontsize=FONTSIZE)
    # axs[llc_axis_num].set_xlabel('# of Words', fontsize=FONTSIZE)
    # # axs[llc_axis_num].set_ylabel('% of Instances correct', fontsize=FONTSIZE)
    # axs[llc_axis_num].legend(fontsize=LEGENDFONT)
    # axs[llc_axis_num].set_xticks(range(0, 21, 2))
    # axs[llc_axis_num].set_yticks(range(0, 101, 20))
    # axs[llc_axis_num].set_ylim(bottom=0)
    # axs[llc_axis_num].set_xlim(left=1, right=20)
    # axs[llc_axis_num].grid(True)
    # axs[llc_axis_num].tick_params(axis='both', which='major', labelsize=FONTSIZE)
    # axs[llc_axis_num].tick_params(axis='both', which='minor', labelsize=FONTSIZE)
    # plt.plot([llc_subdf.min()[x], llc_subdf.max()[x]], [0.5, 0.5])





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
    if llm: df = df.drop(df[df['llm'] != llm].index)
    df.loc[(df.magic =="Let's think step by step.") & (df.cot==""), 'cot'] = 'Zero-Shot CoT'
    df.loc[(df.magic ==" ") & (df.cot==""), 'cot'] = 'Thought Tag Only'
    df.replace({"cot":{"":"direct"}}, inplace=True)
    boolean_table = df.select_dtypes(np.bool_).value_counts(normalize=True).mul(100).astype(str)
    # print(boolean_table+"%")
    # print(df.pivot_table(columns='uniform_token_length', values='correct'))
    # df['binned'] = pd.cut(df['input_length'],5)
    # print(df.pivot_table(index='steps_to_solve',columns='binned', values='correct'))
    # print(f"\n${df['estimated_cost'].sum():.02f} estimated total spend.\n")
    df2 = df.drop(columns=['trial_id', 'temp', 'n_examples', 'output_length', 'well_formed_response', 'timestamp', 'estimated_cost'])
    # print(df2.corr(numeric_only=True))
    # steps_pivot = df.pivot_table(columns='steps_to_solve', values='correct')
    # print(steps_pivot.head())
    # print(steps_pivot.columns)
    # x = 'uniform_token_length'
    subdf = df
    print(idict)
    for k in idict.keys():
        subdf = subdf[subdf[k].isin(idict[k])]
    if domain_name == "coinflip":
        subdf = subdf[subdf.steps_to_solve<30]
    # subdf = subdf[subdf.steps_to_solve>1]
    if values and columns:
        print("\n")
        print(subdf.pivot_table(columns=columns, values=values, aggfunc=aggfunc))
    if graph_it:
        if   graph_it == "line":
            if h and s:
                # df.replace({"cot":{"":"direct"}}, inplace=True)
                # subdf['correct'] = subdf['correct'].map(lambda x: 100*x).fillna(df['correct'])
                # subdf['cot'] = subdf['cot'].map(lambda x: x.capitalize())
                # subdf['relaxation'] = subdf['relaxation'].map({"chain":"Arbitrary", "full":"Arbitrary", "told":"Single-Digit"}).fillna(df['relaxation'])
                # subdf['relaxation'] = subdf['relaxation'].map(lambda x: x.capitalize())
                # subdf['steps_to_solve'] = subdf['steps_to_solve'].map(lambda x: x-1)
                # subdf.rename(columns={"cot": "CoT", "relaxation": "Explanation"}, inplace=True)
                # subd
                g = sns.lineplot(x=x, y=y, style=s, hue=h, data=subdf, palette="deep", err_style=None)
                # axs[2].set_title(f'One-Digit Arithmetic', fontsize=FONTSIZE)
                # axs[2].set_xlabel('# of Operations', fontsize=FONTSIZE)
                # # axs[2].set_ylabel('% of Instances correct', fontsize=FONTSIZE)
                # # a[2]xs[0e)].legend(loc='upper right', markerscale=2)
                # # 0[2]plot
                # # ax2[0].legend(bbox_to_anchor=(1, 1), fontsize=LEGENDFONT)
                # axs[2].legend(fontsize=LEGENDFONT)
                # axs[2].set_xticks(range(2, 31, 3))
                # axs[2].set_yticks(range(0, 101, 20))
                # axs[2].set_ylim(bottom=0)
                # # 0[2]art from 3 so I offset the graph from 2
                # axs[2].set_xlim(left=1, right=29)
                # axs[2].grid(True)
                # #0a[2]bels
                # axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE)
                # axs[2].tick_params(axis='both', which='minor', labelsize=FONTSIZE)
            elif h: g = sns.lineplot(x=x, y=y, hue=h, data=subdf, palette="deep", err_style=None)
            else: g = sns.lineplot(x=x, y=y, data=subdf, err_style=None)
        elif graph_it == "corr":
            ssubdf = df[df.cot.isin(['wei','direct'])]
            ssubdf = ssubdf[ssubdf.bag_correct==True]
            subdf = pd.melt(ssubdf[[x,'correct', 'bag_correct','set_correct','cot']],ssubdf[[x,'cot']])
            g = sns.lineplot(x=x, y='value', hue='variable',style='cot', data=subdf, palette="deep")
            # sns.lineplot(data=df[['correct', 'set_correct', 'bag_correct']])
        elif graph_it == "chain":
            ssubdf = df[df.cot.isin(['basic'])]
            # ssubdf = ssubdf[ssubdf.bag_correct==True]
            subdf = pd.melt(ssubdf[[x,'correct', 'chain_correct','smooth_chain_correct','normalized_chain_length','cot']],ssubdf[[x,'cot']])
            g = sns.lineplot(x=x, y='value', hue='cot',style='variable', data=subdf, palette="deep")
        elif graph_it == "scatter":
            if h and s: g = sns.scatterplot(x=x, y=y, hue=h, style=s, data=subdf, palette="deep")
            elif h: g = sns.scatterplot(x=x, y=y, hue=h, data=subdf, palette="deep")
            else: g = sns.scatterplot(x=x, y=y, data=subdf)
        else: raise ValueError(f"Can't plot something of type {graph_it}")
        # sns.despine(offset=10, trim=True)
        # if domain_name == "coinflip": plt.plot([subdf.min()[x], subdf.max()[x]], [0.5, 0.5])
        # g.axes.xaxis.set_major_locator(ticker.MultipleLocator(2))
        plt.tight_layout()
        plt.savefig(f'analysis/extension_comparison_graph.png')
        plt.show()
        


if __name__=="__main__":
    Fire(evaluate_responses)