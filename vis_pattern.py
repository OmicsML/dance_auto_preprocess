import json
import os
from pathlib import Path
from dance import logger
import pandas as pd
tasks = ["cell type annotation new", "clustering", "imputation_new", "spatial domain", "cell type deconvolution"]
file_root = Path(__file__).resolve().parents[1]
mertic_names = ["test_acc", "acc", "MRE", "ARI", "MSE"]
ascendings = [False, False, True, False, True]
entity = "xzy11632"
project = "dance-dev"
prefix = f'https://wandb.ai/{entity}/{project}'
from functools import partial
import itertools
def update_key(key):
    components = key.split(' x ')
    components = [comp.split("_")[-1] for comp in components]
    return 'x'.join(components)
# def merge_function_forest(original_list):
#     transformed_list = []
#     for key in original_list:
#         transformed_list.append(update_key(key))
#     return transformed_list
def merge_function_apr(original_list):
    transformed_list = []
    for item in original_list:
        transformed_list.append('x'.join(item))
    return transformed_list
def key_sort(item):
    return abs(item[1]["shapiq"])
def update_value(value):
    new_value={}
    new_value["shapiq"]=value["shapiq"]
    new_value["pointbiserialr_r_pb"]=value["pointbiserialr"]["r_pb"]
    return new_value
def filter_pattern_forest(pattern):
    pattern=pattern["pattern"]
    forest_model=pattern["forest_model"]
    total_count = len(forest_model)
    forest_model
    forest_model = {key: value for key, value in forest_model.items() if key.startswith("onehot")}
    top_count = max(total_count // 3,1)
    sorted_items = sorted(forest_model.items(), key=key_sort, reverse=True)
    top_sixth_items = dict(sorted_items[:top_count])
    ans_item={update_key(key): update_value(value) for key, value in top_sixth_items.items()}
    flatten_ans=[{**sub_dict, 'function': key,"type":"forest"} for key, sub_dict in ans_item.items()]
    return flatten_ans
def filter_pattern_apr(pattern,ascending):
    pattern=pattern["pattern"]
    apr_ans=pattern["apr_ans"]
    apr_ans=merge_function_apr(apr_ans)
    ans_item=[{"function":item,"ascending":ascending,"type":"apriori"} for item in apr_ans]
    return ans_item
if __name__ == "__main__":
    
    for i, task in enumerate(tasks):
        task_all=[]
        data = pd.read_excel(file_root / "results.xlsx", sheet_name=task, dtype=str)
        data = data.ffill().set_index(['Methods'])
        for row_idx in range(data.shape[0]):
            for col_idx in range(data.shape[1]):
                method = data.index[row_idx]
                dataset = data.columns[col_idx]
                value = data.iloc[row_idx, col_idx]
                step_name = data.iloc[row_idx]["Unnamed: 1"]
                if isinstance(value, str) and value.startswith(prefix) and (
                        str(step_name).lower() == "step2" or str(step_name).lower() == "step 2"):  #TODO add step3
                    positive_json_path=f"patterns/{task}_{dataset}_{method}_pattern.json"
                    negative_json_path=f"patterns/only_apr_neg_{task}_{dataset}_{method}_pattern.json"
                    if not os.path.exists(positive_json_path) or not os.path.exists(negative_json_path):
                        logger.warning(f"{positive_json_path} or {negative_json_path} not exists")
                        continue
                    with open(positive_json_path,"r") as f:
                        positive_pattern=json.load(f)
                    with open(negative_json_path,"r") as f:
                        negative_pattern=json.load(f)
                    ans_list=filter_pattern_forest(positive_pattern)+filter_pattern_apr(positive_pattern,ascending=ascendings[i])+filter_pattern_apr(negative_pattern,not ascendings[i])
                    vis_list=[{**dict_element,"dataset":dataset,"mehtod":method} for dict_element in ans_list]
                    task_all=task_all+vis_list
                else:
                    continue
        pd.DataFrame(task_all).to_csv(f"{task}_pattern.csv")