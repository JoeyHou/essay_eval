import pandas as pd
import json
import argparse
import os 

# import krippendorff
import numpy as np
import pandas as pd
# from datasets import load_from_disk
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain.prompts import PromptTemplate
# from numpy import mean, std
# from scipy.stats import kendalltau
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

import sys
sys.path.insert(1, 'src/')
# from analysis import calculate_similarity_for_cluster_part, create_embedding, compare_texts
from essay_meta_data import essay_set_descriptions
from evaluation_helpers import qwk


def evaluation_batch(logging_data_dir):
    all_results = []
    for logging_data_path in os.listdir(logging_data_dir):
        if 'mistral_' not in logging_data_path: continue 
        result_df = evaluation(
            args.logging_data_path, 
            args.essay_data_path, 
            args.metric
        )
        result_df['run_id'] = logging_data_path.replace('.json', '')
        all_results.append(result_df)
    return pd.concat(all_results)

def evaluation(
        
        logging_data_path = "./log.json", 
        essay_data_path = "./data/asap/training_set_rel3.xlsx", 
        metric = 'qwk'
    ):

    if metric != 'qwk': raise NotImplementedError

    df = pd.read_excel(essay_data_path, index_col="essay_id")
    df = df.drop(10534)  # this essay is not annotated 
    
    # run_idx = 1
    # var_lst = [
    #     "_var1", 
    #     "_var2", 
    #     "_var3", 
    #     "_var4"
    # ]
    # prompt_lst = [
    #     "holistic_scoring_prompt1",
    #     "holistic_scoring_prompt2",
    #     "holistic_scoring_prompt3",
    #     "explanation_somewhere_prompt",
    #     "explanation_first_prompt",
    #     "feedback_somewhere_prompt",
    #     "feedback_first_prompt",
    #     "feedback_and_explanation_prompt",
    #     # "persona_prompt",
    #     "chain_of_thought_prompt",
    #     "chain_of_thought_detailed_prompt",
    #     "one_shot_prompt",
    # ]
    # new_prompt_lst = [] #[p for p in prompt_lst]
    # for var in var_lst:
    #     for p in prompt_lst:
    #         new_prompt_lst.append(p + var)
    # prompt_lst = new_prompt_lst
    # print('total num of prompt:', len(prompt_lst))
    # experiments = [
    #     {
    #         "experiment_id": "test_{}".format(i),
    #         # "data_path": logging_data_path,
    #         "data_path": "./log/run_{}/mistral_7b_test_v1_{}".format(run_idx, prompt_lst[i]), #log_fp,
    #         "prompt": prompt_lst[i],
    #         "template": 3,
    #         "variant": 1 if '_var' not in prompt_lst[i] else int(prompt_lst[i][-1]),
    #         "model": "mistral",
    #     } for i in range(len(prompt_lst))
    # ]
    # print('total num of experiments:', len(experiments))

    prompt_lst = ['fine_grained']
    experiments = [
        {
            "experiment_id": "run_0",
            # "data_path": logging_data_path,
            "data_path": logging_data_path,
            "prompt": prompt_lst[i],
            "template": -1,
            "variant": -1,
            "model": "",
        } for i in range(len(prompt_lst))
    ]
    print('total num of experiments:', len(experiments))



    file_loading_err = []
    df_data = []
    experiment_results = {}
    for i in tqdm(range(len(experiments))):
        experiment = experiments[i]
        try:
            with (open(experiment["data_path"], "r")) as f:
                results = json.load(f)
        except:
            file_loading_err.append(experiment["data_path"])
            continue 
        # experiments[i]["results"] = results ## EDIT
        # print(1)
        experiment_id = experiment['experiment_id'] ## EDIT
        df[f"{experiment_id}_predicted_score"] = 0 ## EDIT
        experiment_results[experiment_id] = results ## EDIT
        
        for fold in results:
            for j in range(len(results[fold])):
                essay_id = results[fold][j]["id"]
                output = results[fold][j]["parsed_output"]
                essay_set = df.loc[essay_id]['essay_set']
                meta_data = essay_set_descriptions[essay_set - 1]
                try:
                    score = meta_data['full_score_fn'](output, output)
                except Exception as e:
                    score = -1  # mark the score as incorrect to filter it out later
                df.loc[essay_id, f"{experiment_id}_predicted_score"] = score
        # print(2)
        wrong_predictions = 0
        for essay_set in range(1, 9):
            meta_data = essay_set_descriptions[essay_set - 1]
            score_range = meta_data['score_ranges'][0]
            filtered_df = df[df["essay_set"] == essay_set]
            # display(filtered_df.head())
            # print("score_range:", score_range)
            incorrect_predictions = filtered_df[(filtered_df[f"{experiment_id}_predicted_score"] < score_range[0]) | (
                    filtered_df[f"{experiment_id}_predicted_score"] > score_range[1])]
            # change the scores to -1 for the incorrect predictions in the original dataframe
            df.loc[incorrect_predictions.index, f"{experiment_id}_predicted_score"] = -1
            # print(essay_set, len(incorrect_predictions), filtered_df.shape)
            wrong_predictions += len(incorrect_predictions)
        
        # print("num of -1:", sum(df[f"{experiment_id}_predicted_score"] == -1))
        # print("df shape:", df.shape)
        # print(wrong_predictions)
        experiments[i]["wrong_predictions"] = wrong_predictions
        # print(3)
        # calculate the QWK score for each essay set
        qwks = [[] for _ in range(8)]
        for fold in results:
            essay_set = fold.split("_")[-1]
            fold_index = int(fold.split("_")[1])
            # print(3.1)
            filtered_df = df[df["essay_set"] == int(essay_set) + 1]
            # print(3.2)
            # display(filtered_df.head(2))
            # break
            # get the true and predicted scores but filter out the incorrect predictions with -1
            clean_true_scores = filtered_df[filtered_df[f"{experiment_id}_predicted_score"] != -1]["domain1_score"].filter(
                items=[essay['id'] for essay in experiment_results[experiment_id][fold]])
            # print(3.3)
            clean_predicted_scores = filtered_df[filtered_df[f"{experiment_id}_predicted_score"] != -1][
                f"{experiment_id}_predicted_score"].filter(
                items=[essay['id'] for essay in experiment_results[experiment_id][fold]])
            # print(3.4)
            qwks[int(essay_set)].append(qwk(clean_predicted_scores, clean_true_scores))
            # print(essay_set)
            # print(clean_predicted_scores.values)
        # print(4)
        # take the average over all folds
        qwks = [
            sum(score) / len(score) if len(score) > 0 else -1
            for score in qwks
        ]

        # qwk_avg = np.mean([qwk for qwk in qwks if qwk != -1])
        qwk_avg = np.nanmean([qwk for qwk in qwks if qwk != -1])
        # print(qwks)
        # save the results in a pandas dataframe
        new_df_entry = {}
        for essay_set in range(1, 9):
            meta_data = essay_set_descriptions[essay_set - 1]
            score_range = meta_data['score_ranges'][0]
            filtered_df = df[df["essay_set"] == essay_set]
            calculated_qwk_filtered = qwks[essay_set - 1]

            new_df_entry.update({
                'Average': qwk_avg,
                'Incorrect Predictions': wrong_predictions, 'template': experiment['template'],
                'variant': experiment['variant'], 'prompt': experiment['prompt'], f"Essay Set {essay_set}": calculated_qwk_filtered
            })
        df_data.append(new_df_entry)

    # create the dataframe and save it to a csv file
    df_data = pd.DataFrame(df_data)
    # print(file_loading_err)
    # print(df_data)
    # if 'prompt' in df_data
    df_data['prompt_group'] = df_data['prompt'].apply(lambda x: x if '_var' not in x else '_'.join(x.split('_')[:-1]) )
    result_df = df_data.drop(columns = ["template", "variant", "prompt", "Incorrect Predictions"]).groupby('prompt_group').agg('mean')
    
    return result_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    ## run configurations
    # parser.add_argument("--config", type=str, default='')
    parser.add_argument("--logging_data_path", type=str, default="./log.json")
    parser.add_argument("--essay_data_path", type=str, default="./data/asap/training_set_rel3.xlsx")
    parser.add_argument("--metric", type=str, default="qwk")
    parser.add_argument("--qwk_summary_path", type=str)
    args = parser.parse_args()

    result_df = evaluation(
        args.logging_data_path, 
        args.essay_data_path, 
        args.metric
    )
    # if '.json' in args.logging_data_path: # evaluate single output
    #     result_df = evaluation(
    #         args.logging_data_path, 
    #         args.essay_data_path, 
    #         args.metric
    #     )
    # elif os.path.isdir(args.logging_data_path):
    #     result_df = evaluation_batch(args.logging_data_path)
    # else:
    #     raise NotImplementedError
    result_df.to_csv(args.qwk_summary_path)