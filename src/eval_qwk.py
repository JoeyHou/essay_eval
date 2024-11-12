import pandas as pd
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(1, 'src/')
# from analysis import calculate_similarity_for_cluster_part, create_embedding, compare_texts
from essay_meta_data import essay_set_descriptions
# from evaluation_helpers import qwk
from sklearn.metrics import cohen_kappa_score

# def evaluation_batch(logging_data_dir):
#     all_results = []
#     for logging_data_path in os.listdir(logging_data_dir):
#         if 'mistral_' not in logging_data_path: continue 
#         result_df = evaluation(
#             args.logging_data_path, 
#             args.essay_data_path, 
#             args.metric
#         )
#         result_df['run_id'] = logging_data_path.replace('.json', '')
#         all_results.append(result_df)
#     return pd.concat(all_results)

def evaluation(
        logging_data_path = "./log.json", 
        essay_data_path = "./data/asap/training_set_rel3.xlsx", 
        metric = 'qwk'
    ):

    if metric != 'qwk': raise NotImplementedError
    
    # 0. Load relevant data files
    essay_df = pd.read_excel(essay_data_path, index_col="essay_id")
    if 'asap' in essay_data_path:
        essay_df = essay_df.drop(10534)  # this essay is not annotated 
        true_score_col = 'domain1_score'
    else:
        raise NotImplementedError
    
    try:
        with (open(logging_data_path, "r")) as f:
            results = json.load(f)
    except:
        print('[ERROR] cannot load logging data file from: {}'.format(logging_data_path))
        return 
    
    # 1. get prediction scores
    all_prediction_scores = {}
    all_qwks = []
    for fold in results:
        all_prediction_scores[fold] = []
        for dp in results[fold]:

            # 1.1 load scores from logging
            essay_id = dp['id']
            parsed_output = dp['parsed_output']
            essay_set = essay_df.loc[essay_id]['essay_set']
            meta_data = essay_set_descriptions[essay_set - 1]
            try:
                score = meta_data['full_score_fn'](parsed_output, parsed_output)
            except Exception as e:
                score = -1  # mark the score as incorrect to filter it out later
            
            # 1.2 check score range
            score_range = meta_data['score_ranges'][0]
            if score < score_range[0] or score > score_range[1]:
                score = -1

            # 1.3 record the numbers 
            all_prediction_scores[fold].append({
                'essay_id': essay_id,
                'score_pred': score,
                'score_true': essay_df.loc[essay_id][true_score_col]
            })
    
    # 2. calculate qwk
    qwk_data = {
        'run_id': logging_data_path.replace('./log/', '').replace('/cleaned_output.json', ''),
        'avg_qwk': -1
    }
    for i in range(1, 9):
        qwk_data['prompt_{}'.format(i)] = -1
    tmp_valid_qwk = []
    for fold in all_prediction_scores:
        essay_set = int(fold.split("_")[-1])
        valid_data = [dp for dp in all_prediction_scores[fold] if dp['score_pred'] != -1]
        tmp_qwk = cohen_kappa_score(
            [dp['score_pred'] for dp in valid_data], # y_pred
            [dp['score_true'] for dp in valid_data], # y_true
            weights='quadratic'
        )
        qwk_data['prompt_{}'.format(essay_set + 1)] = round(tmp_qwk, 4)
        if tmp_qwk != -1:
            tmp_valid_qwk.append(tmp_qwk)
    qwk_avg = round(np.nanmean(tmp_valid_qwk), 4) if len(tmp_valid_qwk) > 0 else -1
    qwk_data['avg_qwk'] = qwk_avg

    # 3. format to dataframe
    # print(qwk_data)
    qwk_df = pd.DataFrame([qwk_data])
    return qwk_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    ## run configurations
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
    result_df.to_csv(args.qwk_summary_path, index = False) 