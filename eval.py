import pandas as pd 
import numpy as np
import argparse
import json 
import re
import pickle
import readability
from tqdm import tqdm 

# importing module
import logging
logging.basicConfig(
    filename="log/history.log",
    format='%(asctime)s %(message)s',                    
    filemode='a'
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

## local modules ##
import sys
sys.path.insert(1, 'src/')
from utils import (
    load_json,
    write_json,
    openai_call,
    # parsing_delete_uneccessary_json,
    # parsing_preprocessing,
    load_jsonl,
    clean_single_score
)
from essay_meta_data import essay_set_descriptions ## essay meta data ##
from langchain_community.llms import VLLM ## VLLM Support ##
from vllm import LLM, SamplingParams

from fine_grained_prompts import (
    prompt_template_1,
    prompt_template_2,
    prompt_template_3,
    prompt_template_4,
    prompt_template_5,
    analysis_instruction_explanation_holistic,
    analysis_instruction_explanation_fg,
    format_instruction_score_and_analysis,
    parsing_prompt_holistic,
)
from eval_qwk import evaluation
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema

## huggingface setup ##
import os 
home_dir = '/ihome/xli/joh227/'
os.environ['HF_TOKEN'] = load_json(home_dir + 'developer/huggingface_key.json')['personal']
os.environ['HF_HOME'] = home_dir + 'ix_dir/huggingface/'

## openai setup ##
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = load_json(home_dir + 'developer/openai_key.json')['group']

## torch setup ##
import torch 
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

## helper function - prompt ##
def make_prompt(prompt_template, data):
    return prompt_template.format(**data)

## helper function - models ##
def load_llama_vllm(model, max_length=4096, temperature=0.01):
    """
    Load a LLaMA model with specified parameters.

    Parameters:
    - max_length (int, optional): The maximum token length for the model's outputs. Defaults to 4096.
    - temperature (float, optional): The temperature for sampling outputs. Defaults to 0.01.

    Returns:
    - VLLM: The loaded Mistral model.
    """
    if model == 'llama3':
        return VLLM(model="meta-llama/Meta-Llama-3-8B-Instruct", max_length=max_length, temperature=temperature)
    elif model == 'llama31':
        return VLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_length=max_length, temperature=temperature)
    elif model == 'llama3-70':
        return VLLM(model="meta-llama/Meta-Llama-3-70B-Instruct", max_length=max_length, temperature=temperature, tensor_parallel_size=4)

def load_mistral_vllm(max_length=4096, temperature=0.01, tensor_parallel_size = 1):
    """
    Load a Mistral model with specified parameters.

    Parameters:
    - max_length (int, optional): The maximum token length for the model's outputs. Defaults to 4096.
    - temperature (float, optional): The temperature for sampling outputs. Defaults to 0.01.

    Returns:
    - VLLM: The loaded Mistral model.
    """
    return VLLM(
        model="mistralai/Mistral-7B-Instruct-v0.2", 
        max_length=max_length, 
        temperature=temperature, 
        tensor_parallel_size=tensor_parallel_size,
        # vllm_kwargs={'distributed-executor-backend': 'ray'}
    )

def parse_json(json_str):
    json_str = json_str.replace('\n', '')
    
    # find start
    start = json_str.index('{') if '{' in json_str else 0
    json_str = json_str[start:]

    # find end
    end_counter = 0
    end = 0
    for i in range(len(json_str)):
        if json_str[i] == '}':
            end_counter += 1
        if end_counter == 2:
            end = i + 1
            break 

    json_str = json_str[:end]
    try:
        return json.loads(json_str)
    except:
        logger.info('=================\n[ERROR] parsing_output cannot be parsed by json:\n' + json_str)
        return {}

## FineGrainEvaluator ##
class FineGrainEvaluator():

    def __init__(self, args):
        self.essay_meta_data = essay_set_descriptions
        self.args = args
        if not args.skip_llm and not args.test_prompt:
            if args.model == 'mistral':
                self.llm = load_mistral_vllm()
                # self.llm = LLM("mistralai/Mistral-7B-Instruct-v0.2")#, tensor_parallel_size = tensor_parallel_size)
                # self.sampling_params = SamplingParams(temperature=0.00, max_tokens=4096)
            elif 'llama' in args.model:
                pass
                # self.llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct")#, tensor_parallel_size = tensor_parallel_size)
                # self.sampling_params = SamplingParams(temperature=0.00, max_tokens=4096)
            elif 'gpt' in args.model:
                # self.gpt = 
                self.prep_gpt()
                self.llm = load_mistral_vllm()
            else:
                raise NotImplementedError
        
        self.model = args.model 
        self.limit = args.limit if args.limit != -1 else 999999999999
        
        self.load_prompt(args)
        self.hard_code_holistic_prompt_key = 'Overall'

        ## define output location
        self.cleaned_output_path = './log/{}/cleaned_output.json'.format(args.run_id)
        self.raw_output_path = './log/{}/raw_output.json'.format(args.run_id)
        self.parsing_log_path = './log/{}/parsing_log.csv'.format(args.run_id)
        self.qwk_summary_path = './log/{}/qwk.csv'.format(args.run_id)
        os.makedirs('./log/{}'.format(args.run_id), exist_ok = True)

    def load_prompt(self, args):
        # prompt_template
        prompt_templates = [
            prompt_template_1,
            prompt_template_2,
            prompt_template_3,
            prompt_template_4,
            prompt_template_5
        ]
        self.prompt_template = prompt_templates[args.prompt_template - 1]
        
        self.fine_grained = args.fine_grained
        if args.use_machine_rubric:
            self.fine_grained_rubric_key = "fine_grained_rubric_machine"
        elif args.use_asap_pp_rubric:
            self.fine_grained_rubric_key = "fine_grained_rubric_asap_pp"
        else:
            self.fine_grained_rubric_key = "fine_grained_rubric"
        self.verbalize = args.verbalize
        
        # self.parsing_prompt_fg = parsing_prompt_fine_grained
        self.parsing_prompt_holistic = parsing_prompt_holistic
        
        if args.analysis_instruction == "explanation":
            self.analysis_instruction = analysis_instruction_explanation_fg if self.fine_grained else analysis_instruction_explanation_holistic
            self.format_instruction = format_instruction_score_and_analysis
        else:
            raise NotImplementedError

    # from llm_functions.py 
    def generate_rubics(self, meta_data, criteria = ""): 
        """
        Generates small and full rubrics based on essay set metadata.

        Parameters:
        - meta_data (dict): The metadata for the essay set, including scoring rubric details.

        Returns:
        - tuple: A tuple containing the small rubric and the full rubric as strings.
        """
        rubric_small = ""
        rubric_overall_description = ""
        if self.fine_grained:
            for points in meta_data[self.fine_grained_rubric_key][criteria]:
                if points == "overall_description":
                    rubric_overall_description = "(i.e., {})".format(meta_data[self.fine_grained_rubric_key][criteria][points].strip())
                else:
                    description = meta_data[self.fine_grained_rubric_key][criteria][points]['description']
                    rubric_small += f"{points} points: {description}\n"
            return rubric_small, rubric_overall_description
        else:
            for criteria in meta_data["scoring_rubric"]:
                rubric_small += f"Scoring rubric for '{criteria}':\n"
                for points in meta_data["scoring_rubric"][criteria]:
                    if points == "overall_description": continue 
                    description = meta_data["scoring_rubric"][criteria][points]['description']
                    rubric_small += f"{points} points: {description}\n"
            return rubric_small, rubric_overall_description
    
    def prep_gpt(self):
        self.client = OpenAI()

    def prompt_llm(self, inputs, batch_size = 10):
        if isinstance(inputs, dict):
            inputs = [inputs]
        logger.info('Running prompt_llm with model: {}'.format(self.model))
        if 'gpt' in self.model:
            openai_config   = {
                'model': self.model,
                'temperature': 0,
                'max_tokens': 1024
            }
            self.openai_cache_path = './log/{}/openai_cache.pkl'.format(args.run_id)
            try:
                with open(self.openai_cache_path, 'rb') as f:
                    llm_output = pickle.load(f)
            except:
                llm_output = []
            starting_idx = len(llm_output)
            counter = 0
            for prompt in tqdm(inputs[starting_idx:]):
                llm_output.append(openai_call(prompt, self.client, openai_config))
                counter += 1
                if counter % batch_size == 0:
                    pickle.dump(llm_output, open(self.openai_cache_path, 'wb'))
        else:
            llm_output = self.llm.batch(inputs)
            # llm_output = [output.outputs[0].text for output in self.llm.generate(inputs, self.sampling_params)]
        return llm_output

    def parse_output_via_re(self, llm_output, trait = "Overall"):
        try:
            score = float(re.search('### Score: {0,1}\d', llm_output, re.DOTALL).group(0).replace("### Score:", "").strip())
            parsed_score = True
        except:
            score = -1
            parsed_score = False

        try:
            explanation = re.search('### Explanation:.+\d+.+###', llm_output, re.DOTALL).group(0)
            parsed_explanation = True
        except:
            explanation = ""
            parsed_explanation = False
        feedbacks = ""

        result = {
            "Score": {trait: score},
            "Explanation": explanation,
            "Feedbacks": feedbacks,
            "Stats": (parsed_score, parsed_explanation)
        }
        
        return result
    
    def parse_output_via_prompting(self, raw_log_data):
        # print('self.parsing_prompt_template:', self.parsing_prompt_template)
        self.parsing_sep = '\n########\n'
        parsing_inputs = []
        parsed_llm_outputs = {} # {fold: {...}}
        for fold in raw_log_data:
            parsed_llm_outputs[fold] = {} # {essay_id: {prompt_key: parsed_result}}
            for dp in raw_log_data[fold]:
                parsing_input_dp = self.parsing_sep.join([
                    fold, 
                    str(dp['id']), 
                    dp['prompt_key'],
                    self.parsing_prompt_holistic.replace('###LLM_OUTPUT', dp['output'])
                ])
                if self.fine_grained:
                    parsing_input_dp.replace('### Score:', '### Score: -{}'.format(dp['prompt_key']))
                parsing_inputs.append(parsing_input_dp)
                
        if 'gpt' in self.model:
            # raise NotImplementedError # pass 
            cleaned_output = self.llm.batch(parsing_inputs)
        else:
            cleaned_output = self.llm.batch(parsing_inputs)
            # cleaned_output = [output.outputs[0].text for output in self.llm.generate(parsing_inputs, self.sampling_params)]
        
        for i in range(len(parsing_inputs)):
            parsing_output = cleaned_output[i].replace('"""', '').strip()
            parsing_output = parse_json(parsing_output)
            parsing_intput = parsing_inputs[i]
            fold, essay_id, prompt_key, _ = parsing_intput.split(self.parsing_sep)
            parsed_llm_outputs[fold][(essay_id, prompt_key)] = parsing_output

        return parsed_llm_outputs

    def prep_prompt(self, essay_batch):
        '''
        essay_batch: [(essay, essay_id, essay_set, essay_prompt), ...] 
            - essay_prompt only exists for ellipse!
        '''
        all_prompts = []
        all_prompts_id = []
        essay_counter = 0
        counter_per_set = {
            k: 0 for k in range(1, 10)
        }

        if "mistral" in self.model:
            model_prefix= "<s>[INST]"
            model_suffix= "[/INST]"
        elif "llama" in self.model:
            model_prefix = "<|begin_of_text|>"
            model_suffix=""
        else:
            model_prefix=""
            model_suffix=""
        
        ### loading Linguistic Features ### 
        ling_feature_data = {}
        use_ling_feature = False
        if self.args.ling_features != []:
            if self.args.dataset == 'asap':
                try:
                    ling_feature_data = pd.read_csv('./data/asap_ling_feats.csv')
                    ling_feature_data.index = ling_feature_data['essay_id'] # use essay id as the index 
                    use_ling_feature = True
                except:
                    logger.info('[ERROR] Failed to load asap linguistic features!')
            else:
                try:
                    ling_feature_data = pd.read_csv('./data/ELLIPSE_ling_feats.csv')
                    ling_feature_data.index = ling_feature_data['text_id_kaggle'] # use essay id as the index 
                    use_ling_feature = True
                except:
                    logger.info('[ERROR] Failed to load LLIPSE linguistic features!')
            ling_feature_desc_dict = load_json('./data/ling_feature_keys.json')

        for i in range(len(essay_batch)):
            essay, essay_id, essay_set, essay_prompt = essay_batch[i]
            if counter_per_set[essay_set] == self.limit: continue
            if self.args.use_machine_rubric and essay_set in [2, 7, 8]: continue # skip 2, 7, 8 when using machine rubrics
                
            # essay set specific data
            meta_data = self.essay_meta_data[essay_set - 1]
            if self.args.dataset == 'asap':
                essay_prompt = meta_data['prompt']
            elif self.args.dataset == 'ellipse':
                essay_prompt = essay_prompt
            if self.fine_grained:
                fine_grained_prompts = meta_data[self.fine_grained_rubric_key]
            else:
                fine_grained_prompts = {
                    self.hard_code_holistic_prompt_key: "" # use "Overall" as key
                }

            for prompt_key in fine_grained_prompts:

                ## getting scoring range 
                if self.fine_grained:
                    fine_grained_prompt = prompt_key
                    # score_format = str({prompt_key: ""}).replace("'", '"')
                    if essay_set == 2:
                        if self.args.use_asap_pp_rubric:
                            scoring_range = (1, 6)
                        else:
                            if prompt_key == "Writing Applications":
                                scoring_range = (1, 6)
                            elif prompt_key == "Language Conventions":
                                scoring_range = (1, 4)
                    else:
                        scoring_range = meta_data["single_evaluator_score_ranges"][0]
                    scoring_range = f"from {scoring_range[0]} to {scoring_range[1]}"
                else:
                    fine_grained_prompt = ""
                    scoring_range = "\n".join([
                        f"{score_type}: from {meta_data['single_evaluator_score_ranges'][j][0]} to {meta_data['single_evaluator_score_ranges'][j][1]}"
                        for j, score_type in enumerate(meta_data["scoring_rubric"])
                    ])
                if self.args.dataset == 'ellipse':
                    scoring_range += ', inclusive, with 0.5 score increment'
                elif self.args.dataset == 'asap':
                    scoring_range += ', inclusive, with 1.0 score increment'
                
                if not self.fine_grained and self.args.calibration and essay_set in [1, 3, 4, 5, 6, 9]:
                    calibration_data = load_json('./data/score_dist.json')
                    calibration_str = calibration_data[self.args.dataset][str(int(essay_set))]
                    scoring_range += '\n- Score likelyhood based human grading: \n' + calibration_str
                
                ## geting rubrics
                rubric, overall_description = self.generate_rubics(meta_data, prompt_key) ## TODO: check rubric generation, might have some problem 

                ## put together for [analysis instruction]
                if self.fine_grained:
                    if self.verbalize: # fine-grained and verbalized
                        tmp_analysis_instruction = self.analysis_instruction.format(
                            fine_grained_prompt = fine_grained_prompt,
                            scoring_range = scoring_range,
                            rubric = "\n- Here are some grading reference: " + rubric
                        )
                    else: # fine-grained only
                        tmp_analysis_instruction = self.analysis_instruction.format(
                            fine_grained_prompt = fine_grained_prompt,
                            scoring_range = scoring_range,
                            rubric = overall_description
                        )
                else:
                    tmp_analysis_instruction = self.analysis_instruction.format(
                        fine_grained_prompt = fine_grained_prompt,
                        scoring_range = scoring_range,
                        # rubric = "\n- Here are some grading reference: " + rubric
                        rubric = ""
                    )
                    
                ## put together for [linguistic feature]
                additional_information = ''
                if use_ling_feature:
                    additional_information = '### Additional Information:\nStudies show that the following features are highly, positively correlated with the grade of the essay (i.e., higher features typically means higher end score)\n'
                    for ling_feature in self.args.ling_features:
                        if ling_feature not in ling_feature_data.columns:
                            continue
                        ling_feature_desc = ling_feature_desc_dict[ling_feature]
                        additional_information += '- {}: {}\n'.format(
                            ling_feature_desc,
                            ling_feature_data.loc[essay_id][ling_feature],
                        )

                ## put together for [formatting instruction] 
                tmp_format_instruction = self.format_instruction.strip()
                if self.fine_grained:
                    tmp_format_instruction += '\n- {}:'.format(prompt_key)
                else:
                    for tmp_prompt_key in meta_data["scoring_rubric"]:
                        tmp_format_instruction += '\n- {}:'.format(tmp_prompt_key)
                
                ## put together for the entire prompt
                llm_prompt = make_prompt(
                    self.prompt_template,
                    {
                        'essay_prompt': essay_prompt,
                        'essay': essay,
                        'analysis_instruction': tmp_analysis_instruction, 
                        'format_instruction': tmp_format_instruction,
                        'model_prefix': model_prefix, 
                        'model_suffix': model_suffix,
                        'additional_information': additional_information
                    }
                )
                all_prompts.append(llm_prompt)
                all_prompts_id.append((essay_id, float(essay_set), prompt_key, essay))
            counter_per_set[essay_set] += 1
        return all_prompts, all_prompts_id
    
    def process_batch(self, essay_batch):
        
        ## Part 1: LLM prompting (could be skipped)
        if not self.args.skip_llm:
            all_prompts, all_prompts_id = self.prep_prompt(essay_batch)
            if not self.args.test_prompt:
                llm_outputs = self.prompt_llm(all_prompts)
            
                ## result post processing ##
                raw_log_data = {}
                for i in range(len(all_prompts)):
                    tmp_essay_set = int(all_prompts_id[i][1]) - 1
                    if 'fold_0_{}'.format(tmp_essay_set) not in raw_log_data:
                        raw_log_data['fold_0_{}'.format(tmp_essay_set)] = []
                    if self.args.dataset == 'asap':
                        essay_id = float(all_prompts_id[i][0])
                    else:
                        essay_id = all_prompts_id[i][0]
                    tmp_dp = {
                        'essay': all_prompts_id[i][3],
                        'essay_set': all_prompts_id[i][1],
                        'id': essay_id,
                        'output_parsing_info': None,
                        'output': llm_outputs[i],
                        'llm_prompt': all_prompts[i],
                        'prompt_key': all_prompts_id[i][2],
                    }
                    raw_log_data['fold_0_{}'.format(tmp_essay_set)].append(tmp_dp)
                with open(self.raw_output_path, 'w') as f:
                    # print(raw_log_data)
                    f.write(json.dumps(raw_log_data))
            else:
                pickle.dump(all_prompts, open('tmp/tmp.prompts.pkl', 'wb'))
                return all_prompts
        else:
            with open(self.raw_output_path, 'r') as f:
                raw_log_data = json.load(f)
        
        ## Part 2. parsing results
        parsed_llm_outputs = self.parse_output_via_prompting(raw_log_data)
        # write_json(parsed_llm_outputs, 'parsed_llm_outputs.json')
        pickle.dump(parsed_llm_outputs, open('tmp/parsed_llm_outputs.pkl', 'wb'))
        ## merge sub category scores ## 
        processed_log_data = {k: [] for k in list(raw_log_data.keys())}
        parsing_stats = {}
        
        ## Part 3. accumulate the scores and put them together
        for fold in raw_log_data:
            all_essay_id = np.unique([tmp_dp['id'] for tmp_dp in raw_log_data[fold]])
            tmp_dp_dict = {k: [] for k in all_essay_id}
            for tmp_dp in raw_log_data[fold]:
                tmp_dp_dict[tmp_dp['id']].append(tmp_dp)
            
            total_score, total_neg = 0, 0
            for essay_id in tmp_dp_dict:
                tmp_groupped_output = {
                    'essay': tmp_dp_dict[essay_id][0]['essay'],
                    'id': tmp_dp_dict[essay_id][0]['id'],
                    'output_parsing_info': None,
                    'output': [dp['output'] for dp in tmp_dp_dict[essay_id]],
                    'llm_prompt': {dp['prompt_key']: dp['llm_prompt'] for dp in tmp_dp_dict[essay_id]},
                    'raw_parsed_output': {
                        dp['prompt_key']: parsed_llm_outputs[fold][(str(essay_id), dp['prompt_key'])]
                        for dp in tmp_dp_dict[essay_id]
                    }
                }
                
                # get the scores from raw parsed output
                if self.fine_grained:
                    essay_set_number = tmp_dp_dict[essay_id][0]['essay_set']
                    prompt_keys = [prompt_key for prompt_key in tmp_groupped_output['raw_parsed_output']]
                    scores = []
                    for prompt_key in prompt_keys:
                        try:
                            score = tmp_groupped_output['raw_parsed_output'][prompt_key]['Score'][prompt_key]
                        except:
                            try:
                                score = tmp_groupped_output['raw_parsed_output'][prompt_key]['Score'][self.hard_code_holistic_prompt_key]
                            except:
                                score = -1
                        scores.append(score)
                    cleaned_scores = [clean_single_score(score_str) for score_str in scores]
                    total_score += len(list(cleaned_scores))
                    total_neg += sum([score == -1 for score in cleaned_scores])
                    
                    # decide the key of the output is either 'Overall' or '{fine_grained_category}'
                    if essay_set_number in [1, 3, 4, 5, 6]:
                        cleaned_scores = [s for s in cleaned_scores if s != -1]
                        overall_score = -1 if len(cleaned_scores) == 0 else round(np.mean(cleaned_scores))
                        score_dict = {
                            self.hard_code_holistic_prompt_key: overall_score
                        }
                    elif essay_set_number == 2 and self.args.use_asap_pp_rubric:
                        cleaned_scores = [s for s in cleaned_scores if s != -1]
                        overall_score = -1 if len(cleaned_scores) == 0 else round(np.mean(cleaned_scores))
                        score_dict = {
                            self.hard_code_holistic_prompt_key: overall_score
                        }
                    else:
                        score_dict = {
                            prompt_keys[i]: cleaned_scores[i]
                            for i in range(len(cleaned_scores))
                        }
                    tmp_groupped_output['parsed_output'] = score_dict
                else:
                    if 'Score' in tmp_groupped_output['raw_parsed_output'][self.hard_code_holistic_prompt_key]:
                        score_dict = tmp_groupped_output['raw_parsed_output'][self.hard_code_holistic_prompt_key]['Score']
                    else:
                        score_dict = {self.hard_code_holistic_prompt_key: -1}
                    for k in score_dict:
                        try:
                            score_dict[k] = (float(score_dict[k]))
                        except:
                            print('[Error] Cannot convert score into integer, score_dict[k] = {}(k = "{}")'.format(score_dict[k], k))
                            score_dict[k] = -1 
                    tmp_groupped_output['parsed_output'] = score_dict
                    total_score += len(list(score_dict.keys()))
                    total_neg += sum([score == -1 for score in list(score_dict.values())])

                processed_log_data[fold].append(tmp_groupped_output)
            parsing_stats[fold] = {
                'score_count': total_score,
                'invalid_count': total_neg,
                'invalid_prob': round(total_neg / total_score, 4)
            }
            with open(self.cleaned_output_path, 'w') as f:
                f.write(json.dumps(processed_log_data))
        
        # calculating parsing statistics
        parsing_log_df = post_process_parsing_log(parsing_stats)
        parsing_log_df.to_csv(self.parsing_log_path)
        return processed_log_data, raw_log_data

def post_process_parsing_log(parsing_stats):
    def parsing_log_acc(s):
        if 'count' in s.name:
            return s.sum()
        else:
            return s.mean()
    tmp_parsing_log_df = pd.DataFrame(parsing_stats)
    tmp_acc_stats = tmp_parsing_log_df.apply(parsing_log_acc, axis = 1)
    tmp_acc_stats['invalid_prob'] = round(tmp_acc_stats['invalid_count'] / tmp_acc_stats['score_count'], 4)
    tmp_parsing_log_df['acc_stats'] = tmp_acc_stats
    return tmp_parsing_log_df 

def load_asap_id(id_dir, split = 'train'):
    id_lst = []
    with open(id_dir + split + '_ids.txt', 'r') as f:
        for line in f.readlines():
            id_lst.append(int(line.strip()))
    return id_lst

# def run_multiple(config_dir):

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    
    ## run configurations
    # parser.add_argument("--config", type=str, default='')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--run_id", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--fine_grained", action="store_true")
    parser.add_argument("--verbalize", action="store_true")
    parser.add_argument("--use_machine_rubric", action="store_true")
    parser.add_argument("--use_asap_pp_rubric", action="store_true")
    parser.add_argument("--skip_llm", action="store_true") # default to be false
    parser.add_argument("--test_prompt", action="store_true") # default to be false
    
    ## model configurations
    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--temperature", type=float, default=0.00)
    parser.add_argument("--max_length", type=int, default=4096)

    ## prompt configurations
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--prompt_template", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--analysis_instruction", type=str, default="explanation")
    parser.add_argument("--ling_features", type=list, default=[])
    parser.add_argument("--ling_features_w_median", action="store_true")
    # parser.add_argument("--setting", type=str, default="one-shot", choices=["one-shot", "few-shot"])
    # parser.add_argument("--full-rubric", action="store_true")
    # parser.add_argument("--instruction-variant", type=int, default=1, choices=[1, 2, 3, 4])
    
    ## data configurations
    # parser.add_argument("--dataset_split", type=str, default="dev")
    
    ## running in batch only
    parser.add_argument("--config", type=str, default="")
    
    args = parser.parse_args()
    args_config = load_json(args.config)
    args_dataset = args_config['dataset'] if 'dataset' in args_config else 'asap'
    args_split = args_config['split'] if 'split' in args_config else 'dev'
    
    ## Load data ## 
    if args_dataset == 'asap':
        asap_data_dir = './data/asap/'
        train_ids = load_asap_id(asap_data_dir + '5_splits/fold_0/')
        dev_ids = load_asap_id(asap_data_dir + '5_splits/fold_0/', 'dev')
        test_ids = load_asap_id(asap_data_dir + '5_splits/fold_0/', 'test')
        id2split = dict(zip(train_ids, ['train'] * len(train_ids)))
        id2split.update(dict(zip(dev_ids, ['dev'] * len(dev_ids))))
        id2split.update(dict(zip(test_ids, ['test'] * len(test_ids))))

        # print(len(train_ids), len(dev_ids), len(test_ids))
        asap_data_all = pd.read_excel(asap_data_dir + 'training_set_rel3.xlsx', sheet_name = 'training_set')
        asap_data_all['split'] = asap_data_all.essay_id.apply(lambda x: id2split[x] if x in id2split else 'other')

        ## prepare essay data ##
        if args_split == "test":
            asap_data_all = asap_data_all.query('split == "test"')
        elif args_split == "dev":
            asap_data_all = asap_data_all.query('split == "dev"') 
        else:
            raise NotImplementedError
            # asap_data_all = asap_data_all.query('split == "dev"') 
        essay_batch = list(zip(
            asap_data_all['essay'].values, 
            asap_data_all['essay_id'].values, 
            asap_data_all['essay_set'].values,
            [''] * asap_data_all.shape[0] # use dummy for prompt
        ))
    elif args_dataset == 'ellipse':
        # print('[INFO] using ellipse dataset!!')
        ellipse_data = pd.read_csv('./data/ellipse_eval_df.csv')

        essay_batch = list(zip(
            ellipse_data['full_text'].values, 
            ellipse_data['text_id_kaggle'].values, 
            [9] * ellipse_data.shape[0],
            ellipse_data['prompt'].values
        ))

        if args_split == 'test':
            ellipse_gpt4_ids = pickle.load(open('data/ellipse_gpt4_ids.pkl', 'rb'))
            essay_batch = [dp for dp in essay_batch if dp[1] in ellipse_gpt4_ids]
            print('Loaded randomly sampled 500 ellipse essay!!\n\n')
        

    ### batch_running case
    run_list = args_config["experiment_lst"]
    logger.info("\n\n -------------------- List of experiments --------------------")
    for run in run_list:
        logger.info(str(run))
    
    all_result_df = []
    last_dir = ''
    for config in run_list:
        config['dataset'] = args_dataset
        t_args = argparse.Namespace()
        t_args.__dict__.update(config)
        args = parser.parse_args(namespace=t_args)
        logger.info(str(args))

        fge = FineGrainEvaluator(args)
        log_data = fge.process_batch(essay_batch)

        # cleanup
        try:
            del fge.llm 
        except:
            pass
        del fge
        del log_data