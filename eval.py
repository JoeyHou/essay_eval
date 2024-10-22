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
    filename="log/config.history.log",
    format='%(asctime)s %(message)s',                    
    filemode='w'
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.debug("Harmless debug Message")
# logger.info("Just an information")
# logger.warning("Its a Warning")
# logger.error("Did you try to divide by zero")
# logger.critical("Internet is down")

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
from fine_grained_prompts import (
    prompt_template_1,
    prompt_template_2,
    prompt_template_3,
    prompt_template_4,
    prompt_template_5,
    analysis_instruction_simple_holistic,
    analysis_instruction_feedback_holistic,
    analysis_instruction_explanation_holistic,
    analysis_instruction_comprehensive_holistic,
    analysis_instruction_simple_fg,
    analysis_instruction_feedback_fg,
    analysis_instruction_explanation_fg,
    analysis_instruction_comprehensive_fg,
    format_instruction_all,
    format_instruction_score_only,
    format_instruction_score_and_analysis,
    format_instruction_feedbacks
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


    
def load_mistral_vllm(max_length=4096, temperature=0.01):
    """
    Load a Mistral model with specified parameters.

    Parameters:
    - max_length (int, optional): The maximum token length for the model's outputs. Defaults to 4096.
    - temperature (float, optional): The temperature for sampling outputs. Defaults to 0.01.

    Returns:
    - VLLM: The loaded Mistral model.
    """
    return VLLM(model="mistralai/Mistral-7B-Instruct-v0.2", max_length=max_length, temperature=temperature)

## FineGrainEvaluator ##
class FineGrainEvaluator():

    def __init__(self, args):
        self.essay_meta_data = essay_set_descriptions
        self.args = args
        if not args.skip_llm:
            if args.model == 'mistral':
                self.llm = load_mistral_vllm()
            elif 'llama' in args.model:
                self.llm = load_llama_vllm(args.model)
            elif 'gpt' in args.model:
                self.llm = self.prep_gpt()
            else:
                raise NotImplementedError
        
        self.model = args.model 
        self.limit = args.limit if args.limit != -1 else 999999999999
        
        self.load_prompt(args)
        self.hard_code_holistic_prompt_key = 'Overall'

        ## define output location
        self.cleaned_output_path = './log/{}/cleaned_output.json'.format(args.run_id)
        self.raw_output_path = './log/{}/raw_output.json'.format(args.run_id)
        self.parsing_log_path = './log/{}/parsing_log.txt'.format(args.run_id)
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
        else:
            self.fine_grained_rubric_key = "fine_grained_rubric"
        self.verbalize = args.verbalize
        if args.analysis_instruction == "simple":
            self.analysis_instruction = analysis_instruction_simple_fg if self.fine_grained else analysis_instruction_simple_holistic
            self.format_instruction = format_instruction_score_only
        elif args.analysis_instruction == "feedback":
            self.analysis_instruction = analysis_instruction_feedback_fg if self.fine_grained else analysis_instruction_feedback_holistic
            self.format_instruction = format_instruction_feedbacks
        elif args.analysis_instruction == "explanation":
            self.analysis_instruction = analysis_instruction_explanation_fg if self.fine_grained else analysis_instruction_explanation_holistic
            self.format_instruction = format_instruction_score_and_analysis
        elif args.analysis_instruction == "all":
            self.analysis_instruction = analysis_instruction_comprehensive_fg if self.fine_grained else analysis_instruction_comprehensive_holistic
            self.format_instruction = format_instruction_all
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
                # print('meta_data["scoring_rubric"]:', meta_data["scoring_rubric"])
                for points in meta_data["scoring_rubric"][criteria]:
                    try:
                        description = meta_data["scoring_rubric"][criteria][points]['description']
                    except:
                        description = ''
                        print('[ERROR] meta_data["scoring_rubric"][criteria]:', meta_data["scoring_rubric"][criteria])
                    rubric_small += f"{points} points: {description}\n"
            return rubric_small, rubric_overall_description
    
    def prep_gpt(self):
        self.client = OpenAI()

    def prompt_llm(self, inputs, batch_size = 10):
        if isinstance(inputs, dict):
            inputs = [inputs]
        if 'gpt' in self.model:
            openai_config   = {
                'model': self.model,
                'temperature': 0,
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
        return llm_output

    def parse_llm_output(self, llm_output, trait = "Overall"):
        # tmp_out = llm_output.strip()
        # start_idx = tmp_out.find('{')
        # end_idx = tmp_out.rfind('}')
        # tmp_out = tmp_out[start_idx: end_idx + 1]
        # try:
        #     out_json = json.loads(tmp_out, strict = False)
        #     out_json["Stats"] = 1
        # except:
        #     out_json = {
        #         "Score": {self.hard_code_holistic_prompt_key: -1},
        #         "Explanation": "",
        #         "Feedbacks": "",
        #         "Stats": 0
        #     }
        # return out_json
        # print(re.search('### Score: {0,1}\d', llm_output, re.DOTALL))
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
        # print("llm_output", llm_output)
        # print("result", result)
        
        return result
        # llm_output = llm_output.replace('\n', '').strip()
        # print('---------------\n\n', llm_output,'\n------', re.search('### score:.+\{.+\}', llm_output), '\n-----------')
        # try:
        #     score = float(re.search('### score:.+\{.+\}', llm_output)\
        #                   .group(0).replace('### score:', '').strip())
        #     parsed_score = True
        # except:
        #     score = -1
        #     parsed_score = False
        # score = float(re.search('### score:.+\{.+\}', llm_output)\
        #                   .group(0).replace('### score:', '').strip())
        # score = re.search('### score \(json format\):.+\{.+\}', llm_output)
        # if score:
        #     score = float(score.group(0).replace('### score:', '').strip())
        # else:
        #     score = -1 
        # parsed_score = True

        # score = float(re.search('### score:.+\d+', llm_output)\
        #                   .group(0).replace('### score:', '').strip())
        # parsed_score = True
        
        # try:
        #     explanation = re.search('### explanation:.+###', llm_output, re.DOTALL)
        #     if explanation is not None:
        #         explanation = explanation.group(0).replace('### explanation:', '').replace('###', '').strip()
        #         parsed_exp = True
        #     else:
        #         explanation = re.search('### explanation:.+', llm_output, re.DOTALL)
        #         explanation = explanation.group(0).replace('### explanation:', '').strip()
        #         parsed_exp = True
        # except:
        #     explanation = 'n/a'
        #     parsed_exp = False
        
        # try:
        #     feedbacks = re.search('### feedbacks:.+', llm_output, re.DOTALL)\
        #         .group(0).replace('### feedbacks:', '').strip()
        #     parsed_sug = True
        # except:
        #     feedbacks = 'n/a'
        #     parsed_sug = False
        # return {
        #     'score': score,
        #     'explanation': explanation,
        #     'feedbacks': feedbacks,
        #     'stats': (parsed_score, parsed_exp, parsed_sug)
        # }


    def prep_prompt(self, essay_batch):
        '''
        essay_batch: [(essay, essay_id, essay_set), ...]
        '''
        all_prompts = []
        all_prompts_id = []
        essay_counter = 0
        counter_per_set = {
            k: 0 for k in range(1, 9)
        }

        if "mistral" in self.model:
            model_prefix= "<s>[INST]"
            model_suffix= "[/INST]"
        elif "llama" in self.model:
            model_prefix = "<|begin_of_text|>"
            model_suffix=""
        else:
            # raise NotImplementedError
            model_prefix=""
            model_suffix=""
        
        ling_feature_data = {}
        use_ling_feature = False
        if self.args.ling_features != []:
            try:
                ling_feature_data = pd.read_csv('./data/asap/hand_crafted_cleaned.csv')
                ling_feature_data.index = ling_feature_data['item_id'] # use essay id as the index 
                use_ling_feature = True
            except:
                print('[ERROR] Failed to load linguistic features!')

        for i in range(len(essay_batch)):
            essay, essay_id, essay_set = essay_batch[i]
            if counter_per_set[essay_set] == self.limit: continue
            if self.args.use_machine_rubric and essay_set in [2, 7, 8]: continue
                
            # essay set specific data
            meta_data = self.essay_meta_data[essay_set - 1]
            essay_prompt = meta_data['prompt']
            
            if self.fine_grained:
                fine_grained_prompts = meta_data[self.fine_grained_rubric_key]
            else:
                fine_grained_prompts = {
                    self.hard_code_holistic_prompt_key: ""
                }

            for prompt_key in fine_grained_prompts:
                if self.fine_grained:
                    fine_grained_prompt = prompt_key
                    score_format = str({prompt_key: ""}).replace("'", '"')
                    if essay_set == 2:
                        if prompt_key == "Writing Applications":
                            scoring_range = (1, 6)
                        elif prompt_key == "Language Conventions":
                            scoring_range = (1, 4)
                    else:
                        scoring_range = meta_data["single_evaluator_score_ranges"][0]
                    scoring_range = f"from {scoring_range[0]} to {scoring_range[1]}"
                else:
                    fine_grained_prompt = ""
                    score_format = str({k: "" for k in meta_data['scoring_rubric']}).replace("'", '"')
                    scoring_range = "\n".join([
                        f"{score_type}: from {meta_data['single_evaluator_score_ranges'][j][0]} to {meta_data['single_evaluator_score_ranges'][j][1]}"
                        for j, score_type in enumerate(meta_data["scoring_rubric"])
                    ])
                
                rubric, overall_description = self.generate_rubics(meta_data, prompt_key)
                if self.verbalize and self.fine_grained:
                    tmp_analysis_instruction = self.analysis_instruction.format(
                        fine_grained_prompt = fine_grained_prompt,
                        scoring_range = scoring_range,
                        rubric = "\n- Here are some grading reference: " + rubric
                    )
                else:
                    tmp_analysis_instruction = self.analysis_instruction.format(
                        fine_grained_prompt = fine_grained_prompt,
                        scoring_range = scoring_range,
                        rubric = overall_description
                    )
                additional_information = ''

                if use_ling_feature:
                    additional_information = '### Additional Information:\n'
                    # readability_results = readability.getmeasures(essay, lang='en')
                    for ling_feature in self.args.ling_features:
                        if ling_feature in ling_feature_data.columns:
                            tmp_ling_feature = round(ling_feature_data.loc[essay_id][ling_feature], 2) # round to 2nd 
                            group_median = round(ling_feature_data.loc[essay_id][ling_feature + '_median'], 2)
                        else:
                            continue

                        additional_information += '- {}: {} (median: {})\n'.format(
                            ling_feature,
                            tmp_ling_feature,
                            group_median
                        )
                        
                # print('tmp_analysis_instruction:', tmp_analysis_instruction)
                tmp_format_instruction = self.format_instruction.replace('{score_format}', score_format)
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
                all_prompts_id.append((float(essay_id), float(essay_set), prompt_key, essay))
            counter_per_set[essay_set] += 1
        return all_prompts, all_prompts_id
    
    def process_batch(self, essay_batch):
        
        ## option 1: skil the vllm part
        if not self.args.skip_llm:
            all_prompts, all_prompts_id = self.prep_prompt(essay_batch)
            llm_outputs = self.prompt_llm(all_prompts)
            
            ## result post processing ##
            raw_log_data = {}
            for i in range(len(all_prompts)):
                tmp_essay_set = int(all_prompts_id[i][1]) - 1
                if 'fold_0_{}'.format(tmp_essay_set) not in raw_log_data:
                    raw_log_data['fold_0_{}'.format(tmp_essay_set)] = []
                tmp_dp = {
                    'essay': all_prompts_id[i][3],
                    'essay_set': all_prompts_id[i][1],
                    'id': all_prompts_id[i][0],
                    'output_parsing_info': None,
                    'output': llm_outputs[i],
                    'llm_prompt': all_prompts[i],
                    'prompt_key': all_prompts_id[i][2],
                    # 'parsed_output': self.parse_llm_output(llm_outputs[i]),
                }
                raw_log_data['fold_0_{}'.format(tmp_essay_set)].append(tmp_dp)
            with open(self.raw_output_path, 'w') as f:
                f.write(json.dumps(raw_log_data))
        else:
            with open(self.raw_output_path, 'r') as f:
                raw_log_data = json.load(f)
        
        ## merge sub category scores ## 
        processed_log_data = {k: [] for k in list(raw_log_data.keys())}
        
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
                    # 'prompt_key': [dp['prompt_key'] for dp in tmp_dp_dict[essay_id]],
                    'raw_parsed_output': {
                        dp['prompt_key']: self.parse_llm_output(dp['output'], trait = dp['prompt_key']) 
                        for dp in tmp_dp_dict[essay_id]
                    }
                }
                # the parsed score is stored in parsed_output 

                if self.fine_grained:
                    essay_set_number = tmp_dp_dict[essay_id][0]['essay_set']
                    # scores = [
                    #     list(tmp_groupped_output['raw_parsed_output'][prompt_key]['Score'].values())[0]
                    #     for prompt_key in tmp_groupped_output['raw_parsed_output']
                    # ]
                    prompt_keys = [prompt_key for prompt_key in tmp_groupped_output['raw_parsed_output']]
                    # scores = [
                    #     tmp_groupped_output['raw_parsed_output'][prompt_key]['Score'][prompt_key]
                    #     for prompt_key in prompt_keys
                    # ]
                    scores = []
                    for prompt_key in prompt_keys:
                        # if prompt_key in tmp_groupped_output['raw_parsed_output'][prompt_key]['Score']:
                        #     score = tmp_groupped_output['raw_parsed_output'][prompt_key]['Score'][prompt_key]
                        # else:
                        #     score = -1
                        try:
                            score = tmp_groupped_output['raw_parsed_output'][prompt_key]['Score'][prompt_key]
                        except:
                            score = -1
                        scores.append(score)
                    # print("essay_set_number:", essay_set_number)
                    # print("prompt_keys", prompt_keys)
                    # print("scores:", scores)
                    # print('--------')
                    cleaned_scores = [clean_single_score(score_str) for score_str in scores]
                    total_score += len(list(cleaned_scores))
                    total_neg += sum([score == -1 for score in cleaned_scores])
                    if essay_set_number in [1, 3, 4, 5, 6]:
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
                    score_dict = tmp_groupped_output['raw_parsed_output'][self.hard_code_holistic_prompt_key]['Score']
                    for k in score_dict:
                        score_dict[k] = int(float(score_dict[k]))
                    tmp_groupped_output['parsed_output'] = score_dict
                    total_score += len(list(score_dict.keys()))
                    total_neg += sum([score == -1 for score in list(score_dict.values())])

                processed_log_data[fold].append(tmp_groupped_output)
            with open(self.parsing_log_path, 'a') as f:
                f.write('[INFO] fold: {}; total score count: {}; total -1 count: {}\n'.format(fold, total_score, total_neg))
            with open(self.cleaned_output_path, 'w') as f:
                f.write(json.dumps(processed_log_data))
        
        return processed_log_data, raw_log_data
    
def load_asap_id(id_dir, split = 'train'):
    id_lst = []
    with open(id_dir + split + '_ids.txt', 'r') as f:
        for line in f.readlines():
            id_lst.append(int(line.strip()))
    return id_lst

# def run_multiple(config_dir):

if __name__ == '__main__':

    ## texting llama 3 ##
    # prompt = """\nYou are part of an educational research team analyzing the writing skills of students in grades 7 to 10. 
    # You have been given a student's essay and the prompt they responded to.\n\n#### Essay Prompt:\n'''\n
    # More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. 
    # They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people.
    #   Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. \n\nWrite a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.\n'''\n\n#### Analyzed Student Essay:\n'''Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, it's a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listening.'''\n\n### Analysis Task:\n- Rate this essay in the following aspect: {'prompt': 'The essay has fully elaborated reasons with specific details.'}(The essay has fully elaborated reasons with specific details.), with 1 being worst and 6 being good.\n- Give the result in the following format: \n    - Score: \n    - Explanation: \n    - Edit suggestions:\n"""
    # prompt = """hello, what is your name?"""
    
    # # vllm = VLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_length=4096, temperature=0.01)
    # # vllm = VLLM(model="meta-llama/Llama-3.1-8B-Instruct", max_length=4096, temperature=0.01)
    
    # vllm = VLLM(model="mistralai/Mistral-7B-Instruct-v0.2", max_length=4096, temperature=0.01) ## current version
    # # vllm = VLLM(model="meta-llama/Meta-Llama-3-8B-Instruct", max_length=4096, temperature=0.01) ## current version
    # llm_output = vllm.batch([prompt])
    # print(llm_output)
    # exit(0)


    # parse arguments
    parser = argparse.ArgumentParser()
    
    ## run configurations
    parser.add_argument("--config", type=str, default='')
    # parser.add_argument("--logging_data_path", type=str, default="./log.json")
    parser.add_argument("--run_id", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--fine_grained", action="store_true")
    parser.add_argument("--verbalize", action="store_true")
    parser.add_argument("--use_machine_rubric", action="store_true")
    parser.add_argument("--skip_llm", action="store_true") # default to be false
    # parser.add_argument("--raw_log_data_file", type=str, default='raw_log_data.json')
    
    ## model configurations
    parser.add_argument("--model", type=str, default="mistral")
    # parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b"])
    parser.add_argument("--temperature", type=float, default=0.00)
    parser.add_argument("--max_length", type=int, default=4096)

    ## prompt configurations
    # parser.add_argument("--prompt", type=str, default="holistic_scoring_prompt1")
    parser.add_argument("--prompt_template", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--analysis_instruction", type=str, default="simple")
    parser.add_argument("--ling_features", type=list, default=[])
    # parser.add_argument("--setting", type=str, default="one-shot", choices=["one-shot", "few-shot"])
    # parser.add_argument("--full-rubric", action="store_true")
    # parser.add_argument("--instruction-variant", type=int, default=1, choices=[1, 2, 3, 4])
    
    ## data configurations
    parser.add_argument("--dataset_split", type=str, default="test")
    
    ## running in batch only
    parser.add_argument("--batch_json", type=str, default="")
    
    args = parser.parse_args()
    
    if args.config != '':
        with open(args.config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    # print("\n\n=> args:", args, '\n\n')

    
    ## Load data ## 
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
    if args.dataset_split == "test":
        asap_data_all = asap_data_all.query('split == "test"')
        essay_batch = list(zip(asap_data_all['essay'].values, asap_data_all['essay_id'].values, asap_data_all['essay_set'].values))
    else:
        raise NotImplementedError
    
    ### batch_running case
    if args.batch_json != "":
        run_list = load_json(args.batch_json)["experiment_lst"]
        logger.info("\n\n -------------------- List of experiments --------------------")
        for run in run_list:
            logger.info(str(run))
        
        for config in run_list:
            # run_id = config["run_id"]
            t_args = argparse.Namespace()
            t_args.__dict__.update(config)
            args = parser.parse_args(namespace=t_args)
            logger.info(str(args))

            fge = FineGrainEvaluator(args)
            log_data = fge.process_batch(essay_batch)

            ## evaluation ##
            result_df = evaluation(
                fge.cleaned_output_path,
            )
            result_df.to_csv(fge.qwk_summary_path)

            log_dir = './log/' ## TODO: change this!!!

            cleaned_output = load_json(log_dir + args.run_id + '/cleaned_output.json')
            all_fold_stats = {
                "run_id": args.run_id
            }
            del fge.llm 
            del fge
            del log_data

    
    else: ## single task 
        ## initialization ## 
        fge = FineGrainEvaluator(args)
        log_data = fge.process_batch(essay_batch)

        ## evaluation ##
        result_df = evaluation(
            fge.cleaned_output_path,
        )
        result_df.to_csv(fge.qwk_summary_path)

        log_dir = './log/' ## TODO: change this!!!

        cleaned_output = load_json(log_dir + args.run_id + '/cleaned_output.json')
        all_fold_stats = {
            "run_id": args.run_id
        }