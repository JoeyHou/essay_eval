import pandas as pd 
import numpy as np
import argparse
import json 
import re

## local modules ##
import sys
sys.path.insert(1, 'src/')
from utils import (
    load_json,
    write_json
)
from essay_meta_data import essay_set_descriptions ## essay meta data ##
# from langchain.llms import VLLM
from langchain_community.llms import VLLM ## VLLM Support ##
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from llm_functions import load_mistral_vllm
from fine_grained_prompts import (
    fine_grained_template_1,
    fine_grained_template_2,
    fine_grained_template_3,
    fine_grained_template_4
)
from eval_qwk import evaluation

## huggingface setup ##
import os 
home_dir = '/ihome/xli/joh227/'
os.environ['HF_TOKEN'] = load_json(home_dir + 'developer/huggingface_key.json')['personal']
os.environ['HF_HOME'] = home_dir + 'ix_dir/huggingface/'

## torch setup ##
import torch 
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def make_prompt(prompt_template, data):
    return prompt_template.format(**data)

## helper function - models ##
def load_llama_vllm(max_length=4096, temperature=0.01):
    """
    Load a LLaMA model with specified parameters.

    Parameters:
    - max_length (int, optional): The maximum token length for the model's outputs. Defaults to 4096.
    - temperature (float, optional): The temperature for sampling outputs. Defaults to 0.01.

    Returns:
    - VLLM: The loaded Mistral model.
    """
    return VLLM(model="meta-llama/Meta-Llama-3.1-8B", max_length=max_length, temperature=temperature)

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
        if args.model == 'mistral':
            self.vllm = load_mistral_vllm()
        if args.model == 'llama31':
            self.vllm = load_llama_vllm()
        self.limit = args.limit
        # if args.prompt_template == 1
        prompt_templates = [
            fine_grained_template_1,
            fine_grained_template_2,
            fine_grained_template_3,
            fine_grained_template_4
        ]
        self.fine_grained_template = prompt_templates[args.prompt_template - 1]

    def prompt_vllm(self, inputs):
        if isinstance(inputs, dict):
            inputs = [inputs]
        vllm_output = self.vllm.batch(inputs)
        # del chain
        return vllm_output

    def parse_vllm_output(self, vllm_output):
        try:
            score = float(re.search('- score.+\d+\n', vllm_output.lower()).group(0).replace('- score:', '').strip())
            parsed_score = True
        except:
            score = -1
            parsed_score = False
        try:
            explanation = re.search('- explanation.+\n', vllm_output.lower()).group(0).replace('- explanation:', '').strip()
            parsed_exp = True
        except:
            explanation = 'n/a'
            parsed_exp = False
        try:
            suggestion = re.search('- edit suggestion.+', vllm_output.lower()).group(0).replace('- edit suggestions:', '').strip()
            parsed_sug = True
        except:
            suggestion = 'n/a'
            parsed_sug = False
        return {
            'score': score,
            'explanation': explanation,
            'suggestion': suggestion,
            'stats': (parsed_score, parsed_exp, parsed_sug)
        }

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

        if self.args.model in ["mistral", "llama31"]:
            model_prefix="<s>[INST]"
            model_suffix="[/INST]"
        else:
            raise NotImplementedError

        for i in range(len(essay_batch)):
            # prompts = fine_grained_prompts[i]
            essay, essay_id, essay_set = essay_batch[i]
            if counter_per_set[essay_set] > self.limit: continue
            single_evaluator_score_ranges = self.essay_meta_data[essay_set - 1]['single_evaluator_score_ranges'][0]
            essay_prompt = self.essay_meta_data[essay_set - 1]['prompt']
            fine_grained_prompts = self.essay_meta_data[essay_set - 1]['fine_grained_rubric']
            for prompt_key in fine_grained_prompts:
                fine_grained_prompt = '{}({})'.format(
                    fine_grained_prompts[prompt_key],
                    fine_grained_prompts[prompt_key]['prompt']
                )
                vllm_prompt = make_prompt(
                    self.fine_grained_template,
                    {
                        'essay_prompt': essay_prompt,
                        'essay': essay,
                        'fine_grained_prompt': fine_grained_prompt, 
                        'low_score': single_evaluator_score_ranges[0],
                        'high_score': single_evaluator_score_ranges[1],
                        'model_prefix': model_prefix, 
                        'model_suffix': model_suffix
                    }
                )
                # tmp_prompts[prompt_key] = vllm_prompt
                all_prompts.append(vllm_prompt)
                all_prompts_id.append((float(essay_id), float(essay_set), prompt_key, essay))
            counter_per_set[essay_set] += 1
            # essay_counter += 1
            # if essay_counter > self.limit: break 
        return all_prompts, all_prompts_id
    
    def process_batch(self, essay_batch):
        if not self.args.skip_vllm:
            all_prompts, all_prompts_id = self.prep_prompt(essay_batch)
            vllm_outputs = self.prompt_vllm(all_prompts)
            
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
                    'output': vllm_outputs[i],
                    'vllm_prompt': all_prompts[i],
                    'prompt_key': all_prompts_id[i][2],
                    'parsed_output': self.parse_vllm_output(vllm_outputs[i]),
                }
                raw_log_data['fold_0_{}'.format(tmp_essay_set)].append(tmp_dp)
            with open(self.args.raw_log_data_file, 'w') as f:
                f.write(json.dumps(raw_log_data))
        else:
            with open(self.args.raw_log_data_file, 'r') as f:
                raw_log_data = json.load(f)
        
        ## merge sub category scores ## 
        processed_log_data = {k: [] for k in list(raw_log_data.keys())}
        for fold in raw_log_data:
            all_essay_id = np.unique([tmp_dp['id'] for tmp_dp in raw_log_data[fold]])
            tmp_dp_dict = {k: [] for k in all_essay_id}
            for tmp_dp in raw_log_data[fold]:
                tmp_dp_dict[tmp_dp['id']].append(tmp_dp)

            for essay_id in tmp_dp_dict:
                ### check which group it is 
                # if tmp_dp_dict[essay_id][0]['essay_set'] not in [1, 3, 4, 5, 6]: continue 
                tmp_groupped_output = {
                    'essay': tmp_dp_dict[essay_id][0]['essay'],
                    'id': tmp_dp_dict[essay_id][0]['id'],
                    'output_parsing_info': None,
                    'output': [dp['output'] for dp in tmp_dp_dict[essay_id]],
                    'vllm_prompt': {dp['prompt_key']: dp['vllm_prompt'] for dp in tmp_dp_dict[essay_id]},
                    # 'prompt_key': [dp['prompt_key'] for dp in tmp_dp_dict[essay_id]],
                    'raw_parsed_output': {dp['prompt_key']: dp['parsed_output'] for dp in tmp_dp_dict[essay_id]}
                }
                if tmp_dp_dict[essay_id][0]['essay_set'] in [1, 3, 4, 5, 6]:
                    scores = [dp['parsed_output']['score'] for dp in tmp_dp_dict[essay_id] if dp['parsed_output']['score'] != -1]
                    if len(scores) == 0:
                        avg = -1
                    else:
                        avg = np.mean(scores)
                    tmp_groupped_output['parsed_output'] = {'Overall': int(avg)}
                else:
                    tmp_groupped_output['parsed_output'] = {dp['prompt_key']: dp['parsed_output']['score'] for dp in tmp_dp_dict[essay_id]}
                processed_log_data[fold].append(tmp_groupped_output)
            
            with open(self.args.logging_data_path, 'w') as f:
                f.write(json.dumps(processed_log_data))
        return processed_log_data, raw_log_data
    
def load_asap_id(id_dir, split = 'train'):
    id_lst = []
    with open(id_dir + split + '_ids.txt', 'r') as f:
        for line in f.readlines():
            id_lst.append(int(line.strip()))
    return id_lst

if __name__ == '__main__':

    # prompt = """\nYou are part of an educational research team analyzing the writing skills of students in grades 7 to 10. 
    # You have been given a student's essay and the prompt they responded to.\n\n#### Essay Prompt:\n'''\n
    # More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. 
    # They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people.
    #   Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. \n\nWrite a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.\n'''\n\n#### Analyzed Student Essay:\n'''Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, it's a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listening.'''\n\n### Analysis Task:\n- Rate this essay in the following aspect: {'prompt': 'The essay has fully elaborated reasons with specific details.'}(The essay has fully elaborated reasons with specific details.), with 1 being worst and 6 being good.\n- Give the result in the following format: \n    - Score: \n    - Explanation: \n    - Edit suggestions:\n"""
    # prompt = """hello, what is your name?"""
    # vllm = VLLM(model="meta-llama/Meta-Llama-3.1-8B", max_length=4096, temperature=0.01)
    # vllm_output = vllm.batch([prompt])
    # print(vllm_output)
    # exit(0)


    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--logging_data_path", type=str, default="./log.json")
    parser.add_argument("--model", type=str, default="mistral", choices=["llama31", "mistral"])
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b"])
    parser.add_argument("--temperature", type=float, default=0.00)
    parser.add_argument("--max_length", type=int, default=4096)
    # parser.add_argument("--prompt", type=str, default="holistic_scoring_prompt1")
    parser.add_argument("--prompt_template", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_split", type=str, default="test")
    # parser.add_argument("--setting", type=str, default="one-shot", choices=["one-shot", "few-shot"])
    # parser.add_argument("--full-rubric", action="store_true")
    # parser.add_argument("--prompt-template", type=int, default=1, choices=[1, 2, 3, 4])
    # parser.add_argument("--instruction-variant", type=int, default=1, choices=[1, 2, 3, 4])

    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--skip_vllm", action="store_true")
    parser.add_argument("--raw_log_data_file", type=str, default='raw_log_data.json')
    parser.add_argument("--config", type=str, default='')
    args = parser.parse_args()
    if args.config != '':
        with open(args.config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    
    # print("\n\n=> args:", args, '\n\n')
    fge = FineGrainEvaluator(args)
    
    ## Load data ## 
    asap_data_dir = './data/asap/'
    train_ids = load_asap_id(asap_data_dir + '5_splits/fold_0/')
    dev_ids = load_asap_id(asap_data_dir + '5_splits/fold_0/', 'dev')
    test_ids = load_asap_id(asap_data_dir + '5_splits/fold_0/', 'test')
    id2split = dict(zip(train_ids, ['train'] * len(train_ids)))
    id2split.update(dict(zip(dev_ids, ['dev'] * len(dev_ids))))
    id2split.update(dict(zip(test_ids, ['test'] * len(test_ids))))
    # len(train_ids), len(dev_ids), len(test_ids)    
    asap_data_all = pd.read_excel(asap_data_dir + 'training_set_rel3.xlsx', sheet_name = 'training_set')
    asap_data_all['split'] = asap_data_all.essay_id.apply(lambda x: id2split[x] if x in id2split else 'other')

    ## prepare essay data ##
    essay_batch = list(zip(asap_data_all['essay'].values, asap_data_all['essay_id'].values, asap_data_all['essay_set'].values))
    
    ## prompting ## 
    log_data = fge.process_batch(essay_batch)

    ## evaluation ##
    # write_json(log_data)
    result_df = evaluation(
        fge.args.logging_data_path,
    )