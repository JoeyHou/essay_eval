import json
import jsonlines
from datetime import datetime
from pprint import pprint
# import base64
# import mimetypes
# import os
# import requests
# import sys
import re
# from PIL import Image

def load_txt_prompt(filename):
    """
    Load a prompt from local txt file
    """
    prompt = ''.join(open(filename, 'r').readlines())
    return prompt

def load_json(filename):
    """
    Load a JSON file given a filename
    If the file doesn't exist, then return an empty dictionary instead
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def write_json(data, filepath):
    assert isinstance(data, dict), '[ERROR] Expect dictionary data!'
    json_string = json.dumps(data, indent = 4)
    with open(filepath, 'w') as outfile:
        outfile.write(json_string)
    # return 0

def load_jsonl(filename):
    file_content = []
    try:
        with jsonlines.open(filename) as reader:
            for obj in reader:
                file_content.append(obj)
            return file_content
    except FileNotFoundError:
        return []

def write_jsonl(data, filepath):
    with open(filepath, 'w') as jsonl_file:
        for line in data:
            jsonl_file.write(json.dumps(line))
            jsonl_file.write('\n')

def openai_call(prompt, client, config):
    # set parameters
    temperature = config['temperature'] if 'temperature' in config else 0.75
    max_tokens = config['max_tokens'] if 'max_tokens' in config else 256
    # stop_tokens = config['stop_tokens'] if 'stop_tokens' in config else ['###']
    frequency_penalty = config['frequency_penalty'] if 'frequency_penalty' in config else 0
    presence_penalty = config['presence_penalty'] if 'presence_penalty' in config else 0
    wait_time = config['wait_time']  if 'wait_time' in config else 0
    model = config['model'] if 'model' in config else 'text-dacinvi-003'
    return_logprobs = config['return_logprobs'] if 'return_logprobs' in config else False
    logprobs = True if return_logprobs else None

    messages = [
        {  
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages = messages,
        temperature=temperature,
        max_tokens=max_tokens,
        # top_p=1,
        # stop_tokens=stop_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs
    )
    completion = response.choices[0].message.content.strip() 
    if logprobs:
        logprobs = response.choices[0].logprobs
        return completion, logprobs
    else:
        return completion

def make_prompt(data, template):
    return template.format(**data)

# def parsing_delete_uneccessary_json(llm_output):
#     bracket_count = 0
#     idx = 0
#     for char in llm_output:
#         idx += 1
#         if char == '}':
#             bracket_count += 1
#         if bracket_count == 2:
#             break 
#     return llm_output[:idx]

# def parsing_preprocessing(llm_output):
#     llm_output = re.sub('\t+', ' ', llm_output)
#     llm_output = re.sub('\n+', ' ', llm_output)
#     llm_output = re.sub('\s+', ' ', llm_output)
#     # llm_output = delete_uneccessary_json(llm_output)
#     return llm_output.lower().strip()

def clean_single_score(score_str):
    score = -1
    if isinstance(score_str, str):
        if score_str[0] == '-':
            try:
                score = int(float(score_str[:2]))
            except:
                score = -1
        else:
            try:
                score = int(float(score_str[0]))
            except:
                score = -1
    elif isinstance(score_str, float) or isinstance(score_str, int):
        score = int(score_str)
    return score