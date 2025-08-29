# Improving LLM-based Automatic Essay Scoring with Linguistic Features
- [Link to Paper](https://proceedings.mlr.press/v273/hou25a)

## Dependency 
- [vllm v0.6.1.post1](https://docs.vllm.ai/en/v0.6.1.post1/index.html)
- nltk
- readability
- spacy

## Usage
- Prepare linguistic features
    - `python features.py`
- Run essay scoring: 
    - `python eval.py --batch_json={config.json} --limit=-1` 
    - `--batch_json`: any JSON config file from `config/`, see example below (more running configuration can be found in `eval.py`; however, many of them are not part of the experiment and default values would be fine)
    ```
    {    
        "experiment_lst": [
                {
                    "run_id": "final_v0_2/gpt4_asap_default", # unique id, will be used to create a folder under `log/` to store all running outputs
                    "model": "gpt-4" # model name, choose from ["gpt-4", "mistral"]
                },
                {
                    "run_id": "final_v0_2/gpt4_asap_full_feat",
                    "model": "gpt-4",
                    "ling_features": [
                        "HAPAX",
                        "LEMMA_COUNT",
                        "COMPLEX_WORDS_DC"
                    ], # list of ling_features to include

                }
            ] # this config file will run 2 experiments sequentially
    }
    ```    
    - `--limit`: number of datapoint per essay set, set to -1 to run through the entire test split
- Evaluate LLM scores
    - `python src/eval_qwk.py --logging_data_path={cleaned_output.json} --qwk_summary_path={output.csv}`
    - `--logging_data_path`: full file path to `cleaned_output.json` from the previous step
    - `--qwk_summary_path`: intended output filepath (need to be a `.csv` file)
