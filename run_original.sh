echo '+++++++++++++++ now running: holistic_scoring_prompt1 +++++++++++++++'
python run.py --model=mistral --prompt=holistic_scoring_prompt1 --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_holistic_scoring_prompt1

echo '+++++++++++++++ now running: holistic_scoring_prompt2 +++++++++++++++'
python run.py --model=mistral --prompt=holistic_scoring_prompt2 --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_holistic_scoring_prompt2

echo '+++++++++++++++ now running: holistic_scoring_prompt3 +++++++++++++++'
python run.py --model=mistral --prompt=holistic_scoring_prompt3 --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_holistic_scoring_prompt3

echo '+++++++++++++++ now running: explanation_somewhere_prompt +++++++++++++++'
python run.py --model=mistral --prompt=explanation_somewhere_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_explanation_somewhere_prompt

echo '+++++++++++++++ now running: explanation_first_prompt +++++++++++++++'
python run.py --model=mistral --prompt=explanation_first_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_explanation_first_prompt

echo '+++++++++++++++ now running: feedback_somewhere_prompt +++++++++++++++'
python run.py --model=mistral --prompt=feedback_somewhere_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_feedback_somewhere_prompt

echo '+++++++++++++++ now running: feedback_first_prompt +++++++++++++++'
python run.py --model=mistral --prompt=feedback_first_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_feedback_first_prompt

echo '+++++++++++++++ now running: feedback_and_explanation_prompt +++++++++++++++'
python run.py --model=mistral --prompt=feedback_and_explanation_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_feedback_and_explanation_prompt

echo '+++++++++++++++ now running: persona_prompt +++++++++++++++'
python run.py --model=mistral --prompt=persona_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_persona_prompt

echo '+++++++++++++++ now running: chain_of_thought_prompt +++++++++++++++'
python run.py --model=mistral --prompt=chain_of_thought_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_chain_of_thought_prompt

echo '+++++++++++++++ now running: chain_of_thought_detailed_prompt +++++++++++++++'
python run.py --model=mistral --prompt=chain_of_thought_detailed_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_chain_of_thought_detailed_prompt

echo '+++++++++++++++ now running: one_shot_prompt +++++++++++++++'
python run.py --model=mistral --prompt=one_shot_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_one_shot_prompt

echo '+++++++++++++++ now running: feedback_only_prompt +++++++++++++++'
python run.py --model=mistral --prompt=feedback_only_prompt --prompt-template=3 --logging_data_path=log/mistral_7b_test_v1_feedback_only_prompt