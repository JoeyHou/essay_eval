echo "Currently running: $1"

# ### instruction-variant=1
echo '+++++++++++++++ now running: holistic_scoring_prompt1 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt1 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt1_var1 --instruction-variant=1

echo '+++++++++++++++ now running: holistic_scoring_prompt2 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt2 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt2_var1 --instruction-variant=1

echo '+++++++++++++++ now running: holistic_scoring_prompt3 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt3 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt3_var1 --instruction-variant=1

echo '+++++++++++++++ now running: explanation_somewhere_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=explanation_somewhere_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_explanation_somewhere_prompt_var1 --instruction-variant=1

echo '+++++++++++++++ now running: explanation_first_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=explanation_first_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_explanation_first_prompt_var1 --instruction-variant=1

echo '+++++++++++++++ now running: feedback_somewhere_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_somewhere_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_somewhere_prompt_var1 --instruction-variant=1

echo '+++++++++++++++ now running: feedback_first_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_first_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_first_prompt_var1 --instruction-variant=1

echo '+++++++++++++++ now running: feedback_and_explanation_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_and_explanation_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_and_explanation_prompt_var1 --instruction-variant=1

echo '+++++++++++++++ now running: persona_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=persona_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_persona_prompt_var1 --instruction-variant=1

echo '+++++++++++++++ now running: chain_of_thought_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=chain_of_thought_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_chain_of_thought_prompt_var1 --instruction-variant=1

echo '+++++++++++++++ now running: chain_of_thought_detailed_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=chain_of_thought_detailed_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_chain_of_thought_detailed_prompt_var1 --instruction-variant=1

echo '+++++++++++++++ now running: one_shot_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=one_shot_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_one_shot_prompt_var1 --instruction-variant=1

# echo '+++++++++++++++ now running: feedback_only_prompt +++++++++++++++'
# python run_original.py --model=mistral --prompt=feedback_only_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_only_prompt_var1 --instruction-variant=1


# ### instruction-variant=2
echo '+++++++++++++++ now running: holistic_scoring_prompt1 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt1 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt1_var2 --instruction-variant=2

echo '+++++++++++++++ now running: holistic_scoring_prompt2 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt2 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt2_var2 --instruction-variant=2

echo '+++++++++++++++ now running: holistic_scoring_prompt3 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt3 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt3_var2 --instruction-variant=2

echo '+++++++++++++++ now running: explanation_somewhere_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=explanation_somewhere_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_explanation_somewhere_prompt_var2 --instruction-variant=2

echo '+++++++++++++++ now running: explanation_first_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=explanation_first_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_explanation_first_prompt_var2 --instruction-variant=2

echo '+++++++++++++++ now running: feedback_somewhere_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_somewhere_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_somewhere_prompt_var2 --instruction-variant=2

echo '+++++++++++++++ now running: feedback_first_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_first_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_first_prompt_var2 --instruction-variant=2

echo '+++++++++++++++ now running: feedback_and_explanation_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_and_explanation_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_and_explanation_prompt_var2 --instruction-variant=2

echo '+++++++++++++++ now running: persona_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=persona_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_persona_prompt_var2 --instruction-variant=2

echo '+++++++++++++++ now running: chain_of_thought_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=chain_of_thought_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_chain_of_thought_prompt_var2 --instruction-variant=2

echo '+++++++++++++++ now running: chain_of_thought_detailed_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=chain_of_thought_detailed_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_chain_of_thought_detailed_prompt_var2 --instruction-variant=2

echo '+++++++++++++++ now running: one_shot_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=one_shot_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_one_shot_prompt_var2 --instruction-variant=2

# echo '+++++++++++++++ now running: feedback_only_prompt +++++++++++++++'
# python run_original.py --model=mistral --prompt=feedback_only_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_only_prompt_var2 --instruction-variant=2





### instruction-variant=3
echo '+++++++++++++++ now running: holistic_scoring_prompt1 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt1 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt1_var3 --instruction-variant=3

echo '+++++++++++++++ now running: holistic_scoring_prompt2 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt2 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt2_var3 --instruction-variant=3

echo '+++++++++++++++ now running: holistic_scoring_prompt3 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt3 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt3_var3 --instruction-variant=3

echo '+++++++++++++++ now running: explanation_somewhere_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=explanation_somewhere_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_explanation_somewhere_prompt_var3 --instruction-variant=3

echo '+++++++++++++++ now running: explanation_first_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=explanation_first_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_explanation_first_prompt_var3 --instruction-variant=3

echo '+++++++++++++++ now running: feedback_somewhere_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_somewhere_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_somewhere_prompt_var3 --instruction-variant=3

echo '+++++++++++++++ now running: feedback_first_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_first_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_first_prompt_var3 --instruction-variant=3

echo '+++++++++++++++ now running: feedback_and_explanation_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_and_explanation_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_and_explanation_prompt_var3 --instruction-variant=3

echo '+++++++++++++++ now running: persona_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=persona_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_persona_prompt_var3 --instruction-variant=3

echo '+++++++++++++++ now running: chain_of_thought_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=chain_of_thought_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_chain_of_thought_prompt_var3 --instruction-variant=3

echo '+++++++++++++++ now running: chain_of_thought_detailed_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=chain_of_thought_detailed_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_chain_of_thought_detailed_prompt_var3 --instruction-variant=3

echo '+++++++++++++++ now running: one_shot_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=one_shot_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_one_shot_prompt_var3 --instruction-variant=3

# echo '+++++++++++++++ now running: feedback_only_prompt +++++++++++++++'
# python run_original.py --model=mistral --prompt=feedback_only_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_only_prompt_var3 --instruction-variant=3




### instruction-variant=4
echo '+++++++++++++++ now running: holistic_scoring_prompt1 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt1 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt1_var4 --instruction-variant=4

echo '+++++++++++++++ now running: holistic_scoring_prompt2 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt2 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt2_var4 --instruction-variant=4

echo '+++++++++++++++ now running: holistic_scoring_prompt3 +++++++++++++++'
python run_original.py --model=mistral --prompt=holistic_scoring_prompt3 --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_holistic_scoring_prompt3_var4 --instruction-variant=4

echo '+++++++++++++++ now running: explanation_somewhere_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=explanation_somewhere_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_explanation_somewhere_prompt_var4 --instruction-variant=4

echo '+++++++++++++++ now running: explanation_first_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=explanation_first_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_explanation_first_prompt_var4 --instruction-variant=4

echo '+++++++++++++++ now running: feedback_somewhere_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_somewhere_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_somewhere_prompt_var4 --instruction-variant=4

echo '+++++++++++++++ now running: feedback_first_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_first_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_first_prompt_var4 --instruction-variant=4

echo '+++++++++++++++ now running: feedback_and_explanation_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=feedback_and_explanation_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_and_explanation_prompt_var4 --instruction-variant=4

echo '+++++++++++++++ now running: persona_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=persona_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_persona_prompt_var4 --instruction-variant=4

echo '+++++++++++++++ now running: chain_of_thought_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=chain_of_thought_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_chain_of_thought_prompt_var4 --instruction-variant=4

echo '+++++++++++++++ now running: chain_of_thought_detailed_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=chain_of_thought_detailed_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_chain_of_thought_detailed_prompt_var4 --instruction-variant=4

echo '+++++++++++++++ now running: one_shot_prompt +++++++++++++++'
python run_original.py --model=mistral --prompt=one_shot_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_one_shot_prompt_var4 --instruction-variant=4

# echo '+++++++++++++++ now running: feedback_only_prompt +++++++++++++++'
# python run_original.py --model=mistral --prompt=feedback_only_prompt --prompt-template=3 --logging_data_path=log/"$1"/mistral_7b_test_v1_feedback_only_prompt_var4 --instruction-variant=4