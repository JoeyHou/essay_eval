###################### Analysis and Format Instruction ######################

############ Simple ############
# analysis_instruction_simple_holistic = """
# Rate this essay in the following aspect: {fine_grained_prompt}, with {low_score} being worst and {high_score} being good.
# """.strip()

# analysis_instruction_simple_fg = """
# Rate this essay in the following aspect: {fine_grained_prompt}, with {low_score} being worst and {high_score} being good.
# """.strip()

analysis_instruction_simple_holistic = """Given this essay that was written for the given prompt, grade the essay using those ranges: {scoring_range}"""

analysis_instruction_simple_fg = analysis_instruction_simple_holistic + ", with specific focus in the following aspect: {fine_grained_prompt}"

# format_instruction_score_only = """### Score (JSON format): """
format_instruction_score_only = '''### Score: '''

# ###
# {
#     "Score": {score_format}
# }'''

############ Feedbacks ############

# analysis_instruction_feedback_holistic = """
# Rate this essay in the following aspect: {fine_grained_prompt}, with {low_score} being worst and {high_score} being good. 
# Provide comprehensive feedback for the student that helps them to achieve better grades in the future.
# """.strip()

# analysis_instruction_feedback_fg = """
# Rate this essay in the following aspect: {fine_grained_prompt}, with {low_score} being worst and {high_score} being good. 
# Provide comprehensive feedback for the student that helps them to achieve better grades in the future.
# """.strip()

analysis_instruction_feedback_holistic = """Grade the given essay using the following rubric:  {rubric}. Use those score ranges: {scoring_range}. Provide comprehensive feedback for the student that helps them to achieve better grades in the future."""

analysis_instruction_feedback_fg = """Grade the given essay using the following rubric:  {rubric}. Focus on the following aspect: {fine_grained_prompt}; and use those score ranges: {scoring_range}. Provide comprehensive feedback for the student that helps them to achieve better grades in the future."""

# format_instruction_feedbacks = """### Feedbacks:
# ### Score (JSON format): """
format_instruction_feedbacks = '''
### Feedbacks: 
### Score: 
### '''
# ###
# {
#     "Feedbacks": "",
#     "Score": {score_format}
# }
# '''
############ Explanation ############

# analysis_instruction_explanation_holistic = """
# Rate this essay in the following aspect: {fine_grained_prompt}, with {low_score} being worst and {high_score} being good. 
# Provide an explanation for your score as well.
# """.strip()

# analysis_instruction_explanation_fg = """
# Rate this essay in the following aspect: {fine_grained_prompt}, with {low_score} being worst and {high_score} being good. 
# Provide an explanation for your score as well.
# """.strip()

analysis_instruction_explanation_holistic = """Grade the given essay using the following rubric: {rubric}. Use those score ranges: {scoring_range}. Provide an explanation for your score as well."""

analysis_instruction_explanation_fg = """Grade the given essay with the following requirements:
- Focus on one particular aspect: {fine_grained_prompt} {rubric}
- Use those score ranges: {scoring_range}
- Provide an explanation for your score as well."""

format_instruction_score_and_analysis = """
### Explanation: 
### Score: 
###"""

# format_instruction_score_and_analysis = '''
# ###
# {
#     "Explanation": "",
#     "Score": {score_format}
# }'''
############ Comprehensive ############

# analysis_instruction_comprehensive_holistic = """
# Rate this essay in the following aspect: {fine_grained_prompt}, with {low_score} being worst and {high_score} being good. 
# Let's think step by step. First, analyze the quality of the essay in terms of the given rubric. Then, give feedback to the student that explains their mistakes and errors and additionally gives them tips to avoid them in the future. As a final step, output the score at the end.
# """.strip()

# analysis_instruction_comprehensive_fg = """
# Rate this essay in the following aspect: {fine_grained_prompt}, with {low_score} being worst and {high_score} being good. 
# Let's think step by step. First, analyze the quality of the essay in terms of the given rubric. Then, give feedback to the student that explains their mistakes and errors and additionally gives them tips to avoid them in the future. As a final step, output the score at the end.
# """.strip()

# format_instruction_all = """
# ### Explanation: 
# ### Feedbacks:
# ### Score (JSON format): 
# """

analysis_instruction_comprehensive_holistic = """Analyze the given essay using the following rubric and give helpful feedback to the student: {rubric}. Focus on the following aspect: {fine_grained_prompt}; and use those score ranges: {scoring_range}. Let's think step by step. First, analyze the quality of the essay in terms of the given rubric. Then, give feedback to the student that explains their mistakes and errors and additionally gives them tips to avoid them in the future. As a final step, output the score at the end."""

analysis_instruction_comprehensive_fg = """Analyze the given essay using the following rubric and give helpful feedback to the student: {rubric}. Use those score ranges: {scoring_range}. Let's think step by step. First, analyze the quality of the essay in terms of the given rubric. Then, give feedback to the student that explains their mistakes and errors and additionally gives them tips to avoid them in the future. As a final step, output the score at the end."""

format_instruction_all = '''
### Explanation: 
### Score: 
### Feedbacks: 
### '''
# ###
# {
#     "Explanation": "",
#     "Feedbacks": "",
#     "Score": {score_format}
# } ###'''

###################### Prompt Template ######################
prompt_template_1 = """
{model_prefix}

You are given an essay written by a student and the corresponding prompt for the 7th to 10th grade student. 

### Prompt: 
'''{essay_prompt}'''

### Task:
{analysis_instruction}

### Student essay: 
'''{essay}'''

Finally, after everything, give the grade in the following format:
{format_instruction}

\n{model_suffix}"""

# (make sure it is in JSON format): 

prompt_template_2 = """
{model_prefix}

Imagine you are a teacher's assistant in a middle school, tasked with reviewing a 7th to 10th grade student's essay. You have the essay and the prompt that was given to the student.

### Original Prompt Provided to Student:
'''{essay_prompt}'''

### Review Task
{analysis_instruction}

### Student's Essay for Review:
'''{essay}'''

After reviewing, provide feedback and a grade using this format:
{format_instruction}

\n{model_suffix}"""


prompt_template_3 = """
{model_prefix}

You are part of an educational research team analyzing the writing skills of students in grades 7 to 10. You have been given a student's essay and the prompt they responded to.

### Essay Prompt:
'''{essay_prompt}'''

### Analysis Task:
{analysis_instruction}

### Analyzed Student Essay:
'''{essay}'''

Conclude your analysis with a grade and comments in the following format:
{format_instruction}

\n{model_suffix}"""
#  (make sure it is in JSON format):

prompt_template_4 = """
{model_prefix}

You are a creative writing mentor evaluating a piece written by a student in grades 7 to 10. The student's work is based on a specific prompt.

### Creative Prompt Given:
'''{essay_prompt}'''

### Critique Instructions:
{analysis_instruction}

### Student's Creative Piece:
'''{essay}'''

End your critique with a structured assessment and grade, as outlined here:
{format_instruction}

\n{model_suffix}"""

prompt_template_5 = """
{model_prefix}
### Student's Essay
'''{essay}'''

### Anslysis Instructions
{analysis_instruction}

### Output
structured assessment and grade, as outlined here (make sure it is in JSON format):
{format_instruction}

\n{model_suffix}"""