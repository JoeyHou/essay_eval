'''
- Template 
- Formatting instructions 
'''

###################### Analysis Task ######################
analysis_task = """
### Analysis Task:
- Rate this essay in the following aspect: {fine_grained_prompt}, with {low_score} being worst and {high_score} being good.
- Give the result in the following format: 
    - Score: 
    - Explanation: 
    - Edit suggestions:
""".strip()

###################### Prompt Template ######################
fine_grained_template_1 = """
{model_prefix}

You are given an essay written by a student and the corresponding prompt for the 7th to 10th grade student. 

#### Prompt: 
'''{essay_prompt}'''

### Task:
{instruction}

#### Student essay: 
'''{essay}'''
""" + analysis_task + "\n{model_suffix}"

fine_grained_template_2 = """
{model_prefix}

Imagine you are a teacher's assistant in a middle school, tasked with reviewing a 7th to 10th grade student's essay. You have the essay and the prompt that was given to the student.

#### Original Prompt Provided to Student:
'''{prompt}'''

### Review Task:
{essay_prompt}

#### Student's Essay for Review:
'''{essay}'''
""" + analysis_task + "\n{model_suffix}"


fine_grained_template_3 = """
{model_prefix}

You are part of an educational research team analyzing the writing skills of students in grades 7 to 10. You have been given a student's essay and the prompt they responded to.

### Essay Prompt:
'''{essay_prompt}'''

### Analyzed Student Essay:
'''{essay}'''
""" + analysis_task + "\n{model_suffix}"


fine_grained_template_4 = """
{model_prefix}

You are a creative writing mentor evaluating a piece written by a student in grades 7 to 10. The student's work is based on a specific prompt.

### Creative Prompt Given:
'''{prompt}'''

### Critique Instructions:
{essay_prompt}

#### Student's Creative Piece:
'''{essay}'''
"""  + analysis_task + "\n{model_suffix}"