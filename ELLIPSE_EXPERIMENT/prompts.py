# Script modified from the original by Joey Hou
# Modified by Alejandro Ciuba
from pathlib import Path
from langchain.prompts import PromptTemplate
from typing import Union

import json

analysis_instruction = """Grade the given essay using the following rubric:  {rubric}. Use those score ranges: {scoring_range}."""

format_instruction = '''
### Score:
### '''

###################### Prompt Template ######################
prompt_template_3 = """
{model_prefix}

You are part of an educational research team analyzing the writing skills of students in grades 7 to 10. You have been given a student's essay and the prompt they responded to.

### Essay Prompt:
'''{essay_prompt}'''

### Analysis Task:
{analysis_instruction}

### Analyzed Student Essay:
'''{essay}'''

{additional_information}

### Analysis
Conclude your analysis with a grade and comments in the following format:
{format_instruction}

\n{model_suffix}"""

def make_prompt(**kwargs) -> PromptTemplate:
    """
    Make the prompt for the associated essay and essay prompt.

    Parameters
    ---

    essay: `str`
        The student's essay text.

    prompt: `str`
        The prompt associated with the essay text.

    rubric: `str`
        The scoring rubric for the essay.

    range: `tuple[int, int]`
        The minimum and maximum range scores can take.
    """

    ai = PromptTemplate.from_template(template=analysis_instruction).format(**kwargs)
    fi = PromptTemplate.from_template(template=format_instruction).format()

    prompt = PromptTemplate.from_template(template=prompt_template_3)
    return prompt.partial(**kwargs, analysis_instruction=ai, format_instruction=fi)


def batch_prompts(format=True, **kwargs) -> PromptTemplate:
    """
    Batch make a series of prompts.

    Parameters
    ---

    format: `bool`
        Automatically format the prompts before yielding them; useful for lists. Defaults to `True`.

    essays: `Iterable[str]`
        The student essay texts.

    prompts: `Iterable[str]`
        The prompts associated with the essay texts.

    rubric: `str`
        The scoring rubric for the essay.

    range: `tuple[int, int]`
        The minimum and maximum range scores can take.
    """

    for prompt, essay, add_info in zip(kwargs["essay_prompts"], kwargs["essays"], kwargs["additional_information"]):

        output = make_prompt(
            rubric=kwargs['rubric'], 
            scoring_range=kwargs['scoring_range'],
            essay_prompt=prompt,
            essay=essay,
            additional_information=add_info,
            model_prefix=kwargs['model_prefix'], 
            model_suffix=kwargs['model_suffix'],
        )

        yield output.format() if format else output


def make_rubric(file: Union[Path, str]) -> str:
    """
    file: `pathlib.Path | str`
        Filepath to the `JSON` containing the rubric.
    """

    with open(file, 'r') as src:
        rubric_info = json.load(src)

    overall, fine_grained = "Overall:\n", ""
    for section in rubric_info:

        if section == "Overall":
            
            scores = rubric_info[section]
            for score in scores:
                overall += f"{score} points: {scores[score]['description']}\n"

        else:

            categories = rubric_info[section]
            for category in categories:

                fine_grained += f"Scoring rubric for {category}:\n"

                scores = categories[category]
                for score in scores:
                    fine_grained += f"{score} points: {scores[score]['description']}\n"

    return f"{fine_grained}{overall}"


if __name__ == "__main__":

    rubric = make_rubric("data/ELLIPSErubric.json")
    srange = (1, 5)
    essay = "I like to write essays. I just think they're so neat!"
    prompt = "Write about loving essays."

    prompt = make_prompt(
        rubric=rubric, 
        scoring_range=srange,
        essay_prompt=prompt,
        essay=essay,
        )
    
    print(prompt.format(model_prefix="", model_suffix=""))
