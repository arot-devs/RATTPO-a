
from typing import List


def get_prompt_template(version='ours', **kwargs):
    match version:
        case 'ours':
            # kwargs
            initial_prompt: str = kwargs['initial_prompt']
            num_sample: int = kwargs['num_sample']
            num_context: int = kwargs['num_context']
            history_prompts: List[str] = kwargs['history_prompts']
            history_scores: List[float] = kwargs['history_scores']
            hint: str = kwargs.get('hint', None)
            if hint:
                hint = f'(Hint: {hint})\n'
            else:
                hint = ''

            # cleanse.
            initial_prompt = initial_prompt.strip()
            if not initial_prompt.endswith('.'):
                initial_prompt = initial_prompt + '.'

            # base template
            task_description_prompt = \
                f"As an expert prompt engineer for text-to-image generation, rewrite the original prompt in 8 distinct ways to improve the visual quality of the resulting images.\n"

            last_part = (
                f"Return exactly {num_sample} variations, numbered 1 through {num_sample}, each on its own line and ordered from shortest to longest.\n"
                "Preserve the meaning of the original prompt and keep each variation under 70 words. Start your output immediately with the numbered variations.\n"
                "Original Prompt: "
                f"{initial_prompt}\n"
            )

            # histories
            history_prompt = ''
            if num_context > 0 and len(history_prompts) > 0:
                # if given, the length should be the same
                assert len(history_prompts) == len(history_scores) == num_context
                history_pre = (
                    f"To aid you in this task, you will be also given {num_context} history prompts that are already tried before. "
                    "For each history prompt, its score is given as number. Higher score indicates that the prompt is better.\n"
                ) + hint + (
                    "You can use the scores to guide your rewriting process and thus improve the visual quality of the generated image, but your response should be different from histories.\n"
                    "Histories:\n"
                )

                history_each = (
                    "{idx}. Prompt: {prompt} (Score: {score:.3f})\n"
                )
                history_prompt = history_pre
                for i in range(num_context):
                    history_prompt += history_each.format(idx=i+1, prompt=history_prompts[i], score=history_scores[i])

            # 5. combine
            first_round_template = task_description_prompt + last_part
            later_round_template = task_description_prompt + history_prompt + last_part

            return first_round_template, later_round_template

        case 'opt2i_dsg':

            initial_prompt: str = kwargs['initial_prompt']
            num_sample: int = kwargs['num_sample']
            history_list: List[dict] = kwargs['history_list']

            # should be included in user prompt since gemma does not use system role. (https://ai.google.dev/gemma/docs/core/prompt-structure)
            system_instruction = (
                "You are an expert prompt optimizer for text-to-image models. "
                "Text-to-image models take a text prompt as input and generate images depicting the prompt as output. "
                "You translate prompts written by humans into better prompts for the text-to-image models. "
                "Your answers should be concise and effective.\n"
            )

            task_description = (
                f'\nYour task is to optimize this initial prompt written by a human: "{initial_prompt}". '
            )

            # last part
            last_part_pre = (
                f"\nGenerate {num_sample} paraphrases of the initial prompt which keep the semantic meaning and that have higher scores than all the prompts above. "
                "Focus on optimizing for the visual elements that are not consistent. "
                "Favor substitutions and reorderings over additions. "
                "Respond with each new prompt in between <PROMPT> and </PROMPT>, eg:\n"
            )
            last_part_example_each = "{idx}. <PROMPT>paraphrase {idx}</PROMPT>\n"
            last_part = last_part_pre
            for i in range(num_sample):
                last_part += last_part_example_each.format(idx=i+1)

            # histories
            history_prompt = ''
            if len(history_list) > 0:
                history_pre = (
                    "Below are some previous prompts with the consistency of each prompt's visual elements in the generated image via a set of binary questions. "
                    "The prompts are arranged in ascending order based on their overall consistency score, which ranges from 0 to 100 (higher is better)."
                )

                history_each_overall = (
                    "\n{idx}. {prompt}\n"
                    "overall_score: {overall_score:d}\n"
                    'evaluation questions:\n'
                )
                history_each_sub = "{question} {score:d}\n"

                history_prompt = history_pre
                for i, datapoint in enumerate(history_list):
                    history_prompt += history_each_overall.format(idx=i+1,
                                                                  prompt=datapoint['prompt'],
                                                                  overall_score=int(datapoint['reward'] * 100))
                    for j, (key, question) in enumerate(datapoint['details']['qid2question'].items()):
                        score = int(datapoint['details']['qid2scores_after_filtering'][key] * 100)
                        history_prompt += history_each_sub.format(idx=j+1,
                                                                  question=question, score=score)

            # 5. combine
            first_round_template = system_instruction + task_description + last_part
            later_round_template = system_instruction + task_description + history_prompt + last_part

            return first_round_template, later_round_template

        case _:
            raise ValueError(f"Invalid version: {version}")


def get_hint_template(**kwargs):
    history_prompts: List[str] = kwargs['history_prompts']
    history_scores: List[float] = kwargs['history_scores']
    num_context: int = kwargs['num_context']
    # 1. task description and last part
    hint_pre = (
        "As an expert prompt engineer for text-to-image generation, you are trying to rewrite the original prompt to improve the scores.\n"
        "Below are some histories of prompts you tried before. Based on the history prompts with corresponding scores, guess how you can enhance the score.\n"
    )
    hint_post = (
        "Now describe the way how we can increase the score in plain words. Simply output the way in a single line."
    )

    # 2. histories to be used
    assert len(history_prompts) == len(history_scores) == num_context != 0
    history_pre = (
        "Histories:\n"
    )
    history_each = "Prompt: {prompt} (Score: {score:.3f})\n"

    history_prompt = history_pre
    for i in range(num_context):
        history_prompt += history_each.format(idx=i+1, prompt=history_prompts[i], score=history_scores[i])

    ret = hint_pre + history_prompt + hint_post

    return ret
