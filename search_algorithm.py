import torch
import utils
from typing import List, Type
import numpy as np
import requests
import json
from prompt_templates import get_prompt_template, get_hint_template
import os
from google import genai
from google.genai import types, errors
from copy import deepcopy
import time


MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds


class BaseSearch:
    def __init__(self, config: dict, pipeline_name: str, dtype: torch.dtype, verbose=False):
        self.verbose = verbose
        self.config = config
        self.pipeline_name = pipeline_name
        self.dtype = dtype
        self.latent_prep_fn = utils.get_latent_prep_fn(pipeline_name)

    def propose(self, *args, **kwargs) -> dict:
        '''
        Generate next pipeline call arguments based on the search algorithm
        Return a dictionary including keys: 'prompts'
        '''
        raise NotImplementedError

    def update(self, *args, **kwargs):
        '''
        Update internal state based on the round output
        '''
        raise NotImplementedError

    def v_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class LLMQuerySearch(BaseSearch):
    '''
    Query LLM with search history as a context.
    '''
    nickname = 'llm_icl'

    def __init__(self, config, pipeline_name: str, dtype: torch.dtype, verbose=False):
        super().__init__(config, pipeline_name, dtype, verbose)

        self.initial_prompt = None
        search_args = config["search_args"]
        self.num_samples_per_round = search_args["num_samples_per_round"]

        self.history = []
        self.history_selection_strategy = search_args['history_selection_strategy']
        self.history_num_selection = search_args.get('history_num_selection', self.num_samples_per_round)
        self.template_version = search_args.get('template_version', 'ours')
        self.last_used_prompt = None

        self.llm_name = search_args['llm_name']
        self.ollama_port = config['ollama_port']
        self.genai_client = None
        if search_args.get('use_genai_api', False):
            self.genai_client = genai.Client(
                api_key=os.environ['GENAI_API_KEY'],
            )

        self.num_total_required = self.num_samples_per_round * search_args['search_rounds']

    def _get_prompt(self, search_round: int) -> str:
        '''
        Generate prompt to query LLM.
        '''
        history = deepcopy(self.history)  # make a copy of the history
        unique_history = {}
        for each in history:
            if each['prompt'] not in unique_history:
                unique_history[each['prompt']] = each
        history = list(unique_history.values())

        num_context = min(self.history_num_selection, len(history))  # prevent out of index error
        history = history[:self.history_num_selection]  # select the best
        num_sample = self.num_samples_per_round

        # template
        history_prompts = [each['prompt'] for each in history]
        history_scores = [each['reward'] for each in history]

        num_context = min(self.history_num_selection, len(history_prompts))
        first_round_prompt, later_round_prompt = \
            get_prompt_template(version=self.template_version,
                                initial_prompt=self.initial_prompt,
                                num_sample=num_sample,
                                num_context=num_context,
                                history_prompts=history_prompts,
                                history_scores=history_scores,
                                hint=self.hint if hasattr(self, 'hint') else None,  # TODO: make this configurable
                                )
        if search_round == 1 or self.history_selection_strategy == 'none':
            return first_round_prompt
        else:
            return later_round_prompt

    def _parse_helper(self, raw_response: str) -> List[str]:
        '''
        Parse the raw response from LLM to get the prompts.
        '''
        if self.genai_client is not None:
            parsed_response = raw_response.text.strip()
            # remove response until '1.' appears, and split by new line
            parsed_response = parsed_response[parsed_response.find('1.'):].split('\n')
            ret = ['. '.join(each.split('. ')[1:]).strip() for each in parsed_response]  # remove the index
        else:
            raw_response = raw_response.json()['response']
            parsed_response = raw_response.strip()

            # remove response until '1.' appears, and split by new line
            parsed_response = parsed_response[parsed_response.find('1.'):].strip().split('\n')
            ret = ['. '.join(each.split('. ')[1:]).strip() for each in parsed_response]  # remove the index

        # delete empty strings
        ret = [each for each in ret if each != '']

        return ret

    def _query_llm(self, search_round) -> List[str]:
        '''
        Query LLM with the full history as context.
        if context length is None, use the full history.
        Return: list of upscaled prompts, number of prompts equal to the number of noises to sample.
        '''
        post_args = utils.base_post_args(model_name=self.llm_name, port=self.ollama_port)

        # prepare prompt for LLM
        _prompt = self._get_prompt(search_round)
        self.last_used_prompt = _prompt  # save the last used prompt for debugging
        # import ipdb; ipdb.set_trace()
        post_args['json']['prompt'] = _prompt

        # query the LLM api and parse
        success = False
        num_tries = 0
        while not success:
            try:
                if self.genai_client is not None:
                    contents = [
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=_prompt),
                            ],
                        ),
                    ]
                    raw_response = self.genai_client.models.generate_content(
                        model=self.llm_name,
                        contents=contents,
                    )

                else:
                    raw_response = requests.post(**post_args)

                # # delete empty strings <- now do it at parse helper
                # ret = [each for each in ret if each != '']
                ret = self._parse_helper(raw_response)  # parse the response

                assert len(ret) >= self.num_samples_per_round, f"Got {len(ret)} responses but requires {self.num_samples_per_round}. \n {ret}"
                if len(ret) > self.num_samples_per_round:
                    # print warning message
                    print(f'Got {len(ret)} responses but requires {self.num_samples_per_round}. Using first {self.num_samples_per_round}. Total response: \n {ret}')
                    ret = ret[:self.num_samples_per_round]  # only keep the first num_samples_per_round samples
                success = True

            except Exception as e:
                if not isinstance(e, errors.ServerError):
                    num_tries += 1

                if num_tries > 10:
                    if isinstance(e, errors.ClientError) and e.code == 429:
                        print(f"Cannot proceed due to quota limit")
                        print(self.genai_client._api_client.api_key)
                        raise e
                    print(f"Failed to query LLM after {num_tries} tries. Using the initial prompt as fallback.")
                    ret = [self.initial_prompt] * self.num_samples_per_round
                    success = True
                    break

                print(f"Error in querying LLM: {e}")
                print(f"Retrying...")
                time.sleep(5)

        return ret

    def _sort_history(self):
        '''
        Sort the history based on the reward.
        '''
        self.history = sorted(self.history, key=lambda x: x['reward'], reverse=True)

        return None

    def propose(self, search_round: int, prompt: str, prompt_idx: int):
        # Generate prompts to use
        if search_round == 1:
            self.initial_prompt = prompt  # save the initial prompt
        prompts = self._query_llm(search_round)

        return {"noises": None, "prompts": prompts}

    def update(self, search_round, datapoints, cur_dir=None, prompt_idx=None):
        # save history
        round_output = datapoints

        for each in round_output:
            one_inst = {
                'prompt': each['prompt'],
                'round': search_round,
            }
            one_inst['reward'] = each['reward']
            self.history.append(one_inst)
        self._sort_history()

        # save internal state
        if cur_dir is not None and prompt_idx is not None:
            to_save = {}
            to_save['history'] = self.history
            to_save['last_used_prompt'] = self.last_used_prompt
            with open(os.path.join(cur_dir, f"prompt_{prompt_idx:04d}_algo_state.json"), 'w') as f:
                json.dump(to_save, f, indent=4)

        return None


class LLMQueryHintSearch(LLMQuerySearch):
    '''
    Query LLM similar to LLMQuerySearch, but also produce 'hints' about the meaning of score from search history.
    '''
    nickname = 'llm_hint'

    def __init__(self, config, pipeline_name: str, dtype: torch.dtype, verbose=False):
        super().__init__(config, pipeline_name, dtype, verbose)
        search_args = config["search_args"]
        self.hint_history_selection_strategy = search_args.hint_history_selection_strategy
        self.hint_history_num_selection = search_args.hint_history_num_selection
        self.hint_update_rounds = [i for i in range(1, search_args.search_rounds)]

        self.hint = None  # initialize as none, to be updated later
        self.last_used_history = None
        self.last_used_hint_prompt = None  # save the last used hint prompt for debugging

    def _query_llm_hint(self, search_round: int, prompt_idx: int) -> str:
        '''
        Query LLM to generate hint
        '''
        # import ipdb; ipdb.set_trace()
        history = deepcopy(self.history)  # make a copy of the history

        # deduplicate the history based on the prompt
        unique_history = {}
        for each in history:
            if each['prompt'] not in unique_history:
                unique_history[each['prompt']] = each
        history = list(unique_history.values())  # preserves the order of the dict since python 3.7

        num_context = min(self.hint_history_num_selection, len(history))  # prevent out of index error
        # first get the history based on selection strategy
        match self.hint_history_selection_strategy:
            case 'best':
                history = history[:num_context]  # select the best
            case 'all':
                history = history
            case 'random':  # randomly select
                history = np.random.choice(history, size=num_context, replace=False)
            case _:
                raise NotImplementedError(f"History selection strategy '{self.history_selection_strategy}' not implemented yet.")
        self.last_used_history = history.tolist() if isinstance(history, np.ndarray) else history  # save the last used history for debugging

        history_prompts = [each['prompt'] for each in history]
        history_scores = [each['reward'] for each in history]

        _prompt = get_hint_template(history_prompts=history_prompts,
                                    history_scores=history_scores,
                                    num_context=num_context,
                                    )
        # now call api
        self.ollama_port = os.environ.get('OLLAMA_HOST', self.ollama_port)
        post_args = utils.base_post_args(model_name=self.llm_name, port=self.ollama_port)
        post_args['json']['prompt'] = _prompt
        self.last_used_hint_prompt = _prompt  # save the last used hint prompt for debugging

        success = False
        num_tries = 0
        while not success:
            try:
                if self.genai_client is not None:
                    contents = [
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=_prompt),
                            ],
                        ),
                    ]
                    raw_response = self.genai_client.models.generate_content(
                        model=self.llm_name,
                        contents=contents,
                    )
                    ret = raw_response.text

                else:
                    raw_response = requests.post(**post_args)
                    ret = raw_response.json()['response']
                assert ret is not None, 'LLM response is None'
                success = True

            except Exception as e:
                if not isinstance(e, errors.ServerError):
                    num_tries += 1

                if num_tries > 10:
                    if isinstance(e, errors.ClientError) and e.code == 429:
                        print(f"Cannot proceed due to quota limit")
                        print(self.genai_client._api_client.api_key)
                    raise e

                print(f"Error in querying LLM for hint: {e}")
                print(f"Retrying...")
                time.sleep(5)

        # clean up the response
        while ret.count('\n\n') > 1:
            ret = ret.replace('\n\n', '\n')
        ret = ret.strip()

        return ret

    def update(self, search_round, datapoints, cur_dir=None, prompt_idx=None):
        # save history
        for each in datapoints:
            one_inst = {
                'prompt': each['prompt'],
                'reward': each['reward'],
                'round': search_round,
            }
            self.history.append(one_inst)
        self._sort_history()

        # update hint if needed
        if search_round in self.hint_update_rounds:
            self.hint = self._query_llm_hint(search_round, prompt_idx)

        # save internal state
        if cur_dir is not None and prompt_idx is not None:
            to_save = {}
            to_save['hint'] = self.hint
            to_save['last_used_prompt'] = self.last_used_prompt
            to_save['last_used_hint_prompt'] = self.last_used_hint_prompt
            to_save['last_used_history'] = self.last_used_history
            to_save['history'] = self.history
            with open(os.path.join(cur_dir, f"prompt_{prompt_idx:04d}_algo_state.json"), 'w') as f:
                json.dump(to_save, f, indent=4)

        return None


class LLMQueryHintCacheSearch(LLMQueryHintSearch):
    '''
    LLMQueryHintSearch but use hint from other experiment. For analysis purpose.
    '''
    nickname = 'llm_hint_cache'

    def __init__(self, config, pipeline_name: str, dtype: torch.dtype, verbose=False):
        super().__init__(config, pipeline_name, dtype, verbose)
        search_args = config["search_args"]
        search_rounds = search_args['search_rounds']
        self.hint_cache_exp_root = search_args['hint_cache_exp_root']
        self.hint_update_rounds = list(range(1, search_rounds+1))  # update every round, just skip if none in cache.

    def _query_llm_hint(self, search_round: int, prompt_idx: int) -> str:
        '''
        Query LLM to generate hint
        '''
        # import ipdb; ipdb.set_trace()
        algo_state_path = os.path.join(
            self.hint_cache_exp_root,
            f"round_{search_round}",
            f"prompt_{prompt_idx:04d}_algo_state.json"
        )
        with open(algo_state_path, 'r') as f:
            algo_state = json.load(f)
        hint = algo_state['hint']
        if hint is None:
            hint = self.hint  # do not update hint if none in cache

        return hint


class OPT2ISearch(LLMQuerySearch):
    '''
    Query LLM with search history as a context.
    '''
    nickname = 'opt2i'

    def __init__(self, config, pipeline_name, dtype, verbose=False):
        super().__init__(config, pipeline_name, dtype, verbose)
        assert 'opt2i_dsg' in self.template_version, 'opt2i_dsg is the only template version supported for OPT2I'

    def _parse_helper(self, raw_response: str) -> List[str]:
        if self.genai_client is not None:
            parsed_response = raw_response.text.strip()
        else:
            parsed_response = raw_response.json()['response'].strip()
        ret = []
        # find all paraphrase between <PROMPT> and </PROMPT>
        start_idx = 0
        while True:
            start_idx = parsed_response.find('<PROMPT>', start_idx)
            if start_idx == -1:
                break
            end_idx = parsed_response.find('</PROMPT>', start_idx)
            if end_idx == -1:
                break
            ret.append(parsed_response[start_idx+len('<PROMPT>'):end_idx].strip())
            start_idx = end_idx + len('</PROMPT>')

        # delete empty strings
        ret = [each for each in ret if each != '']

        return ret

    def _get_prompt(self, search_round: int) -> str:
        '''
        Generate prompt to query LLM.
        '''
        # first get the history based on selection strategy
        history = self.history[:self.history_num_selection][::-1]  # select the best and reverse the order

        num_sample: int = self.num_samples_per_round
        first_round_prompt, later_round_prompt = \
            get_prompt_template(version=self.template_version,  # opt2i_dsg
                                initial_prompt=self.initial_prompt,
                                num_sample=num_sample,
                                history_list=history,
                                )
        if search_round == 1:
            return first_round_prompt
        else:
            return later_round_prompt

    def update(self, search_round, datapoints, cur_dir=None, prompt_idx=None):
        # save history
        round_output = datapoints

        for each in round_output:
            one_inst = deepcopy(each)
            one_inst['round'] = search_round
            self.history.append(one_inst)

        self._sort_history()

        # save internal state
        if cur_dir is not None and prompt_idx is not None:
            to_save = {}
            to_save['history'] = self.history
            to_save['last_used_prompt'] = self.last_used_prompt
            with open(os.path.join(cur_dir, f"prompt_{prompt_idx:04d}_algo_state.json"), 'w') as f:
                json.dump(to_save, f, indent=4)

        return None


class PromptRandomParaphraseSearch(BaseSearch):
    nickname = 'prompt_random_paraphrase'

    def __init__(self, config, pipeline_name: str, dtype: torch.dtype, verbose=False):
        super().__init__(config, pipeline_name, dtype, verbose)
        search_args = config["search_args"]
        self.num_noises_to_sample = search_args["num_samples_per_round"]
        cache_path = search_args.get('cache_path', None)
        self.num_total_required = self.num_noises_to_sample * config['search_args']['search_rounds']

        self.cache = None
        self.cache_path = cache_path
        self.start_pointer = 0
        self.start_with_initial_prompt = search_args.get('start_with_initial_prompt', True)

        with open(cache_path, 'r') as f:
            self.cache = json.load(f)
        cache_dataset = self.cache.get('dataset', 'drawbench')
        # 'to prevent future bug: indexing could be done across different dataset'
        assert cache_dataset == self.config['dataset'], 'Cache dataset does not match the current dataset.'

    def propose(self, search_round: int, prompt: str, prompt_idx: int):
        candidates_list = self.cache[str(prompt_idx)]

        # check validity
        if search_round == 1:
            cur_len = len(candidates_list)
            assert cur_len >= self.num_total_required, \
                f"Not enough paraphrases in the cache; found {cur_len} but need {self.num_total_required}."

            if self.start_with_initial_prompt:
                candidates_list[0] = prompt

        prompts = candidates_list[self.start_pointer: self.start_pointer+self.num_noises_to_sample]
        self.start_pointer += self.num_noises_to_sample

        return {"prompts": prompts}

    def update(self, *args, **kwargs):
        '''
        No need to update any internal state.
        '''
        return


def get_search_algo(nickname: str) -> Type[BaseSearch]:
    match nickname:
        case 'llm_icl':
            return LLMQuerySearch
        case 'llm_icl_hint':
            return LLMQueryHintSearch
        case 'llm_icl_hint_cache':
            return LLMQueryHintCacheSearch
        case 'opt2i':
            return OPT2ISearch
        case 'prompt_random_paraphrase':
            return PromptRandomParaphraseSearch
        case _:
            raise ValueError(f"Search algorithm '{nickname}' not found.")
