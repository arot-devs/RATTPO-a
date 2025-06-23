import json
from typing import Dict, Union, List
from PIL import Image
import torch
from . import SUPPORTED_VERIFIERS

class PromptAdaptationVerifier():
    nickname = "prompt_adaptation"
    SUPPORTED_METRIC_CHOICES = [
        "reward",
    ]
    def __init__(self, verifier_args, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # search_target = None
        self.eval_target = verifier_args['eval_target']
        self.verifiers = {k: SUPPORTED_VERIFIERS[k](device=device
            ) for k in self.eval_target}

        assert 'clipscore' in self.eval_target, "clipscore must be in eval_target"
        assert 'aesthetic' in self.eval_target, "aesthetic must be in eval_target"

        self.base_score_path = verifier_args['base_score_path']
        with open(self.base_score_path, 'r') as f: # json file
            self.base_scores = json.load(f)

        
    def prepare_inputs(
        self,
        images: Union[list[Union[str, Image.Image]], Union[str, Image.Image]],
        prompts: Union[list[str], str],
        **kwargs,
    ):  
        return self.verifiers['clipscore'].prepare_inputs(images, prompts, **kwargs)

    def aggregate_to_one(self, results: List[Dict], method='mean') -> Dict:
        '''
        Aggregate given results to one dict.
        '''
        assert len(results) > 0, "results should not be empty"

        ret = {
                "reward": [],
                "verifier_scores": {
                        vname: []
                        for vname in self.verifiers.keys()
                    }
              }
        # append
        for single_result in results:
            ret['reward'].append(single_result['reward'])
            for vname in self.verifiers.keys():
                ret['verifier_scores'][vname].append(single_result['verifier_scores'][vname])
                    
        # mean
        assert method == 'mean', "only mean is supported for now"
        ret['reward'] = sum(ret['reward']) / len(ret['reward'])
        for vname in self.verifiers.keys():
            ret['verifier_scores'][vname] = sum(ret['verifier_scores'][vname]) / len(ret['verifier_scores'][vname])

        
        return ret

    def score(self, inputs: list[tuple[str, Union[str, Image.Image]]], ret_type='float', **kwargs):
        prompt_idx = kwargs['prompt_idx'] # should be in kwargs
        result = {}
        for vname, verifier in self.verifiers.items():
            result[vname] = verifier.score(inputs, ret_type=ret_type, **kwargs)

        prompt_adaptation_reward = []
        for i in range(len(inputs)):
            _clip = result["clipscore"][i]['reward']
            _aesthetic = result["aesthetic"][i]['reward']
            _base_aesthetic = self.base_scores["aesthetic"][str(prompt_idx)]
            if isinstance(_base_aesthetic, list): # use mean as cache. seems okay if we don't know inference noise.
                _base_aesthetic = sum(_base_aesthetic) / len(_base_aesthetic) 
            f_rel = min(_clip / 100 * 20 - 5.6, 0) # clamped
            f_aes = _aesthetic - _base_aesthetic # delta aesthetic
            prompt_adaptation_reward.append(f_rel + f_aes)

        results = [
            {
                "reward": prompt_adaptation_reward[i],
                "verifier_scores": {
                    vname: result[verifier.nickname][i]['reward']
                    for vname, verifier in self.verifiers.items()
                }
            }
            for i in range(len(inputs))
        ]

        return results
