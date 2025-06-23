import os
import sys
import json
from typing import Dict, Union, List, Union

import torch
import ImageReward as RM
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
)

import typing_extensions as typing
from tqdm import tqdm


class Score(typing.TypedDict):
    explanation: str
    score: float


class Grading(typing.TypedDict):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_or_thematic_resonance: Score
    overall_score: Score


def get_preprocess(n_px=224):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


class ImageRewardVerifier:
    nickname = "imagereward"
    SUPPORTED_METRIC_CHOICES = [
        "reward",
    ]
    def __init__(self, device: str = None, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RM.load("ImageReward-v1.0")
        self.device = device
        self.model.to(self.device)
        self.model.device = device
        self.model.eval()

    def prepare_inputs(
        self,
        images: Union[list[Union[str, Image.Image]], Union[str, Image.Image]],
        prompts: Union[list[str], str],
        **kwargs,
    ):

        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]

        inputs = list(zip(prompts, images))

        return inputs

    def score(self, inputs: list[tuple[str, Union[str, Image.Image]]], **kwargs):

        def compute_grading(pair: tuple[str, Union[str, Image.Image]]):

            prompt, image_data = pair

            if isinstance(image_data, str):
                pil_image = Image.open(image_data)
            else:
                pil_image = image_data        

            rewards = self.model.score(prompt, pil_image)

            result: Grading = {
                    "explanation": "ImageReward.",
                    "reward": rewards,
                }
            return result

        results = [compute_grading(inp) for inp in inputs]
        return results


    def aggregate_to_one(self, results: List[Dict], method='mean') -> Dict:
        '''
        Aggregate given results to one dict.
        '''
        assert len(results) > 0, "results should not be empty"

        ret = {
                "reward": [],
                "verifier_scores": {
                        'imagereward': []
                        # for vname in self.verifiers.keys()
                    }
              }
        # append
        for single_result in results:
            ret['reward'].append(single_result['reward'])
            ret['verifier_scores']['imagereward'].append(single_result['reward'])
                    
        # mean
        assert method == 'mean', "only mean is supported for now"
        ret['reward'] = sum(ret['reward']) / len(ret['reward'])

        ret['verifier_scores']['imagereward'] = sum(ret['verifier_scores']['imagereward']) / len(ret['verifier_scores']['imagereward'])

        
        return ret

