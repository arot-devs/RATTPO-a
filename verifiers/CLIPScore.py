import os
import sys
import json
from typing import Dict, Union, List, Union

import torch
import numpy as np
import clip
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


class CLIPScoreVerifier:
    nickname = "clipscore"
    SUPPORTED_METRIC_CHOICES = [
        "reward",
    ]
    def __init__(self, device: str = None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        self.model.eval()
        print('CLIP model loaded.')

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

    def score(self, inputs: list[tuple[str, Union[str, Image.Image]]], ret_type='float', negative=False, **kwargs):

        def compute_clip_grading(pair: tuple[str, Union[str, Image.Image]]):
            prompt, image_data = pair

            if isinstance(image_data, str):
                image = Image.open(image_data)
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            elif isinstance(image_data, Image.Image):
                # already PIL.Image
                image = image_data
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            else: # is tensor?
                if image_data.ndim == 3:
                    image_tensor = image_data.unsqueeze(0).to(self.device)
                else: # nidm=4
                    image_tensor = image_data
            text_tokens = clip.tokenize([prompt], truncate=True).to(self.device)


            if ret_type == 'float':
                with torch.no_grad():
                    if self.device == "cuda":
                        image_tensor = image_tensor.half()
                    image_features = self.model.encode_image(image_tensor)
                    text_features = self.model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features * text_features).sum(dim=-1).item()
                if negative:
                    similarity = -similarity

                clipscore = float(similarity * 100.0)
            
            elif ret_type == 'tensor':
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features * text_features).sum(dim=-1).mean()
                if negative:
                    similarity = -similarity

                clipscore = similarity * 100.0

            result: Score = {
                    "explanation": "CLIPScore.",
                    "reward": clipscore,
                }
            return result

        results = [compute_clip_grading(inp) for inp in inputs]
        return results

    def aggregate_to_one(self, results: List[Dict], method='mean') -> Dict:
        '''
        Aggregate given results to one dict.
        '''
        assert len(results) > 0, "results should not be empty"

        ret = {
                "reward": [],
                "verifier_scores": {
                        'clipscore': []
                        # for vname in self.verifiers.keys()
                    }
              }
        # append
        for single_result in results:
            ret['reward'].append(single_result['reward'])
            ret['verifier_scores']['clipscore'].append(single_result['reward'])
                    
        # mean
        assert method == 'mean', "only mean is supported for now"
        ret['reward'] = sum(ret['reward']) / len(ret['reward'])

        ret['verifier_scores']['clipscore'] = sum(ret['verifier_scores']['clipscore']) / len(ret['verifier_scores']['clipscore'])
        
        return ret
