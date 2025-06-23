import os
import sys
import json
from typing import Union
import numpy as np

import typing_extensions as typing
from PIL import Image
import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip


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


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def normalize_tensor(a: torch.Tensor, axis: int = -1, order: int = 2) -> torch.Tensor:
    norm_sq = a.abs().pow(order).sum(dim=axis, keepdim=True)
    l2 = norm_sq.pow(1.0 / order)
    l2 = torch.where(l2 == 0, torch.ones_like(l2), l2)
    return a / l2

class AestheticVerifier:
    nickname = "aesthetic"    
    SUPPORTED_METRIC_CHOICES = [
        "reward",
    ]
    def __init__(
        self,
        aesthetic_model_path="verifiers/sac+logos+ava1-l14-linearMSE.pth",
        device=None,
        **kwargs
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # TODO load (ViT-L/14)
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()

        self.aesthetic_model = MLP(input_size=768)
        state_dict = torch.load(aesthetic_model_path, map_location=self.device)
        self.aesthetic_model.load_state_dict(state_dict)
        self.aesthetic_model.to(self.device)
        self.aesthetic_model.eval()

    def prepare_inputs(
        self,
        images: Union[list[Union[str, Image.Image]], Union[str, Image.Image]],
        prompts: Union[list[str], str],
        **kwargs
    ):

        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]

        inputs = []
        for prompt, image in zip(prompts, images):

            inputs.append((prompt, image))
        return inputs

    def score(self, inputs, ret_type='float', **kwargs) -> list[Grading]:

        def call_aesthetic_score(pair, ret_type='float'):
            prompt, image_data = pair

            if isinstance(image_data, str):
                pil_image = Image.open(image_data)
                img_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            elif isinstance(image_data, Image.Image):
                # already PIL.Image
                pil_image = image_data
                img_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            else: # is torch.tensor. already normalized (preprocessed)
                if image_data.ndim == 3:
                    img_tensor = image_data.unsqueeze(0).to(self.device)
                else: # is tensor?
                    img_tensor = image_data
            
            if ret_type == 'float':
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(img_tensor)
                image_features = image_features.cpu().numpy() # maybe fix later; it is fast enough.
                image_features = normalized(image_features)

                with torch.no_grad():
                    emb_tensor = torch.from_numpy(image_features).to(self.device).float()
                    score_tensor = self.aesthetic_model(emb_tensor)
                    aesthetic_score = float(score_tensor.item())
            elif ret_type == 'tensor':
                image_features = self.clip_model.encode_image(img_tensor)
                
                image_features = normalize_tensor(image_features)
                emb_tensor = image_features.float()
                score_tensor = self.aesthetic_model(emb_tensor)
                aesthetic_score = torch.mean(score_tensor) # ret as torch.floatTensor

            grading: Score = {
                    "explanation": "Aesthetic score",
                    "reward": aesthetic_score,
                }   

            return grading

        results = [call_aesthetic_score(pair, ret_type) for pair in inputs]
        return results
