
import json
from typing import Dict, Union, List, Union
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from verifiers.DSG.dsg.vqa_utils import InstructBLIP
import typing_extensions as typing
from tqdm import tqdm


class Score(typing.TypedDict):
    explanation: str
    score: float


def evaluate_with_question(generated_image, vqa_model, qid2tuple, qid2dependency, qid2question, verbose=True):        
    if verbose:
        print("#"*50)
        print("2) Answer questions given the generated image, with VQA")
        print("#"*50)

    qid2answer = {}
    qid2scores = {}

    for id, question in tqdm(qid2question.items(), disable=not verbose):
        answer = vqa_model.vqa(generated_image, question)
        answer = answer.replace(question, '').replace(',', '').strip() # remove question and strip
        qid2answer[id] = answer
        qid2scores[id] = float(answer[:3] in ['yes', 'Yes'])
            
    average_score_without_dep = sum(qid2scores.values()) / len(qid2scores)
        
    if verbose:
        print("#"*50)
        print("3) Zero-out scores from invalid questions")
        print("#"*50)
        
 
    # 3) zero-out scores from invalid questions 
    qid2validity = {}
    qid2scores_after_filtering = deepcopy(qid2scores)

    for id, parent_ids in qid2dependency.items():
        # zero-out scores if parent questions are answered 'no'
        any_parent_answered_no = False
        for parent_id in parent_ids:
            if parent_id == 0:
                continue
            # if qid2scores[parent_id] == 0:
            if qid2scores[str(parent_id)] == 0:
                any_parent_answered_no = True
                break
        if any_parent_answered_no:
            qid2scores_after_filtering[id] = 0.0
            qid2validity[id] = False
        else:
            qid2validity[id] = True
            
    if verbose:
        print("Per-quesiton eval results (after using dependency)")
        for id in qid2question:
            print("ID", id)
            print("question", qid2question[id])
            print("answer", qid2answer[id])
            print("validity", qid2validity[id])
            print("score (before filtering)", qid2scores[id])
            print("score (after filtering)", qid2scores_after_filtering[id])
            print()
        

    if verbose:
        print("#"*50)
        print("4) Calculate the final score by averaging")
        print("#"*50)

    average_score_with_dep = sum(qid2scores_after_filtering.values()) / len(qid2scores)
        
    return {
        "explanation": "DSGScore",
        'reward': average_score_with_dep,
        'details': {
            'qid2tuple': qid2tuple,
            'qid2dependency': qid2dependency,
            'qid2question': qid2question,
            'qid2answer': qid2answer,
            'qid2scores': qid2scores,
            'qid2scores_after_filtering': qid2scores_after_filtering,
            'qid2validity': qid2validity,
            'average_score_with_dependency': average_score_with_dep,
            'average_score_without_dependency': average_score_without_dep
        }
    }


class DSGScoreVerifier:
    nickname = "dsgscore"
    SUPPORTED_METRIC_CHOICES = [
        "reward",
    ]
    def __init__(self, verifier_args, device:str=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.vqa_model = InstructBLIP()
        self.vqa_model.model.eval()
        self.vqa_model.model.to(device)

        question_cache_path = verifier_args['question_cache_path'] # must be provided
        with open(question_cache_path, 'r') as f:
            self.cache = json.load(f)


    def prepare_inputs(
        self,
        images: Union[list[Union[str, Image.Image]], Union[str, Image.Image]],
        prompts: Union[list[str], str],
        **kwargs,
    ):

        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]
        assert len(images) == len(prompts), "images and prompts must have the same length"

        inputs = list(zip(prompts, images))

        return inputs


    @torch.inference_mode()
    def score(self, inputs: list[tuple[str, Union[str, Image.Image]]], prompt_idx, verbose=False, **kwargs):
        
        qid2tuple = self.cache[str(prompt_idx)]['qid2tuples']
        qid2dependency = self.cache[str(prompt_idx)]['qid2dependencies']
        qid2question = self.cache[str(prompt_idx)]['qid2questions']
        
        results = []
        for (_, image_data) in inputs:
            ret = evaluate_with_question(
                image_data, 
                self.vqa_model, 
                qid2tuple, 
                qid2dependency, 
                qid2question, 
                verbose=verbose)
            results.append(ret)
        
        return results


    def aggregate_to_one(self, results: List[Dict], method='mean') -> Dict:
        '''
        Aggregate given results to one dict.
        '''
        assert len(results) > 0, "results should not be empty"


        ret = {
            'reward': [],
            'details': {
                'qid2question': results[0]['details']['qid2question'],
                'qid2scores': {qid:[] for qid in results[0]['details']['qid2scores'].keys()},
                'qid2scores_after_filtering': {qid:[] for qid in results[0]['details']['qid2scores_after_filtering'].keys()},
                'average_score_with_dependency': [],
                'average_score_without_dependency': []
            }
        }

        # append
        for single_result in results:
            ret['reward'].append(single_result['reward'])
            ret['details']['average_score_without_dependency'].append(single_result['details']['average_score_without_dependency'])
            ret['details']['average_score_with_dependency'].append(single_result['details']['average_score_with_dependency'])
            for qid in ret['details']['qid2scores'].keys():
                ret['details']['qid2scores'][qid].append(single_result['details']['qid2scores'][qid])
                ret['details']['qid2scores_after_filtering'][qid].append(single_result['details']['qid2scores_after_filtering'][qid])
                    
        # mean
        assert method == 'mean', "only mean is supported for now"

        ret['reward'] = np.mean(ret['reward'])
        ret['details']['average_score_without_dependency'] = np.mean(ret['details']['average_score_without_dependency'])
        ret['details']['average_score_with_dependency'] = np.mean(ret['details']['average_score_with_dependency'])
        for qid in ret['details']['qid2scores'].keys():
            ret['details']['qid2scores'][qid] = np.mean(ret['details']['qid2scores'][qid])
            ret['details']['qid2scores_after_filtering'][qid] = np.mean(ret['details']['qid2scores_after_filtering'][qid])
        
        return ret
    