import os
import json
import spacy
import torch
import numpy as np
import PIL.Image as Image

from tqdm import tqdm
from word2number import w2n
from typing import Dict, Union, List, Union
from accelerate import Accelerator

from UniDet_eval.experts.obj_detection.generate_dataset import Dataset as Dataset2D
from UniDet_eval.experts.obj_detection.generate_dataset_3d import Dataset as Dataset3D
from UniDet_eval.experts.model_bank import load_expert_model as load_expert_model_2d
from UniDet_eval.experts.model_bank_3d import load_expert_model as load_expert_model_3d

from UniDet_eval.experts.depth.generate_dataset import Dataset as Dataset_depth
from UniDet_eval.spatial_eval_3D import determine_position as determine_position_3d
from UniDet_eval.spatial_eval_2D import determine_position as determine_position_2d
from UniDet_eval.spatial_eval_2D import get_mask_labels
from UniDet_eval.numeracy_eval import calculate_iou

obj_label_map = torch.load('UniDet_eval/dataset/detection_features.pt', weights_only=False)['labels']


with open("UniDet_eval/examples/dataset/new_objects.txt", "r") as f:
    objects = f.read().splitlines()
    object_s, object_p = [obj.split(" - ")[0].strip().lower() for obj in objects], [obj.split(" - ")[1].strip().lower() for obj in objects]


def collate_fn(batch):
    image_list = []
    for image in batch:
        image_list.append(image)
    return image_list


class CompBenchVerifier:
    nickname = "compbench"
    SUPPORTED_METRIC_CHOICES = [
        "reward",
    ]

    def __init__(self, category:str=None):
        self.category = category
        self.accelerator = Accelerator(mixed_precision='fp16')
        match category:
            case 'numeracy':
                obj_det_model_numeracy, self.obj_det_transform_numeracy = load_expert_model_2d(task='obj_detection', ckpt="R50")
                self.obj_det_model_numeracy = self.accelerator.prepare(obj_det_model_numeracy)
            case '2d':
                obj_det_model_2d, self.obj_det_transform_2d = load_expert_model_2d(task='obj_detection', ckpt="RS200")
                self.obj_det_model_2d = self.accelerator.prepare(obj_det_model_2d)
            case '3d':
                depth_model, self.depth_transform = load_expert_model_3d(task='depth')
                self.depth_model = self.accelerator.prepare(depth_model)

                obj_det_model_3d, self.obj_det_transform_3d = load_expert_model_3d(task='obj_detection')
                self.obj_det_model_3d = self.accelerator.prepare(obj_det_model_3d)
            case _:
                raise ValueError("Invalid category")

        self.batch_size = 8 #TODO
        self.nlp = spacy.load("en_core_web_sm") # load once and reuse


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
    

    def score(self, inputs: list[tuple[str, Union[str, Image.Image]]], prompt_idx=None, verbose=False, **kwargs):
        temp_dir = self._init_temp_dir(inputs)
        if verbose:
            print('temp dir created in {}'.format(temp_dir.name))
        data_path = temp_dir.name

        match self.category:
            case 'numeracy':
                score_list = self._get_numeracy_score(data_path, verbose=verbose)

            case '2d':
                score_list = self._get_2d_score(data_path, verbose=verbose)

            case '3d':
                # additional depth estimation needed for 3d relation
                self._save_depth_map(temp_dir, verbose=verbose)
                score_list = self._get_3d_score(data_path, verbose=verbose)
            
        result = [{
                "explanation": f"CompBench:{self.category}",
                "reward": score,
            } for score in score_list]

        temp_dir.cleanup()
        return result


    def aggregate_to_one(self, results: List[Dict], method='mean') -> Dict:
        
        ret = {
            'reward': [],
        }

        # append
        for single_result in results:
            ret['reward'].append(single_result['reward'])
                    
        # mean
        assert method == 'mean', "only mean is supported for now"

        ret['reward'] = np.mean(ret['reward'])

        return ret


    def _init_temp_dir(self, inps):
        '''
        Initialize temp directory and save images on there.
        '''
        import tempfile
        temp_dir = tempfile.TemporaryDirectory()
        img_path = os.path.join(f'{temp_dir.name}', 'samples')
        os.makedirs(img_path, exist_ok=True)
        for i, (prompt, img) in enumerate(inps):
            img.save(os.path.join(img_path, f'{prompt}_{i:04d}.png'))
        
        return temp_dir


    @torch.no_grad()
    def _save_depth_map(self, temp_dir, verbose=False):

        outpath = data_path = temp_dir.name
        save_path = os.path.join(f'{outpath}/labels', 'depth')
        dataset = Dataset_depth(data_path, self.depth_transform)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        data_loader = self.accelerator.prepare(data_loader)

        for i, (test_data, img_path, img_size) in enumerate(tqdm(data_loader, disable=not verbose)):
            test_pred = self.depth_model(test_data)

            for k in range(len(test_pred)):
                img_path_split = img_path[k].split('/')
                ps = img_path[k].split('.')[-1]
                im_save_path = os.path.join(save_path, img_path_split[-3], img_path_split[-2])
                os.makedirs(im_save_path, exist_ok=True)

                im_size = img_size[0][k].item(), img_size[1][k].item()
                depth = test_pred[k]
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(1), size=(im_size[1], im_size[0]),
                                                        mode='bilinear', align_corners=True)
                depth_im = Image.fromarray(255 * depth[0, 0].detach().cpu().numpy()).convert('L')
                depth_im.save(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.png')))
        if verbose:
            print('depth map saved in {}'.format(im_save_path))


    @torch.no_grad()
    def _get_3d_score(self, data_path, verbose=False):

        save_path = os.path.join(data_path, 'labels', 'depth')
        depth_path = os.path.join(save_path, data_path.split('/')[-1])


        dataset = Dataset3D(data_path, depth_path, self.obj_det_transform_3d)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        data_loader = self.accelerator.prepare(data_loader)

        #obj detection
        result = []
        map_result = []
        for _, test_data in enumerate(tqdm(data_loader, disable=not verbose)):
            test_pred = self.obj_det_model_3d(test_data)
            for k in range(len(test_pred)):
                
                instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  
                instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
                depth = test_data[k]['depth']

                # get score
                instance_score = test_pred[k]['instances'].get_fields()['scores']

                obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)

                obj = []  
                for i in range(len(obj_bounding_box)):
                    obj_name = obj_label_map[obj_labels_dict[i]]  
                    obj.append(obj_name)


                img_path_split = test_data[k]['image_path'].split('/')
                prompt = img_path_split[-1].split('_')[0] # get prompt from file names
                
                vocab_spatial_3d = ["in front of", "behind", "hidden"] 

                locality = None

                for word in vocab_spatial_3d:
                    if word in prompt:
                        locality = word
                        break

                # nlp = spacy.load("en_core_web_sm")
                doc = self.nlp(prompt)
                obj1= [token.text for token in doc if token.pos_=='NOUN'][0]
                obj2= [token.text for token in doc if token.pos_=='NOUN'][-1]

                person = ['girl','boy','man','woman']
                if obj1 in person:
                    obj1 = "person"
                if obj2 in person:
                    obj2 = "person"
                # transform obj list to str
                obj_str = " ".join(obj)
                obj1_pos = None
                obj2_pos = None
                if obj1 in obj_str and obj2 in obj_str:
                    # get obj_pos
                    for i in range(len(obj)):
                        if obj1 in obj[i]:
                            obj1_pos = i
                        if obj2 in obj[i]:
                            obj2_pos = i
                        if (obj1_pos is not None) and (obj2_pos is not None):
                            break
                        
                    obj1_bb = obj_bounding_box[obj1_pos]
                    obj2_bb = obj_bounding_box[obj2_pos]
                    box1, box2={},{}

                    box1["x_min"] = obj1_bb[0]
                    box1["y_min"] = obj1_bb[1]
                    box1["x_max"] = obj1_bb[2]
                    box1["y_max"] = obj1_bb[3]
                    box2["x_min"] = obj2_bb[0]
                    box2["y_min"] = obj2_bb[1]
                    box2["x_max"] = obj2_bb[2]
                    box2["y_max"] = obj2_bb[3]


                    score = 0.25 * instance_score[obj1_pos].item() + 0.25 * instance_score[obj2_pos].item()  # score = avg across two objects score
                    score += determine_position_3d(locality, box1, box2, depth_map=depth) / 2
                elif obj1 in obj_str:
                    # get obj_pos
                    for i in range(len(obj)):
                        if obj1 in obj[i]:
                            obj1_pos = i
                            break
                    # obj1_pos = obj.index(obj1)  
                    score = 0.25 * instance_score[obj1_pos].item()
                elif obj2 in obj_str:
                    # get obj_pos
                    for i in range(len(obj)):
                        if obj2 in obj[i]:
                            obj2_pos = i
                            break
                    # obj2_pos = obj.index(obj2)
                    score = 0.25 * instance_score[obj2_pos].item()
                else:
                    score = 0


                image_dict = {}
                image_dict['question_id']=int(img_path_split[-1].split('_')[-1].split('.')[0])
                image_dict['answer'] = score
                result.append(image_dict)

        im_save_path = os.path.join(save_path, 'annotation_obj_detection_3d')
        os.makedirs(im_save_path, exist_ok=True)

        with open(os.path.join(im_save_path, 'vqa_result.json'), 'w') as f:
            json.dump(result, f)

        # get avg score
        score_list = []
        for i in range(len(result)):
            score_list.append(result[i]['answer'])
        with open(os.path.join(im_save_path, 'avg_result.txt'), 'w') as f:
            f.write('avg score is {}'.format(np.mean(score_list)))
        if verbose:
            print('avg score is {}'.format(np.mean(score_list)))
            print('result saved in {}'.format(im_save_path))

        return score_list


    @torch.no_grad()
    def _get_2d_score(self, data_path, verbose=False):
        save_path = os.path.join(data_path, 'labels', 'depth')
        depth_path = os.path.join(save_path, data_path.split('/')[-1])


        dataset = Dataset2D(data_path,  self.obj_det_transform_2d)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        data_loader = self.accelerator.prepare(data_loader)


        result = []
        map_result = []
        for i, test_data in enumerate(tqdm(data_loader, disable=not verbose)):
            test_pred = self.obj_det_model_2d(test_data)
            for k in range(len(test_pred)):
                instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  # get the bbox of list
                instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
                depth = test_data[k]['image'][0]

                # get score
                instance_score = test_pred[k]['instances'].get_fields()['scores']

                obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)

                obj = []  
                for i in range(len(obj_bounding_box)):
                    obj_name = obj_label_map[obj_labels_dict[i]]  
                    obj.append(obj_name)


                img_path_split = test_data[k]['image_path'].split('/')
                prompt = img_path_split[-1].split('_')[0] # get prompt from file names
                vocab_spatial = ['on side of', 'next to', 'near', 'on the left of', 'on the right of', 'on the bottom of', 'on the top of','on top of'] #locality words

                locality = None
                for word in vocab_spatial:
                    if word in prompt:
                        locality = word
                        break

                # if (args.complex):
                if False:
                    #for complex structure
                    nlp = spacy.load('en_core_web_sm')
                    # Define the sentence
                    sentence = prompt
                    # Process the sentence using spaCy
                    doc = nlp(sentence)
                    # Define the target prepositions
                    prepositions = ["on top of", "on bottom of", "on the left", "on the right",'next to','on side of','near']
                    # Extract objects before and after the prepositions
                    objects = []
                    for i in range(len(doc)):
                        if doc[i:i + 3].text in prepositions or doc[i:i + 2].text in prepositions or doc[i:i + 1].text in prepositions:
                            if doc[i:i + 3].text in prepositions:
                                k=3
                            elif doc[i:i + 2].text in prepositions:
                                k=2
                            elif doc[i:i + 1].text in prepositions:
                                k=1
                            preposition_phrase = doc[i:i + 3].text
                            for j in range(i - 1, -1, -1):
                                if doc[j].pos_ == 'NOUN':
                                    objects.append(doc[j].text)
                                    break
                                elif doc[j].pos_ == 'PROPN':
                                    objects.append(doc[j].text)
                                    break
                            flag=False
                            for j in range(i + k, len(doc)):
                                if doc[j].pos_ == 'NOUN':
                                    objects.append(doc[j].text)
                                    break
                                if(j==len(doc)-1):
                                    flag=True 
                            if flag:
                                for j in range(i + k, len(doc)):
                                    if (j+1<len(doc)) and doc[j].pos_ == 'PROPN' and doc[j+1].pos_ != 'PROPN':
                                        objects.append(doc[j].text)
                                        break
                    if (len(objects)==2):
                        obj1=objects[0]
                        obj2=objects[1]
                    else:
                        obj1=None
                        obj2=None
                else:
                    #for simple structure
                    # nlp = spacy.load("en_core_web_sm")
                    doc = self.nlp(prompt)
                    obj1= [token.text for token in doc if token.pos_=='NOUN'][0]
                    obj2= [token.text for token in doc if token.pos_=='NOUN'][-1]

                person = ['girl','boy','man','woman']
                if obj1 in person:
                    obj1 = "person"
                if obj2 in person:
                    obj2 = "person"
                if obj1 in obj and obj2 in obj:
                    obj1_pos = obj.index(obj1)
                    obj2_pos = obj.index(obj2)
                    obj1_bb = obj_bounding_box[obj1_pos]
                    obj2_bb = obj_bounding_box[obj2_pos]
                    box1, box2={},{}

                    box1["x_min"] = obj1_bb[0]
                    box1["y_min"] = obj1_bb[1]
                    box1["x_max"] = obj1_bb[2]
                    box1["y_max"] = obj1_bb[3]
                    box2["x_min"] = obj2_bb[0]
                    box2["y_min"] = obj2_bb[1]
                    box2["x_max"] = obj2_bb[2]
                    box2["y_max"] = obj2_bb[3]


                    score = 0.25 * instance_score[obj1_pos].item() + 0.25 * instance_score[obj2_pos].item()  # score = avg across two objects score
                    score += determine_position_2d(locality, box1, box2) / 2
                elif obj1 in obj:
                    obj1_pos = obj.index(obj1)  
                    score = 0.25 * instance_score[obj1_pos].item()
                elif obj2 in obj:
                    obj2_pos = obj.index(obj2)
                    score = 0.25 * instance_score[obj2_pos].item()
                else:
                    score = 0
                if (score<0.5):
                    score=0

                image_dict = {}
                image_dict['question_id']=int(img_path_split[-1].split('_')[-1].split('.')[0])
                image_dict['answer'] = score
                result.append(image_dict)

                # add mapping
                map_dict = {}
                map_dict['image'] = img_path_split[-1]
                map_dict['question_id']=int(img_path_split[-1].split('_')[-1].split('.')[0])
                map_result.append(map_dict)
        

        im_save_path = os.path.join(save_path, 'annotation_obj_detection_2d')
        os.makedirs(im_save_path, exist_ok=True)

        with open(os.path.join(im_save_path, 'vqa_result.json'), 'w') as f:
            json.dump(result, f)
        if verbose:
            print('vqa result saved in {}'.format(im_save_path))

        # get avg score
        score_list = []
        for i in range(len(result)):
            score_list.append(result[i]['answer'])

        with open(os.path.join(im_save_path, 'avg_score.txt'), 'w') as f:
            f.write('score avg:'+str(sum(score_list)/len(result)))
        if verbose:
            print("avg score:",sum(score_list)/len(result))
        
        # save mapping
        with open(os.path.join(im_save_path, 'mapping.json'), 'w') as f:
            json.dump(map_result, f)

        return score_list


    @torch.no_grad()
    def _get_numeracy_score(self, data_path, verbose=False):

        dataset = Dataset2D(data_path,  self.obj_det_transform_numeracy)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        data_loader = self.accelerator.prepare(data_loader)
        cnt = 0
        score_map = []
        total_score = 0
        # nlp = spacy.load('en_core_web_sm')
        for i, test_data in enumerate(tqdm(data_loader, disable=not verbose)):
            flag = 0

            test_pred = self.obj_det_model_numeracy(test_data)
            for k in range(len(test_pred)):
                instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  # get the bbox of list
                instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
                depth = test_data[k]['image'][0]

                obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)

                obj = []  
                for i in range(len(obj_bounding_box)):
                    obj_name = obj_label_map[obj_labels_dict[i]]  
                    obj.append(obj_name)
                new_obj = []
                new_bbox = []
                for i in range(len(obj)):
                    flag = 0
                    for j in range(len(new_obj)):
                        if calculate_iou(obj_bounding_box[i], new_bbox[j]) and obj[i] == new_obj[j]:
                            flag = 1
                            break
                    if flag == 0:
                        new_obj.append(obj[i])
                        new_bbox.append(obj_bounding_box[i])

                img_path_split = test_data[k]['image_path'].split('/')
                
                prompt = img_path_split[-1].split('_')[0] # get prompt from file names
            
                doc = self.nlp(prompt)
                number = ["a", "an", "one", "two", "three", "four", "five", "six", "seven", "eight"]
                num_obj = []
                my_obj = []
                for i in range(len(doc)):
                    if doc[i].text in number:
                        if (i < len(doc) - 2) and (doc[i+1].text + " " + doc[i+2].text in object_s or doc[i+1].text + " " + doc[i+2].text in object_p):
                            if doc[i+1].text + " " + doc[i+2].text in object_p and doc[i].text not in ["a", "an", "one"]:
                                my_obj.append(object_s[object_p.index(doc[i+1].text + " " + doc[i+2].text)])
                                try:
                                    num_obj.append(w2n.word_to_num(doc[i].text))
                                except:
                                    pass
                            else:
                                num_obj.append(1)
                                my_obj.append(doc[i+1].text + " " + doc[i+2].text)
                        elif doc[i+1].text in object_s or doc[i+1].text in object_p:
                            if doc[i+1].text in object_s and doc[i].text in ["a", "an", "one"]:
                                num_obj.append(1)
                                my_obj.append(doc[i+1].text)
                            else:
                                my_obj.append(object_s[object_p.index(doc[i+1].text)])
                                try:
                                    num_obj.append(w2n.word_to_num(doc[i].text))
                                except:
                                    pass
                score = 0
                weight = 1.0 / len(my_obj)             
                for i, my_obj_i in enumerate(my_obj):
                    if my_obj_i in ["boy", "girl", "man", "woman"]:
                        my_obj_i = "person"
                    if my_obj_i == "ship":
                        my_obj_i = "boat"
                    if my_obj_i == "telivision":
                        my_obj_i = "tv"
                    if my_obj_i == "goldfish":
                        my_obj_i = "fish"
                    if my_obj_i == "painting":
                        my_obj_i = "picture"

                    if my_obj_i not in new_obj:
                        for j, obj_i in enumerate(new_obj):
                            if my_obj_i in obj_i:
                                new_obj[j] = my_obj_i

                    if my_obj_i in new_obj:
                        score += 0.5* weight
                        num_det = new_obj.count(my_obj_i)
                        if num_det == num_obj[i]:
                            score += 0.5* weight
                
                from copy import copy
                score_map.append({"question_id": int(img_path_split[-1].split(".png")[0].split("_")[1]), "answer": score})
                cnt += 1
                total_score += score
    
        os.makedirs(os.path.join(data_path, "annotation_num"), exist_ok=True)
        p = os.path.join(data_path, "annotation_num")
        with open(os.path.join(p, 'vqa_result.json'), 'w') as f:
            json.dump(score_map, f)

        with open(os.path.join(p, 'score.txt'), 'w') as f:
            f.write(f"total:{total_score} num:{cnt} avg:{str(total_score / cnt)}")

        # get avg score
        score_list = []
        for i in range(len(score_map)):
            score_list.append(score_map[i]['answer'])
        
        return score_list
