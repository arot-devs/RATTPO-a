
import requests
from PIL import Image
from copy import deepcopy
from pprint import pprint
from tqdm.auto import tqdm
from DSG.dsg.query_utils import generate_dsg
from DSG.dsg.parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output
from DSG.dsg.vqa_utils import InstructBLIP
import os
import json
import sys
import random
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

INPUT_TEXT_PROMPT = "A baby is sleeping on a covered mattress. His shirt is black white and blue. The mattress cover has designs all over it. There is a doll with a blue and purple hoody nearby."
generated_image = Image.open("DSG/assets/example_img.png")


def make_question(id2prompts, LLM, num_workers=16, verbose=True):

    if verbose:
        print("#"*50)
        print("1) Generate DSG from text with LLM")
        print("#"*50)
    # id2prompts = {
    #     'custom_0': {
    #         'input': text_prompt,
    #     }
    # }
    # import ipdb; ipdb.set_trace()

    id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
        id2prompts, generate_fn=LLM,
        verbose=verbose,
        N_parallel_workers=num_workers,
    )
    qid2tuples = [
        parse_tuple_output(id2tuple_outputs[k]['output'])
        for k in id2tuple_outputs.keys()
    ]
    qid2dependencies = [
        parse_dependency_output(id2dependency_outputs[k]['output'])
        for k in id2dependency_outputs.keys()
    ]
    qid2questions = [
        parse_question_output(id2question_outputs[k]['output'])
        for k in id2question_outputs.keys()
    ]

    return qid2tuples, qid2dependencies, qid2questions


global COUNTER
COUNTER = 0


if __name__ == "__main__":

    # hardcoded for now
    avilable_ports = [11434]
    model_name = 'gemma3:27b'
    num_workers = 1
    verbose = False

    def gemma_generate(_prompt):
        global COUNTER
        port = avilable_ports[COUNTER % len(avilable_ports)]
        COUNTER += 1
        # port = 11434
        headers = {'Content-Type': 'application/json'}
        url = f"http://localhost:{port}/api/generate"
        data = {
            "model": model_name,
            "stream": False,
        }
        post_args = {"url": url, "json": data, "headers": headers}
        post_args['json']['prompt'] = _prompt

        success = False
        while not success:
            try:
                raw_response = requests.post(**post_args).json()
                output = raw_response['response']
                success = True
            except KeyboardInterrupt:
                exit(1)
            except Exception as e:
                print("Connection error, retrying...")
                print(raw_response)
                pass

        return output

    # init data: also hardcoded for now, use partiprompt
    parti = []
    with open("../dataset/eval_prompt_parti.jsonl", "r") as f:
        for line in f:
            parti.append(json.loads(line)['initial_prompt'])

    
    id2prompts = {}
    for i, prompt in enumerate(parti):
        id2prompts[f'{i}'] = {
            'input': prompt,
        }

    qid2tuples, qid2dependencies, qid2questions = make_question(
        id2prompts = id2prompts,
        LLM = gemma_generate,
        num_workers= num_workers,
        verbose = verbose
    )

    output = {}
    for i, k in enumerate(id2prompts.keys()):
        output[k] = {
            'qid2tuples': qid2tuples[i],
            'qid2dependencies': qid2dependencies[i],
            'qid2questions': qid2questions[i],
        }


    with open("parti_decomposed.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Done")