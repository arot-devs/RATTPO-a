import json
import os
import pandas as pd
from torch.utils.data import Dataset
import time


_URLS = {
    "lexica": "https://raw.githubusercontent.com/dbsxodud-11/PAG/refs/heads/main/prompts/eval_prompt_lexica.jsonl",
    "diffusiondb": "https://raw.githubusercontent.com/dbsxodud-11/PAG/refs/heads/main/prompts/eval_prompt_diffusiondb.jsonl",
    "parti": None,
    "compbench_2d": None,
    "compbench_3d": None,
    "compbench_numeracy": None,
}


class JsonlDataset(Dataset):
    def __init__(self, dataset_name='lexica'):
        super().__init__()
        assert dataset_name in _URLS, f"dataset_name {dataset_name} not in {_URLS.keys()}"
        file_path=f'dataset/eval_prompt_{dataset_name}.jsonl'
        if not os.path.exists(file_path):            
            url = _URLS[dataset_name]
            if dataset_name == 'parti':
                os.system('python dataset/convert_partiprompt.py')
            elif dataset_name.startswith('compbench'):
                assert False, 'Should be here.'
            else:
                os.system(f"wget {url} -O {file_path}")
            # wait for download
            while not os.path.exists(file_path):
                time.sleep(1)
                pass
        # read it and save as list
        all_prompts = []
        with open(file_path, "r") as f:
            for line in f:
                all_prompts.append(json.loads(line)['initial_prompt'])

        self.all_prompts = all_prompts

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, i):
        return self.all_prompts[i]


def get_all_prompts(dataset_name='lexica'):
    return JsonlDataset(dataset_name).all_prompts
