import json
import os
import pandas as pd
from torch.utils.data import Dataset
import time


_URL = "https://raw.githubusercontent.com/dbsxodud-11/PAG/refs/heads/main/prompts/eval_prompt_lexica.jsonl"


class Lexica(Dataset):
    def __init__(self, file_path='dataset/eval_prompt_lexica.jsonl'):
        super().__init__()
        # download csv if not exist
        if not os.path.exists(file_path):
            os.system(f"wget {_URL} -O {file_path}")
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

def get_lexica_list():
    # print("Deprecated, use jsonl_dataset instead.")
    return Lexica().all_prompts
