import os
import json
from datasets import load_dataset
# check if 'dataset' dir is in the current working directory
# change working dir to parent dir
import os
if not os.path.exists('dataset'):
    os.chdir("..")

ds = load_dataset("nateraw/parti-prompts", split="train")

challenge_count = {
    "Properties & Positioning": 0,
    "Quantity": 0,
    "Fine-grained Detail": 0,
    "Complex": 0,
}


# helper for filtering
def count_filter(example, first_n=50):
    if example['Challenge'] in challenge_count.keys():
        if challenge_count[example['Challenge']] >= first_n:
            return False
        challenge_count[example['Challenge']] += 1
        return True
    return False

print("Filtering dataset...")
# fitler first 50 for each challenge
ds = ds.filter(count_filter)

out = []
for each in ds:
    out.append({
        "initial_prompt": each['Prompt'],
        "challenge": each['Challenge'],
    })

# save to jsonl
with open("dataset/eval_prompt_parti.jsonl", "w") as f:
    for each in out:
        f.write(json.dumps(each) + "\n")

print("Done.")