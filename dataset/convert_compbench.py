import json


eval_prompt_compbench_3d = "UniDet_eval/examples/dataset/3d_spatial_val.txt"
eval_prompt_compbench_2d = "UniDet_eval/examples/dataset/spatial_val.txt"
eval_prompt_compbench_numeracy = "UniDet_eval/examples/dataset/numeracy_val.txt"

with open(eval_prompt_compbench_3d, 'r') as f:
    eval_prompt_compbench_3d = f.readlines()
with open(eval_prompt_compbench_2d, 'r') as f:
    eval_prompt_compbench_2d = f.readlines()
with open(eval_prompt_compbench_numeracy, 'r') as f:
    eval_prompt_compbench_numeracy = f.readlines()


# strip and save as a jsonl
eval_prompt_compbench_3d = [{'initial_prompt': x.strip()} for i, x in enumerate(eval_prompt_compbench_3d) if i % 3 == 0]
eval_prompt_compbench_2d = [{'initial_prompt': x.strip()} for i, x in enumerate(eval_prompt_compbench_2d) if i % 3 == 0]
eval_prompt_compbench_numeracy = [{'initial_prompt': x.strip()} for i, x in enumerate(eval_prompt_compbench_numeracy) if i % 3 == 0]


# manual filter for avoiding safety filter of Gemma
num_substitute = 0
for i in range(len(eval_prompt_compbench_3d)):
    if eval_prompt_compbench_3d[i]['initial_prompt'] == 'a girl hidden by a refrigerator':
        eval_prompt_compbench_3d[i]['initial_prompt'] = "a bowl behind a refrigerator"
        num_substitute += 1

print(len(eval_prompt_compbench_3d), len(eval_prompt_compbench_2d), len(eval_prompt_compbench_numeracy), num_substitute)


with open('dataset/eval_prompt_compbench_3d.jsonl', 'w') as f:
    for item in eval_prompt_compbench_3d:
        f.write(json.dumps(item) + '\n')
with open('dataset/eval_prompt_compbench_2d.jsonl', 'w') as f:
    for item in eval_prompt_compbench_2d:
        f.write(json.dumps(item) + '\n')
with open('dataset/eval_prompt_compbench_numeracy.jsonl', 'w') as f:
    for item in eval_prompt_compbench_numeracy:
        f.write(json.dumps(item) + '\n')

print("Saved eval prompts to dataset/eval_prompt_compbench_3d.jsonl, dataset/eval_prompt_compbench_2d.jsonl, dataset/eval_prompt_compbench_numeracy.jsonl")
