# Reward-Agnostic Prompt Optimization for Text-to-Image Diffusion Models

This repository contains the official implementation of the paper "Reward-Agnostic Prompt Optimization for Text-to-Image Diffusion Models"

[Paper Link](https://arxiv.org/abs/2506.16853)

## Quick Start
If you want to run RATTPO with your own prompt or want to skim the codebase, we refer to the demo notebook (`RATTPO_demo.ipynb`).

### Repository Structure
The default search algorithm for RATTPO can be found in the class `LLMQueryHintSearch` of `search_algorithm.py`.
Datasets and preprocessing codes are in `dataset` directory, for example `eval_prompt_lexica.jsonl`.

### Setup
Our code is tested on `PyTorch 2.5.1+cu118`. We refer to `requirements.txt` for installing other dependencies.

## Usage
An example code usage, for Promptist Reward on Lexica dataset, is shown below:
```
OLLAMA_URL="localhost:11434"
python main.py  --config configs/diffusion_configs/sd1.4.json \
                        configs/search_configs/rattpo.json \
                        configs/verifier_configs/prompt_adaptation/lexica_sd1.4_seed42.json \
                --dataset lexica \
                --seed 42 \
                --exp_name example_exp_name
```
For other experiment setups, compose three configs (diffusion, search, and verifier) as desired.
Also make sure to change `--dataset` flag and `--exp_name`.

Note that the (local) ollama server should be running.
We refer to [official ollama repo](https://github.com/ollama/ollama) for setting up LLM server.

You may otherwise prefer to use Google GenAI API.
In this case, set `use_genai_api : false` in the search config and also set the environment variable `GENAI_API_KEY` as your API key.


### Running UniDet Evaluator
For using UniDet evaluators, you need to install detectron and download expert weights (for detph estimation and object detection).
Below commands would be sufficient, but we refer to [official T2I-Compbench repo](https://github.com/Karine-Huang/T2I-CompBench) for details. 
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --user
mkdir -p UniDet_eval/experts/expert_weights
cd UniDet_eval/experts/expert_weights
wget https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt
pip install gdown
gdown https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq
```

## Acknowledgement
This repository is largely based on several public repositories that we do not own.
We approciate them for sharing great codes!

- [tt-scale-flux](https://github.com/sayakpaul/tt-scale-flux) for overall repository structure
- [ImageReward](https://github.com/THUDM/ImageReward) for ImageReward verifier
- [DAS](https://github.com/krafton-ai/DAS) for combining RATTPO with test-time alignment. 
- [T2I-CompBench++](https://github.com/Karine-Huang/T2I-CompBench) for UniDet-based verifiers.
- [DSG](https://github.com/j-min/DSG) for DSG verifier.
- [Improved-Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) for Aesthetic verifier.


## Citation
If you find this repository helpful, please consider citing us:

```bibtex
@misc{Kim2025RATTPO,
Author = {Semin Kim and Yeonwoo Cha and Jaehoon Yoo and Seunghoon Hong},
Title = {Reward-Agnostic Prompt Optimization for Text-to-Image Diffusion Models},
Year = {2025},
Eprint = {arXiv:2506.16853},
}
```