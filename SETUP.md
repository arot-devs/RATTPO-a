**RATTPO‑a Setup**

- Python 3.10, GPU recommended (CUDA 12.4 compatible). Tested with PyTorch 2.5.1 + cu12.4.
- This fork adds an env file, a quick smoke‑test script, and smaller configs to validate the pipeline end‑to‑end.

**Quick Start**
- Create the conda env:
  - `conda env create -f environment.yml`
  - `conda activate rattpo`
  - `python -m spacy download en_core_web_sm`
  - Optional kernel: `python -m ipykernel install --user --name rattpo --display-name "Python (rattpo)"`

- Smoke test with Ollama (local LLM):
  - `bash scripts/run_ollama_quick.sh`
  - This will:
    - Install Ollama if missing
    - Use custom data dir at `/local/yada/ollama`
    - Start the server at `127.0.0.1:11434`
    - Pull a small model (`llama3.2:1b`) and run a 1‑round, 1‑sample job
  - Artifacts path is printed, e.g. `output/<host>/custom/sd-v1.4/aesthetic/<exp>`
  - Tunables (env vars): `MODEL`, `PROMPT`, `EXP_NAME`, `OLLAMA_HOME`, `OLLAMA_PORT`, `KEEP_OLLAMA=1`

**Environment Details**
- File: `environment.yml` (Python 3.10)
  - Core: `torch==2.5.x`, `torchvision==0.20.x`, `pytorch-cuda=12.4`
  - Diffusion stack: `diffusers` (from Git), `transformers` (pinned Git SHA), `safetensors`, `accelerate`
  - Verifiers: `pytorch-lightning`, `fairscale`, `datasets`, `sentencepiece`, `spacy` (plus model), `protobuf`, `timm`, `einops`, `ruamel.yaml`
  - Other: `omegaconf`, `numpy`, `pillow`, `tqdm`, `matplotlib`, `requests`, `numba`, `scipy`, `google-genai`, `outlines`, `word2number`, `clip`

- GPU requirements: NVIDIA driver compatible with CUDA 12.4. The conda env provides the CUDA runtime (`pytorch-cuda=12.4`).

**Running The Notebook**
- Launch Jupyter and pick the rattpo kernel:
  - `jupyter lab`, open `RATTPO_demo.ipynb`, select kernel "Python (rattpo)".

**Running From CLI**
- Example (uses defaults in README):
  - Ensure a local LLM is available. For Ollama, set `OLLAMA_URL` (host:port), e.g. `export OLLAMA_URL=127.0.0.1:11434` and have a model pulled that matches your search config (`llm_name`).
  - `python main.py  --config configs/diffusion_configs/sd1.4.json \
                               configs/search_configs/rattpo.json \
                               configs/verifier_configs/prompt_adaptation/lexica_sd1.4_seed42.json \
                     --dataset lexica \
                     --seed 42 \
                     --exp_name <your_exp_name>`

**LLM Options**
- Local (Ollama):
  - Install: `curl -fsSL https://ollama.com/install.sh | sh`
  - Data dir: set `OLLAMA_HOME=/local/yada/ollama` to store models there.
  - Run: `OLLAMA_URL=127.0.0.1:11434 ollama serve` (in shell) and `ollama pull <model>` (e.g., `llama3.2:1b` or the model named in your search config, e.g., `gemma3:27b`).
  - The code reads `OLLAMA_URL` (host:port) to contact the server.

- Google GenAI API:
  - Set `GENAI_API_KEY` and switch the search config to `"use_genai_api": true`.

**UniDet Evaluator (Optional)**
- For UniDet‑based evaluators (CompBench), install detectron2 and expert weights as in the README:
  - `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --user`
  - `mkdir -p UniDet_eval/experts/expert_weights`
  - `cd UniDet_eval/experts/expert_weights`
  - `wget https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth`
  - `wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt`
  - `pip install gdown && gdown https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq`

**Tips**
- First run downloads large models (SD weights, Ollama model). Ensure disk space and network access.
- If you prefer CPU‑only, remove `pytorch-cuda` from `environment.yml` and recreate the env (generation will be much slower).

