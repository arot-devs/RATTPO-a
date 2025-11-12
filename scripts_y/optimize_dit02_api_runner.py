#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Optional

import torch
from PIL import Image
from tqdm import tqdm
import shutil

# --- Path setup: use repo root for imports ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Local utility for API-based generation
from scripts_y.api_image_generator import (
    create_client,
    generate_image_via_api,
    DEFAULTS as API_DEFAULTS,
    sanitize,
    save_optimizer_chat,
)

# RATTPO utilities & verifiers
from verifiers import SUPPORTED_VERIFIERS
from prompt_templates import get_prompt_template, get_hint_template


class PromptistRewardVerifier:
    """Aggregate Aesthetic + CLIPScore into a single reward.

    reward = min(clip/100*20 - 5.6, 0) + (aesthetic - base_aesthetic)
    base_aesthetic may be provided via a JSON file mapping prompt_idx -> score or list of scores.
    """
    nickname = "promptist_reward"

    def __init__(self, base_score_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_target = ["clipscore", "aesthetic"]
        # Resolve aesthetic checkpoint robustly
        ckpt_name = 'sac+logos+ava1-l14-linearMSE.pth'
        candidate_dirs = [
            REPO_ROOT / 'verifiers',
            REPO_ROOT / 'modules' / 'RATTPO' / 'verifiers',
            Path.cwd() / 'verifiers',
        ]
        aest_pth = None
        for d in candidate_dirs:
            p = d / ckpt_name
            if p.exists():
                aest_pth = str(p)
                break
        if aest_pth is None:
            aest_pth = str(REPO_ROOT / 'verifiers' / ckpt_name)
        self.verifiers = {
            'clipscore': SUPPORTED_VERIFIERS['clipscore'](device=device),
            'aesthetic': SUPPORTED_VERIFIERS['aesthetic'](aesthetic_model_path=aest_pth, device=device),
        }
        self.base_scores = None
        if base_score_path:
            with open(base_score_path, 'r') as f:
                self.base_scores = json.load(f)

    def prepare_inputs(self, images: Union[List[Union[str, Image.Image]], Union[str, Image.Image]],
                       prompts: Union[List[str], str], **kwargs):
        return self.verifiers['clipscore'].prepare_inputs(images, prompts, **kwargs)

    def aggregate_to_one(self, results: List[Dict], method='mean') -> Dict:
        assert len(results) > 0
        ret = {"reward": [], "verifier_scores": {v: [] for v in self.verifiers.keys()}}
        for r in results:
            ret["reward"].append(r["reward"])
            for v in self.verifiers.keys():
                ret["verifier_scores"][v].append(r["verifier_scores"][v])
        ret["reward"] = sum(ret["reward"]) / len(ret["reward"])  # mean
        for v in self.verifiers.keys():
            vs = ret["verifier_scores"][v]
            ret["verifier_scores"][v] = sum(vs) / len(vs)
        return ret

    def score(self, inputs: List[tuple[str, Union[str, Image.Image]]], ret_type='float', **kwargs):
        prompt_idx = kwargs.get('prompt_idx', 0)
        result = {vname: verifier.score(inputs, ret_type=ret_type, **kwargs)
                  for vname, verifier in self.verifiers.items()}
        rewards = []
        for i in range(len(inputs)):
            _clip = result['clipscore'][i]['reward']
            _aesthetic = result['aesthetic'][i]['reward']
            base_aes = 0
            if self.base_scores is not None:
                base_aes = self.base_scores["aesthetic"][str(prompt_idx)]
                if isinstance(base_aes, list):
                    base_aes = sum(base_aes) / len(base_aes)
            f_rel = min(_clip / 100 * 20 - 5.6, 0)
            f_aes = _aesthetic - base_aes
            rewards.append(f_rel + f_aes)
        outputs = [{
            "reward": rewards[i],
            "verifier_scores": {
                vname: result[vname][i]['reward'] for vname in self.verifiers.keys()
            }
        } for i in range(len(inputs))]
        return outputs


class RATTPO:
    def __init__(self, *, search_rounds: int, num_samples_per_round: int,
                 template_version: str = 'ours', history_selection_strategy: str = 'best', history_num_selection: int = 4,
                 hint_history_selection_strategy: str = 'best', hint_history_num_selection: int = 3,
                 use_ollama: bool = True, genai_api_key: str = ''):
        self.search_rounds = search_rounds
        self.num_samples_per_round = num_samples_per_round
        self.template_version = template_version
        self.history_selection_strategy = history_selection_strategy
        self.history_num_selection = history_num_selection
        self.hint_history_selection_strategy = hint_history_selection_strategy
        self.hint_history_num_selection = hint_history_num_selection
        self.use_ollama = use_ollama
        self.genai_api_key = genai_api_key
        self.history: List[Dict] = []
        self.hint: Optional[str] = None
        if not use_ollama:
            assert genai_api_key, "Set --genai-api-key when not using Ollama"
            from google import genai
            self.genai_client = genai.Client(api_key=genai_api_key)
        else:
            import ollama  # lazy import
            self._ollama = ollama
        # logging targets (set by runner)
        self._exp_dir: Optional[str] = None
        self._round_dir: Optional[str] = None

    def set_log_dirs(self, exp_dir: str, round_dir: Optional[str] = None):
        self._exp_dir = exp_dir
        self._round_dir = round_dir

    def propose(self, search_round: int, prompt: str):
        if search_round == 1:
            self.history = []
            self.hint = None
            self.initial_prompt = prompt
        prompts = self._query_optimizer_llm(search_round)
        return {"prompts": prompts}

    def update(self, search_round: int, datapoints: List[Dict]):
        for d in datapoints:
            self.history.append({"prompt": d["prompt"], "reward": d["reward"], "round": search_round})
        self.history = sorted(self.history, key=lambda x: x['reward'], reverse=True)
        self.hint = self._query_hint_llm(search_round)

    def _query(self, text: str, *, search_round: Optional[int] = None, label: str = 'optimizer') -> str:
        tries = 0
        while True:
            try:
                if self.use_ollama:
                    messages = [{"role": "user", "content": text}]
                    out = self._ollama.chat(model=os.environ.get('RATTPO_OLLAMA_MODEL', 'gemma3:27b'),
                                             messages=messages)
                    ret = out['message']['content']
                else:
                    from google.genai import types
                    contents = [types.Content(role='user', parts=[types.Part.from_text(text=text)])]
                    out = self.genai_client.models.generate_content(model=os.environ.get('RATTPO_GENAI_MODEL', 'gemma-3-27b-it'), contents=contents)
                    ret = out.text
                ret = ret.strip()
                # Log chat (request and response)
                try:
                    odir = self._round_dir or self._exp_dir
                    if odir is not None and search_round is not None:
                        if self.use_ollama:
                            request = messages
                            response = out
                        else:
                            request = [{"role": "user", "content": text}]
                            response = ret
                        save_optimizer_chat(
                            output_dir=odir,
                            search_round=int(search_round),
                            request_messages=request,
                            response=response,
                            label=label,
                        )
                except Exception:
                    pass
                while ret.count('\n\n') > 1:
                    ret = ret.replace('\n\n', '\n')
                return ret
            except Exception as e:
                tries += 1
                if tries >= 5:
                    raise e

    def _query_optimizer_llm(self, search_round: int) -> List[str]:
        history = self.history[:]
        uniq = {}
        for h in history:
            if h['prompt'] not in uniq:
                uniq[h['prompt']] = h
        history = list(uniq.values())
        num_context = min(self.history_num_selection, len(history))
        history = history[:self.history_num_selection]
        history_prompts = [h['prompt'] for h in history]
        history_scores = [h['reward'] for h in history]
        first_t, later_t = get_prompt_template(version=self.template_version,
                                               initial_prompt=self.initial_prompt,
                                               num_sample=self.num_samples_per_round,
                                               num_context=num_context,
                                               history_prompts=history_prompts,
                                               history_scores=history_scores,
                                               hint=self.hint)
        meta_prompt = first_t if len(self.history) == 0 else later_t
        raw = self._query(meta_prompt, search_round=search_round, label='optimizer')
        parsed = raw[raw.find('1.'):].split('\n')
        prompts = ['. '.join(s.split('. ')[1:]).strip() for s in parsed if s.strip()]
        if len(prompts) < self.num_samples_per_round:
            prompts = prompts + [self.initial_prompt] * (self.num_samples_per_round - len(prompts))
        return prompts[:self.num_samples_per_round]

    def _query_hint_llm(self, search_round: int) -> Optional[str]:
        history = self.history[:]
        uniq = {}
        for h in history:
            if h['prompt'] not in uniq:
                uniq[h['prompt']] = h
        history = list(uniq.values())
        if not history:
            return None
        num_context = min(self.hint_history_num_selection, len(history))
        if self.hint_history_selection_strategy == 'best':
            history = history[:num_context]
        elif self.hint_history_selection_strategy == 'random' and num_context > 0:
            import numpy as np
            history = list(np.random.choice(history, size=num_context, replace=False))
        history_prompts = [h['prompt'] for h in history]
        history_scores = [h['reward'] for h in history]
        prompt = get_hint_template(history_prompts=history_prompts, history_scores=history_scores, num_context=num_context)
        return self._query(prompt, search_round=search_round, label='hint')


def generate_images_via_api(api_client, prompts: List[str], seeds: List[int], round_dir: str,
                            width: int, height: int, neg_prompt: str = ''):
    """Generate images via Gradio API.

    Copies the original file returned by gradio_client (preserving PNG metadata)
    into the round directory using a stable filename. Also shows a per-round
    progress bar, since generation can be slow.
    """
    os.makedirs(round_dir, exist_ok=True)
    images: List[Image.Image] = []
    img_paths: List[str] = []

    total = max(1, len(prompts) * len(seeds))
    desc = f"Generating images ({os.path.basename(round_dir)})"
    with tqdm(total=total, desc=desc, leave=False) as pbar:
        for i, p in enumerate(prompts):
            for j, s in enumerate(seeds):
                # Call API; returns path to a local tmp file with rich PNG metadata
                src_path = generate_image_via_api(
                    api_client,
                    prompt=p,
                    seed=int(s),
                    width=int(width),
                    height=int(height),
                    output_dir=round_dir,
                    image_id=f"prompt_{i}_{j}.png",
                )

                # Copy the original file (preserves metadata) to round_dir
                dst_path = os.path.join(round_dir, f"prompt_{i}_{j}.png")
                if os.path.abspath(src_path) == os.path.abspath(dst_path):
                    # Already copied by API helper
                    img_paths.append(dst_path)
                else:
                    try:
                        shutil.copy2(src_path, dst_path)
                        img_paths.append(dst_path)
                    except Exception:
                        # Fall back to the source path if copy fails
                        img_paths.append(src_path)

                # Load the image object for scoring/visualization (OK if metadata not carried in-memory)
                try:
                    img = Image.open(img_paths[-1])
                except Exception:
                    img = None
                if img is not None:
                    images.append(img)

                pbar.update(1)

    return images, img_paths


def aggregate_by_prompt(verifier: PromptistRewardVerifier, outputs: List[Dict], prompts: List[str], img_paths: List[str],
                        nip: int, initial_prompt: str, search_round: int):
    agg = []
    for i in range(len(prompts)):
        cur_out = outputs[i * nip: (i + 1) * nip]
        cur_paths = img_paths[i * nip: (i + 1) * nip]
        a = verifier.aggregate_to_one(cur_out, method='mean')
        a['initial_prompt'] = initial_prompt
        a['prompt'] = prompts[i]
        a['search_round'] = search_round
        a['img_path'] = cur_paths
        a['generation_idx'] = i
        agg.append(a)
    return agg


def load_prompts(dataset: str, prompt_path: Optional[str], num_prompts: Union[int, str], prompt_div: int, prompt_mod: int):
    if dataset == 'custom':
        assert prompt_path, "--prompt-path required for dataset=custom"
        with open(prompt_path, 'r') as f:
            all_prompts = [ln.strip() for ln in f if ln.strip()]
    else:
        from dataset.jsonl_dataset import get_all_prompts
        all_prompts = get_all_prompts(dataset)

    prompts = []
    global_indices = []
    for i, p in enumerate(all_prompts):
        if i % prompt_div == prompt_mod:
            prompts.append(p)
            global_indices.append(i)
    if isinstance(num_prompts, int):
        prompts = prompts[:num_prompts]
        global_indices = global_indices[:num_prompts]
    return prompts, global_indices


@dataclass
class Args:
    output_dir: str
    dataset: str
    prompt_path: Optional[str]
    num_prompts: Union[int, str]
    prompt_div: int
    prompt_mod: int
    width: int
    height: int
    nip: int
    seed: int
    use_ollama: bool
    genai_api_key: str
    search_rounds: int
    nspr: int
    neg_prompt: str
    api_url: Optional[str]


def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Optimize prompts using RATTPO with API-based image generation")
    ap.add_argument('--output-dir', default='output_dit02', help='Root output directory')
    ap.add_argument('--dataset', choices=['lexica', 'diffusiondb', 'parti', 'compbench_2d', 'compbench_3d', 'compbench_numeracy', 'custom'], default='custom')
    ap.add_argument('--prompt-path', default=None, help='Path to prompts file (one prompt per line) when dataset=custom')
    ap.add_argument('--num-prompts', default='all', help="Number of prompts or 'all'")
    ap.add_argument('--prompt-div', type=int, default=1)
    ap.add_argument('--prompt-mod', type=int, default=0)
    ap.add_argument('--width', type=int, default=API_DEFAULTS['width'])
    ap.add_argument('--height', type=int, default=API_DEFAULTS['height'])
    ap.add_argument('--nip', type=int, default=3, help='num images per prompt')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--use-ollama', action='store_true', help='Use local ollama backend for LLM (default off)')
    ap.add_argument('--genai-api-key', default='', help='Google GenAI API key (when not using Ollama)')
    ap.add_argument('--search-rounds', type=int, default=3)
    ap.add_argument('--nspr', type=int, default=8, help='num samples per round (candidate prompts)')
    ap.add_argument('--neg-prompt', default=API_DEFAULTS['neg_t2i'], help='Negative prompt (used as reference; generation handled by API)')
    ap.add_argument('--api-url', default=None, help='Gradio server URL (default env RATTPO_API_URL or http://127.0.0.1:7860)')
    a = ap.parse_args()
    n = a.num_prompts
    if isinstance(n, str) and n.lower() == 'all':
        n_val: Union[int, str] = 'all'
    else:
        n_val = int(n)
    return Args(
        output_dir=a.output_dir,
        dataset=a.dataset,
        prompt_path=a.prompt_path,
        num_prompts=n_val,
        prompt_div=a.prompt_div,
        prompt_mod=a.prompt_mod,
        width=a.width,
        height=a.height,
        nip=a.nip,
        seed=a.seed,
        use_ollama=a.use_ollama,
        genai_api_key=a.genai_api_key,
        search_rounds=a.search_rounds,
        nspr=a.nspr,
        neg_prompt=a.neg_prompt,
        api_url=a.api_url,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prompts, global_indices = load_prompts(args.dataset, args.prompt_path, args.num_prompts, args.prompt_div, args.prompt_mod)
    print(f"Using {len(prompts)} prompt(s)")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    verifier = PromptistRewardVerifier(device=device)

    # Create Gradio client
    api_client = create_client(args.api_url)

    algo = None  # created per-prompt, since it keeps history
    seeds = [int(args.seed + k) for k in range(args.nip)]

    for idx, prompt in enumerate(prompts):
        prompt_idx = global_indices[idx]
        exp_dir = os.path.join(args.output_dir, sanitize(prompt))
        os.makedirs(exp_dir, exist_ok=True)

        # New algorithm instance per prompt
        algo = RATTPO(search_rounds=args.search_rounds, num_samples_per_round=args.nspr,
                      template_version='ours', history_selection_strategy='best', history_num_selection=4,
                      hint_history_selection_strategy='best', hint_history_num_selection=3,
                      use_ollama=args.use_ollama, genai_api_key=args.genai_api_key)

        # Round 0: evaluate initial prompt
        round0_dir = os.path.join(exp_dir, 'round_0')
        init_images, init_paths = generate_images_via_api(api_client, [prompt], seeds, round0_dir, args.width, args.height, args.neg_prompt)
        vinp = verifier.prepare_inputs(images=init_images, prompts=[prompt] * len(init_images))
        init_scores_detail = verifier.score(inputs=vinp, ret_type='float', prompt_idx=prompt_idx)
        with open(os.path.join(round0_dir, 'detail.json'), 'w') as f:
            json.dump(init_scores_detail, f, indent=4)

        # Initialize proposer
        # Set logging dirs for round 1 before proposing (to log the optimizer chat into round_1)
        round1_dir = os.path.join(exp_dir, 'round_1')
        os.makedirs(round1_dir, exist_ok=True)
        algo.set_log_dirs(exp_dir, round1_dir)
        ret = algo.propose(1, prompt=prompt)
        candidates = ret['prompts'][:args.nspr]

        for r in range(1, args.search_rounds + 1):
            round_dir = os.path.join(exp_dir, f'round_{r}')
            # Update logger to current round dir
            algo.set_log_dirs(exp_dir, round_dir)
            images, img_paths = generate_images_via_api(api_client, candidates, seeds, round_dir, args.width, args.height, args.neg_prompt)
            vinp = verifier.prepare_inputs(images=images, prompts=[prompt] * len(images))
            outputs = verifier.score(inputs=vinp, ret_type='float', prompt_idx=prompt_idx)
            aggregated = aggregate_by_prompt(verifier, outputs, candidates, img_paths, args.nip, prompt, r)
            order = sorted(range(len(aggregated)), key=lambda x: aggregated[x]['reward'], reverse=True)
            sorted_aggr = [aggregated[i] for i in order]
            with open(os.path.join(round_dir, 'result.json'), 'w') as f:
                json.dump(sorted_aggr, f, indent=4)

            # decorate per-image details
            for i in range(len(images)):
                cand_idx, noise_idx = divmod(i, args.nip)
                outputs[i]['prompt'] = candidates[cand_idx]
                outputs[i]['search_round'] = r
                outputs[i]['img_path'] = img_paths[i]
                outputs[i]['generation_idx'] = cand_idx
                outputs[i]['noise_idx'] = noise_idx
            with open(os.path.join(round_dir, 'detail.json'), 'w') as f:
                json.dump(outputs, f, indent=4)

            algo.update(r, sorted_aggr)
            print(f"[Prompt {prompt_idx}] Round {r} best: {algo.history[0]['prompt']} (reward={algo.history[0]['reward']:.3f})")

            if r < args.search_rounds:
                ret = algo.propose(r + 1, prompt=prompt)
                candidates = ret['prompts'][:args.nspr]

    print("Done.")


if __name__ == '__main__':
    main()
