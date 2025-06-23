import os
import json
from typing import List, Tuple
from tqdm.auto import tqdm
from datetime import datetime, timezone, timedelta
import torch
import utils
from diffusers import DiffusionPipeline
from verifiers import SUPPORTED_VERIFIERS
from verifiers import AestheticVerifier, ImageRewardVerifier, DSGScoreVerifier, CompBenchVerifier
from dataset.jsonl_dataset import get_all_prompts
from PIL import Image
import sys


UTC = timezone(timedelta(hours=0))  # UTC


def eval_plain(pipe: DiffusionPipeline, initial_prompt: str, prompt_idx: int,
               verifiers, num_noise=3, sync_seed:torch.Tensor=None) -> Tuple[List[List[float] | None], List[Image.Image]]:
    '''
    Evaluate given initial prompt with verifiers. Return the scores in the same order as the verifiers.
    Verifiers can be None, in which case the score will be None.
    Return: (List of List of scores, List of images)
    '''
    latents = sync_seed.clone()
    # generate images
    plain_images = pipe(
        prompt=initial_prompt,
        num_images_per_prompt=num_noise,
        latents=latents,
    ).images
    # evaluate clip and aesthetic scores
    out = []
    for veri in verifiers:
        if veri is None:
            out.append(None)
            continue
        # prepare inputs
        inps = veri.prepare_inputs(images=plain_images,
                                    prompts=[initial_prompt] * num_noise)
        cur_out = [float(each['reward']) for each in veri.score(inps, prompt_idx=prompt_idx)]

        out.append(cur_out)

    return out, plain_images


def main(args):
    # === Initialize the verifiers ===
    diffusion_config = utils.load_if_needed(args.diffusion_config)

    pipe = utils.init_diffusion(diffusion_config)

    aes_veri = AestheticVerifier() if "aesthetic" in args.verifiers else None
    ir_veri = ImageRewardVerifier() if "imagereward" in args.verifiers else None
    dsg_veri = None
    if "dsgscore" in args.verifiers:
        dsg_veri = DSGScoreVerifier(verifier_args={
            'question_cache_path': f"verifiers/{args.dataset}_decomposed.json" })
    comp_veri = None
    if "compbench" in args.verifiers:
        category = args.dataset.replace('compbench_', '')
        comp_veri = CompBenchVerifier(category=category)
    all_verifiers = [each for each in [aes_veri, ir_veri, dsg_veri, comp_veri] if each is not None]
    

    # === datasets ===
    initial_prompts = get_all_prompts(args.dataset)

    # === Initialize the output directory ===
    output_dir = os.path.join('baseline_output', args.dataset, args.name)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_img:
        os.makedirs(os.path.join(output_dir, 'plain_images'), exist_ok=True)
    # save args in output directory
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    eval_result = {}
    # save args in eval result
    for k, v in vars(args).items():
        eval_result[k] = v

    eval_datetime = datetime.now(UTC)
    eval_result['eval_datetime'] = eval_datetime.strftime("%Y%m%d_%H%M%S")
    for veri_nickname in args.verifiers:
        eval_result[veri_nickname] = {}

    model_name = diffusion_config["pretrained_model_name_or_path"]
    pipeline_call_args = diffusion_config["pipeline_call_args"]
    dtype = utils.TORCH_DTYPE_MAP[diffusion_config["torch_dtype"]]
    num_noise = pipeline_call_args.num_images_per_prompt
    assert num_noise == 3
    latent_prep_fn = utils.get_latent_prep_fn(model_name)
    fixed_noise = utils.get_noises(
        noise_seed=args.seed,
        num_images_per_prompt=num_noise,
        height=pipeline_call_args.height,
        width=pipeline_call_args.width,
        dtype=dtype,
        fn=latent_prep_fn,
    )

    for i, _p in tqdm(enumerate(initial_prompts), total=len(initial_prompts)):
        list_of_veri_scores, plain_images = eval_plain(
            pipe=pipe,
            initial_prompt=_p,
            prompt_idx=i,
            verifiers=all_verifiers,
            num_noise=num_noise,
            sync_seed=fixed_noise,
        )
        
        # parse and store to dict
        for veri, scores in zip(all_verifiers, list_of_veri_scores):
            if scores is None:
                continue
            eval_result[veri.nickname][i] = scores

        # save plain images
        if args.save_img:
            for j, img in enumerate(plain_images):
                img.save(os.path.join(output_dir, 'plain_images', f"plain_{i}_{j}.png"))

        # import ipdb; ipdb.set_trace() # debug

    # save eval result as json
    path = os.path.join('baseline_output', args.dataset, args.name, f'basescore_seed{args.seed}.json')
    with open(path, "w") as f:
        json.dump(eval_result, f, indent=4)
    print(f"Saved eval result to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Base score evaluation script")
    parser.add_argument("--name", type=str, required=True, help="Name for determining output path")
    parser.add_argument("--dataset", type=str, default="lexica", help="Dataset to use for evaluation") # TODO add other datasets as choices
    parser.add_argument("--diffusion_config", type=str, default="configs/diffusion_configs/sd1.4.json")
    parser.add_argument("--save_img", action='store_true', help="If set, save generated images")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--verifiers", nargs='+', default=["clipscore", "aesthetic", "imagereward"], 
                        choices=list(SUPPORTED_VERIFIERS.keys()))
    # TODO: add other arguments maybe...

    args = parser.parse_args()
    original_command = "python " + ' '.join(sys.argv)
    args.original_command = original_command
    try:
        main(args)
    except Exception as e:
        import ipdb
        print(f"Error: {e}")
        ipdb.post_mortem()
