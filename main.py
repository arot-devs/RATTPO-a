from utils import (
    TORCH_DTYPE_MAP,
    MODEL_NAME_MAP,
    parse_cli_args,
    preload_ollama,
    init_diffusion,
    get_noises,
    get_latent_prep_fn,
)
import os
import sys
import json
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
from verifiers.Aesthetic import AestheticVerifier
from verifiers.CompBench import CompBenchVerifier
from verifiers.ImageReward import ImageRewardVerifier
from verifiers.prompt_adaptation import PromptAdaptationVerifier
from verifiers.DSGScore import DSGScoreVerifier
from verifiers.VLM import VLMVerifier
from search_algorithm import get_search_algo, BaseSearch
from lightning_fabric import seed_everything
from glob import glob
from omegaconf import OmegaConf

# disable torch grad tracking
torch.set_grad_enabled(False)


# Non-configurable constants
UTC = timezone(timedelta(hours=0))  # UTC
VERBOSE = False
HOSTNAME = os.uname()[1]  # for debugging purpose


def v_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def sample(
    prompts: list[str],
    prompt_idx: int,
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    root_dir: str,
    config: dict,
    original_prompt: str,
    sync_seed: torch.Tensor = None,
) -> dict:
    verifier_args = config.verifier_args
    choice_of_metric = verifier_args.get("choice_of_metric", None)
    batch_size_for_img_gen = config.get("batch_size_for_img_gen", 1)

    # pipeline call args handling
    pipeline_call_args = config.pipeline_call_args
    num_images_per_prompt = pipeline_call_args.num_images_per_prompt if not config.get("das", False) else pipeline_call_args.num_particles
    num_total_images_gen = len(prompts) * num_images_per_prompt
    # just for sd1.5 negative prompt handling
    if isinstance(config["pipeline_call_args"]["negative_prompt"], str):
        config["pipeline_call_args"]["negative_prompt"] = [config["pipeline_call_args"]["negative_prompt"]] * num_total_images_gen

    if sync_seed is not None:
        if 'flux' not in config.pretrained_model_name_or_path.lower():
            assert sync_seed.shape[0] == num_images_per_prompt and len(sync_seed.shape) == 4, sync_seed.shape
        else:
            # FLUX.1-dev has different shape
            assert sync_seed.shape[0] == num_images_per_prompt and len(sync_seed.shape) == 3, sync_seed.shape

    # Generation in batch.
    # NOTE: order of all_images: (prompt0, noise0), (prompt0, noise1),  ..., (prompt1, noise0), ...
    all_images = []
    for i in range(0, len(prompts), batch_size_for_img_gen):
        batched_prompts = prompts[i: i + batch_size_for_img_gen]
        # repeat for each prompt
        if sync_seed is None:
            latents = None
        elif 'flux' not in config.pretrained_model_name_or_path.lower():
            latents = sync_seed.clone().repeat(len(batched_prompts), 1, 1, 1)  # [num_images_per_prompt * batch_size, 4, 64, 64]
        else:
            # NOTE: this is for FLUX.1-dev
            latents = sync_seed.clone().repeat(len(batched_prompts), 1, 1)
        if not config.get('das', False):
            batch_result = pipe(prompt=batched_prompts,
                                latents=latents,
                                **config["pipeline_call_args"])
            # batch_images = batch_result.images
        else:

            batch_result = pipe(prompt=batched_prompts,
                                prompt_idx=prompt_idx,
                                latents=latents,
                                ori_prompt=original_prompt,
                                **config["pipeline_call_args"])
            # batch_images = [pipe_image_process(batch_result.images[i]) for i in range(len(batch_result.images))]

        # defualt return image type is PIL Image
        batch_images = batch_result.images
        all_images.extend(batch_images)

    img_paths = []
    assert len(all_images) == num_total_images_gen, len(all_images)  # should be same
    # save all generated images

    for i in range(len(all_images)):
        candidate_idx, noise_idx = divmod(i, num_images_per_prompt)
        img_paths.append(os.path.join(root_dir, f"prompt_{prompt_idx:04d}_{candidate_idx}_{noise_idx}.png"))
    # save all generated images
    for _img, _path in zip(all_images, img_paths):
        _img.save(_path)
    # Prepare verifier inputs and perform inference.
    if isinstance(verifier, VLMVerifier):
        # VLMVerifier requires image path and prompt as input
        # # not relative path, but absolute path
        # abs_img_paths = [os.path.abspath(p) for p in img_paths]
        # NOTE: this is for VLMVerifier, which requires image path and prompt as input
        verifier_inputs = verifier.prepare_inputs(images=img_paths, prompts=[original_prompt] * num_total_images_gen)
        # verifier_inputs = verifier.prepare_inputs(images=img_paths, prompts=prompts)
    else:
        verifier_inputs = verifier.prepare_inputs(images=all_images,
                                                  prompts=[original_prompt] * num_total_images_gen)
    v_print("Scoring with the verifier.")
    outputs = verifier.score(
        inputs=verifier_inputs,
        prompt_idx=prompt_idx,
        ret_type='float'
    )
    for o in outputs:
        assert choice_of_metric in o, o.keys()

    assert len(outputs) == num_total_images_gen, f"Got {len(outputs)} != {num_total_images_gen}."

    # aggregate outputs from same prompt idx for easy evaluation
    aggregated_outputs = []
    for i in range(len(prompts)):
        # slice all images generated from this prompt
        cur_outputs = outputs[i * num_images_per_prompt: (i + 1) * num_images_per_prompt]
        cur_img_paths = img_paths[i * num_images_per_prompt: (i + 1) * num_images_per_prompt]
        aggr = verifier.aggregate_to_one(cur_outputs, method='mean')
        aggr["initial_prompt"] = original_prompt  # this is the original prompt
        aggr["prompt"] = prompts[i]  # later this is subject to change (prompt search)
        aggr["prompt_idx"] = prompt_idx  # this should be unique for each prompt
        aggr["search_round"] = search_round
        aggr["img_path"] = cur_img_paths
        aggr["generation_idx"] = i
        aggregated_outputs.append(aggr)

    # sort with score, NOTE: assume higher is better (maybe later we can sort with verifier method)
    indices = list(range(len(aggregated_outputs)))
    # Final scores are all saved as 'reward' in the output config regardless of 'choice_of_metric'/
    order_pos = sorted(indices, key=lambda x: aggregated_outputs[x]['reward'], reverse=True)
    sorted_aggr_outputs = [aggregated_outputs[i] for i in order_pos]  # NOTE: not a deep copy
    # save
    result_json_filename = os.path.join(root_dir, f"prompt_{prompt_idx:04d}.json")
    with open(result_json_filename, "w") as f:
        json.dump(sorted_aggr_outputs, f, indent=4)
    # Print debug information.
    v_print(f"Prompt='{sorted_aggr_outputs[0]['prompt']}' | Score={(sorted_aggr_outputs[0]['reward'])}")

    # also save detailed outputs for each image
    for i in range(len(all_images)):
        candidate_idx, noise_idx = divmod(i, num_images_per_prompt)
        cur_prompt = prompts[candidate_idx]
        outputs[i]["prompt"] = cur_prompt  # later this is subject to change (prompt search)
        outputs[i]["prompt_idx"] = prompt_idx  # this should be unique for each prompt
        outputs[i]["search_round"] = search_round
        outputs[i]["img_path"] = img_paths[i]
        outputs[i]["generation_idx"] = candidate_idx
        outputs[i]["noise_idx"] = noise_idx

    # sort with score
    indices = list(range(len(outputs)))
    order_pos = sorted(indices, key=lambda x: outputs[x][choice_of_metric], reverse=True)
    # order_pos = sorted(indices, key=lambda x: outputs[x][choice_], reverse=True)
    # sorted_outputs = [outputs[i] for i in order_pos] # no need to sort detailed scores
    # save
    result_json_filename = os.path.join(root_dir, f"prompt_{prompt_idx:04d}_detail.json")
    with open(result_json_filename, "w") as f:
        json.dump(outputs, f, indent=4)
    v_print(f"Serialized JSON configuration and images to {root_dir}.")

    datapoints = sorted_aggr_outputs

    return datapoints


@torch.no_grad()
def main():
    # === Load configuration and CLI arguments ===
    config = parse_cli_args()

    search_args = config["search_args"]
    search_rounds = search_args["search_rounds"]
    search_method = search_args["search_method"]
    num_prompts = config["num_prompts"]

    # === Create output directory ===
    start_datetime = datetime.now(UTC)
    config['start_datetime'] = start_datetime.strftime("%Y%m%d_%H%M%S")
    config['original_command'] = "python " + ' '.join(sys.argv)
    # current_datetime = start_datetime.strftime("%Y%m%d_%H%M%S")
    pipeline_name = config["pretrained_model_name_or_path"]
    pipeline_call_args = config["pipeline_call_args"]
    verifier_name = config["verifier_args"]["name"]

    output_dir = os.path.join(
        "output",
        HOSTNAME,
        config['dataset'],
        MODEL_NAME_MAP[pipeline_name],
        verifier_name,
        config['exp_name'],
    )


    os.makedirs(output_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {output_dir}")

    # === Load prompts ===
    match config.dataset:
        case "lexica" | "diffusiondb" | "parti":
            from dataset.jsonl_dataset import get_all_prompts
            all_prompts = get_all_prompts(config.dataset)

        case "compbench_2d" | "compbench_3d" | "compbench_numeracy":
            from dataset.jsonl_dataset import get_all_prompts
            all_prompts = get_all_prompts(config.dataset)
            config.verifier_args.category = config.dataset.split('_')[1]

        case "custom":
            all_prompts = [config.prompt]
            assert config['prompt_div'] == 1, config['prompt_div']  # expect no division; this dataset is for sanity check

        case _:
            # TODO: check this and uncomment it later
            raise NotImplementedError

    # filter for sharding
    prompts = []
    global_prompt_indices = []
    for i in range(len(all_prompts)):
        if i % config['prompt_div'] == config['prompt_mod']:
            prompts.append(all_prompts[i])
            global_prompt_indices.append(i)

    if num_prompts != "all":
        prompts = prompts[:num_prompts]

    print(f"Using {len(prompts)} prompt(s).")

    # small hack: preload ollama model if needed (fire-and-forget)
    use_ollama = 'llm' in search_method  # for all llm-based methods
    if search_args.get('use_genai_api', False):
        use_ollama = False

    if use_ollama:
        ollama_port = config['ollama_port']
        llm_name = config['search_args']['llm_name']  # maybe "llama3.3"
        preload_ollama(model_name=llm_name, port=ollama_port)

    # === Set up the image-generation pipeline ===
    torch_dtype = TORCH_DTYPE_MAP[config.torch_dtype]
    pipe = init_diffusion(config)

    # Fix seed across prompts and rounds
    model_name = config["pretrained_model_name_or_path"]
    latent_prep_fn = get_latent_prep_fn(model_name)
    fixed_noise = None
    if config.sync_all_seed:
        fixed_noise = get_noises(
            noise_seed=config.seed,
            num_images_per_prompt=pipeline_call_args.num_images_per_prompt if not config.get("das", False) else pipeline_call_args.num_particles,
            height=pipeline_call_args.height,
            width=pipeline_call_args.width,
            dtype=torch_dtype,
            fn=latent_prep_fn,
        )

    # === Load verifier model ===
    verifier_args = config["verifier_args"]

    # verifiers init
    metric_choice = verifier_args["choice_of_metric"]

    match vname := verifier_args["name"]:
        case "aesthetic":
            verifier = AestheticVerifier()
        case "imagereward":
            verifier = ImageRewardVerifier()
        case "prompt_adaptation":
            verifier = PromptAdaptationVerifier(verifier_args)
        case "dsgscore":
            verifier = DSGScoreVerifier(verifier_args)
        case "compbench":
            verifier = CompBenchVerifier(category=verifier_args.category)
        case "vlmscore":
            verifier = VLMVerifier(verifier_args)
        case _:
            raise ValueError(f"Unknown verifier: {vname}")
    pipe.verifier = verifier
    # Timeit for initialization
    end_datetime = datetime.now(UTC)
    elapsed_time = end_datetime - start_datetime
    print(f"Initialization done! Elapsed time: {elapsed_time}")
    propose_time = []
    sample_time = []
    update_time = []

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(config, resolve=True), f, indent=4)

    # === Main loop: For each prompt and each search round ===
    for idx, prompt in tqdm(enumerate(prompts), desc="Processing prompts", total=len(prompts)):
        prompt_idx = idx if config.dataset == "custom" else global_prompt_indices[idx]

        # for reproducibility
        seed_everything(config.seed, verbose=False)

        # search algorithm init. we do it here because there would be internal states to keep.
        algo_type: BaseSearch = get_search_algo(nickname=search_method)
        algo = algo_type(config, pipeline_name=pipeline_name, dtype=torch_dtype, verbose=VERBOSE)

        # search round loop
        # when verbose, disable the tqdm
        for search_round in tqdm(range(1, search_rounds + 1), leave=False, disable=VERBOSE, desc=f'prompt_{prompt_idx}'):
            v_print(f"\n=== Prompt: {prompt} | Round: {search_round} ===")

            # set output dir for this round; skip if already generated
            round_output_dir = os.path.join(output_dir, f"round_{search_round}")
            # check for skip-or-not
            if search_round == 1 and os.path.exists(os.path.join(round_output_dir, f"prompt_{prompt_idx:04d}.json")):
                skip_flag = True  # if already generated, check whether all rounds exist or not
                for rd in range(2, search_rounds + 1):
                    if not os.path.exists(os.path.join(output_dir, f"round_{rd}", f"prompt_{prompt_idx:04d}.json")):
                        skip_flag = False  # some later rounds are missing; need to re-generate
                        break

                if skip_flag:
                    print(f"Skipping prompt {prompt_idx} as it is already generated.")
                    break  # skip this prompt
                else:
                    print(f"Warning: prompt {prompt_idx} is generated partially. Re-generating it.")
                    # gather all the partial results and delete them
                    partial_results = glob(os.path.join(output_dir, f"round_*", f"prompt_{prompt_idx:04d}*"))  # both json and png
                    for pr in partial_results:
                        os.remove(pr)

            os.makedirs(round_output_dir, exist_ok=True)

            propose_start_time = datetime.now(UTC)
            # propose noises and prompts from the search algorithm
            proposal = algo.propose(search_round, prompt, prompt_idx)
            prompts = proposal["prompts"]

            sample_start_time = propose_end_time = datetime.now(UTC)
            # --- Sampling, verifying, and saving artifacts ---
            datapoints = sample(
                prompts=prompts,
                prompt_idx=prompt_idx,
                search_round=search_round,
                pipe=pipe,
                verifier=verifier,
                root_dir=round_output_dir,
                config=config,
                original_prompt=prompt,
                sync_seed=fixed_noise,
            )
            sample_end_time = update_start_time = datetime.now(UTC)
            # update search algorithm internal state with round output
            algo.update(search_round, datapoints, cur_dir=round_output_dir, prompt_idx=prompt_idx)
            update_end_time = datetime.now(UTC)

            # === Print elapsed time for each step ===
            propose_elapsed_time = propose_end_time - propose_start_time
            sample_elapsed_time = sample_end_time - sample_start_time
            update_elapsed_time = update_end_time - update_start_time
            propose_time.append(propose_elapsed_time.total_seconds())
            update_time.append(update_elapsed_time.total_seconds())

            # separately handle compile time, do not add it to sample time
            if config.compile and idx == 0 and search_round == 1:  # 2 since
                compile_time = sample_elapsed_time
                print(f"First-Loop sample time including compile: {compile_time.total_seconds()} sec.")
            else:
                sample_time.append(sample_elapsed_time.total_seconds())

    # === Timeit ===
    end_datetime = datetime.now(UTC)
    elapsed_time = end_datetime - start_datetime
    print(f"Done! Elapsed time: {elapsed_time}")
    print(f"Propose time (sec. per round): {np.mean(propose_time)} ± {np.std(propose_time)}")
    print(f"Sample time (sec. per round): {np.mean(sample_time)} ± {np.std(sample_time)}")
    print(f"Update time (sec. per round): {np.mean(update_time)} ± {np.std(update_time)}")
    print(end_datetime.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S'))
        print(e)
        import ipdb; ipdb.post_mortem()
