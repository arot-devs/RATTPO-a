import os
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxPipeline
import re
import hashlib
from typing import Dict
import json
from typing import Union
from PIL import Image
import requests
import argparse
import io
import numpy as np
import threading
from torch.utils.data import Dataset
from glob import glob
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, PNDMScheduler
from omegaconf import OmegaConf
from functools import partial
from das.diffusers_patch.pipeline_using_SMC import pipeline_using_smc
from das.diffusers_patch.pipeline_using_SMC_SDXL import pipeline_using_smc_sdxl
from das.diffusers_patch.pipeline_using_SMC_LCM import pipeline_using_smc_lcm


TORCH_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
MODEL_NAME_MAP = {
    "black-forest-labs/FLUX.1-dev": "flux.1-dev",
    # "HighCWu/FLUX.1-dev-4bit": "flux.1-dev-4bit",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "pixart-sigma-1024-ms",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl-base",
    "CompVis/stable-diffusion-v1-4": "sd-v1.4",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "sd-v1.5",
    "stabilityai/stable-diffusion-2-1-base": "sd-v2.1",
    "stabilityai/stable-diffusion-3.5-large": "sd-v3.5",
    "stabilityai/sdxl-turbo": "sdxl-turbo",
    "segmind/SSD-1B": "ssd-1b",
    "black-forest-labs/FLUX.1-schnell": "flux.1-schnell",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "pixart-sigma",

}
MANDATORY_CONFIG_KEYS = [
    "pretrained_model_name_or_path",
    "torch_dtype",
    "pipeline_call_args",
    "verifier_args",
    "search_args",
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_cli_args():
    """
    Parse and return CLI arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_name",
        default=None,
        # required=True, # handle later
        help="Experiment Name.",
    )

    parser.add_argument(
        "--sync_all_seed",
        default=True,
        type=str2bool,
        help="Whether or not to sync initial noises"
    )

    parser.add_argument(
        "--config",
        type=str,
        nargs='+',
        default=["configs/diffusion_configs/sd1.4.json",
                 "configs/search_configs/rattpo.json",
                 "configs/verifier_configs/prompt_adaptation/lexica_sd1.4_seed42.json"
                 ],
    )
    parser.add_argument("--prompt", type=str, default=None, help="Use your own prompt.")

    parser.add_argument(
        "--seed",
        default=42,
        help="Random seed to use",
    )

    parser.add_argument(
        "--compile",
        default=True,
        action='store_true',
        help="If set, do compile",
    )

    parser.add_argument(
        "--num_prompts",
        type=lambda x: None if x.lower() == "none" else x if x.lower() == "all" else int(x),
        default='all',
        help="Number of prompts to use (or 'all' to use all prompts from file).",
    )

    parser.add_argument(
        "--dataset",
        choices=['lexica', 'diffusiondb', 'parti', 'compbench_2d', 'compbench_3d', 'compbench_numeracy'],
        # default='lexica',
        required=True,
        help="Prompt dataset to use for search.",
    )

    parser.add_argument(
        "--prompt_div",
        default=1,
        type=int,
        help="For parallel sampling of prompts. Will execute if prompt_index % prompt_div == prompt_mod.",
    )
    parser.add_argument(
        "--prompt_mod",
        default=0,
        type=int,
        help="For parallel sampling of prompts. Will execute if prompt_index % prompt_div == prompt_mod.",
    )

    cur_gpu = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    _default_port = 11434 + cur_gpu % 2  # assume 11434 or 11435
    parser.add_argument(
        "--ollama_port",
        default=_default_port,
        type=int,
        help="port number for Ollama API Endpoint.",
    )

    args = parser.parse_args()

    if args.exp_name is None and not args.final:
        raise ValueError("Please provide an experiment name.")

    if args.prompt:
        args.dataset = "custom"


    # load config
    config = load_config(args.config)
    config.update(vars(args))

    validate_config(config)

    return config


def load_config(config_paths: list[str]):
    """
    Load configs with omegaconf, return the configuration from the given JSON file.
    Later file will overwrite the previous one if there are any conflicts.
    """
    config = OmegaConf.create()
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config.update(json.load(f))
        else:
            raise FileNotFoundError(f"Config file {config_path} not found.")
    return config


def validate_config(config):
    if config.prompt and config.num_prompts:
        raise ValueError("Both `prompt` and `num_prompts` cannot be specified.")
    if not config.prompt and not config.num_prompts:
        raise ValueError("Both `prompt` and `num_prompts` cannot be None.")

    # with open(args.pipeline_config_path, "r") as f:
    #     config = json.load(f)

    config_keys = list(config.keys())
    assert all(element in config_keys for element in MANDATORY_CONFIG_KEYS), (
        f"Expected the following keys to be present: {MANDATORY_CONFIG_KEYS} but got: {config_keys}."
    )

    _validate_verifier_args(config)
    _validate_search_args(config)

    assert 0 <= config.prompt_mod < config.prompt_div, f"0 <= {config.prompt_mod} < {config.prompt_div}"


def _validate_verifier_args(config):
    return

def _validate_search_args(config):
    return


# Adapted from Diffusers.
def prepare_latents_for_flux(
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_latent_channels = 16
    vae_scale_factor = 8

    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    shape = (batch_size, num_latent_channels, height, width)
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)  # (B, 16, 128, 128)
    latents = FluxPipeline._pack_latents(latents, batch_size, num_latent_channels, height, width)  # (B, 4096, 64)
    return latents


# Adapted from Diffusers.
def prepare_latents(
    batch_size: int, height: int, width: int, generator: torch.Generator, device: str, dtype: torch.dtype
):
    if height == 1024 and width == 1024:  # only for sd-v3.5
        num_channels_latents = 16
    else:
        num_channels_latents = 4
    vae_scale_factor = 8
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)
    return latents


def prepare_latents_for_pixart(
    batch_size: int, height: int, width: int, generator: torch.Generator, device: str, dtype: torch.dtype
):
    num_channels_latents = 4
    vae_scale_factor = 8
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)
    return latents


def get_latent_prep_fn(pretrained_model_name_or_path: str) -> callable:
    fn_map = {
        "CompVis/stable-diffusion-v1-4": prepare_latents,
        "stable-diffusion-v1-5/stable-diffusion-v1-5": prepare_latents,
        "stabilityai/stable-diffusion-2-1-base": prepare_latents,
        "stabilityai/sdxl-turbo": prepare_latents,
        "black-forest-labs/FLUX.1-schnell": prepare_latents_for_flux,
    }[pretrained_model_name_or_path]
    return fn_map


def load_verifier_prompt(path: str) -> str:
    with open(path, "r") as f:
        verifier_prompt = f.read().replace('"""', "")

    return verifier_prompt


def load_image(path_or_url: Union[str, Image.Image]) -> Image.Image:
    """
    Load an image from a local path or a URL and return a PIL Image object.

    `path_or_url` is returned as is if it's an `Image` already.
    """
    if isinstance(path_or_url, Image.Image):
        return path_or_url
    elif path_or_url.startswith("http"):
        response = requests.get(path_or_url, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    return Image.open(path_or_url)


def base_post_args(port=11434, model_name='gemma3:27b') -> Dict[str, str]:
    '''
    return a dict of url, json, headers for a POST request to the Ollama API
    '''
    headers = {'Content-Type': 'application/json'}
    host = os.environ.get('OLLAMA_URL', None)
    # port = 11434 + int(os.environ.get('CUDA_VISIBLE_DEVICES', 0))
    url = f"http://{host}/api/generate"
    data = {
        "model": model_name,
        "stream": False,
    }

    return {"url": url, "json": data, "headers": headers}


def preload_ollama(port=11434, model_name='gemma3:27b'):
    '''
    Fire-and-Forget request for preloading
    '''
    base_payload = base_post_args(port, model_name)
    # Create and start a daemon thread that will perform the HTTP request

    thread = threading.Thread(target=requests.post, kwargs=base_payload)
    thread.daemon = True  # Daemon threads will automatically shut down when the main program exits
    thread.start()


def find_json(root_dir):
    '''
    Find all experiment result (.json) files under experiment root directory.
    '''

    # sort with prompt number
    json_files = glob(f"{root_dir}/round*/*.json", recursive=True)

    # to eliminate _detail.json file
    pattern = re.compile(r'.*[^a-zA-Z](\d+)\.json$|^(\d+)\.json$')

    json_files = [f for f in json_files if pattern.search(os.path.basename(f))]

    json_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
    print(f"Found {len(json_files)} json files from {root_dir}")

    # first divide with respect to prompt
    json_per_prompt = {}
    for each in json_files:
        prompt = each.split("/")[-1].split(".")[0]
        if prompt not in json_per_prompt:
            json_per_prompt[prompt] = []
        json_per_prompt[prompt].append(each)

    # next, sort w.r.t to round
    for prompt in json_per_prompt:
        json_per_prompt[prompt].sort(key=lambda x: int(x.split("/")[-2].split("_")[-1]))

    return json_per_prompt


def load_if_needed(config: Union[str, dict, OmegaConf]):
    """
    Load the configuration if it is a string, otherwise return the config as is.
    """
    if isinstance(config, str):
        return OmegaConf.load(config)
    elif isinstance(config, dict):
        return OmegaConf.create(config)
    return config


def init_diffusion(config: Union[str, OmegaConf]):
    """
    Initialize the diffusion pipeline with the given configuration.
    """
    config = load_if_needed(config)
    das_mode = config.get("das", False)

    if das_mode:
        model_name = config["pretrained_model_name_or_path"]
        dtype = TORCH_DTYPE_MAP[config["torch_dtype"]]

        if "xl" in model_name.lower():
            base_pipe = AutoPipelineForText2Image.from_pretrained(
                model_name, torch_dtype=dtype
            )
            smc_fn = pipeline_using_smc_sdxl
        elif "lcm" in model_name.lower():
            base_pipe = StableDiffusionPipeline.from_pretrained(
                model_name, torch_dtype=dtype
            )
            smc_fn = pipeline_using_smc_lcm
        else:                                   # SD 1.x, 2.x
            from diffusers import StableDiffusionPipeline
            base_pipe = StableDiffusionPipeline.from_pretrained(
                model_name, torch_dtype=dtype,
                revision='main'
            )
            smc_fn = pipeline_using_smc
        from verifiers.prompt_adaptation import PromptAdaptationVerifier
        base_pipe.verifier = PromptAdaptationVerifier(config.verifier_args)
        from diffusers import DDIMScheduler
        base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
        base_pipe.scheduler.set_timesteps(config.pipeline_call_args.num_inference_steps)
        base_pipe.safety_checker = None
        base_pipe = base_pipe.to("cuda:0")
        base_pipe.set_progress_bar_config(disable=True)

        smc_defaults = dict(
            num_inference_steps=config.pipeline_call_args.get("num_inference_steps", 20),
            guidance_scale=config.pipeline_call_args.get("guidance_scale", 7.5),
            eta=config.pipeline_call_args.get("eta", 0.0),
            num_particles=config.pipeline_call_args.get("num_particles", 16),
            resample_strategy=config.pipeline_call_args.get("resample_strategy", "multinomial"),
            ess_threshold=config.pipeline_call_args.get("ess_threshold", 0.5),
            tempering=config.pipeline_call_args.get("tempering", False),
            kl_coeff=config.pipeline_call_args.get("kl_coeff", 1.0),
            verbose=False,
            reward_fn=lambda veri_input, ret_type, prompt_idx: base_pipe.verifier.score(veri_input, ret_type, prompt_idx=prompt_idx)
            # reward_fn = lambda
            # [torch.mean(img).to(img.device) for img in imgs]
        )
        smc_defaults.update(config.get("pipeline_call_args", {}))

        if config.get("compile", False):  # do copile
            print("Will compile the model.")
            base_pipe.unet = torch.compile(base_pipe.unet, mode='reduce-overhead', fullgraph=True)
        pipe = partial(smc_fn, base_pipe, **smc_defaults)

        return pipe

    # Load the model name and dtype from the configuration
    model_name = config["pretrained_model_name_or_path"]
    dtype = TORCH_DTYPE_MAP[config["torch_dtype"]]
    pipe = AutoPipelineForText2Image.from_pretrained(model_name,
                                                     torch_dtype=dtype)

    # options for sampler
    sampler = config.get("sampler", 'dpm_solver')
    pipeline_call_args = config.pipeline_call_args

    if sampler == "dpm_solver" and model_name != "stabilityai/sdxl-turbo":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif sampler == 'pndm_solver':

        assert model_name == "stabilityai/sdxl-turbo", f"PNDMSolverMultistepScheduler is only supported for SDXL, but got {model_name}."
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

    elif sampler is None:
        # No specific scheduler is set, use the default one
        pass
    else:
        raise ValueError(f"Unknown sampler: {sampler}. Supported ones are: dpm_solver, None.")

    pipe.safety_checker = None
    pipe = pipe.to('cuda:0')
    pipe.set_progress_bar_config(disable=True)

    if config.get("compile", False) and 'flux' not in model_name.lower():  # do copile
        print("Will compile the model.")
        pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead', fullgraph=True)

    # make inference step and guidance scale configurable
    pipe = partial(pipe, **pipeline_call_args)  # still can access original pipe with pipe.func

    return pipe


def get_noises(
    noise_seed: int,
    num_images_per_prompt: int,
    height: int,
    width: int,
    device="cuda",
    dtype: torch.dtype = torch.float16,
    fn: callable = prepare_latents,
    verbose=True,
) -> Dict[int, torch.Tensor]:

    latents = fn(
        batch_size=num_images_per_prompt,
        height=height,
        width=width,
        generator=torch.manual_seed(int(noise_seed)),
        device=device,
        dtype=dtype,
    )

    assert len(latents) == num_images_per_prompt, latents.shape

    if verbose:
        if latents.ndim == 4:
            print(f"Generated initial noise with shape {latents.shape}, seed {noise_seed}.",
                #   f"debug: {latents[0].mean().item():.5f}, {latents[0, 0, 0, 0].item():.5f}"   # -0.00946, 0.38696
                  )
        elif latents.ndim == 3:
            assert height == 1024 and width == 1024, f"latents shape {latents.shape} is not supported."
            print(f"Generated initial noise with shape {latents.shape}, seed {noise_seed}.",
                  f"debug: {latents[0].mean().item():.5f}, {latents[0, 0, 0].item():.5f}")
        else:
            raise ValueError(f"latents shape {latents.shape} is not supported.")

    return latents


def pipe_image_process(image, cut_out_size=224):  # Same as clip preprocess
    from torchvision import transforms
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    # image = image.permute(0, 3, 1, 2)  # (B, C, H, W) -> (B, C, H, W)
    image = transforms.Resize(cut_out_size)(image)
    image = normalize(image).to(image.dtype)
    return image

def visualize_images(images, scores, title=None):
    assert len(images) == len(scores), "images, scores and prompts should have the same length"
    import matplotlib.pyplot as plt
    from PIL import Image
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    for i in range(num_images):
        axs[i].imshow(images[i])
        axs[i].set_title(f"Reward: {scores[i]:.2f}", y=-0.1)
        axs[i].axis('off')

    if title:
        plt.suptitle(title, fontsize=16, wrap=True)
    plt.tight_layout()
    plt.show()
