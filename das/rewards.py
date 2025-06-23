from PIL import Image
import os
from pathlib import Path
import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from transformers import pipeline
from diffusers.utils import load_image
from importlib import resources
ASSETS_PATH = resources.files("assets")

def jpeg_compressibility(inference_dtype=None, device=None):
    import io
    import numpy as np
    def loss_fn(images):
        if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        loss = torch.tensor(sizes, dtype=inference_dtype, device=device)
        rewards = -1 * loss

        return loss, rewards

    return loss_fn
