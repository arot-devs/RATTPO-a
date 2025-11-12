from __future__ import annotations
import os
from pathlib import Path
import shutil
from typing import Optional, Dict, Any, Union
import json
from datetime import datetime

try:
    from gradio_client import Client
except Exception as e:  # pragma: no cover
    Client = None  # type: ignore


DEFAULTS: Dict[str, Any] = {
    "t2i_model_path_selected": "/data/yada/models/merge_p14_qft_desu/ema_desu-refdbr-150k-r4-cont__hl700_s14400_0.40__ema_desu-refdbr-150k-r5__hl700_s9600_0.60_f36a4de8/model.pt",
    "neg_t2i": "chromatic aberration, sketch, extra digits, unfinished, artistic error, rough, jpeg artifacts, scan, text, username, signature, year_2005, year_2006, lowres, bad quality, worst quality, explicit",
    "quality_preset_t2i": "masterpiece, polish, best quality",
    "neg_i2i": "chromatic aberration, sketch, extra digits, unfinished, artistic error, rough, jpeg artifacts, scan, text, username, signature, year_2005, year_2006, lowres, bad quality, worst quality, explicit",
    "seed": 114514,
    "width": 832,
    "height": 1216,
    "t2i_steps": 28,
    "t2i_guidance": 5,
    "t2i_guidance_schedule_text": "5@0,7@0.75,8@0.95, 10@1",
    "strength": 0.65,
    "i2i_scale": 1.5,
    "i2i_model_path": "/data/yada/checkpoints/dit02_qft/ema/ema_desu-refdbr-150k-r4-cont__hl700_s14400.safetensors",
    "quality_preset_i2i": "masterpiece, polish, best quality",
    "i2i_steps": 28,
    "i2i_guidance": 7.5,
    "i2i_guidance_schedule_text": "5@0,7@0.75,8@0.95, 10@1",
    "first_stage_only": True,
    "latent_upscale": False,
    "t2i_cfg_enable": False,
    "t2i_cfg_rescale": 0.6,
    "i2i_cfg_enable": False,
    "i2i_cfg_rescale": 0.6,
    "use_token_weighting": True,
    "clamp": False,
    "api_name": "/infer_chain",
}


def sanitize(text: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in text)


def create_client(api_url: Optional[str] = None):
    """Create a Gradio Client given an API URL or env var RATTPO_API_URL."""
    if Client is None:
        raise ImportError("gradio_client is required. Install with: pip install gradio_client")
    url = api_url or os.environ.get("RATTPO_API_URL", "http://127.0.0.1:7860")
    return Client(url)


def generate_image_via_api(
    client,
    *,
    prompt: str,
    seed: int,
    width: int = DEFAULTS["width"],
    height: int = DEFAULTS["height"],
    output_dir: str = "output_api",
    image_id: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    # Logging options (request payload only; LLM chat is logged via save_optimizer_chat)
    log_request: bool = True,
    request_log_name: str = "request_log.jsonl",
) -> str:
    """Call the T2I chain API to generate a single image and return a local path.

    If `image_id` is provided, copy the tmp file returned by gradio_client
    (which contains rich PNG metadata) into `output_dir` preserving metadata,
    and return the copied path.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = {**DEFAULTS}
    if overrides:
        cfg.update(overrides)

    # Build payload (what we feed into the model)
    payload = dict(
        t2i_model_path_selected=cfg["t2i_model_path_selected"],
        prompt=prompt,
        neg_t2i=cfg["neg_t2i"],
        quality_preset_t2i=cfg["quality_preset_t2i"],
        neg_i2i=cfg["neg_i2i"],
        seed=int(seed),
        width=int(width),
        height=int(height),
        t2i_steps=cfg["t2i_steps"],
        t2i_guidance=cfg["t2i_guidance"],
        t2i_guidance_schedule_text=cfg["t2i_guidance_schedule_text"],
        strength=cfg["strength"],
        i2i_scale=cfg["i2i_scale"],
        i2i_model_path=cfg["i2i_model_path"],
        quality_preset_i2i=cfg["quality_preset_i2i"],
        i2i_steps=cfg["i2i_steps"],
        i2i_guidance=cfg["i2i_guidance"],
        i2i_guidance_schedule_text=cfg["i2i_guidance_schedule_text"],
        first_stage_only=cfg["first_stage_only"],
        latent_upscale=cfg["latent_upscale"],
        t2i_cfg_enable=cfg["t2i_cfg_enable"],
        t2i_cfg_rescale=cfg["t2i_cfg_rescale"],
        i2i_cfg_enable=cfg["i2i_cfg_enable"],
        i2i_cfg_rescale=cfg["i2i_cfg_rescale"],
        use_token_weighting=cfg["use_token_weighting"],
        clamp=cfg["clamp"],
        api_name=cfg["api_name"],
    )

    res = client.predict(
        **payload,
    )

    img_path = res[0]
    copied_to: Optional[str] = None
    # Copy into output_dir using a stable name if requested (preserve metadata)
    if image_id:
        try:
            os.makedirs(output_dir, exist_ok=True)
            name = image_id
            target = str(Path(output_dir) / name)
            shutil.copy2(img_path, target)
            copied_to = target
        except Exception:
            # Fall back to returning the original path
            copied_to = None

    # Record what was fed into the model (per-round = per output_dir)
    if log_request:
        try:
            rec = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "request": payload,
                "overrides": overrides or {},
                "image_id": image_id,
                "result": {
                    "tmp_img_path": img_path,
                    "copied_to": copied_to,
                },
            }
            log_path = str(Path(output_dir) / request_log_name)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # do not break generation if logging fails
            pass

    # Return copied path if available; else the original tmp path
    return copied_to or img_path


# Helper to save the LLM optimization chat (what is fed to/returned from the optimizer LLM)
def save_optimizer_chat(
    output_dir: str,
    search_round: int,
    *,
    request_messages: Union[str, Dict[str, Any], list],
    response: Optional[Union[str, Dict[str, Any], list]] = None,
    label: str = "optimizer",
    jsonl_name: Optional[str] = None,
    txt_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.utcnow().isoformat() + "Z"
    # JSONL (machine‑readable)
    jsonl = jsonl_name or f"{label}_chat.jsonl"
    jsonl_path = str(Path(output_dir) / jsonl)
    rec = {
        "ts": ts,
        "round": int(search_round),
        "label": label,
        "request": request_messages,
        "response": response,
    }
    if extra is not None:
        rec["extra"] = extra
    try:
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # TXT (human‑readable per‑round transcript)
    txt = txt_name or f"{label}_chat_round_{int(search_round)}.txt"
    txt_path = str(Path(output_dir) / txt)
    try:
        def _stringify(obj: Union[str, Dict[str, Any], list]) -> str:
            if isinstance(obj, str):
                return obj
            try:
                return json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                return str(obj)

        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write(f"==== {label.upper()} CHAT | round={int(search_round)} | {ts} ====\n")
            f.write("-- REQUEST --\n")
            f.write(_stringify(request_messages))
            if response is not None:
                f.write("\n-- RESPONSE --\n")
                f.write(_stringify(response))
            if extra is not None:
                f.write("\n-- EXTRA --\n")
                f.write(_stringify(extra))
            f.write("\n\n")
    except Exception:
        pass

    return {"jsonl": jsonl_path, "txt": txt_path}
