#!/usr/bin/env python
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation script for a `train_lora_jit.py` LoRA checkpoint.

Generates `--num_images` samples with the fine-tuned transformer, then scores them with:
  - the LAION "improved aesthetic predictor" (CLIP ViT-L/14 embedding -> linear MLP regressor), and
  - HPSv2 (Human Preference Score v2).

HPSv2 is a text-image preference score (trained to compare images generated from the same prompt),
but this LoRA is class-conditional, not text-conditional -- there is no natural prompt for the images
it generates. `--hps_prompt` is a generic stand-in (default: "a photo of a face"); treat the HPSv2
number here as a rough general-quality proxy, not a prompt-faithfulness score in the usual HPS sense.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm


try:
    import diffusers  # noqa: F401
except ImportError:
    # `JiTTransformer2DModel`/`JiTPipeline` only exist in this repo checkout, not on PyPI yet.
    _repo_src = Path(__file__).resolve().parents[3] / "src"
    if _repo_src.is_dir():
        sys.path.insert(0, str(_repo_src))

from diffusers import FlowMatchEulerDiscreteScheduler, JiTPipeline, JiTTransformer2DModel


AESTHETIC_PREDICTOR_REPO = "Geonmo/laion-aesthetic-predictor"
AESTHETIC_PREDICTOR_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_CLIP_MODEL = "openai/clip-vit-large-patch14"


class AestheticMLP(nn.Module):
    """CLIP ViT-L/14 image embedding -> scalar aesthetic score.

    Architecture matches the `sac+logos+ava1-l14-linearMSE` checkpoint from
    https://github.com/christophschuhmann/improved-aesthetic-predictor exactly (including the
    `layers.*` state dict key names), so the pretrained weights load directly with no remapping.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a JiT LoRA checkpoint: generate + score images.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="JiT-diffusers/JiT-B-16",
        help="Path to the base JiT variant folder the LoRA was fine-tuned from.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--class_label",
        type=int,
        default=0,
        help="Class label to condition generation on. Defaults to the model's null class "
        "(`transformer.config.num_classes`) -- match whatever class the LoRA was actually trained with.",
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument(
        "--hps_prompt",
        type=str,
        default="a photo of a face",
        help="Stand-in text prompt for HPSv2 scoring (see module docstring: this LoRA isn't text-conditional).",
    )
    return parser.parse_args()


def resolve_transformer_path(path):
    """Accept a run dir, a `checkpoint-N` dir, or a `transformer/` folder; return the folder holding `config.json`."""
    path = Path(path)
    if (path / "config.json").is_file():
        return path
    if (path / "transformer" / "config.json").is_file():
        return path / "transformer"
    raise FileNotFoundError(f"No transformer `config.json` found in {path} or {path / 'transformer'}")


def generate_images(args, device):
    if args.transformer_path is not None:
        transformer = JiTTransformer2DModel.from_pretrained(resolve_transformer_path(args.transformer_path))
    else:
        transformer = JiTTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision
        )
    if args.lora_path is not None:
        transformer.load_lora_adapter(args.lora_path, prefix=None, weight_name="pytorch_lora_weights.safetensors")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=noise_scheduler.config.shift)
    pipeline = JiTPipeline(transformer=transformer, scheduler=scheduler)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)

    class_label = args.class_label if args.class_label is not None else transformer.config.num_classes

    images_dir = Path(args.output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    image_paths = []
    with tqdm(total=args.num_images, desc="Generating") as progress_bar:
        while len(image_paths) < args.num_images:
            batch = min(args.batch_size, args.num_images - len(image_paths))
            images = pipeline(
                class_labels=[class_label] * batch,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                output_type="pil",
            ).images
            for image in images:
                path = images_dir / f"{len(image_paths):04d}.png"
                image.save(path)
                image_paths.append(path)
            progress_bar.update(len(images))

    del pipeline
    return image_paths


def compute_aesthetic_scores(image_paths, device):
    from huggingface_hub import hf_hub_download
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(AESTHETIC_CLIP_MODEL).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(AESTHETIC_CLIP_MODEL)

    mlp = AestheticMLP().to(device).eval()
    checkpoint_path = hf_hub_download(
        repo_id=AESTHETIC_PREDICTOR_REPO, filename=AESTHETIC_PREDICTOR_FILENAME, repo_type="space"
    )
    mlp.load_state_dict(torch.load(checkpoint_path, map_location=device))

    scores = []
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Aesthetic score"):
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            image_features = clip_model.get_image_features(**inputs).pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores.append(mlp(image_features.float()).item())
    return scores


def compute_hpsv2_scores(image_paths, prompt):
    import hpsv2

    scores = hpsv2.score([str(path) for path in image_paths], prompt, hps_version="v2.1")
    return [float(score) for score in scores]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = generate_images(args, device)
    aesthetic_scores = compute_aesthetic_scores(image_paths, device)
    hps_scores = compute_hpsv2_scores(image_paths, args.hps_prompt)

    aesthetic_mean = sum(aesthetic_scores) / len(aesthetic_scores)
    aesthetic_std = (sum((s - aesthetic_mean) ** 2 for s in aesthetic_scores) / len(aesthetic_scores)) ** 0.5
    hps_mean = sum(hps_scores) / len(hps_scores)
    hps_std = (sum((s - hps_mean) ** 2 for s in hps_scores) / len(hps_scores)) ** 0.5

    summary = {
        "transformer_path": args.transformer_path,
        "lora_path": args.lora_path,
        "num_images": len(image_paths),
        "aesthetic_score_mean": aesthetic_mean,
        "aesthetic_score_std": aesthetic_std,
        "hpsv2_mean": hps_mean,
        "hpsv2_std": hps_std,
        "hps_prompt": args.hps_prompt,
        "per_image": [
            {"path": str(path), "aesthetic_score": a, "hpsv2": h}
            for path, a, h in zip(image_paths, aesthetic_scores, hps_scores)
        ],
    }
    summary_path = Path(args.output_dir) / "scores.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nGenerated {len(image_paths)} images from {args.lora_path or args.transformer_path or args.pretrained_model_name_or_path}")
    print(f"LAION aesthetic score: {aesthetic_mean:.4f} +/- {aesthetic_std:.4f}")
    print(f"HPSv2 ({args.hps_prompt!r}):  {hps_mean:.4f} +/- {hps_std:.4f}")
    print(f"Full results written to {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
