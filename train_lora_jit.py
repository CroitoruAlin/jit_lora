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
"""LoRA fine-tuning for `JiTTransformer2DModel`.

JiT operates directly on image pixels (no VAE) and is trained with flow matching: the model is fed
a noisy image `z_t` and the timestep `t`, and predicts a clean target image directly. Unlike the JiT
paper ("Back to Basics: Let Denoising Generative Models Denoise", https://arxiv.org/abs/2511.13720),
which samples `t` from a logit-normal distribution biased toward noisier timesteps, `t` is sampled
uniformly from (0, 1) here: the paper's bias was tuned for a plain x0-MSE loss, where the near-noise
task is the hard, undertrained one. With the frequency-masked target below, the near-noise task is
made easier on its own (a much simpler target), so biasing `t` sampling the same way would starve
higher-`t`, higher-detail steps of training signal -- concretely, combining the paper's
logit-normal(-0.8) with the frequency-mask schedule left >99% of training steps with a target that
never reaches even half of the full spectrum.

The target is not plain `x0` but a frequency-masked version of it: its 2D FFT is radially cropped to
a cutoff that grows with `t`, so at high noise levels (small `t`) the model is only asked to recover
low-frequency structure, and at low noise levels (`t` close to 1) the target includes the full
spectrum. The noised input is built from this same masked image (`z_t = t * target + (1 - t) *
noise`), not the true `x0`: at inference there is no true `x0` for high-frequency content to leak in
from, so noising with the true image would train on inputs inconsistent with what the model actually
sees when sampling -- building both `z_t` and the target from the same masked image keeps the
corruption process self-consistent, as in blurring/soft-diffusion formulations, rather than asking
the model to discard real detail that's diluted into its own input. The loss is weighted by
`1 / (1 - t)^2`, the JiT paper's implicit loss weighting (which the paper derives from a
velocity-space reformulation of the x0-prediction loss; applied here directly as an explicit weight
since the target is no longer plain `x0`).

Since JiT is class-conditional (ImageNet, 1000 classes) and most fine-tuning datasets are not,
every example is conditioned on the model's built-in null class (index `num_classes`) - the same
"unconditional" token used for classifier-free guidance at pretraining time.
"""

import argparse
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, set_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm


try:
    import diffusers  # noqa: F401
except ImportError:
    # `JiTTransformer2DModel`/`JiTPipeline` only exist in this repo checkout, not on PyPI yet.
    # Fall back to the local `src/` if `diffusers` isn't installed (e.g. no editable install available).
    _repo_src = Path(__file__).resolve().parents[3] / "src"
    if _repo_src.is_dir():
        sys.path.insert(0, str(_repo_src))

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
    JiTPipeline,
    JiTTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import check_min_version, is_wandb_available


check_min_version("0.36.0.dev0")

logger = get_logger(__name__)


def radial_frequency_grid(size, device):
    """Normalized radial distance (in [0, 1]) of each 2D FFT bin from the DC component, after fftshift."""
    coords = torch.arange(size, device=device) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radius = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    return (radius / radius.max()).view(1, 1, size, size)


def frequency_mask(images, radius_grid, cutoff_fraction, transition_width):
    """Attenuate FFT bins beyond a per-sample radial cutoff, then invert.

    The rolloff is a smooth sigmoid rather than a hard step: a brick-wall cutoff creates a sharp
    discontinuity in the spectrum, which shows up as ringing (Gibbs phenomenon) in the target image
    that the model then learns to reproduce. The mask is renormalized so it is always exactly 1 at
    the DC bin (radius 0) regardless of how small `cutoff_fraction` is relative to
    `transition_width` -- otherwise the mean color/brightness itself gets attenuated at low cutoffs,
    which looks like the image washing out toward gray rather than blurring.

    `transition_width` is capped per-sample at `cutoff_fraction` (with a floor at 20% of its passed-in
    value): a transition much wider than the cutoff itself lets meaningful low/mid-frequency content
    leak through regardless of how small the cutoff is, making tiny cutoffs far less aggressive than
    intended; capping keeps the transition proportional to the passband instead.

    `radius_grid` is normalized so its maximum (the corner frequency) is exactly 1.0, i.e. the same
    upper bound as `cutoff_fraction` after clamping. The sigmoid's midpoint sits at `r = cutoff`, so
    right at that shared boundary (`cutoff_fraction == 1.0`, `r == 1.0`) it evaluates to exactly 0.5,
    not 1.0 -- there's no "outside the passband" left to fade into once the cutoff already covers the
    whole spectrum. Left alone, the "full detail" case is never actually full detail: even the
    corner frequency is permanently attenuated by half, and everything from roughly r=0.7 up is
    softened too. That silently caps how sharp the model can ever learn to be, at every `t`. Samples
    at the saturation point are therefore passed through unmodified instead of through the sigmoid.
    """
    cutoff_fraction = cutoff_fraction.view(-1, 1, 1, 1)
    transition_width = cutoff_fraction.clamp(min=0.2 * transition_width, max=transition_width)
    freqs = torch.fft.fftshift(torch.fft.fft2(images.float()), dim=(-2, -1))
    mask = torch.sigmoid((cutoff_fraction - radius_grid) / transition_width)
    mask = mask / torch.sigmoid(cutoff_fraction / transition_width)
    mask = torch.where(cutoff_fraction >= 1.0, torch.ones_like(mask), mask).to(freqs.dtype)
    freqs = torch.fft.ifftshift(freqs * mask, dim=(-2, -1))
    return torch.fft.ifft2(freqs).real.to(images.dtype)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning script for JiTTransformer2DModel.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/colligo/jit/JiT-diffusers/JiT-L-16",
        help="Path to a JiT variant folder (must contain a `transformer` and a `scheduler` subfolder).",
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="Revision of the pretrained model to use (branch, tag, commit id)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="korexyz/celeba-hq-256x256",
        help="Name of a dataset from the HuggingFace hub, or a path to a local `imagefolder`-compatible directory.",
    )
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image", help="Name of the image column in the dataset.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="jit-lora-celeba-hq")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Training resolution. Defaults to the pretrained transformer's native `sample_size`.",
    )
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=64, help="Rank of the LoRA update matrices.")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (scaling = lora_alpha / rank).")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help="Comma-separated list of module names to apply LoRA to. Defaults to attention and MLP projections.",
    )
    parser.add_argument(
        "--t_eps",
        type=float,
        default=0.05,
        help="Clips `(1 - t)` in the loss weight `1 / (1 - t)^2`, matching the JiT paper's clipping to avoid a "
        "division blowup near t=1.",
    )
    parser.add_argument(
        "--freq_mask_alpha",
        type=float,
        default=2.0,
        help="Assumed natural-image spectral falloff exponent (power spectrum ~ 1/f^alpha; 2.0 is the standard "
        "value for natural images). Shapes the FFT cutoff curve between t=0 and `--freq_mask_saturation_t`, "
        "derived from where per-frequency signal power (~t^2 / f^alpha) equals injected noise power (~(1-t)^2, "
        "flat across frequencies).",
    )
    parser.add_argument(
        "--freq_mask_saturation_t",
        type=float,
        default=0.7,
        help="The t value at which the FFT cutoff first reaches 1.0 (full detail, no masking). This calibrates "
        "the signal-vs-noise power comparison used by `--freq_mask_alpha` -- without it there's an arbitrary "
        "implicit assumption about how much energy real images have at the highest frequency relative to the "
        "injected noise. Lower it to shrink the fraction of (uniformly-sampled) training steps that see a "
        "heavily blurred target -- with uniform t sampling, 0.7 (default) means ~19% of steps stay heavily "
        "blurred (cutoff_fraction < 0.1) and ~31% see near-full detail (cutoff_fraction > 0.95), vs. ~47% / "
        "~10% at 0.9. Raise it back toward 1.0 for a more aggressive, longer low-frequency-only schedule.",
    )
    parser.add_argument(
        "--freq_mask_transition",
        type=float,
        default=0.1,
        help="Maximum width (in normalized radius units) of the smooth sigmoid rolloff around the FFT cutoff. "
        "Larger values give a gentler transition (less ringing, less precise frequency selectivity); smaller "
        "values approach a hard cutoff (more ringing). On a synthetic hard-edge test image, ringing (measured "
        "as overshoot beyond the original pixel range) only becomes negligible around 0.08-0.1. For small "
        "cutoffs (low `t`), the actual transition used is capped at the cutoff itself (floor: 20% of this "
        "value) so the rolloff doesn't dominate and swamp a tiny passband, which would otherwise make small "
        "cutoffs far less aggressive than intended.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=[None, "no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb", choices=["tensorboard", "wandb"])
    parser.add_argument("--all_dir", type=str, default="logs")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a `checkpoint-*` directory saved by `--checkpointing_steps`, or `latest`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Generate and log validation samples every N epochs. Set to 0 to disable.",
    )
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_inference_steps", type=int, default=50)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)

    args = parser.parse_args()
    return args


def main(args):
    logging_dir = Path(args.output_dir, "log")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging: `pip install wandb`.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the pretrained transformer and read the flow-matching shift used for validation sampling
    # (the shift only affects the inference-time schedule spacing, not the training-time loss below).
    transformer = JiTTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision
    )
    noise_scheduler = FlowMatchHeunDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )
    shift = noise_scheduler.config.shift
    resolution = args.resolution or transformer.config.sample_size
    null_class = 0 #transformer.config.num_classes

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    transformer.requires_grad_(False)
    transformer.to(accelerator.device, dtype=weight_dtype)
    radius_grid = radial_frequency_grid(resolution, accelerator.device)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0", "w12", "w3"]

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_config)

    def unwrap_model(model):
        return accelerator.unwrap_model(model)

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                unwrap_model(model).save_lora_adapter(output_dir, weight_name="pytorch_lora_weights.safetensors")
                weights.pop()

    def load_model_hook(models, input_dir):
        from safetensors.torch import load_file

        while len(models) > 0:
            model = models.pop()
            lora_state_dict = load_file(os.path.join(input_dir, "pytorch_lora_weights.safetensors"))
            incompatible_keys = set_peft_model_state_dict(model, lora_state_dict, adapter_name="default")
            if incompatible_keys is not None and getattr(incompatible_keys, "unexpected_keys", None):
                logger.warning(f"Unexpected keys while loading LoRA weights: {incompatible_keys.unexpected_keys}")

        if args.mixed_precision == "fp16":
            cast_training_params([transformer], dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, split="train")

    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess(examples):
        examples["pixel_values"] = [train_transforms(image.convert("RGB")) for image in examples[args.image_column]]
        return examples

    dataset.set_transform(preprocess)

    def collate_fn(examples):
        return {"pixel_values": torch.stack([example["pixel_values"] for example in examples])}

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("jit_lora", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Resolution = {resolution}, flow-matching shift = {shift}, null class = {null_class}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            path = checkpoints[-1] if len(checkpoints) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                bsz = pixel_values.shape[0]

                t = torch.rand(bsz, device=pixel_values.device)
                t_view = t.view(-1, 1, 1, 1).to(pixel_values.dtype)

                # Target: x0 with its FFT radially cropped to a cutoff that grows with t, so the model is
                # only asked to recover low-frequency structure at high noise levels (small t) and the full
                # spectrum as noise drops. The noised input z_t is built from this SAME masked image, not
                # the true x0: at inference there is no true x0 for high-frequency content to leak in from,
                # so training z_t from the true image would teach the model on inputs inconsistent with what
                # it actually sees when sampling. Building both z_t and the target from the same masked image
                # keeps the corruption process self-consistent (as in blurring/soft-diffusion formulations),
                # rather than asking the model to discard real detail that's diluted into its own input.
                snr_ratio = (t / (1.0 - t)) / (
                    args.freq_mask_saturation_t / (1.0 - args.freq_mask_saturation_t)
                )
                cutoff_fraction = (snr_ratio ** (2.0 / args.freq_mask_alpha)).clamp(max=1.0)
                target = frequency_mask(pixel_values, radius_grid, cutoff_fraction, args.freq_mask_transition)

                noise = torch.randn_like(pixel_values)
                z_t = t_view * target + (1.0 - t_view) * noise
                class_labels = torch.full((bsz,), null_class, device=pixel_values.device, dtype=torch.long)

                model_pred = transformer(z_t, timestep=t.to(pixel_values.dtype), class_labels=class_labels).sample

                # Weighted by 1/(1-t)^2, the JiT paper's implicit loss weight (normally obtained "for free"
                # via a velocity-space reformulation of the x0-MSE; applied here explicitly since the target
                # is no longer plain x0).
                weight = 1.0 / (1.0 - t_view).clamp(min=args.t_eps) ** 2
                loss = (weight * (model_pred.float() - target.float()) ** 2).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            (d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")),
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for checkpoint in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, checkpoint))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved LoRA checkpoint to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        if (
            accelerator.is_main_process
            and args.validation_epochs > 0
            and (epoch + 1) % args.validation_epochs == 0
        ):
            logger.info("Running validation... generating sample images.")
            pipeline = JiTPipeline(
                transformer=unwrap_model(transformer),
                scheduler=FlowMatchEulerDiscreteScheduler(shift=shift),
            )
            pipeline.set_progress_bar_config(disable=True)
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            images = pipeline(
                class_labels=[null_class] * args.num_validation_images,
                guidance_scale=3.0,
                num_inference_steps=args.validation_inference_steps,
                generator=generator,
                output_type="np",
            ).images
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    tracker.writer.add_images("validation", images, epoch, dataformats="NHWC")
                elif tracker.name == "wandb":
                    import wandb

                    tracker.log({"validation": [wandb.Image(image) for image in images]})
            del pipeline
            free_memory()

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrap_model(transformer).save_lora_adapter(
            args.output_dir, weight_name="pytorch_lora_weights.safetensors"
        )
        logger.info(f"Saved final LoRA weights to {args.output_dir}")

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of LoRA fine-tuning",
                ignore_patterns=["checkpoint-*", "logs"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
