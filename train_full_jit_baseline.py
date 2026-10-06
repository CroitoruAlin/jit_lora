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
"""Full-model (no LoRA) fine-tuning for `JiTTransformer2DModel`, following the exact training recipe
of the JiT paper ("Back to Basics: Let Denoising Generative Models Denoise",
https://arxiv.org/abs/2511.13720). This is the BASELINE script: it contains no frequency-domain loss
weighting and no other additions beyond what the paper describes -- it exists to be compared against
`train_jit_freqweighted.py`, which is identical except for the loss.

JiT operates directly on image pixels (no VAE, no tokenizer) and is trained with a rectified-flow
interpolant between data and noise: `z_t = t * x0 + (1 - t) * noise`, where `t = 1` is the clean
image and `t = 0` is pure noise. The network is fed `(z_t, t)` and predicts `x0` directly (not the
velocity) -- the paper's central claim is that this plain "x-prediction" parameterization, with a
large enough patch size and enough compute, needs none of the usual crutches (latent tokenizers,
REPA-style representation losses, EMA-heavy schedules) to denoise well.

Loss. The paper trains with a velocity loss `L = E||v_theta(z_t, t) - v||^2`, where the true velocity
along the linear interpolant is the constant `v = x0 - noise`, and the model's velocity is recovered
from its x0-prediction as `v_theta(z_t, t) = (net_theta(z_t, t) - z_t) / (1 - t)`. Substituting the
interpolant algebraically collapses this to a plain x0-MSE loss with an explicit weight:
`L = E[ ||net_theta(z_t, t) - x0||^2 / (1 - t)^2 ]` -- i.e. the velocity reformulation is equivalent
to x0-MSE weighted by `1 / (1 - t)^2`, which is what's implemented directly below (the `(1 - t)`
denominator is clipped at `--t_eps` to avoid the blowup as `t -> 1`).

Timestep sampling. `t` is *not* sampled uniformly: the paper samples `logit(t) ~ N(mu, sigma^2)` with
`mu = -0.8, sigma = 0.8` (tuned for ImageNet 256x256), which biases sampling toward small `t` (noisier
images) -- under the `1 / (1 - t)^2` weighting, the near-`t=1` steps already dominate the loss, so the
sampler compensates by spending more steps in the harder, higher-noise regime. `--t_sampling uniform`
and `uniform_bounded` are kept as simple ablation toggles over the same interpolant/loss; they are not
part of the paper's recipe, only `logit_normal` (the default) is.

Since JiT is class-conditional (ImageNet, 1000 classes) and most fine-tuning datasets are not,
every example is conditioned on the model's built-in null class (index `num_classes`) - the same
"unconditional" token used for classifier-free guidance at pretraining time.

Full fine-tuning vs. LoRA. Every parameter in the transformer is trainable here -- there is no adapter,
no frozen base model, and no rank restriction. This is far more memory-hungry than a LoRA run of the
same model (expect to need gradient checkpointing and/or a smaller batch size), but it also means there
is no ceiling on how much the model's internals -- including the patch-embedding projection at the
input -- can shift to fit the fine-tuning data.
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
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
from diffusers.training_utils import free_memory
from diffusers.utils import check_min_version, is_wandb_available


check_min_version("0.36.0.dev0")

logger = get_logger(__name__)


def sample_logit_normal_t(batch_size, mean, std, device, dtype):
    """Sample `t in (0, 1)` via `t = sigmoid(logit)`, `logit ~ N(mean, std^2)` -- the JiT paper's
    timestep sampler (`mean=-0.8, std=0.8` for ImageNet 256x256), biased toward small `t` (noisier
    images) to counterbalance the `1 / (1 - t)^2` loss weighting, which otherwise concentrates nearly
    all training signal near `t=1`.
    """
    logits = torch.randn(batch_size, device=device, dtype=dtype) * std + mean
    return torch.sigmoid(logits)


def parse_args():
    parser = argparse.ArgumentParser(description="Full-model fine-tuning script for JiTTransformer2DModel.")
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
    parser.add_argument("--output_dir", type=str, default="jit-full-finetune-celeba-hq")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--class_label",
        type=int,
        default=0,
        help="ImageNet class index to condition every training example on. Leave unset to condition on the "
        "model's null class, which is simple but makes classifier-free guidance a no-op at inference: "
        "`JiTPipeline` forms guidance from `v(class) - v(null)`, so if the trained class *is* null both "
        "branches are identical and `guidance_scale` only doubles compute. Since guidance is the standard "
        "mechanism for tightening global structure, set this (to any index in [0, num_classes), e.g. an "
        "unused one) together with --class_dropout_prob to make guidance available.",
    )
    parser.add_argument(
        "--class_dropout_prob",
        type=float,
        default=0.1,
        help="Probability of replacing --class_label with the null class for a given training example, the "
        "standard classifier-free-guidance recipe: the null embedding keeps its 'unconditional' meaning "
        "while --class_label carries the fine-tuned domain, and guidance at inference is the contrast "
        "between them. Ignored when --class_label is unset (every example is already the null class).",
    )
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
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Strongly recommended for full-model fine-tuning (as opposed to LoRA) -- every parameter is "
        "trainable here, so activation memory is the dominant cost.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument(
        "--t_eps",
        type=float,
        default=0.05,
        help="Clips `(1 - t)` in the loss weight `1 / (1 - t)^2`, matching the JiT paper's clipping to avoid a "
        "division blowup near t=1.",
    )
    parser.add_argument(
        "--t_sampling",
        type=str,
        default="logit_normal",
        choices=["logit_normal", "uniform", "uniform_bounded"],
        help="How t is sampled. 'logit_normal' (default, the JiT paper's choice): t = sigmoid(N(--logit_mean, "
        "--logit_std^2)), which -- paired with the 1/(1-t)^2 loss weight already dominating near t=1 -- "
        "deliberately *under*-samples t near 1, so the --t_eps clamp on (1 - t) essentially never fires "
        "(P(t > 1 - t_eps) ~ 5e-7 at the default mean/std). 'uniform': t ~ Uniform(0, 1), ignoring "
        "--logit_mean/--logit_std; note this makes the clamp fire on ~t_eps of all steps (5% at the default "
        "t_eps=0.05), and whenever it fires the velocity residual picks up a spurious term proportional to "
        "(x0 - noise) -- white, i.e. flat across every frequency band -- that is clamp arithmetic, not model "
        "error. 'uniform_bounded': t ~ Uniform(0, 1 - t_eps), which keeps uniform coverage but never lets "
        "the clamp fire. These two alternatives are simple ablation toggles, not part of the paper's recipe.",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=-0.8,
        help="Mean of the Gaussian applied to `logit(t)` before sigmoid, i.e. `t = sigmoid(N(logit_mean, "
        "logit_std^2))`. The JiT paper's timestep sampler, default -0.8 (its ImageNet 256x256 value), biases "
        "sampling toward small t (noisier images/harder denoising steps). Ignored if --t_sampling=uniform.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=0.8,
        help="Std of the Gaussian applied to `logit(t)` before sigmoid. The JiT paper's default, 0.8, matches "
        "its ImageNet 256x256 recipe. Ignored unless --t_sampling=logit_normal.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The JiT paper's Adam beta1.")
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The JiT paper's Adam beta2 (note: not AdamW's usual 0.999)."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.0, help="The JiT paper trains with zero weight decay."
    )
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
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale for validation samples. Forced to 1.0 when --class_label is "
        "unset, where guidance cannot do anything (see --class_label).",
    )
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
    null_class = transformer.config.num_classes
    if args.class_label is not None and not 0 <= args.class_label < null_class:
        raise ValueError(
            f"--class_label must be in [0, {null_class}); got {args.class_label}. {null_class} is the null "
            "class itself -- leave --class_label unset to train on it, but note that makes guidance a no-op."
        )
    # The class every example is conditioned on, before dropout. Also what validation samples with, so the
    # validation grid reflects the same conditioning the model is actually being fit for.
    train_class = null_class if args.class_label is None else args.class_label

    # Full fine-tuning: every parameter is trainable and the whole model is kept in fp32. Under
    # --mixed_precision fp16/bf16, Accelerate autocasts the forward pass internally once the model is
    # passed through accelerator.prepare() below; there is no separate "cast trainable params to fp32"
    # step here the way a LoRA run needs, because the entire model *is* the trainable params.
    transformer.requires_grad_(True)
    transformer.to(accelerator.device)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        return accelerator.unwrap_model(model)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
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
        accelerator.init_trackers("jit_full_finetune", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training (full model, baseline JiT recipe) *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Trainable parameters = {sum(p.numel() for p in trainable_params):,} (full model)")
    logger.info(f"  Resolution = {resolution}, flow-matching shift = {shift}, null class = {null_class}")
    if args.class_label is None:
        logger.info(
            "  Conditioning on the null class; classifier-free guidance will be a no-op at inference "
            "(set --class_label to enable it)."
        )
    else:
        logger.info(
            f"  Conditioning on class {train_class} with {args.class_dropout_prob:.0%} dropout to null. "
            f"Pass --class_label={train_class} to the pipeline to sample with guidance."
        )

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
                pixel_values = batch["pixel_values"]
                bsz = pixel_values.shape[0]

                if args.t_sampling == "uniform":
                    t = torch.rand(bsz, device=pixel_values.device, dtype=pixel_values.dtype)
                elif args.t_sampling == "uniform_bounded":
                    # Uniform over (0, 1 - t_eps): the open interval on which the `--t_eps` clamp below
                    # never fires, so v_theta - v stays exactly (net_theta - x0) / (1 - t) everywhere.
                    t = torch.rand(bsz, device=pixel_values.device, dtype=pixel_values.dtype) * (1.0 - args.t_eps)
                else:
                    # Logit-normal t sampling (JiT paper): biases sampling toward small t (noisier images),
                    # counterbalancing the 1/(1-t)^2 weight below, which otherwise concentrates most of the
                    # loss near t=1.
                    t = sample_logit_normal_t(
                        bsz, args.logit_mean, args.logit_std, pixel_values.device, pixel_values.dtype
                    )
                t_view = t.view(-1, 1, 1, 1)

                # z_t = t * x0 + (1 - t) * noise: the rectified-flow interpolant between the clean image
                # (t=1) and pure noise (t=0). The model predicts x0 directly from (z_t, t).
                target = pixel_values
                noise = torch.randn_like(pixel_values)
                z_t = t_view * target + (1.0 - t_view) * noise
                class_labels = torch.full((bsz,), train_class, device=pixel_values.device, dtype=torch.long)
                if args.class_label is not None and args.class_dropout_prob > 0:
                    # Standard CFG training: some examples are shown as unconditional so the null embedding
                    # keeps meaning something distinct from --class_label, which is what guidance contrasts.
                    dropped = torch.rand(bsz, device=pixel_values.device) < args.class_dropout_prob
                    class_labels = torch.where(dropped, null_class, class_labels)

                model_pred = transformer(z_t, timestep=t, class_labels=class_labels).sample

                # x0-MSE weighted by 1/(1-t)^2 -- algebraically equivalent to the paper's velocity loss
                # ||v_theta(z_t, t) - v||^2, where v_theta(z_t, t) = (net_theta(z_t, t) - z_t) / (1 - t) and
                # v = x0 - noise is the (t-independent) true velocity along the linear interpolant. `(1 - t)`
                # is clipped at `--t_eps` to avoid the division blowup as t -> 1. This is the ONLY loss term
                # in this script -- no frequency weighting, no adaptive schedules.
                weight = 1.0 / (1.0 - t_view).clamp(min=args.t_eps) ** 2
                loss = (weight * (model_pred.float() - target.float()) ** 2).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
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
                            import shutil

                            for checkpoint in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, checkpoint))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    # accelerator.save_state saves the full model + optimizer + scheduler + RNG state, so
                    # training can be resumed exactly. It is *not* the same as a diffusers-loadable
                    # checkpoint, which is why we also drop a `save_pretrained` snapshot alongside it.
                    accelerator.save_state(save_path)
                    unwrap_model(transformer).save_pretrained(os.path.join(save_path, "transformer"))
                    logger.info(f"Saved checkpoint to {save_path}")

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
                class_labels=[train_class] * args.num_validation_images,
                # Guidance is only meaningful when the trained class differs from the null class it is
                # contrasted against; at train_class == null_class it is exactly a no-op that doubles cost.
                guidance_scale=args.validation_guidance_scale if args.class_label is not None else 1.0,
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
        unwrap_model(transformer).save_pretrained(os.path.join(args.output_dir, "transformer"))
        logger.info(f"Saved final full model weights to {args.output_dir}")

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of full-model fine-tuning",
                ignore_patterns=["checkpoint-*", "logs"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)