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
"""Full fine-tuning of `JiTTransformer2DModel` with a frequency-weighted velocity loss and a phase-coherence loss.

Rectified flow: `z_t = t * x0 + (1 - t) * noise` (t=1 clean, t=0 noise). The model predicts x0; the velocity
is recovered as `v_theta = (pred - z_t) / (1 - t)` against the target `v = x0 - noise`.
"""

import argparse
import logging
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import FlowMatchEulerDiscreteScheduler, JiTPipeline, JiTTransformer2DModel
from diffusers.training_utils import free_memory


logger = get_logger(__name__)


# ----------------------------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------------------------


def radial_frequency_band_index(height, width, num_bands, device):
    """`(H, W)` map assigning each (non-fftshifted) FFT bin to one of `num_bands` radial bands, 0 = DC."""
    fy, fx = torch.meshgrid(
        torch.fft.fftfreq(height, device=device), torch.fft.fftfreq(width, device=device), indexing="ij"
    )
    radius = torch.sqrt(fy**2 + fx**2)
    return torch.clamp((radius / radius.max() * num_bands).long(), max=num_bands - 1)


def velocity_residual(model_pred, z_t, x0, noise, t_view, t_eps):
    """`v_theta - v`, with `(1 - t)` clipped at `t_eps`."""
    v_pred = (model_pred - z_t) / (1.0 - t_view).clamp(min=t_eps)
    return (v_pred - (x0 - noise)).float()


def frequency_weighted_velocity_loss(model_pred, z_t, x0, noise, t_view, weight_map, t_eps):
    """Mean of `weight_map`-weighted `|FFT(v_theta - v)|^2`. An all-ones map equals plain velocity MSE."""
    residual_fft = torch.fft.fft2(velocity_residual(model_pred, z_t, x0, noise, t_view, t_eps), norm="ortho")
    power = residual_fft.real.pow(2) + residual_fft.imag.pow(2)
    return (weight_map * power).mean()


def phase_coherence_loss(model_pred, target, t, band_idx, band_gammas, eps=1e-8):
    """Signal-power-weighted `1 - cos(phase(pred) - phase(target))` over FFT bins.

    Each bin is faded in by `t ** band_gammas[band]` in the numerator only; the ungated denominator keeps a
    uniform fade from cancelling out of the ratio.
    """
    pred_fft = torch.fft.fft2(model_pred.float(), norm="ortho")
    target_fft = torch.fft.fft2(target.float(), norm="ortho")
    target_mag = target_fft.abs()
    cos_phase_diff = (pred_fft * target_fft.conj()).real / (pred_fft.abs() * target_mag + eps)

    gamma_map = band_gammas.to(device=model_pred.device, dtype=model_pred.dtype)[band_idx]  # (H, W)
    t_col = t.to(model_pred.dtype).clamp(0.0, 1.0).view(-1, 1, 1)
    fade = (t_col ** gamma_map.unsqueeze(0)).unsqueeze(1)  # (B, 1, H, W)

    signal_weight = target_mag.pow(2)
    numerator = ((1.0 - cos_phase_diff) * signal_weight * fade).flatten(1).sum(dim=1)
    denominator = signal_weight.flatten(1).sum(dim=1).clamp(min=eps)
    return (numerator / denominator).mean()


# ----------------------------------------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------------------------------------


def measure_patch_bottleneck_transmission(transformer, num_bands):
    """Per-band fraction of patch energy that survives `x_embedder.proj1` (1 = fully transmitted).

    Projects every 2D Fourier mode of a patch onto `proj1`'s row space. `transformer` must be unwrapped.
    """
    proj1 = transformer.x_embedder.proj1.weight.detach().float().cpu()  # (bottleneck_dim, C, P, P)
    channels, patch = proj1.shape[1], proj1.shape[2]

    _, s, vh = torch.linalg.svd(proj1.flatten(1), full_matrices=False)
    basis = vh[s > s.max() * 1e-6]

    freq = torch.fft.fftfreq(patch)
    fy, fx = torch.meshgrid(freq, freq, indexing="ij")
    radius = torch.sqrt(fy**2 + fx**2)
    radius = radius / radius.max()
    rows, cols = torch.meshgrid(torch.arange(patch), torch.arange(patch), indexing="ij")

    transmitted = torch.zeros(num_bands)
    counts = torch.zeros(num_bands)
    for a in range(patch):
        for b in range(patch):
            band = min(int(radius[a, b].item() * num_bands), num_bands - 1)
            phase = 2.0 * math.pi * (a * rows / patch + b * cols / patch)
            for mode in (torch.cos(phase), torch.sin(phase)):
                if mode.abs().max() < 1e-6:  # sine of a self-conjugate mode is zero
                    continue
                for channel in range(channels):
                    probe = torch.zeros(channels, patch, patch)
                    probe[channel] = mode
                    probe = (probe / probe.norm()).flatten()
                    transmitted[band] += (basis @ probe).pow(2).sum()
                    counts[band] += 1
    return transmitted / counts.clamp(min=1)


def residual_band_power(residual, band_idx, num_bands):
    """Per-sample mean FFT power of a `(B, C, H, W)` tensor within each radial band -> `(B, num_bands)`."""
    residual_fft = torch.fft.fft2(residual, norm="ortho")
    power = (residual_fft.real.pow(2) + residual_fft.imag.pow(2)).mean(dim=1).flatten(start_dim=1)  # (B, H*W)

    bsz = power.shape[0]
    band_sum = torch.zeros(bsz, num_bands, device=power.device, dtype=power.dtype)
    band_sum.scatter_add_(1, band_idx.flatten().unsqueeze(0).expand(bsz, -1), power)
    band_count = torch.bincount(band_idx.flatten(), minlength=num_bands).clamp(min=1).to(power.dtype)
    return band_sum / band_count


def _to_display_image(x):
    """`(C, H, W)` tensor in `[-1, 1]` -> numpy image in `[0, 1]` (clamped, not auto-stretched)."""
    x = ((x.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    if x.shape[0] == 1:
        return x[0].cpu().numpy()
    if x.shape[0] == 3:
        return x.permute(1, 2, 0).cpu().numpy()
    return x.mean(dim=0).cpu().numpy()


def log_figure(accelerator, tag, fig, step):
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    accelerator.log({tag: wandb.Image(image)}, step=step)


def spread_over_t(t, num_samples):
    """Batch indices of `num_samples` samples evenly spaced across the sorted t range."""
    bsz = t.shape[0]
    num_samples = min(num_samples, bsz)
    return torch.argsort(t)[torch.linspace(0, bsz - 1, num_samples, device=t.device).long()].tolist()


def log_residual_fft(residual, t, picks, accelerator, step):
    fig, axes = plt.subplots(1, len(picks), figsize=(3 * len(picks), 3), squeeze=False)
    for ax, idx in zip(axes[0], picks):
        magnitude = torch.fft.fft2(residual[idx], norm="ortho").abs().mean(dim=0)
        im = ax.imshow(torch.fft.fftshift(magnitude).log1p().cpu().numpy(), cmap="viridis")
        ax.set_title(f"t={t[idx].item():.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"|FFT(v_theta - v)| -- step {step}")
    fig.tight_layout()
    log_figure(accelerator, "diagnostics/residual_fft", fig, step)


def log_reconstruction_spatial(model_pred, target, residual, t, picks, accelerator, step):
    fig, axes = plt.subplots(3, len(picks), figsize=(3 * len(picks), 9), squeeze=False)
    for col, idx in enumerate(picks):
        axes[0, col].imshow(_to_display_image(target[idx]))
        axes[0, col].set_title(f"t={t[idx].item():.2f}")
        axes[1, col].imshow(_to_display_image(model_pred[idx]))
        im = axes[2, col].imshow(residual[idx].abs().mean(dim=0).cpu().numpy(), cmap="inferno")
        fig.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    for row, label in enumerate(["clean x0", "reconstruction", "|v_theta - v|"]):
        axes[row, 0].set_ylabel(label, fontsize=10)
    fig.suptitle(f"Reconstruction vs. target vs. residual -- step {step}")
    fig.tight_layout()
    log_figure(accelerator, "diagnostics/reconstruction_spatial", fig, step)


def bucket_by_t(band_power, t, num_t_buckets):
    """Mean `(N, num_bands)` band power per equal-width t bucket -> `(num_bands, num_t_buckets)`, plus counts."""
    t_bucket = torch.clamp((t * num_t_buckets).long(), max=num_t_buckets - 1)
    heatmap = torch.zeros(band_power.shape[1], num_t_buckets)
    for bucket in range(num_t_buckets):
        mask = t_bucket == bucket
        if mask.any():
            heatmap[:, bucket] = band_power[mask].mean(dim=0)
    return heatmap, torch.bincount(t_bucket, minlength=num_t_buckets)


def plot_band_t_heatmap(values, counts, colorbar_label, title, **imshow_kwargs):
    num_bands, num_t_buckets = values.shape
    fig, ax = plt.subplots(figsize=(1.2 * num_t_buckets + 2, 0.6 * num_bands + 2))
    im = ax.imshow(values, aspect="auto", origin="lower", **imshow_kwargs)
    ax.set_xlabel(f"t bucket (0 = noisiest ... {num_t_buckets - 1} = cleanest); n = samples per bucket")
    ax.set_ylabel(f"frequency band (0 = DC ... {num_bands - 1} = Nyquist)")
    ax.set_xticks(range(num_t_buckets))
    ax.set_xticklabels(
        [f"{(b + 0.5) / num_t_buckets:.2f}\n(n={int(counts[b])})" for b in range(num_t_buckets)],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(num_bands))
    cbar = fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, cbar


def log_band_vs_t_heatmap(band_power_history, t_history, num_t_buckets, accelerator, step, tag, title):
    """Mean raw residual power per (band, t bucket), log color scale."""
    band_power = torch.cat(band_power_history)
    heatmap, counts = bucket_by_t(band_power, torch.cat(t_history), num_t_buckets)
    heatmap = heatmap.numpy()
    positive = heatmap[heatmap > 0]
    floor = positive.min() if positive.size > 0 else 1e-8
    norm = matplotlib.colors.LogNorm(vmin=floor, vmax=max(heatmap.max(), floor * 10))
    fig, _ = plot_band_t_heatmap(
        np.clip(heatmap, floor, None),
        counts,
        "mean |FFT(residual)|^2 (log scale)",
        f"{title} -- step {step}, n={band_power.shape[0]}",
        cmap="magma",
        norm=norm,
    )
    log_figure(accelerator, tag, fig, step)


def log_normalized_band_heatmap(
    x0_band_power_history, signal_band_power_history, t_history, num_t_buckets, accelerator, step
):
    """Per-band NMSE = x0 residual power / signal power; 0 on the log scale = as good as predicting zero."""
    x0_band_power = torch.cat(x0_band_power_history)
    signal_band_power = torch.cat(signal_band_power_history).mean(dim=0)
    measured, counts = bucket_by_t(x0_band_power, torch.cat(t_history), num_t_buckets)
    nmse = (measured / signal_band_power.clamp(min=1e-12).unsqueeze(1)).numpy()

    finite = nmse[np.isfinite(nmse) & (nmse > 0)]
    extent = max(abs(np.log10(finite).max()), abs(np.log10(finite).min()), 0.1) if finite.size else 1.0
    fig, cbar = plot_band_t_heatmap(
        np.log10(np.clip(nmse, 1e-12, None)),
        counts,
        "log10(NMSE)  --  0 = as good as predicting zero, red = worse",
        f"Per-band normalized MSE (x0 residual / signal power) -- step {step}, n={x0_band_power.shape[0]}",
        cmap="RdBu_r",
        vmin=-extent,
        vmax=extent,
    )
    cbar.ax.axhline(0.0, color="black", linewidth=1)
    log_figure(accelerator, "diagnostics/normalized_band_mse", fig, step)


# ----------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------


def collate_fn(examples):
    return {"pixel_values": torch.stack([example["pixel_values"] for example in examples])}


def parse_args():
    parser = argparse.ArgumentParser(description="Full fine-tuning of JiT with frequency-weighted and phase losses.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="JiT-diffusers/JiT-B-16")
    parser.add_argument("--dataset_name", type=str, default="korexyz/celeba-hq-256x256")
    parser.add_argument("--output_dir", type=str, default="jit-full-finetune-freqloss-celeba-hq")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class_label", type=int, default=0, help="Class every example is conditioned on.")
    parser.add_argument("--class_dropout_prob", type=float, default=0.1, help="Probability of using the null class.")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--max_train_steps", type=int, default=15000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--t_eps", type=float, default=0.05, help="Lower clip on (1 - t).")
    parser.add_argument("--t_sampling", type=str, default="logit_normal", choices=["logit_normal"])
    parser.add_argument("--logit_mean", type=float, default=-0.8)
    parser.add_argument("--logit_std", type=float, default=0.8)
    parser.add_argument("--freq_loss", action="store_true", help="Use the frequency-weighted velocity loss.")
    parser.add_argument("--freq_schedule", type=str, default="static", choices=["static"])
    parser.add_argument("--num_freq_bands", type=int, default=8)
    parser.add_argument(
        "--freq_weight_power", type=float, default=1.0, help="weight(band) = 1 + scale * (band / (n - 1)) ** power"
    )
    parser.add_argument("--freq_weight_scale", type=float, default=3.0)
    parser.add_argument("--phase_loss_weight", type=float, default=0.0, help="0 disables the phase loss.")
    parser.add_argument("--phase_loss_gamma", type=float, default=3.0, help="t-fade exponent at the highest band.")
    parser.add_argument("--phase_loss_gamma_low", type=float, default=0.2, help="t-fade exponent at band 0.")
    parser.add_argument("--transmission_log_steps", type=int, default=0, help="0 disables.")
    parser.add_argument("--freq_log_steps", type=int, default=0, help="0 disables.")
    parser.add_argument("--freq_log_num_samples", type=int, default=3)
    parser.add_argument("--freq_log_t_buckets", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_epochs", type=int, default=1, help="0 disables.")
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_guidance_scale", type=float, default=3.0)
    parser.add_argument("--validation_inference_steps", type=int, default=50)
    return parser.parse_args()


def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=str(Path(args.output_dir, "log"))
        ),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    transformer = JiTTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    shift = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    ).config.shift
    resolution = transformer.config.sample_size
    null_class = transformer.config.num_classes
    if not 0 <= args.class_label < null_class:
        raise ValueError(f"--class_label must be in [0, {null_class}); got {args.class_label}.")

    num_bands = args.num_freq_bands
    band_idx = radial_frequency_band_index(resolution, resolution, num_bands, accelerator.device)

    if args.freq_loss:
        band_weights = [
            1.0 + args.freq_weight_scale * (band / max(num_bands - 1, 1)) ** args.freq_weight_power
            for band in range(num_bands)
        ]
        freq_weight_map = torch.tensor(band_weights, device=accelerator.device, dtype=torch.float32)[band_idx]
        logger.info(f"  Frequency loss band weights (DC -> Nyquist) = {band_weights}")

    if args.phase_loss_weight > 0:
        phase_band_gammas = args.phase_loss_gamma_low + (
            args.phase_loss_gamma - args.phase_loss_gamma_low
        ) * torch.linspace(0.0, 1.0, num_bands, device=accelerator.device)
        logger.info(
            f"  Phase loss weight = {args.phase_loss_weight}, "
            f"band gammas (DC -> Nyquist) = {[round(g, 3) for g in phase_band_gammas.tolist()]}"
        )

    transformer.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=1e-8,
    )

    dataset = load_dataset(args.dataset_name, split="train")
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
        examples["pixel_values"] = [train_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples

    dataset.set_transform(preprocess)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)
    num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))

    if accelerator.is_main_process:
        accelerator.init_trackers("jit_full_finetune_freqloss", config=vars(args))

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}, num epochs = {num_train_epochs}")
    logger.info(f"  Total train batch size = {args.train_batch_size * accelerator.num_processes}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Resolution = {resolution}, shift = {shift}, class = {args.class_label}, null class = {null_class}")

    # Accumulated between --freq_log_steps flushes.
    history_v, history_x0, history_signal, history_t = [], [], [], []

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Steps")

    for epoch in range(num_train_epochs):
        transformer.train()
        for batch in train_dataloader:
            target = batch["pixel_values"]
            bsz = target.shape[0]

            t = torch.sigmoid(
                torch.randn(bsz, device=target.device, dtype=target.dtype) * args.logit_std + args.logit_mean
            )
            t_view = t.view(-1, 1, 1, 1)
            noise = torch.randn_like(target)
            z_t = t_view * target + (1.0 - t_view) * noise
            class_labels = torch.full((bsz,), args.class_label, device=target.device, dtype=torch.long)
            dropped = torch.rand(bsz, device=target.device) < args.class_dropout_prob
            class_labels = torch.where(dropped, null_class, class_labels)

            model_pred = transformer(z_t, timestep=t, class_labels=class_labels).sample

            if args.freq_log_steps > 0 and accelerator.is_main_process:
                with torch.no_grad():
                    pred = model_pred.detach()
                    residual_v = velocity_residual(pred, z_t, target, noise, t_view, args.t_eps)
                    history_v.append(residual_band_power(residual_v, band_idx, num_bands).cpu())
                    history_x0.append(residual_band_power(pred.float() - target.float(), band_idx, num_bands).cpu())
                    history_signal.append(residual_band_power(target.float(), band_idx, num_bands).cpu())
                    history_t.append(t.cpu())

            if args.freq_loss:
                loss = frequency_weighted_velocity_loss(
                    model_pred, z_t, target, noise, t_view, freq_weight_map, args.t_eps
                )
            else:
                weight = 1.0 / (1.0 - t_view).clamp(min=args.t_eps) ** 2
                loss = (weight * (model_pred.float() - target.float()) ** 2).mean()

            if args.phase_loss_weight > 0:
                loss = loss + args.phase_loss_weight * phase_coherence_loss(
                    model_pred, target, t, band_idx, phase_band_gammas
                )

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if args.transmission_log_steps > 0 and global_step % args.transmission_log_steps == 0:
                    with torch.no_grad():
                        transmission = measure_patch_bottleneck_transmission(
                            accelerator.unwrap_model(transformer), num_bands
                        ).tolist()
                    accelerator.log(
                        {f"diagnostics/transmission_band_{b}": v for b, v in enumerate(transmission)},
                        step=global_step,
                    )
                    logger.info(f"  [transmission] step {global_step}: {[round(v, 4) for v in transmission]}")

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    accelerator.unwrap_model(transformer).save_pretrained(os.path.join(save_path, "transformer"))
                    logger.info(f"Saved checkpoint to {save_path}")

                if args.freq_log_steps > 0 and global_step % args.freq_log_steps == 0:
                    with torch.no_grad():
                        pred = model_pred.detach()
                        residual_v = velocity_residual(pred, z_t, target, noise, t_view, args.t_eps)
                        picks = spread_over_t(t, args.freq_log_num_samples)
                        log_residual_fft(residual_v, t, picks, accelerator, global_step)
                        log_reconstruction_spatial(pred, target, residual_v, t, picks, accelerator, global_step)
                    log_band_vs_t_heatmap(
                        history_v,
                        history_t,
                        args.freq_log_t_buckets,
                        accelerator,
                        global_step,
                        tag="diagnostics/band_vs_t_heatmap_velocity",
                        title="Velocity residual |FFT(v_theta - v)|^2 vs band and t (rescaled by 1/(1-t))",
                    )
                    log_band_vs_t_heatmap(
                        history_x0,
                        history_t,
                        args.freq_log_t_buckets,
                        accelerator,
                        global_step,
                        tag="diagnostics/band_vs_t_heatmap_x0",
                        title="Raw x0 residual |FFT(model_pred - x0)|^2 vs band and t (unscaled)",
                    )
                    log_normalized_band_heatmap(
                        history_x0, history_signal, history_t, args.freq_log_t_buckets, accelerator, global_step
                    )
                    for history in (history_v, history_x0, history_signal, history_t):
                        history.clear()

            progress_bar.set_postfix(loss=loss.detach().item())
            accelerator.log({"loss": loss.detach().item()}, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        if accelerator.is_main_process and args.validation_epochs > 0 and (epoch + 1) % args.validation_epochs == 0:
            logger.info("Running validation...")
            pipeline = JiTPipeline(
                transformer=accelerator.unwrap_model(transformer),
                scheduler=FlowMatchEulerDiscreteScheduler(shift=shift),
            )
            pipeline.set_progress_bar_config(disable=True)
            images = pipeline(
                class_labels=[args.class_label] * args.num_validation_images,
                guidance_scale=args.validation_guidance_scale,
                num_inference_steps=args.validation_inference_steps,
                generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
                output_type="np",
            ).images
            accelerator.log({"validation": [wandb.Image(image) for image in images]})
            del pipeline
            free_memory()

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(transformer).save_pretrained(os.path.join(args.output_dir, "transformer"))
        logger.info(f"Saved final model to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
