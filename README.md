# JiT fine-tuning

Full fine-tuning of [JiT](https://arxiv.org/abs/2511.13720) (pixel-space, class-conditional diffusion transformer) on
CelebA-HQ 256×256, with:

- `train_full_jit_baseline.py` — the plain JiT recipe (x0-prediction, `1/(1-t)^2`-weighted loss).
- `train_full_freq.py` — the same, plus a frequency-band-weighted velocity loss (`--freq_loss`), a phase-coherence
  loss (`--phase_loss_weight`), and frequency diagnostics logged to wandb.

## Setup

### 1. Python environment

Python 3.12. Install a PyTorch build matching your CUDA version first (see [pytorch.org](https://pytorch.org/get-started/locally/));
this repo was run with torch 2.13 + CUDA 13.0 on RTX 5090s.

```bash
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 2. diffusers with JiT

JiT is not in released diffusers yet. The `jit/` folder holds its source files (from the
`add-jit-diffusion` branch of [AlanPonnachan/diffusers](https://github.com/AlanPonnachan/diffusers), with
`PeftAdapterMixin` added so LoRA adapters can be loaded), laid out at the same paths as in diffusers:

```
jit/
├── src/diffusers/models/transformers/jit_transformer_2d.py   # JiTTransformer2DModel
├── src/diffusers/pipelines/jit/__init__.py
├── src/diffusers/pipelines/jit/pipeline_jit.py               # JiTPipeline
├── register_jit.py                                           # adds the exports to diffusers' __init__.py files
└── convert_checkpoint.py                                     # converts the pretrained weights (step 3)
```

Clone diffusers, copy the JiT source in, register it, and install diffusers in editable mode:

```bash
git clone https://github.com/huggingface/diffusers.git
cp -r jit/src/. diffusers/src/
python jit/register_jit.py diffusers
uv pip install -e ./diffusers
uv pip install -r requirements.txt
```

Check it worked:

```bash
python -c "from diffusers import JiTPipeline, JiTTransformer2DModel"
```

### 3. Pretrained checkpoints

Download the JiT checkpoints (needs `git lfs`), then convert the transformer weights to the attention layout used by
`jit_transformer_2d.py` (splits the fused `qkv` projection; the original file is kept as `*.bak`):

```bash
git lfs install
git clone https://huggingface.co/BiliSakura/JiT-diffusers
python jit/convert_checkpoint.py JiT-diffusers/JiT-B-16
```

Pass more variant folders (e.g. `JiT-diffusers/JiT-L-16`) to convert them too. Re-running on an already converted
folder is a no-op.

### 4. Logging

Both scripts log to wandb:

```bash
wandb login
```

## Training

The CelebA-HQ dataset (`korexyz/celeba-hq-256x256`) is downloaded from the Hugging Face Hub on first run.
`accelerate launch` uses every visible GPU; restrict it with `CUDA_VISIBLE_DEVICES=0` or set it up once with
`accelerate config`. The effective batch size is `--train_batch_size` (default 16) × number of GPUs.

### Baseline

```bash
accelerate launch train_full_jit_baseline.py \
  --pretrained_model_name_or_path=JiT-diffusers/JiT-B-16 \
  --dataset_name=korexyz/celeba-hq-256x256 \
  --seed=0 \
  --class_label=0 --class_dropout_prob=0.1 \
  --mixed_precision=bf16 \
  --learning_rate=1e-4 --max_train_steps=15000 \
  --checkpointing_steps=5000 --validation_epochs=5 \
  --t_sampling=logit_normal \
  --output_dir=runs/00_baseline
```

### Frequency-weighted + phase loss

Same shared settings, plus the frequency loss, the phase loss and the diagnostics:

```bash
accelerate launch train_full_freq.py \
  --pretrained_model_name_or_path=JiT-diffusers/JiT-B-16 \
  --dataset_name=korexyz/celeba-hq-256x256 \
  --seed=0 \
  --class_label=0 --class_dropout_prob=0.1 \
  --mixed_precision=bf16 \
  --learning_rate=1e-4 --max_train_steps=15000 \
  --checkpointing_steps=5000 --validation_epochs=5 \
  --t_sampling=logit_normal \
  --freq_log_steps=200 --num_freq_bands=8 --transmission_log_steps=1000 \
  --freq_loss --freq_schedule=static --freq_weight_power=1.0 --freq_weight_scale=3.0 \
  --phase_loss_weight=0.2 --phase_loss_gamma=3.0 --phase_loss_gamma_low=0.5 \
  --output_dir=runs/D1_w0.2_gl0.5
```

| Flag | Meaning |
| --- | --- |
| `--freq_loss` | Replace the baseline loss with an FFT-domain velocity loss weighted per radial frequency band. |
| `--freq_weight_scale`, `--freq_weight_power` | Band weights: `1 + scale * (band / (num_bands - 1)) ** power` (DC = 1, highest band = `1 + scale`). |
| `--phase_loss_weight` | Weight of the phase-coherence loss; `0` disables it. |
| `--phase_loss_gamma_low`, `--phase_loss_gamma` | Phase loss is faded in by `t ** gamma`, with gamma ramping from `gamma_low` (DC) to `gamma` (highest band). |
| `--freq_log_steps` | Log residual spectra, reconstructions and band-vs-t error heatmaps every N steps; `0` disables. |
| `--transmission_log_steps` | Log how much of each band survives the patch-embedding bottleneck every N steps; `0` disables. |

`run.sh` holds the full sweep (commented-out blocks are earlier runs).

### Outputs

```
runs/<name>/
├── checkpoint-5000/transformer/   # intermediate weights (plus accelerate optimizer state)
├── ...
└── transformer/                   # final weights
```

Each `transformer/` folder loads with `JiTTransformer2DModel.from_pretrained(...)`. To score one with the LAION
aesthetic predictor and HPSv2:

```bash
python eval_lora_jit.py --transformer_path runs/D1_w0.2_gl0.5 --output_dir eval_output/D1
```
