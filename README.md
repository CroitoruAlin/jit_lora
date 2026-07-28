# LoRA fine-tuning for JiT

[`JiTTransformer2DModel`](../../../src/diffusers/models/transformers/jit_transformer_2d.py) is a pixel-space,
class-conditional (ImageNet, 1000 classes) transformer trained with flow matching: it takes a noisy image and a
timestep and predicts the clean image directly, with no VAE involved. `train_lora_jit.py` fine-tunes a pretrained
JiT checkpoint with [LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora) on an arbitrary image dataset,
loosely following the training recipe from the JiT paper,
["Back to Basics: Let Denoising Generative Models Denoise"](https://arxiv.org/abs/2511.13720).

The target is not plain `x0` but a frequency-masked version of it: its 2D FFT is radially cropped to a cutoff that
grows with `t`, so at high noise levels the model is only asked to recover low-frequency structure, and at low
noise levels the target includes the full spectrum. The noised input is built from this **same masked image**, not
the true `x0` (`z_t = t * target + (1 - t) * noise`): at inference there is no true `x0` for high-frequency content
to leak in from, so noising with the true image would train on inputs inconsistent with what the model actually
sees when sampling. Building both `z_t` and the target from the same masked image keeps the corruption process
self-consistent, as in blurring/soft-diffusion formulations, rather than asking the model to discard real detail
that's diluted into its own input. This is combined with the paper's `1 / (1 - t)^2` implicit loss weight (normally
obtained "for free" from a velocity-space reformulation of the x0-MSE; applied here as an explicit weight since the
target is no longer plain `x0`).

Unlike the paper, `t` is sampled **uniformly** from `(0, 1)` rather than from a logit-normal distribution biased
toward noisier timesteps. That bias was tuned for a plain x0-MSE loss, where the near-noise task is the hard,
undertrained one; the frequency-masked target above already makes the near-noise task easier on its own (a much
simpler target), so keeping the paper's bias on top would starve higher-`t`, higher-detail steps of training
signal. Concretely: combining the paper's `logit_normal(mean=-0.8)` with the frequency-mask schedule left over 99%
of training steps with a target that never reached even half of the full spectrum — the model never learned to
add detail because it was (almost) never asked to.

Since most fine-tuning datasets aren't labeled with ImageNet classes, every training example is conditioned on the
model's built-in null class (the same token JiT was trained to use for unconditional/classifier-free-guidance
sampling), effectively adapting the pretrained backbone to a new, unconditional image domain.

## Setup

`JiTTransformer2DModel` and `JiTPipeline` only exist in this local checkout, not yet in a released `diffusers`
version on PyPI, so install *this* repo in editable mode rather than a plain `pip install diffusers` (which would
silently give you a `diffusers` without JiT support). From the repo root:

```bash
pip install -e ".[dev]"
pip install -r examples/research_projects/jit_lora/requirements.txt
```

If you already have a different `diffusers` installed, `pip install -e .` will replace it in your environment with
this checkout (verify with `python -c "import diffusers; print(diffusers.__file__)"` — it should point inside this
`diffusers/src/diffusers` directory).

Logging defaults to [Weights & Biases](https://wandb.ai); run `wandb login` (or set `WANDB_API_KEY`) beforehand, or
pass `--report_to=tensorboard` to use TensorBoard instead.

## Usage

The default configuration fine-tunes [`JiT-diffusers/JiT-B-16`](https://huggingface.co/JiT-diffusers/JiT-B-16) on
[`korexyz/celeba-hq-256x256`](https://huggingface.co/datasets/korexyz/celeba-hq-256x256), which already ships
256x256 images matching JiT-B/16's native resolution:

```bash
accelerate launch train_lora_jit.py \
  --pretrained_model_name_or_path="JiT-diffusers/JiT-B-16" \
  --dataset_name="korexyz/celeba-hq-256x256" \
  --output_dir="jit-lora-celeba-hq" \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --rank=16 \
  --num_train_epochs=50 \
  --checkpointing_steps=500 \
  --validation_epochs=5 \
  --mixed_precision="bf16"
```

Notable options:

- `--rank` / `--lora_alpha` / `--lora_dropout`: standard LoRA hyperparameters.
- `--lora_layers`: comma-separated module names to attach LoRA to. Defaults to the attention projections
  (`to_q`, `to_k`, `to_v`, `to_out.0`) and the SwiGLU MLP projections (`w12`, `w3`) in every block.
- `--t_eps`: clips `(1 - t)` in the loss weight `1 / (1 - t)^2` to avoid a division blowup near `t=1`, matching the
  JiT paper's clipping (default `0.05`).
- `--freq_mask_alpha` / `--freq_mask_saturation_t`: the FFT cutoff is derived from where per-frequency signal power
  (natural images fall off as `~1/f^alpha`, `alpha` default `2.0`) equals the injected noise power, which is flat
  across frequencies. `--freq_mask_saturation_t` (default `0.7`) is the `t` value at which the cutoff first reaches
  1.0 (full detail); it exists because the signal-vs-noise comparison needs a calibration constant that can't be
  derived from `alpha` alone. With uniform `t` sampling, `0.7` means ~19% of training steps stay heavily blurred
  (`cutoff_fraction < 0.1`) and ~31% see near-full detail (`cutoff_fraction > 0.95`); raise it toward `1.0` for a
  longer, more aggressive low-frequency-only curriculum (e.g. `0.9` leaves ~47% of steps heavily blurred and only
  ~10% near-full detail).
- `--freq_mask_transition`: width of the smooth rolloff around the FFT cutoff. The cutoff is a sigmoid, not a hard
  step, to avoid ringing (Gibbs phenomenon) artifacts in the target that a brick-wall cutoff would otherwise bake
  in and the model would learn to reproduce. Default `0.1`; going much below that (e.g. `0.05`) reintroduces
  visible ringing.
- `--validation_epochs` / `--num_validation_images`: periodically samples images with the in-training LoRA weights
  and logs them to Weights & Biases (or TensorBoard with `--report_to=tensorboard`).

Only the LoRA adapter weights are saved (as `pytorch_lora_weights.safetensors`), both in intermediate
`checkpoint-*` directories and in `--output_dir` at the end of training.

## Inference

```python
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, JiTPipeline, JiTTransformer2DModel

transformer = JiTTransformer2DModel.from_pretrained("JiT-diffusers/JiT-B-16", subfolder="transformer")
transformer.load_lora_adapter("jit-lora-celeba-hq", prefix=None, weight_name="pytorch_lora_weights.safetensors")

pipe = JiTPipeline(transformer=transformer, scheduler=FlowMatchEulerDiscreteScheduler(shift=4.0))
pipe.to("cuda")

null_class = transformer.config.num_classes  # unconditional token used during fine-tuning
image = pipe(class_labels=null_class, guidance_scale=1.0, num_inference_steps=50).images[0]
image.save("sample.png")
```

## Evaluation

`eval_lora_jit.py` generates a batch of images with a trained LoRA checkpoint and scores them with the LAION
"improved aesthetic predictor" and [HPSv2](https://github.com/tgxs002/HPSv2):

```bash
python eval_lora_jit.py \
  --pretrained_model_name_or_path="JiT-diffusers/JiT-B-16" \
  --lora_path="jit-lora-celeba-hq" \
  --num_images=100 \
  --guidance_scale=1.0
```

`--class_label` defaults to the null class (`transformer.config.num_classes`) — pass the same class the LoRA was
actually trained with if it differs. Results (per-image and aggregate mean/std) are written to
`<output_dir>/scores.json`; generated images go to `<output_dir>/images/`.

HPSv2 is a text-image preference score (it compares images generated from the *same prompt*), but this LoRA is
class-conditional, not text-conditional, so there's no natural prompt for the generated images. `--hps_prompt`
(default `"a photo of a face"`) is a generic stand-in — treat the HPSv2 number as a rough general-quality proxy,
not a prompt-faithfulness score in the usual HPS sense.

**Known issue in `hpsv2==1.2.0`:** the PyPI package vendors its own `open_clip` copy but the wheel is missing its
BPE vocab data file, so `hpsv2.score(...)` fails with `FileNotFoundError: ... bpe_simple_vocab_16e6.txt.gz`. Fix by
copying the file from a real `open_clip_torch` install:

```bash
pip install open_clip_torch
python -c "
import shutil, hpsv2, open_clip, pathlib
src = pathlib.Path(open_clip.__file__).parent / 'bpe_simple_vocab_16e6.txt.gz'
dst = pathlib.Path(hpsv2.__file__).parent / 'src' / 'open_clip' / 'bpe_simple_vocab_16e6.txt.gz'
shutil.copy(src, dst)
"
```
