<div align="center">

# TurboT2AV

Fast text-to-audio-video generation distilled from LTX-2 19B.

</div>

## TurboDiffusion-Style Acceleration

![TurboT2AV TD-style acceleration decomposition at 1024x1792](assets/turbot2av_td_style_1024x1792.png)

Measured on a single NVIDIA H20 at `1024x1792`:

| Stage | Latency | Speedup vs previous | Speedup vs teacher | What changes |
| --- | ---: | ---: | ---: | --- |
| LTX-2-19B teacher (40 steps) | 318.7405s | - | 1.00x | Full teacher baseline with dense attention. |
| + W8A8 & FastNorm | 233.3424s | 1.37x | 1.37x | Add TileLang W8A8 Linear and FastNorm to the teacher. |
| + rCM (4-step student) | 11.7628s | 19.84x | 27.10x | Switch to the distilled student while retaining W8A8/FastNorm. |
| + SageSLA final | 5.8689s | 2.00x | 54.31x | Add SageSLA `topk=0.3` self-attention to the accelerated student. |

At this resolution the video latent is `[1,16,128,32,56]`, corresponding to
28,672 video self-attention tokens. The stages are cumulative: rCM keeps the
W8A8/FastNorm stack, and the final stage adds SageSLA. For reference, the pure
4-step student without these inference optimizations takes 16.1096s/video, so
the final path is also 2.75x faster than the pure student.

See [TurboDiffusion Integration Notes](docs/acceleration.md) for the reused
components, LTX-2-specific adaptations, and interpretation of these results.

## Overview

TurboT2AV generates synchronized audio-video from text prompts in 4 steps.
The demo compares the 40-step teacher with the 4-step student.
This repository provides single-GPU inference for the distilled checkpoint.
On an NVIDIA H20 at 1024x1792, generator-only latency falls from 318.74
seconds/video for the 40-step teacher to 5.87 seconds/video for the accelerated
4-step student.

Main contributions:

- Combines the diversity of consistency models (DCM/SCM) with the high
  perceptual quality of score-model distillation (DMD), taking advantage of both
  families of methods by using CM as a forward-divergence offline method that
  complements DMD as a reverse-KL on-policy method.
- First extends this combined distillation strategy to a large-scale joint
  audio-video generation model at the 14B-video + 5B-audio scale.
- Integrates a TurboDiffusion-style inference stack with SageSLA, FastNorm, and
  TileLang W8A8 Linear. On a single NVIDIA H20 at 1024x1792, the final
  accelerated student is 54.31x faster than the 40-step teacher and 2.75x
  faster than the pure 4-step student.

## 1. Setup

```bash
cd TurboDiffusion/TurboT2AV/LTX-2
pixi install
pixi run install-acceleration
```

This single task installs the local LTX packages, CUDA 12.8 PyTorch,
SageAttention, SpargeAttn, and TileLang. It provides everything required by the
recommended SageSLA + FastNorm + TileLang W8A8 inference path.

## 2. Download Weights

| Model Name | Checkpoint Link |
| --- | --- |
| TurboT2AV-14BVideo-5BAudio | [Hugging Face Model](https://huggingface.co/luyu1021/turbo-t2av) |
| LTX-2-19B | [Hugging Face Model](https://huggingface.co/Lightricks/LTX-2) |
| Gemma-3-12B-IT-QAT-Q4_0 | [Hugging Face Model](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) |

Gemma is a gated Hugging Face model. Before downloading, visit the model page,
accept the access terms, and export a Hugging Face token with access permission:

```bash
export HF_TOKEN=your_huggingface_token
```

Base model weights:

```bash
hf download Lightricks/LTX-2 ltx-2-19b-dev.safetensors --local-dir /path/to/checkpoints/LTX-2
hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir /path/to/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
```

Distilled checkpoint:

```bash
hf download luyu1021/turbo-t2av checkpoints/scm_dmd_checkpoint_001000/model.pth --local-dir /path/to/checkpoints
```

## 3. Run Inference

Run the following commands from `TurboDiffusion/TurboT2AV/LTX-2`:

```bash
export TURBO_CHECKPOINT_PATH=/path/to/ltx-2-19b-dev.safetensors
export TURBO_GEMMA_PATH=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
export PYTHONPATH=../..:../../turbodiffusion:$PYTHONPATH
```

`--prompts_file` accepts a text file with one prompt per line or a CSV file with
a `prompt` column.

### Accelerated Student (Recommended)

```bash
CUDA_VISIBLE_DEVICES=0 pixi run python -m ltx_distillation.tools.run_av_inference_eval \
  --config_path packages/ltx-distillation/configs/bidirectional_rcm.yaml \
  --prompts_file /path/to/prompts.csv \
  --output_dir /path/to/student_output \
  --model_kind student \
  --student_checkpoint /path/to/checkpoints/scm_dmd_checkpoint_001000/model.pth \
  --student_param auto \
  --num_prompts 8 \
  --video_height 1024 \
  --video_width 1792 \
  --attention_type sagesla \
  --attention_scope self \
  --sla_topk 0.3 \
  --fast_norm \
  --quant_linear \
  --quant_linear_scope all \
  --quant_linear_backend tilelang_postscale
```

### Teacher Baseline (40 Steps)

```bash
CUDA_VISIBLE_DEVICES=0 pixi run python -m ltx_distillation.tools.run_av_inference_eval \
  --config_path packages/ltx-distillation/configs/bidirectional_rcm.yaml \
  --prompts_file /path/to/prompts.csv \
  --output_dir /path/to/teacher_output \
  --model_kind teacher \
  --teacher_mode native_rf \
  --teacher_steps 40 \
  --num_prompts 8 \
  --video_height 1024 \
  --video_width 1792
```

## Demos

<table>
  <thead>
    <tr>
      <th align="center" width="50%">Teacher (40 steps)</th>
      <th align="center" width="50%">Student (4 steps)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/116d8f07-96a3-4a68-b0bd-fb03eb385269" alt="1" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/2b0509e0-216c-4575-ad03-d183a673a9ec" alt="1" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/e5330419-2147-48a9-9225-03113c6d1488" alt="2" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/9bbbf2ad-b16a-42ea-b63d-2faf1ef0abb2" alt="2" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/3a7e87b7-5c72-496c-9e6c-205e1ad31bb5" alt="3" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/eb21d65c-3bf9-4999-b73d-6c1f825f1549" alt="3" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/f04e2e34-b503-4557-8700-18ce5a058ff9" alt="4" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/24ba0beb-f362-4e13-8b03-cac9f63a5410" alt="4" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/820cd365-1737-4126-89aa-af9b120a0c22" alt="5" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/de460bd2-c41a-4888-ad0c-735d0358787b" alt="5" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/32bba4ef-ebd2-4a58-a574-352f22ba64b9" alt="6" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/1cb3b685-b4cf-4455-b421-623a7ccb39c0" alt="6" width="100%" controls></video></td>
    </tr>
    <tr>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/59a77c01-e237-4cdc-8099-c8cd2e364b70" alt="7" width="100%" controls></video></td>
      <td align="center" width="50%"><video src="https://github.com/user-attachments/assets/f692a251-0f63-48eb-a51e-cdb0deaedcd7" alt="7" width="100%" controls></video></td>
    </tr>
  </tbody>
</table>
