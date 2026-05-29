<div align="center">

# TurboT2AV

Fast text-to-audio-video generation distilled from LTX-2 19B.

</div>

## Overview

TurboT2AV generates synchronized audio-video from text prompts in 4 steps.
This repository provides single-GPU inference for the distilled checkpoint.

Main contributions:

- Combines the diversity of consistency models (DCM/SCM) with the high
  perceptual quality of score-model distillation (DMD), taking advantage of both
  families of methods.
- Shows that CM, as a forward-divergence offline method, can complement DMD, a
  reverse-KL on-policy method.
- First extends this combined distillation strategy to a large-scale joint
  audio-video generation model at the 14B-video + 5B-audio scale.

<table>
  <thead>
    <tr>
      <th align="center" width="34%">Prompt</th>
      <th align="center" width="33%">Teacher</th>
      <th align="center" width="33%">Student</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="34%">The woman with long brown hair and wearing white pants and a white jacket is performing on stage, singing into a microphone. As she sings, her voice is accompanied by background music, showcasing her talent as a female vocalist.</td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/teacher_p5.mp4" alt="teacher_p5" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/student_p5.mp4" alt="student_p5" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">The man is seen sitting and playing an acoustic guitar in front of a green shirt hanging on a white wall, the sound of the guitar being strummed and finger-picked with various chords and melodies fills the air.</td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/teacher_p6.mp4" alt="teacher_p6" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/student_p6.mp4" alt="student_p6" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">As the man speaks, the soothing sound of birds flapping their wings fills the background. In the video, a close-up of an animal surrounded by lush, green grass in a peaceful field is shown. The animal appears calm and observes its natural surroundings, creating a serene and tranquil scene.</td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/teacher_p73.mp4" alt="teacher_p73" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/student_p73.mp4" alt="student_p73" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">Pigeons are cooing and flapping their wings while the wind blows and traffic passes by. In the video, a flock of white and black pigeons can be seen standing closely together. The pigeons appear calm and seem to be pecking at the ground for food. The birds vary in color, with some being predominantly white and others predominantly black. They are standing in a public area, possibly a park or square. The pigeons exhibit typical behaviors such as grooming themselves and fluttering their wings. Overall, the scene captures a peaceful coexistence between the white and black pigeons.</td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/teacher_p79.mp4" alt="teacher_p79" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/student_p79.mp4" alt="student_p79" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">The sound of a truck accelerating can be heard, along with the sound of its tires squealing as it speeds up. A man is seen standing next to a garbage truck parked in front of a house on a driveway.</td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/teacher_p92.mp4" alt="teacher_p92" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/student_p92.mp4" alt="student_p92" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">The man wearing a camouflage shirt and hat sits in front of a sign as he speaks for a few minutes before a horn is blown, interrupting him.</td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/teacher_p99.mp4" alt="teacher_p99" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/student_p99.mp4" alt="student_p99" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">The speaker is a woman, wearing a pink top with a butterfly design on the chest, stands in front of a camera.</td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/teacher_p165.mp4" alt="teacher_p165" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://cdn.jsdelivr.net/gh/liuyuxiang1021/turbo-t2av@main/assets/videos/demo/student_p165.mp4" alt="student_p165" width="320" controls></video></td>
    </tr>
  </tbody>
</table>

## 1. Setup

```bash
git clone https://github.com/liuyuxiang1021/turbo-t2av.git
cd turbo-t2av/LTX-2
pixi install
pixi run pip install -e packages/ltx-core
pixi run pip install -e packages/ltx-pipelines
pixi run pip install -e packages/ltx-causal
pixi run pip install -e packages/ltx-distillation
```

## 2. Download Weights

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

Set environment variables:

```bash
export TURBO_CHECKPOINT_PATH=/path/to/ltx-2-19b-dev.safetensors
export TURBO_GEMMA_PATH=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

### Teacher (40-step reference)

```bash
cd LTX-2
PYTHONPATH=packages/ltx-distillation/src:packages/ltx-core/src:packages/ltx-pipelines/src:$PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  pixi run python -m ltx_distillation.tools.run_av_inference_eval \
  --config_path packages/ltx-distillation/configs/bidirectional_rcm.yaml \
  --prompts_file /path/to/prompts.csv \
  --output_dir /path/to/teacher_output \
  --model_kind teacher \
  --teacher_mode native_rf \
  --teacher_steps 40 \
  --num_prompts 8
```

### Student (4-step)

```bash
cd LTX-2
PYTHONPATH=packages/ltx-distillation/src:packages/ltx-core/src:packages/ltx-pipelines/src:$PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  pixi run python -m ltx_distillation.tools.run_av_inference_eval \
  --config_path packages/ltx-distillation/configs/bidirectional_rcm.yaml \
  --prompts_file /path/to/prompts.csv \
  --output_dir /path/to/student_output \
  --model_kind student \
  --student_checkpoint /path/to/checkpoint.pt \
  --student_param auto \
  --num_prompts 8
```

`--prompts_file` supports CSV (`video_id,prompt`) or plain text (one prompt per line).
