<div align="center">

# TurboT2AV

</div>

## Overview

TurboT2AV is a distillation workspace for audio-conditioned text-to-video generation based on LTX-2. It focuses on:

- Distilling an audio-conditioned text-to-video teacher into a faster student model.
- Keeping the maintained training path centered on DCM, SCM, DMD, and rCM-style SCM+DMD training.
- Using DCM as a 500-step warmup before SCM-only training to step 1000, then continuing with the full rCM-style objective.
- Supporting standalone DCM, SCM, and DMD launchers for debugging and ablation runs.
- Providing a single-GPU inference entrypoint for checking trained checkpoints with prompt files.

At a high level, the workflow is:

1. Set up the Pixi environment and install the local LTX packages.
2. Download the LTX-2 and Gemma assets, then prepare `prompts.txt` and the SCM latent LMDB from the video dataset.
3. Edit the YAML configs with the local asset, data, output, and optional WandB settings.
4. Start the default distillation recipe or one of the standalone training modes.
5. Run single-GPU inference from the maintained LTX-2 inference script.

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
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/45db5c23-fbaa-4c71-bc51-c1785a6c87eb" alt="teacher" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/6a110256-1eee-45c8-8bd6-e1166627fb99" alt="student_p5" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">The man is seen sitting and playing an acoustic guitar in front of a green shirt hanging on a white wall, the sound of the guitar being strummed and finger-picked with various chords and melodies fills the air.</td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/03c7ceb9-a557-40ef-a8c9-984f790ae893" alt="teacher" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/754cebcd-663f-4d2b-804e-54bc5f3921cd" alt="student_p5" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">As the man speaks, the soothing sound of birds flapping their wings fills the background. In the video, a close-up of an animal surrounded by lush, green grass in a peaceful field is shown. The animal appears calm and observes its natural surroundings, creating a serene and tranquil scene.</td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/abfcc196-530b-472d-99a0-0f1c1c9126e6" alt="teacher" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/9b37fee1-74a6-4767-8db0-0a1cd84cd0bf" alt="student_p5" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">Pigeons are cooing and flapping their wings while the wind blows and traffic passes by. In the video, a flock of white and black pigeons can be seen standing closely together. The pigeons appear calm and seem to be pecking at the ground for food. The birds vary in color, with some being predominantly white and others predominantly black. They are standing in a public area, possibly a park or square. The pigeons exhibit typical behaviors such as grooming themselves and fluttering their wings. Overall, the scene captures a peaceful coexistence between the white and black pigeons.</td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/0b3a7855-a4e8-484f-9f0c-6221d4306099" alt="teacher" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/a61c2047-146a-4cdb-af13-0c04613dfd5f" alt="student_p5" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">The sound of a truck accelerating can be heard, along with the sound of its tires squealing as it speeds up. A man is seen standing next to a garbage truck parked in front of a house on a driveway.</td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/f9d8e1f4-d5a6-4c4e-b697-cf569e03b4b4" alt="teacher" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/0d946d16-bb18-45c1-937b-d6f9a233f6f8" alt="student_p5" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">The man wearing a camouflage shirt and hat sits in front of a sign as he speaks for a few minutes before a horn is blown, interrupting him.</td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/b5b4c3fc-b139-4fa4-ab25-d69f2c343e0c" alt="teacher" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/aab1465f-7328-493f-b679-4f6d87e02e75" alt="student_p5" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">As the woman speaks in the audio, the serene animation of a pool filled with clear and pristine blue water is displayed. The water flows gracefully out of the pool, creating gentle ripples on the surface, while the music plays in the background, adding to the tranquil atmosphere of the scene.</td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/d8a5d98c-53e7-4725-8c3e-d8c6caa34b43" alt="teacher" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/b6af3013-bb33-4b00-8b2c-62b21821e4a3" alt="student_p5" width="320" controls></video></td>
    </tr>
    <tr>
      <td width="34%">The speaker is a woman, wearing a pink top with a butterfly design on the chest, stands in front of a camera.</td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/621486c1-d56d-4be5-9a83-0f5baeaab3d1" alt="teacher" width="320" controls></video></td>
      <td align="center" width="33%"><video src="https://github.com/user-attachments/assets/c7caf2db-700f-4b52-9466-2952ecfbcc9a" alt="student_p5" width="320" controls></video></td>
    </tr>
  </tbody>
</table>

## 1. Set Up The Environment

Clone the repo and enter the LTX-2 workspace:

```bash
git clone https://github.com/liuyuxiang1021/TurboT2AV.git
cd TurboT2AV/LTX-2
```

Create or update the Pixi environment:

```bash
pixi install
```

Install the local packages inside that Pixi environment:

```bash
pixi run pip install -e packages/ltx-core
pixi run pip install -e packages/ltx-pipelines
pixi run pip install -e packages/ltx-causal
pixi run pip install -e packages/ltx-distillation
```

## 2. Download Weights And Prepare Data

Download the required model weights from HuggingFace:

| Asset | HF Repo | Config key |
| --- | --- | --- |
| `ltx-2-19b-dev.safetensors` | [`Lightricks/LTX-2`](https://huggingface.co/Lightricks/LTX-2) | `checkpoint_path` |
| `gemma-3-12b-it-qat-q4_0-unquantized` | [`google/gemma-3-12b-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) | `gemma_path` |

```bash
export HF_TOKEN="your_huggingface_token"

huggingface-cli download Lightricks/LTX-2 \
    ltx-2-19b-dev.safetensors \
    --local-dir /path/to/checkpoints/LTX-2

huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir /path/to/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
```

Prepare the distillation data:

**mapping.csv** — one row per sample, header `video_id,prompt`:

```csv
video_id,prompt
001.mp4,"A dog barking in a park."
002.mp4,"A piano solo performance."
003.mp4,"A woman singing on stage."
```

- `video_id` — filename relative to `--video_dir` in the LMDB tool.
- `prompt` — free-text caption used for both LMDB creation and training/inference.

**SCM latent LMDB** — precomputed VAE latents for SCM/DCM training. Generate from mapping.csv:

```bash
python -m ltx_distillation.tools.create_scm_latent_lmdb \
  --mapping_csv /path/to/mapping.csv \
  --video_dir /path/to/videos/ \
  --checkpoint_path /path/to/ltx-2-19b-dev.safetensors \
  --output_lmdb /path/to/scm_latent_lmdb \
  --num_workers 8
```

## 3. Start Training

Set paths once via environment variables — no need to edit YAML files:

```bash
export TURBO_CHECKPOINT_PATH=/path/to/ltx-2-19b-dev.safetensors
export TURBO_GEMMA_PATH=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
export TURBO_DATA_PATH=/path/to/mapping.csv              # same CSV from Section 2
export TURBO_SCM_DATA_PATH=/path/to/scm_latent_lmdb
export TURBO_OUTPUT_PATH=/path/to/outputs
```

The default recipe is **DCM 500-step warmup → SCM-only to step 1000 → full rCM training**:

```bash
cd LTX-2/packages/ltx-distillation
NUM_GPUS=8 MASTER_PORT=29500 ./scripts/train_default_distillation.sh
```

Starting SCM directly from the base model can be unstable. The default therefore runs DCM for 500 steps as discrete consistency warmup, then SCM-only to step 1000, then rCM from that checkpoint.

The recipe produces three checkpoint directories under `output_path`:

```text
<prefix>_dcm500_warmup/checkpoints/checkpoint_000500/model.pth
<prefix>_scm1000_from_dcm500/checkpoints/checkpoint_001000/model.pth
<prefix>_rcm_from_scm1000/
```

## 4. Run Inference

### Teacher (40-step reference)

```bash
cd LTX-2
PYTHONPATH=packages/ltx-distillation/src:$PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  pixi run python -m ltx_distillation.tools.run_av_inference_eval \
  --config_path packages/ltx-distillation/configs/bidirectional_rcm.yaml \
  --prompts_file /path/to/mapping.csv \
  --output_dir /path/to/teacher_output \
  --model_kind teacher \
  --teacher_mode native_rf \
  --teacher_steps 40 \
  --num_prompts 8
```

### Student (4-step, single GPU)

```bash
cd LTX-2
PYTHONPATH=packages/ltx-distillation/src:$PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  pixi run python -m ltx_distillation.tools.run_av_inference_eval \
  --config_path packages/ltx-distillation/configs/bidirectional_rcm.yaml \
  --prompts_file /path/to/mapping.csv \
  --output_dir /path/to/student_output \
  --model_kind student \
  --student_checkpoint /path/to/checkpoint.pt \
  --student_param auto \
  --num_prompts 8
```

## 5. Run Evaluation

TBA. The repository currently keeps training and single-GPU inference entrypoints. A formal evaluation script and metric workflow still need to be added.

## Repository Structure

```text
TurboT2AV/
├── README.md
├── static/
└── LTX-2/
    ├── scripts/
    └── packages/
        ├── ltx-core/
        ├── ltx-causal/
        ├── ltx-pipelines/
        ├── pe/
        └── ltx-distillation/
            ├── configs/
            ├── scripts/
            └── src/ltx_distillation/
```
