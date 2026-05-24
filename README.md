<div align="center">

# turbo-t2av

<img src="static/images/teaser_2.png" width="100%">

<a href="https://github.com/liuyuxiang1021/TurboT2AV"><img src="https://img.shields.io/badge/GitHub-TurboT2AV-blue.svg" alt="GitHub"></a>

</div>

**turbo-t2av** is a text-to-audio-video training and distillation project built around LTX-2 components. The repository contains data preparation tools, bidirectional DCM/SCM/DMD distillation recipes, causal ODE training, causal/self-forcing DMD training, and standalone inference scripts.

## Setup

```bash
git clone https://github.com/liuyuxiang1021/TurboT2AV.git
cd turbo-t2av/LTX-2

# Recommended when uv/pixi is available.
uv sync

# Or install the local packages into an existing Python environment.
pip install -e packages/ltx-core
pip install -e packages/ltx-pipelines
pip install -e packages/ltx-causal
pip install -e packages/ltx-distillation
```

Download the base assets and update the paths in the YAML files under `LTX-2/packages/ltx-distillation/configs/`:

| Asset | Default config key |
| --- | --- |
| `ltx-2-19b-dev.safetensors` | `checkpoint_path` |
| `gemma-3-12b-it-qat-q4_0-unquantized` | `gemma_path` or `text_encoder_checkpoint` |

WandB credentials should be passed through the environment, not committed into configs:

```bash
export WANDB_API_KEY=...
```

## Data Flow

Run these commands from `LTX-2/packages/ltx-distillation`.

1. Prepare prompts.

```bash
cd ../pe
python batch_enhance.py captions.txt --duration 5s
python enhance_prompts_light.py --input captions.txt --output prompts.txt
cd ../ltx-distillation
```

2. Reconstruct/download a video-caption dataset when using TAVGBench-style sources.

```bash
CAPTIONS_FILE=/path/to/release_captions.txt \
OUTPUT_DIR=/path/to/tavgbench \
./scripts/reconstruct_tavgbench_dataset.sh

CAPTIONS_FILE=/path/to/release_captions.txt \
VIDEO_DIR=/path/to/tavgbench/video_clips \
OUTPUT_FILE=/path/to/tavgbench/turbo-t2av_video_caption_manifest.jsonl \
./scripts/build_video_caption_manifest.sh
```

3. Build SCM latent data from real video/audio samples.

```bash
MANIFEST_PATH=/path/to/tavgbench/turbo-t2av_video_caption_manifest.jsonl \
OUTPUT_LMDB=/path/to/turbo-t2av_latent_100k \
CHECKPOINT_PATH=/path/to/ltx-2-19b-dev.safetensors \
./scripts/create_scm_latent_lmdb_8gpu.sh

LMDB_ROOT=/path/to/turbo-t2av_latent_100k \
CHECKPOINT_PATH=/path/to/ltx-2-19b-dev.safetensors \
./scripts/verify_scm_latent_decode.sh
```

4. Optionally build pseudo-SCM latents from teacher generations.

```bash
PROMPTS_FILE=/path/to/prompts.txt \
OUTPUT_LMDB=/path/to/scm_latent_teacher_native_rf_100000_shards \
PREVIEW_DIR=/path/to/scm_latent_teacher_preview \
./scripts/launch_teacher_scm_latent_100000_8gpu.sh

SHARDS_ROOT=/path/to/scm_latent_teacher_native_rf_100000_shards \
OUTPUT_LMDB=/path/to/scm_latent_teacher_native_rf_100000 \
./scripts/merge_teacher_scm_latent_shards.sh
```

5. Build ODE data for causal training.

```bash
TEACHER_CHECKPOINT=/path/to/bidirectional/checkpoints/checkpoint_XXXXXX/model.pth \
GEMMA_PATH=/path/to/gemma-3-12b-it-qat-q4_0-unquantized \
PROMPTS_FILE=/path/to/prompts.txt \
OUTPUT_DIR=/path/to/ode_pairs \
./scripts/generate_ode_pairs.sh

DATA_PATH=/path/to/ode_pairs \
LMDB_PATH=/path/to/ode_lmdb \
./scripts/create_ode_lmdb.sh
```

## Bidirectional Training

The unified launcher is `./scripts/train_bidirectional.sh`. The short wrapper scripts call the same launcher:

| Mode | Wrapper | Default config |
| --- | --- | --- |
| DCM warmup | `./scripts/train_dcm.sh` | `configs/stage1_bidirectional_dcm.yaml` |
| SCM only | `./scripts/train_scm.sh` | `configs/stage1_bidirectional_scm.yaml` |
| DMD only | `./scripts/train_dmd.sh` | `configs/stage1_bidirectional_dmd.yaml` |
| SCM + DMD | `./scripts/train_scm_dmd.sh` | `configs/stage1_bidirectional_scm_dmd.yaml` |

Basic launch:

```bash
cd LTX-2/packages/ltx-distillation

NUM_GPUS=8 MASTER_PORT=29500 ./scripts/train_dcm.sh
NUM_GPUS=8 MASTER_PORT=29501 ./scripts/train_scm.sh
NUM_GPUS=8 MASTER_PORT=29502 ./scripts/train_dmd.sh
NUM_GPUS=8 MASTER_PORT=29503 ./scripts/train_scm_dmd.sh
```

SCM, DMD, and SCM+DMD can all start from a DCM checkpoint. Treat DCM as the warmup stage and pass the DCM checkpoint through `DCM_CHECKPOINT`. The launcher injects:

```yaml
resume_checkpoint: <DCM_CHECKPOINT>
checkpoint_load_mode: parallel
resume_training_state: false
skip_initial_checkpoint: true
```

Examples:

```bash
DCM_CHECKPOINT=/path/to/dcm_run/checkpoints/checkpoint_001000/model.pth \
NUM_GPUS=8 MASTER_PORT=29601 \
./scripts/train_scm.sh

DCM_CHECKPOINT=/path/to/dcm_run/checkpoints/checkpoint_001000/model.pth \
NUM_GPUS=8 MASTER_PORT=29602 \
./scripts/train_dmd.sh

DCM_CHECKPOINT=/path/to/dcm_run/checkpoints/checkpoint_001000/model.pth \
NUM_GPUS=8 MASTER_PORT=29603 \
./scripts/train_scm_dmd.sh
```

For multi-node jobs, set the same launcher variables used by `torchrun`:

```bash
NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 NUM_GPUS=8 \
./scripts/train_scm_dmd.sh
```

Bidirectional checkpoints are saved as:

```text
<output_path>/<run_dir>/checkpoints/checkpoint_XXXXXX/model.pth
```

## Causal Training

After a bidirectional model is trained, generate ODE pairs and LMDB data, then train the causal ODE model:

```bash
./scripts/train_stage2_causal_ode.sh configs/stage2_causal_ode.yaml
```

Update `configs/stage2_causal_ode.yaml` before launching:

```yaml
checkpoint_path: /path/to/ltx-2-19b-dev.safetensors
bidirectional_model_checkpoint: /path/to/bidirectional/checkpoints/checkpoint_XXXXXX/model.pth
text_encoder_checkpoint: /path/to/gemma-3-12b-it-qat-q4_0-unquantized
data_path: /path/to/ode_lmdb
```

Causal ODE checkpoints are saved as:

```text
<output_path>/checkpoints/checkpoint_XXXXXX/model.pt
```

Then train causal/self-forcing DMD:

```bash
./scripts/train_stage3_causal_dmd.sh configs/stage3_causal_dmd.yaml
```

Update `configs/stage3_causal_dmd.yaml`:

```yaml
checkpoint_path: /path/to/ltx-2-19b-dev.safetensors
gemma_path: /path/to/gemma-3-12b-it-qat-q4_0-unquantized
stage1_ckpt_path: /path/to/stage2_causal_ode/checkpoints/checkpoint_XXXXXX/model.pt
generator_ckpt: /path/to/bidirectional/checkpoints/checkpoint_XXXXXX/model.pth
bootstrap_bidirectional_ckpt_path: /path/to/bidirectional/checkpoints/checkpoint_XXXXXX/model.pth
data_path: /path/to/ode_lmdb
benchmark_prompt_file: /path/to/prompts.txt
```

## Inference

Use `./scripts/infer_av.sh` for standalone inference. Student inference requires a trained checkpoint:

```bash
STUDENT_CHECKPOINT=/path/to/run/checkpoints/checkpoint_XXXXXX/model.pth \
CONFIG_PATH=configs/stage1_bidirectional_dmd.yaml \
PROMPTS_FILE=/path/to/prompts.txt \
OUTPUT_DIR=./outputs/infer_student \
NUM_PROMPTS=8 \
./scripts/infer_av.sh student
```

Teacher/base-model inference:

```bash
MODEL_KIND=teacher \
CONFIG_PATH=configs/stage1_bidirectional_dmd.yaml \
PROMPTS_FILE=/path/to/prompts.txt \
OUTPUT_DIR=./outputs/infer_teacher \
TEACHER_MODE=native_rf \
TEACHER_STEPS=40 \
NUM_PROMPTS=8 \
./scripts/infer_av.sh
```

For sharded inference:

```bash
NUM_SHARDS=8 SHARD_ID=0 CUDA_VISIBLE_DEVICES=0 ./scripts/infer_av.sh student
NUM_SHARDS=8 SHARD_ID=1 CUDA_VISIBLE_DEVICES=1 ./scripts/infer_av.sh student
```

Outputs are written as `sample_XXXX.mp4` plus prompt index files in `OUTPUT_DIR`.

## Repository Structure

```text
turbo-t2av/
├── README.md
├── static/
└── LTX-2/
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

## Acknowledgements

turbo-t2av builds on LTX-2, Self-Forcing, CausVid, and DMD-style distillation work.
