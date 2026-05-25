<div align="center">

# TurboT2AV

<a href="https://github.com/liuyuxiang1021/TurboT2AV"><img src="https://img.shields.io/badge/GitHub-TurboT2AV-blue.svg" alt="GitHub"></a>

</div>

## TurboT2AV — Consistency Distillation Demos

DCM warmup → SCM → SCM+DMD on LTX-2 19B AV. [Full gallery →](https://liuyuxiang1021.github.io/TurboT2AV.github.io/)

| Prompt | DCM500 | SCM1000 | DMD-only | SCM+DMD |
|:---|:---:|:---:|:---:|:---:|
| Prompt 5 | <video src="https://github.com/user-attachments/assets/6d8c1b6a-b0dd-437f-a2d6-15e76660a9e4" controls></video> | <video src="https://github.com/user-attachments/assets/f8e85d37-0f3c-4e9e-b7cc-0bd20acb9423" controls></video> | <video src="https://github.com/user-attachments/assets/26fdf548-fa0f-41ac-a029-333ad2683bf0" controls></video> | <video src="https://github.com/user-attachments/assets/b24a7826-10b8-46a2-806d-2a8a7239cd5b" controls></video> |
| Prompt 6 | <video src="https://github.com/user-attachments/assets/108a7e57-a6a7-427a-8054-9fccfe014139" controls></video> | <video src="https://github.com/user-attachments/assets/f2b4fb0e-eaac-4afe-8a12-48eb266eb5aa" controls></video> | <video src="https://github.com/user-attachments/assets/847ec948-25ba-44a6-a759-8ed990030d64" controls></video> | <video src="https://github.com/user-attachments/assets/beaa0569-1671-4382-8e3b-94ada731bdef" controls></video> |
| Prompt 73 | <video src="https://github.com/user-attachments/assets/d90a43cb-f426-490e-87fa-1ae4286eb683" controls></video> | <video src="https://github.com/user-attachments/assets/7fc68b85-a43f-407e-a918-a41b10707aab" controls></video> | <video src="https://github.com/user-attachments/assets/d1935401-cb50-463d-9b91-f20ec99055e5" controls></video> | <video src="https://github.com/user-attachments/assets/a2d7a977-8ae6-4527-a87b-2261b5e5df75" controls></video> |
| Prompt 79 | <video src="https://github.com/user-attachments/assets/6241fec0-62c7-449c-a64f-c86a63bbf4b0" controls></video> | <video src="https://github.com/user-attachments/assets/f1ec4f83-b171-4341-92de-b2d5caad92c3" controls></video> | <video src="https://github.com/user-attachments/assets/ff249534-04cc-4936-bf31-4b88d1304905" controls></video> | <video src="https://github.com/user-attachments/assets/0112b842-abee-4f64-bfd3-ee4a9f674955" controls></video> |
| Prompt 92 | <video src="https://github.com/user-attachments/assets/26f3c929-736c-48e7-8a8a-6918672e8c71" controls></video> | <video src="https://github.com/user-attachments/assets/e8c24bd1-7d78-4da0-83ea-63751201df0c" controls></video> | <video src="https://github.com/user-attachments/assets/36908aae-a8b0-474f-827f-62d7984269e3" controls></video> | <video src="https://github.com/user-attachments/assets/caa3a56c-b33e-4d31-aced-41aaf9f0ad90" controls></video> |
| Prompt 99 | <video src="https://github.com/user-attachments/assets/4af594c8-7c24-496d-843d-f27d470d3cb7" controls></video> | <video src="https://github.com/user-attachments/assets/428e7505-98f3-422d-b48c-2b9e717df6e3" controls></video> | <video src="https://github.com/user-attachments/assets/2a326ebe-65a2-451a-a1fb-3f589f160326" controls></video> | <video src="https://github.com/user-attachments/assets/1e08cecc-dd26-4ca9-8b32-b39e906ec2d1" controls></video> |
| Prompt 108 | <video src="https://github.com/user-attachments/assets/07d46837-3068-4e60-925f-726d3a941e49" controls></video> | <video src="https://github.com/user-attachments/assets/77051699-8a8e-465c-8ebe-8d0ba94b5882" controls></video> | <video src="https://github.com/user-attachments/assets/cfc73e14-616f-436c-b6a7-dec7743f9984" controls></video> | <video src="https://github.com/user-attachments/assets/b583d52b-ee51-4687-b6cf-8cc332c61e6c" controls></video> |
| Prompt 165 | <video src="https://github.com/user-attachments/assets/45f5072e-7982-4ea3-8f0a-a789bafbe8cc" controls></video> | <video src="https://github.com/user-attachments/assets/d04c0c4b-fd9a-428c-b2e0-0aa65b2914c0" controls></video> | <video src="https://github.com/user-attachments/assets/57815744-7cd2-4568-a805-2b4901b8778f" controls></video> | <video src="https://github.com/user-attachments/assets/fe5ec9a1-67dd-46a0-8d65-14309982aa00" controls></video> |



**TurboT2AV** is a text-to-audio-video distillation project built around LTX-2 components. The maintained workflow is bidirectional distillation: DCM warmup, SCM, DMD, and rCM-style joint SCM+DMD.

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

Download the base assets and keep their paths available for the config files:

| Asset | Config key |
| --- | --- |
| `ltx-2-19b-dev.safetensors` | `checkpoint_path` |
| `gemma-3-12b-it-qat-q4_0-unquantized` | `gemma_path` |

Prepare the distillation data from a processed video dataset:

| Input | Config key | Used by |
| --- | --- | --- |
| Prompt text file, one prompt per line | `data_path` | DCM, DMD, rCM |
| SCM latent LMDB/root | `scm_data_path` | SCM, rCM |

### Convert Videos To SCM Latents

SCM and rCM do not read raw videos directly during training. They read a latent LMDB generated from real video/audio samples. LMDB is a local key-value database; here it stores the pre-encoded video/audio latents and prompts so training can load them quickly.

The CSV-based layout below is an example of the maintained input format. Replace `/path/to/dataset` with the local dataset directory.

```text
/path/to/dataset/
├── mapping.csv
└── videos/
    ├── 001.mp4
    ├── 002.mp4
    └── ...
```

The pipeline is:

```text
dataset/mapping.csv + dataset/videos/*.mp4 -> manifest.jsonl + prompts.txt -> SCM latent LMDB
```

By default, `manifest.jsonl` and `prompts.txt` are written under `/path/to/dataset`.

Build the manifest and prompt file:

```bash
cd LTX-2

pixi run python -m ltx_distillation.tools.prepare_dataset \
  --dataset_root /path/to/dataset
```

Then encode the videos/audio into SCM latents:

```bash
pixi run python -m ltx_distillation.tools.create_scm_latent_lmdb \
  --manifest_path /path/to/dataset/manifest.jsonl \
  --output_lmdb /path/to/scm_latent_lmdb \
  --checkpoint_path /path/to/ltx-2-19b-dev.safetensors \
  --num_frames 121 \
  --video_height 512 \
  --video_width 768 \
  --video_fps 24 \
  --device cuda \
  --dtype bfloat16 \
  --batch_size 1 \
  --resume
```

Point the configs to the generated files:

```yaml
data_path: /path/to/dataset/prompts.txt
scm_data_path: /path/to/scm_latent_lmdb
```

Optionally decode a few latent samples to verify the LMDB before training:

```bash
pixi run python -m ltx_distillation.tools.verify_scm_latent_decode \
  --lmdb_root /path/to/scm_latent_lmdb \
  --checkpoint_path /path/to/ltx-2-19b-dev.safetensors \
  --output_dir /path/to/latent_decode_preview \
  --num_samples 8
```

The config directory intentionally keeps only the default bidirectional recipes:

```text
LTX-2/packages/ltx-distillation/configs/
├── bidirectional_dcm.yaml
├── bidirectional_scm.yaml
├── bidirectional_dmd.yaml
└── bidirectional_rcm.yaml
```

## 3. Edit The Configs

Before training, update the selected YAML under `LTX-2/packages/ltx-distillation/configs/`:

```yaml
checkpoint_path: /path/to/ltx-2-19b-dev.safetensors
gemma_path: /path/to/gemma-3-12b-it-qat-q4_0-unquantized
data_path: /path/to/dataset/prompts.txt
scm_data_path: /path/to/scm_latent_lmdb
output_path: /path/to/outputs
wandb_api_key: ""  # optional; fill only when WandB login is needed
```

WandB logging is optional. If the machine is already logged in, leave `wandb_api_key` empty. If a run needs an explicit key, put it in the config you are launching, for example `LTX-2/packages/ltx-distillation/configs/bidirectional_rcm.yaml`:

```yaml
wandb_project: your_wandb_project
wandb_entity: ""  # optional; leave empty for the default account/team
wandb_name: your_run_name
wandb_api_key: ""  # optional; fill only when explicit login is needed
```

Use these configs for the maintained training entrypoints:

| Mode | Config |
| --- | --- |
| Full default recipe | `configs/bidirectional_dcm.yaml`, `configs/bidirectional_scm.yaml`, `configs/bidirectional_rcm.yaml` |
| DCM warmup | `configs/bidirectional_dcm.yaml` |
| SCM only | `configs/bidirectional_scm.yaml` |
| DMD only | `configs/bidirectional_dmd.yaml` |

## 4. Start Training

Run training commands from `LTX-2/packages/ltx-distillation`:

```bash
cd LTX-2/packages/ltx-distillation
```

The default recipe is:

```text
DCM 500-step warmup -> SCM-only training to step 1000 -> full rCM training
```

In practice, starting SCM directly from the base model can be unstable during the early steps. The project default therefore first runs DCM for 500 steps as a discrete consistency warmup, then switches to SCM-only training until `checkpoint_001000`, and then initializes the full rCM run from that SCM checkpoint.

Launch the default recipe:

```bash
NUM_GPUS=8 MASTER_PORT=29500 ./scripts/train_default_distillation.sh
```

Useful overrides:

```bash
OUTPUT_PATH=/path/to/outputs \
PIPELINE_RUN_PREFIX=0524_TurboT2AV \
NUM_GPUS=8 MASTER_PORT=29500 \
./scripts/train_default_distillation.sh
```

The default recipe writes phase run directories under `output_path`:

```text
<prefix>_dcm500_warmup/checkpoints/checkpoint_000500/model.pth
<prefix>_scm1000_from_dcm500/checkpoints/checkpoint_001000/model.pth
<prefix>_rcm_from_scm1000/
```

Standalone DCM, SCM, and DMD commands are also available:

```bash
NUM_GPUS=8 MASTER_PORT=29500 ./scripts/train_dcm.sh
NUM_GPUS=8 MASTER_PORT=29501 ./scripts/train_scm.sh
NUM_GPUS=8 MASTER_PORT=29502 ./scripts/train_dmd.sh
```

To pass a specific config:

```bash
NUM_GPUS=8 MASTER_PORT=29510 \
./scripts/train_scm.sh /path/to/config.yaml
```

SCM and DMD can start from an earlier checkpoint by setting `INIT_CHECKPOINT`, `WARMUP_CHECKPOINT`, or `DCM_CHECKPOINT`. The launcher writes a temporary config with:

```yaml
resume_checkpoint: <CHECKPOINT>
checkpoint_load_mode: parallel
resume_training_state: false
skip_initial_checkpoint: true
```

Examples:

```bash
DCM_CHECKPOINT=/path/to/dcm500_warmup/checkpoints/checkpoint_000500/model.pth \
NUM_GPUS=8 MASTER_PORT=29601 \
./scripts/train_scm.sh

INIT_CHECKPOINT=/path/to/checkpoints/checkpoint_001000/model.pth \
NUM_GPUS=8 MASTER_PORT=29602 \
./scripts/train_dmd.sh
```

## 5. Run Inference

Single-GPU inference is kept in the original project layout:

```text
LTX-2/scripts/run_inference_single_gpu.sh
```

The default command sequentially evaluates the default comparison checkpoints on one GPU:

```bash
bash LTX-2/scripts/run_inference_single_gpu.sh 0 200
```

Arguments:

```text
0    GPU id
200  number of prompts
```

Useful overrides:

```bash
PYTHON_BIN=/path/to/python \
RUN_ROOT=/path/to/training_runs \
OUTPUT_ROOT=/path/to/inference_outputs \
PROMPTS_FILE=/path/to/prompts.txt \
bash LTX-2/scripts/run_inference_single_gpu.sh 0 200
```

Each model writes `sample_*.mp4`, `sample_*.json`, `samples.csv`, and `run.log` under `OUTPUT_ROOT/<index>_<name>/`. Separate `sample_*.wav` files are removed after MP4 writing.

## 6. Run Evaluation

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
