<div align="center">

# TurboT2AV

<a href="https://github.com/liuyuxiang1021/TurboT2AV"><img src="https://img.shields.io/badge/GitHub-TurboT2AV-blue.svg" alt="GitHub"></a>

</div>

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

Prepare the distillation data from the Seedance dance packet:

| Input | Config key | Used by |
| --- | --- | --- |
| Prompt text file, one prompt per line | `data_path` | DCM, DMD, rCM |
| SCM latent LMDB/root | `scm_data_path` | SCM, rCM |

### Convert Videos To SCM Latents

SCM and rCM do not read raw videos directly during training. They read a latent LMDB generated from real video/audio samples. LMDB is a local key-value database; here it stores the pre-encoded video/audio latents and prompts so training can load them quickly.

The maintained packet layout is shown below. The path is an example; replace `/path/to/packet` with the local packet directory.

```text
/path/to/packet/
├── mapping.csv
└── dance_dataset/
    ├── 001.mp4
    ├── 002.mp4
    └── ...
```

The pipeline is:

```text
packet/mapping.csv + packet/dance_dataset/*.mp4 -> manifest.jsonl + prompts.txt -> SCM latent LMDB
```

By default, `manifest.jsonl` and `prompts.txt` are written under `/path/to/packet`.

Build the manifest and prompt file:

```bash
cd LTX-2

pixi run python -m ltx_distillation.tools.prepare_seedance_packet \
  --packet_root /path/to/packet
```

Then encode the videos/audio into SCM latents:

```bash
pixi run python -m ltx_distillation.tools.create_scm_latent_lmdb \
  --manifest_path /path/to/packet/manifest.jsonl \
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
data_path: /path/to/packet/prompts.txt
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
data_path: /path/to/packet/prompts.txt
scm_data_path: /path/to/scm_latent_lmdb
output_path: /path/to/outputs
wandb_api_key: ""  # optional; fill only when WandB login is needed
```

WandB logging is optional. If the machine is already logged in, leave `wandb_api_key` empty. If a run needs an explicit key, put it in the config you are launching, for example `LTX-2/packages/ltx-distillation/configs/bidirectional_rcm.yaml`:

```yaml
wandb_project: TurboT2AV
wandb_entity: liuyuxiang1021-tianjin-university
wandb_name: bidirectional_rcm
wandb_api_key: "wandb_..."
```

Use these four configs for the standard modes:

| Mode | Config |
| --- | --- |
| DCM warmup | `configs/bidirectional_dcm.yaml` |
| SCM only | `configs/bidirectional_scm.yaml` |
| DMD only | `configs/bidirectional_dmd.yaml` |
| rCM-style SCM + DMD | `configs/bidirectional_rcm.yaml` |

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

Standalone mode commands are also available:

```bash
NUM_GPUS=8 MASTER_PORT=29500 ./scripts/train_dcm.sh
NUM_GPUS=8 MASTER_PORT=29501 ./scripts/train_scm.sh
NUM_GPUS=8 MASTER_PORT=29502 ./scripts/train_dmd.sh
NUM_GPUS=8 MASTER_PORT=29503 ./scripts/train_rcm.sh
```

To pass a specific config:

```bash
NUM_GPUS=8 MASTER_PORT=29510 \
./scripts/train_bidirectional.sh scm /path/to/config.yaml
```

SCM, DMD, and rCM can start from an earlier checkpoint by setting `INIT_CHECKPOINT`, `WARMUP_CHECKPOINT`, or `DCM_CHECKPOINT`. The launcher writes a temporary config with:

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

WARMUP_CHECKPOINT=/path/to/scm1000_from_dcm500/checkpoints/checkpoint_001000/model.pth \
NUM_GPUS=8 MASTER_PORT=29603 \
./scripts/train_rcm.sh
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

## Script Reference

Distillation entrypoints live in `LTX-2/packages/ltx-distillation/scripts/`:

```text
train_bidirectional.sh
train_default_distillation.sh
train_dcm.sh
train_scm.sh
train_dmd.sh
train_rcm.sh
train_scm_dmd.sh
```

Single-GPU inference lives in:

```text
LTX-2/scripts/run_inference_single_gpu.sh
```

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

## Acknowledgements

TurboT2AV builds on:

- [rCM](https://github.com/NVlabs/rcm)
- [OmniForcing](https://github.com/OmniForcing/OmniForcing)
- [LTX-2](https://github.com/Lightricks/LTX-2)
