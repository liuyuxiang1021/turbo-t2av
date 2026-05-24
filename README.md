<div align="center">

# turbo-t2av

<img src="static/images/teaser_2.png" width="100%">

<a href="https://github.com/liuyuxiang1021/TurboT2AV"><img src="https://img.shields.io/badge/GitHub-TurboT2AV-blue.svg" alt="GitHub"></a>

</div>

**turbo-t2av** is a text-to-audio-video distillation project built around LTX-2 components. The current script surface is intentionally limited to bidirectional distillation: DCM warmup, SCM, DMD, and rCM-style joint SCM+DMD.

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
| `gemma-3-12b-it-qat-q4_0-unquantized` | `gemma_path` |

WandB credentials should be passed through the environment, not committed into configs:

```bash
export WANDB_API_KEY=...
```

## Distillation Data

The distillation recipes use two data inputs:

| Input | Config key | Used by |
| --- | --- | --- |
| Prompt text file, one prompt per line | `data_path` | DCM, DMD, rCM |
| SCM latent LMDB/root | `scm_data_path` | SCM, rCM |

Before launching training, edit the selected config and point these fields to existing local data:

```yaml
checkpoint_path: /path/to/ltx-2-19b-dev.safetensors
gemma_path: /path/to/gemma-3-12b-it-qat-q4_0-unquantized
data_path: /path/to/prompts.txt
scm_data_path: /path/to/scm_latent_lmdb_or_root
output_path: /path/to/outputs
```

## Distillation Training

Run commands from `LTX-2/packages/ltx-distillation`.

The unified launcher is `./scripts/train_bidirectional.sh`. The short wrapper scripts call the same launcher:

| Mode | Wrapper |
| --- | --- |
| DCM warmup | `./scripts/train_dcm.sh` |
| SCM only | `./scripts/train_scm.sh` |
| DMD only | `./scripts/train_dmd.sh` |
| rCM-style SCM + DMD | `./scripts/train_rcm.sh` |

`./scripts/train_scm_dmd.sh` is kept as a compatibility alias for `./scripts/train_rcm.sh`.

Basic launch:

```bash
cd LTX-2/packages/ltx-distillation

NUM_GPUS=8 MASTER_PORT=29500 ./scripts/train_dcm.sh
NUM_GPUS=8 MASTER_PORT=29501 ./scripts/train_scm.sh
NUM_GPUS=8 MASTER_PORT=29502 ./scripts/train_dmd.sh
NUM_GPUS=8 MASTER_PORT=29503 ./scripts/train_rcm.sh
```

You can also pass an explicit config:

```bash
NUM_GPUS=8 MASTER_PORT=29510 \
./scripts/train_bidirectional.sh scm /path/to/config.yaml
```

## DCM As Warmup

SCM, DMD, and rCM can all start from a DCM checkpoint. Treat DCM as the warmup stage and pass the DCM checkpoint through `DCM_CHECKPOINT`.

The launcher injects these config values into a temporary YAML:

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
./scripts/train_rcm.sh
```

For multi-node jobs, set the same launcher variables used by `torchrun`:

```bash
NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 NUM_GPUS=8 \
./scripts/train_rcm.sh
```

Distillation checkpoints are saved as:

```text
<output_path>/<run_dir>/checkpoints/checkpoint_XXXXXX/model.pth
```

## Kept Scripts

Only distillation scripts are kept in `LTX-2/packages/ltx-distillation/scripts/`:

```text
train_bidirectional.sh
train_dcm.sh
train_scm.sh
train_dmd.sh
train_rcm.sh
train_scm_dmd.sh
```

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
