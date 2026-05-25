<div align="center">

# TurboT2AV

<a href="https://github.com/liuyuxiang1021/TurboT2AV"><img src="https://img.shields.io/badge/GitHub-TurboT2AV-blue.svg" alt="GitHub"></a>

</div>

| Prompt | Teacher (50-step) | SCM+DMD (4-step) |
|:---|:---:|:---:|
| The woman with long brown hair and wearing white pants and a white jacket is performing on stage, singing into a microphone. As she sings, her voice is accompanied by background music, showcasing her talent as a female vocalist. | <video src="https://github.com/user-attachments/assets/45db5c23-fbaa-4c71-bc51-c1785a6c87eb" width="320" height="180" controls></video> | <video src="https://github.com/user-attachments/assets/30784eb1-634a-4eee-a718-3961f7077e72" width="320" height="180" controls></video> |
| The man is seen sitting and playing an acoustic guitar in front of a green shirt hanging on a white wall, the sound of the guitar being strummed and finger-picked with various chords and melodies fills the air. | <video src="https://github.com/user-attachments/assets/03c7ceb9-a557-40ef-a8c9-984f790ae893" width="320" height="180" controls></video> | <video src="https://github.com/user-attachments/assets/408dcc48-8c83-42ee-8008-fd362d2b752d" width="320" height="180" controls></video> |
| As the man speaks, the soothing sound of birds flapping their wings fills the background. In the video, a close-up of an animal surrounded by lush, green grass in a peaceful field is shown. The animal appears calm and observes its natural surroundings, creating a serene and tranquil scene. | <video src="https://github.com/user-attachments/assets/abfcc196-530b-472d-99a0-0f1c1c9126e6" width="320" height="180" controls></video> | <video src="https://github.com/user-attachments/assets/f85d4bd0-c25e-4a21-a6ef-8dc7d3bc113e" width="320" height="180" controls></video> |
| Pigeons are cooing and flapping their wings while the wind blows and traffic passes by. In the video, a flock of white and black pigeons can be seen standing closely together. The pigeons appear calm and seem to be pecking at the ground for food. The birds vary in color, with some being predominantly white and others predominantly black. They are standing in a public area, possibly a park or square. The pigeons exhibit typical behaviors such as grooming themselves and fluttering their wings. Overall, the scene captures a peaceful coexistence between the white and black pigeons. | <video src="https://github.com/user-attachments/assets/0b3a7855-a4e8-484f-9f0c-6221d4306099" width="320" height="180" controls></video> | <video src="https://github.com/user-attachments/assets/159bdcd8-bcc4-443d-a495-5248934cf107" width="320" height="180" controls></video> |
| The sound of a truck accelerating can be heard, along with the sound of its tires squealing as it speeds up. A man is seen standing next to a garbage truck parked in front of a house on a driveway. | <video src="https://github.com/user-attachments/assets/f9d8e1f4-d5a6-4c4e-b697-cf569e03b4b4" width="320" height="180" controls></video> | <video src="https://github.com/user-attachments/assets/4f30969c-b979-4981-80b4-b8244bf207e2" width="320" height="180" controls></video> |
| The man wearing a camouflage shirt and hat sits in front of a sign as he speaks for a few minutes before a horn is blown, interrupting him. | <video src="https://github.com/user-attachments/assets/b5b4c3fc-b139-4fa4-ab25-d69f2c343e0c" width="320" height="180" controls></video> | <video src="https://github.com/user-attachments/assets/edaae47a-de4f-40cc-b469-b46bd78967d1" width="320" height="180" controls></video> |
| As the woman speaks in the audio, the serene animation of a pool filled with clear and pristine blue water is displayed. The water flows gracefully out of the pool, creating gentle ripples on the surface, while the music plays in the background, adding to the tranquil atmosphere of the scene. | <video src="https://github.com/user-attachments/assets/d8a5d98c-53e7-4725-8c3e-d8c6caa34b43" width="320" height="180" controls></video> | <video src="https://github.com/user-attachments/assets/d319b7d9-f1d7-4cb9-ad20-b7ab53799321" width="320" height="180" controls></video> |
| The speaker is a woman, wearing a pink top with a butterfly design on the chest, stands in front of a camera. | <video src="https://github.com/user-attachments/assets/621486c1-d56d-4be5-9a83-0f5baeaab3d1" width="320" height="180" controls></video> | <video src="https://github.com/user-attachments/assets/e98289ce-01ed-4a10-b5f0-4815b973a212" width="320" height="180" controls></video> |

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
