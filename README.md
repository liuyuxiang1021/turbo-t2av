<div align="center">

# turbo-t2av

<img src="static/images/teaser_2.png" width="100%">

<a href="https://github.com/liuyuxiang1021/TurboT2AV"><img src="https://img.shields.io/badge/GitHub-TurboT2AV-blue.svg" alt="GitHub"></a>

</div>

**turbo-t2av** is a text-to-audio-video training and distillation project built around LTX-2 components. The repository contains causal model wrappers, distillation configs, data preparation scripts, and training code for bidirectional DMD, causal ODE regression, and self-forcing DMD workflows.


## Method Overview

<div align="center">
<img src="static/images/method.png" width="100%">
</div>

turbo-t2av employs a **three-stage distillation pipeline** to progressively transform the bidirectional teacher into a causal streaming engine:

- **Stage 1 -- Bidirectional DMD:** Distribution Matching Distillation compresses the multi-step diffusion sampling into few-step denoising, while preserving the original global attention.

- **Stage 2 -- Causal ODE Regression:** The model is equipped with our **Asymmetric Block-Causal Mask** and trained via ODE trajectory regression to adapt to causal attention. An **Audio Sink Token** mechanism with **Identity RoPE** is introduced to resolve the Softmax collapse and gradient explosion caused by extreme audio token sparsity.

- **Stage 3 -- Joint Self-Forcing DMD:** The model autoregressively unrolls its own generations during training, enabling dynamic self-correction of cumulative cross-modal errors from exposure bias. Two variants are provided:
  - **Self-Forcing DMD** (`main` branch): Autoregressive self-forcing rollout with DMD loss (recommended).
  - **Causal DMD** (`causal-dmd` branch): Block-wise DMD training without self-forcing rollout.

## Getting Started

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.2.0
- 8x or 32x GPUs with 96GB+ memory (H200 recommended). Causal DMD branch may work on lower-memory GPUs.

### Installation

```bash
git clone https://github.com/liuyuxiang1021/TurboT2AV.git
cd turbo-t2av/LTX-2

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e packages/ltx-core
pip install -e packages/ltx-pipelines
pip install -e packages/ltx-causal
pip install -e packages/ltx-distillation
```

### Download Models

Download the following pretrained models and update the paths in the config files:

| Model | Description |
|-------|-------------|
| `ltx-2-19b-dev.safetensors` | LTX-2 base model (19B), from [Lightricks/LTX-2](https://huggingface.co/collections/Lightricks/ltx-2) |
| `gemma-3-12b-it-qat-q4_0-unquantized` | Gemma 3 12B text encoder (unquantized QAT variant) |




## Training Pipeline

> **Note:** Support for LTX-2.3, improved inference pipeline, and future new work will be released soon. Multi-node launch scripts may need modification depending on your cluster scheduler (SLURM, etc.).

The training follows a three-stage pipeline. We recommend **32 GPUs** (4 nodes x 8 GPUs) for optimal performance. You can also train with **8 GPUs** by setting `gradient_accumulation_steps: 4` in the config.

### Stage 1: Bidirectional DMD

Distills the LTX-2 teacher model from multi-step to 4-step inference while preserving global attention.

**Data preparation:** Prepare a text prompts file (one prompt per line). You can use our prompt enhancement tools to expand short captions into detailed LTX-2 prompts:

```bash
cd LTX-2/packages/pe

# Option A: Heavy mode - vLLM + LLM (recommended, higher quality)
# First start a vLLM server with your preferred LLM:
#   vllm serve /path/to/your/llm --tensor-parallel-size 8
python batch_enhance.py captions.txt --duration 5s

# Option B: Light mode - local Gemma via vLLM (simpler, faster)
python enhance_prompts_light.py --input captions.txt --output prompts.txt
```

**Training:**

```bash
cd LTX-2/packages/ltx-distillation

# Edit configs/stage1_bidirectional_dmd.yaml with your paths, then:
./scripts/train_stage1_bidirectional_dmd.sh

# Or specify config explicitly:
./scripts/train_stage1_bidirectional_dmd.sh configs/stage1_bidirectional_dmd.yaml
```

### Stage 2: Causal ODE Regression

Converts the bidirectional model to causal autoregressive using ODE trajectory regression. This stage requires generating ODE trajectory pairs from the Stage 1 teacher.

**Step 1: Generate ODE trajectory pairs**

```bash
cd LTX-2/packages/ltx-distillation

# Single GPU:
TEACHER_CHECKPOINT=/path/to/stage1_checkpoint/model.pt \
GEMMA_PATH=/path/to/gemma \
PROMPTS_FILE=/path/to/prompts.txt \
OUTPUT_DIR=./ode_pairs \
    ./scripts/generate_ode_pairs.sh

# Multi-node (faster):
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
TEACHER_CHECKPOINT=/path/to/stage1_checkpoint/model.pt \
    ./scripts/generate_ode_pairs_multi_node.sh
```

**Step 2: Create LMDB dataset**

```bash
DATA_PATH=./ode_pairs LMDB_PATH=./ode_lmdb ./scripts/create_ode_lmdb.sh
```

**Step 3: Train**

```bash
# Edit configs/stage2_causal_ode.yaml with your paths, then:
./scripts/train_stage2_causal_ode.sh
```

### Stage 3: Causal DMD

Trains the causal autoregressive model with DMD loss using the ODE-initialized generator and bidirectional teacher/critic.

Two variants are available:

| Variant | Branch | Description |
|---------|--------|-------------|
| **Self-Forcing DMD** | `main` | Autoregressive self-forcing rollout with DMD loss (recommended) |
| **Causal DMD** | `causal-dmd` | Block-wise DMD training without self-forcing rollout |

**Training (Self-Forcing DMD, default):**

```bash
# Edit configs/stage3_causal_dmd.yaml with your paths, then:
./scripts/train_stage3_causal_dmd.sh
```

**Training (Causal DMD, alternative):**

```bash
git checkout causal-dmd
./scripts/train_stage3_causal_dmd.sh
```

### Hardware Recommendations

| Setup | GPUs | Config Change |
|-------|------|---------------|
| Recommended | 32 (4 nodes x 8) | Default settings |
| Minimum | 8 (1 node) | Set `gradient_accumulation_steps: 4` |

All training scripts auto-detect SLURM and multi-node environment variables. For multi-node training, set `NNODES`, `NODE_RANK`, and `MASTER_ADDR`.


## Repository Structure

```
turbo-t2av/
├── README.md
├── static/                              # Demo images and videos
└── LTX-2/
    └── packages/
        ├── ltx-core/                    # Base model components (transformer, VAE, text encoder)
        ├── ltx-causal/                  # Causal wrapper, attention masks, block-causal architecture
        ├── ltx-distillation/            # Training pipeline
        │   ├── configs/
        │   │   ├── stage1_bidirectional_dmd.yaml
        │   │   ├── stage2_causal_ode.yaml
        │   │   └── stage3_causal_dmd.yaml
        │   ├── scripts/
        │   │   ├── train_stage1_bidirectional_dmd.sh
        │   │   ├── train_stage2_causal_ode.sh
        │   │   ├── train_stage3_causal_dmd.sh
        │   │   ├── generate_ode_pairs.sh
        │   │   ├── generate_ode_pairs_multi_node.sh
        │   │   └── create_ode_lmdb.sh
        │   └── src/ltx_distillation/
        │       ├── train_distillation.py    # DMD training loop (Stages 1 & 3)
        │       ├── dmd.py                   # DMD model, loss, generator/critic
        │       ├── ode/                     # ODE regression (Stage 2)
        │       │   ├── train_ode.py
        │       │   ├── generate_ode_pairs.py
        │       │   └── create_lmdb.py
        │       ├── inference/               # Benchmark pipelines
        │       └── models/                  # Model wrappers
        ├── ltx-pipelines/               # Inference pipeline utilities
        └── pe/                          # Prompt enhancement tools
            ├── batch_enhance.py         # Heavy mode (vLLM + LLM, higher quality)
            ├── prompt_enhancer.py       # Heavy mode core (duration-aware)
            ├── enhance_prompts_light.py # Light mode (simpler, faster)
            └── light_system_prompt.txt  # Light mode system prompt
```


## Acknowledgements

turbo-t2av builds upon several outstanding works. We thank the authors of [LTX-2](https://github.com/Lightricks/LTX-2), [Self-Forcing](https://github.com/guandeh17/Self-Forcing), [CausVid](https://github.com/tianweiy/CausVid), and [DMD](https://github.com/tianweiy/DMD2) for their pioneering contributions.
