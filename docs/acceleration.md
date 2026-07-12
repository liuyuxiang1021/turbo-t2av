# TurboDiffusion Integration Notes

TurboT2AV adapts TurboDiffusion's inference stack to the LTX-2 joint
audio-video transformer. The integration is intentionally limited to inference;
the TurboT2AV distillation and training pipeline remains independent.

## Reused Components

- **SageSLA:** TurboDiffusion's sparse-linear attention design and SLA
  compensation path are wrapped for LTX-2 video and audio self-attention.
- **FastNorm:** module and functional RMSNorm/LayerNorm operations are replaced
  with fused implementations, together with LTX-specific modulation, residual,
  and rotary-embedding helpers.
- **W8A8 Linear:** weights are stored as INT8 and activations are dynamically
  quantized to INT8 before transformer Linear operations.

The recommended path combines SageSLA `topk=0.3`, FastNorm, and TileLang W8A8.

## Differences From TurboDiffusion

| Area | TurboDiffusion | TurboT2AV integration |
| --- | --- | --- |
| Model | Wan text-to-video transformer | LTX-2 joint audio-video transformer |
| Tested GPU | RTX 5090 in the published latency figure | NVIDIA H20 |
| Memory | Includes a CPU-offload stage | Full model remains on the H20; no CPU offload |
| Attention | Wan self-attention layout | LTX video/audio self-attention adapters; masked text cross-attention remains native |
| W8A8 kernel | Original CUTE/SM80 blockwise kernel | TileLang post-scale kernel tuned for H20 and LTX FFN shapes |
| Distillation | rCM is a separate acceleration stage | Uses the existing four-step TurboT2AV student checkpoint |

The TileLang kernel keeps INT8 accumulation continuous across the K dimension
and applies activation/weight scales in the epilogue. This adaptation is used
because TurboDiffusion's original W8A8 kernel did not outperform BF16 Linear on
the tested H20 LTX shapes.

## End-to-End Result

The following generator-only measurements use one H20, 121 frames, and a
`1024x1792` output resolution. Stages are cumulative.

| Stage | Latency | Speedup vs previous | Speedup vs teacher |
| --- | ---: | ---: | ---: |
| LTX-2-19B teacher, 40 steps | 318.7405s | - | 1.00x |
| + TileLang W8A8 and FastNorm | 233.3424s | 1.37x | 1.37x |
| + four-step TurboT2AV student | 11.7628s | 19.84x | 27.10x |
| + SageSLA `topk=0.3` | 5.8689s | 2.00x | 54.31x |

The pure four-step student without the inference optimizations takes
16.1096s/video, making the final path 2.75x faster than the pure student.
SageSLA has the largest inference-kernel benefit at this resolution because the
video self-attention sequence contains 28,672 tokens. `topk=0.3` was selected
as the practical speed/quality setting from paired visual comparisons; sparse
attention is not numerically lossless and should be revalidated for a different
resolution or prompt distribution.
