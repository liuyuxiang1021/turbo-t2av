# Acceleration Reference

TurboT2AV's recommended H20 inference path combines:

- SageSLA self-attention with `topk=0.3`
- FastNorm and fused Ada/RoPE helpers
- text-context trimming
- TileLang post-scale W8A8 Linear

The main README contains the shortest working inference commands. This document
covers the tuning and implementation details.

## SageSLA

`--attention_scope self` replaces the 96 video/audio self-attention modules.
Masked text cross-attention remains on the native backend because its mask path
is not supported by the SageAttention/SageSLA integration.

`--sla_topk 1.0` is the quality-first dense-block setting. Lower values are
faster on long video sequences but can change generated content because SageSLA
is a sparse-linear approximation rather than a numerically equivalent dense
attention kernel. The reported H20 speed/quality setting is `0.3`.

A per-layer schedule can keep selected layers denser. Unmatched layers use the
global `--sla_topk` value:

```bash
--sla_topk 0.3 --sla_topk_schedule 0-15:0.35,16-31:0.3,32-47:0.3
```

Validate lower top-k values visually on the target prompt distribution.

## W8A8 Linear

The recommended H20 configuration is:

```bash
--quant_linear \
--quant_linear_scope all \
--quant_linear_backend tilelang_postscale
```

Available scopes are:

| Scope | Replaced Linear layers |
| --- | --- |
| `all` | Broad TurboDiffusion-style replacement |
| `ffn` | All transformer feed-forward layers |
| `video_ffn` | Video feed-forward layers only |
| `audio_ffn` | Audio feed-forward layers only |
| `non_attention` | Linear layers outside attention projections |

The TileLang backend stores W8 weights once, dynamically quantizes activations
to A8, performs the INT8 GEMM with continuous K accumulation, and applies scales
in the epilogue.

### Strict TurboDiffusion Backend

`--quant_linear_backend turbodiffusion` uses TurboDiffusion's strict
`Int8Linear`: precompressed `int8_weight` and `scale` buffers, `quant_cuda` A8
activation quantization, and `gemm_cuda_swizzle_bias`. It is real W8A8 and does
not recompress weights on every forward.

Create a prequantized student checkpoint with:

```bash
python -m ltx_distillation.tools.prequantize_av_student \
  --student_checkpoint /path/to/model.pth \
  --output_path /path/to/model_w8a8_video_ffn_prequant.pth \
  --quant_linear_scope video_ffn
```

Load it with:

```bash
--quant_linear_prequantized \
--quant_linear_scope video_ffn \
--quant_linear_backend turbodiffusion
```

The strict kernel precompresses weights correctly, but it was slower than BF16
cuBLASLt for TurboT2AV's FFN shapes on the tested H20. The integrated path
therefore uses `tilelang_postscale`. The compiled torchao backend remains
experimental because of its additional dependency and first-sample compile
cost.

## H20 Measurements

Generator-only measurements use 121 frames and exclude VAE decoding. The
teacher uses one measured generation; student values are medians from repeated
generations. Acceleration stages are cumulative.

| Stage | Generator time | Speedup vs previous | Speedup vs teacher |
| --- | ---: | ---: | ---: |
| LTX-2-19B teacher (40 steps) | 318.7405s/video | - | 1.00x |
| + W8A8/FastNorm | 233.3424s/video | 1.37x | 1.37x |
| + rCM (4-step student) | 11.7628s/video | 19.84x | 27.10x |
| + SageSLA `topk=0.3` | 5.8689s/video | 2.00x | 54.31x |

The non-cumulative pure 4-step student baseline is 16.1096s/video.

Component-level validation at `1024x1792` (28,672 video tokens):

| Component | Shape / setting | Baseline | Accelerated | Speedup |
| --- | --- | ---: | ---: | ---: |
| Self-attention | 28,672 tokens | SDPA 37.70ms | SageSLA 7.818ms | 4.82x |
| W8A8 GEMM | `M=28672,N=16384,K=4096` | BF16 5.281ms | 3.375ms | 1.56x |
| W8A8 GEMM | `M=28672,N=4096,K=16384` | BF16 5.456ms | 3.397ms | 1.61x |

FFN layers benefit from reduced weight bandwidth and INT8 GEMM throughput.
Attention contains additional QKV layout changes, softmax/masking, and memory
traffic, so quantizing its projections has less direct end-to-end impact.

Text K/V caching can be enabled experimentally with
`TURBOT2AV_CACHE_TEXT_KV=1`, but it did not improve measured generator latency
on this workload and is disabled by default.
