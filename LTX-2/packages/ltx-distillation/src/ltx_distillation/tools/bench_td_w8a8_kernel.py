from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from turbo_diffusion_ops import gemm_cuda, gemm_cuda_swizzle, gemm_cuda_swizzle_bias, quant_cuda
from ltx_distillation.acceleration import _TurboDiffusionInt8Linear
from turbodiffusion.ops.core import Int8Linear, int8_linear, int8_quant


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    m: int
    k: int
    n: int


SHAPES = (
    ShapeSpec("video512_ffn_up", 6144, 4096, 16384),
    ShapeSpec("video512_ffn_down", 6144, 16384, 4096),
    ShapeSpec("video1024_ffn_up", 28672, 4096, 16384),
    ShapeSpec("video1024_ffn_down", 28672, 16384, 4096),
)


def _cuda_time_ms(fn: Callable[[], torch.Tensor | None], repeat: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _tops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / (ms * 1.0e9)


def _bench_shape(spec: ShapeSpec, repeat: int, warmup: int, swizzle_logs: list[int]) -> dict[str, object]:
    torch.manual_seed(0)
    x = torch.randn(spec.m, spec.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(spec.n, spec.k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(spec.n, device="cuda", dtype=torch.bfloat16)

    linear = torch.nn.Linear(spec.k, spec.n, bias=True, dtype=torch.bfloat16, device="cuda")
    linear.weight.data.copy_(weight)
    linear.bias.data.copy_(bias)
    td_linear = Int8Linear.from_linear(linear, quantize=True).cuda()
    opt_td_linear = _TurboDiffusionInt8Linear.from_linear(linear, quantize=True).cuda()

    w_q = td_linear.int8_weight
    w_s = td_linear.scale
    x_q, x_s = int8_quant(x)
    y = torch.empty(spec.m, spec.n, dtype=x.dtype, device=x.device)
    x_q_buf = torch.empty_like(x_q)
    x_s_buf = torch.empty_like(x_s)
    torch.cuda.synchronize()

    result: dict[str, object] = {
        "shape": spec.__dict__,
        "bf16_linear": _cuda_time_ms(lambda: F.linear(x, weight, bias), repeat, warmup),
        "td_quant_alloc": _cuda_time_ms(lambda: quant_cuda(x, None, None), repeat, warmup),
        "td_quant_prealloc": _cuda_time_ms(lambda: quant_cuda(x, x_q_buf, x_s_buf), repeat, warmup),
        "td_gemm_only": _cuda_time_ms(lambda: gemm_cuda(x_q, x_s, w_q, w_s, y), repeat, warmup),
        "td_full_no_bias": _cuda_time_ms(lambda: int8_linear(x, w_q, w_s), repeat, warmup),
        "td_full_bias": _cuda_time_ms(lambda: td_linear(x), repeat, warmup),
        "td_full_bias_opt_wrapper": _cuda_time_ms(lambda: opt_td_linear(x), repeat, warmup),
    }

    def prealloc_full() -> torch.Tensor:
        quant_cuda(x, x_q_buf, x_s_buf)
        gemm_cuda(x_q_buf, x_s_buf, w_q, w_s, y)
        y.add_(bias)
        return y

    result["td_full_prealloc_bias"] = _cuda_time_ms(prealloc_full, repeat, warmup)

    swizzle_results: dict[str, dict[str, float]] = {}
    swizzle_bias_results: dict[str, dict[str, float]] = {}
    for direction in (0, 1):
        for log_size in swizzle_logs:
            key = f"dir{direction}_log{log_size}"
            swizzle_results[key] = _cuda_time_ms(
                lambda direction=direction, log_size=log_size: gemm_cuda_swizzle(
                    x_q, x_s, w_q, w_s, y, direction, log_size
                ),
                repeat,
                warmup,
            )
            swizzle_bias_results[key] = _cuda_time_ms(
                lambda direction=direction, log_size=log_size: gemm_cuda_swizzle_bias(
                    x_q, x_s, w_q, w_s, y, bias, direction, log_size
                ),
                repeat,
                warmup,
            )
    result["td_gemm_swizzle"] = swizzle_results
    result["td_gemm_swizzle_bias"] = swizzle_bias_results

    for key in (
        "bf16_linear",
        "td_gemm_only",
        "td_full_no_bias",
        "td_full_bias",
        "td_full_bias_opt_wrapper",
        "td_full_prealloc_bias",
    ):
        value = result[key]
        assert isinstance(value, dict)
        value["tops"] = _tops(spec.m, spec.n, spec.k, value["median_ms"])

    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark strict TurboDiffusion W8A8 kernels.")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--shape",
        choices=[shape.name for shape in SHAPES] + ["all"],
        default="all",
    )
    parser.add_argument("--swizzle_logs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    selected = SHAPES if args.shape == "all" else tuple(shape for shape in SHAPES if shape.name == args.shape)
    results = {
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "results": [_bench_shape(shape, args.repeat, args.warmup, args.swizzle_logs) for shape in selected],
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
