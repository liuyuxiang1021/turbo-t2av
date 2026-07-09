"""Prequantize TurboT2AV student Linear weights for TurboDiffusion W8A8 inference."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from ltx_core.loader.registry import DummyRegistry
from ltx_distillation.acceleration import QUANT_LINEAR_SCOPES, replace_ltx_linears
from ltx_distillation.models.ltx_trig_wrapper import create_ltx2_trig_wrapper
from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
from ltx_distillation.tools.run_av_inference_eval import _load_generator_state


def _apply_env_overrides(cfg: Any) -> None:
    for key, env in [
        ("checkpoint_path", "TURBO_CHECKPOINT_PATH"),
        ("gemma_path", "TURBO_GEMMA_PATH"),
        ("data_path", "TURBO_DATA_PATH"),
        ("scm_data_path", "TURBO_SCM_DATA_PATH"),
        ("output_path", "TURBO_OUTPUT_PATH"),
    ]:
        if os.environ.get(env):
            cfg[key] = os.environ[env]


def _state_dict_quant_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    stats = {
        "int8_weight_tensors": 0,
        "int8_weight_bytes": 0,
        "scale_tensors": 0,
        "scale_bytes": 0,
        "bf16_weight_tensors": 0,
        "bf16_weight_bytes": 0,
    }
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor_bytes = tensor.numel() * tensor.element_size()
        if key.endswith(".int8_weight") and tensor.dtype == torch.int8:
            stats["int8_weight_tensors"] += 1
            stats["int8_weight_bytes"] += tensor_bytes
        elif key.endswith(".scale") and tensor.dtype == torch.float32:
            stats["scale_tensors"] += 1
            stats["scale_bytes"] += tensor_bytes
        elif key.endswith(".weight") and tensor.dtype == torch.bfloat16:
            stats["bf16_weight_tensors"] += 1
            stats["bf16_weight_bytes"] += tensor_bytes
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_path",
        default="packages/ltx-distillation/configs/bidirectional_rcm.yaml",
        help="TurboT2AV inference/training config used to build the student wrapper.",
    )
    parser.add_argument("--student_checkpoint", required=True, help="Input BF16 student checkpoint.")
    parser.add_argument("--output_path", required=True, help="Output prequantized state dict path.")
    parser.add_argument(
        "--quant_linear_scope",
        choices=QUANT_LINEAR_SCOPES,
        default="video_ffn",
        help=(
            "Linear layers to prequantize. transformer_blocks matches TurboDiffusion's "
            "block-local replacement; video_ffn targets the largest video FFN matrices."
        ),
    )
    parser.add_argument(
        "--student_param",
        choices=["auto", "legacy", "rcm_trig"],
        default="auto",
        help="Student wrapper parametrization, matching run_av_inference_eval.py.",
    )
    parser.add_argument("--student_strict", action="store_true", help="Strictly load the input student checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because TurboDiffusion Int8Linear quantizes weights on CUDA.")

    cfg = OmegaConf.load(args.config_path)
    _apply_env_overrides(cfg)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if bool(getattr(cfg, "mixed_precision", True)) else torch.float32
    dmd_style = str(getattr(cfg, "dmd_style", "legacy")).lower()
    force_trig = args.student_param == "rcm_trig" or (
        args.student_param == "auto" and dmd_style in {"rcm", "rcm_trig", "trig"}
    )
    wrapper_factory = create_ltx2_trig_wrapper if force_trig else create_ltx2_wrapper

    print(f"[TurboT2AV][prequant] building student wrapper device={device} dtype={dtype}", flush=True)
    model = wrapper_factory(
        checkpoint_path=cfg.checkpoint_path,
        gemma_path=cfg.gemma_path,
        device=device,
        dtype=dtype,
        video_height=int(cfg.video_height),
        video_width=int(cfg.video_width),
        registry=DummyRegistry(),
    ).eval()

    print(f"[TurboT2AV][prequant] loading {args.student_checkpoint}", flush=True)
    _load_generator_state(model, args.student_checkpoint, args.student_strict)
    model.eval()

    print(
        "[TurboT2AV][prequant] replacing Linear -> TurboDiffusion Int8Linear "
        f"scope={args.quant_linear_scope}",
        flush=True,
    )
    replaced = replace_ltx_linears(
        model,
        quantize=True,
        scope=args.quant_linear_scope,
        backend="turbodiffusion",
    )
    state_dict = model.state_dict()
    stats = _state_dict_quant_stats(state_dict)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[TurboT2AV][prequant] saving {output_path}", flush=True)
    torch.save(state_dict, output_path)
    size_gib = output_path.stat().st_size / 1024**3
    print(
        "[TurboT2AV][prequant] "
        f"replaced_linear={replaced} size_gib={size_gib:.2f} stats={stats}",
        flush=True,
    )

    del state_dict, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
