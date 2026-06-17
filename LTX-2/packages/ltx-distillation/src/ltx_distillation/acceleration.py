"""Optional SageAttention and FastNorm acceleration for LTX-2 inference."""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from ltx_core.model.transformer.attention import Attention

ATTENTION_TYPES = ("default", "sageattn", "sla", "sagesla")
ATTENTION_SCOPES = ("self", "self_av")
DEFAULT_SLA_TOPK = 1.0


@dataclass(frozen=True)
class AccelerationReport:
    attention_type: str
    attention_scope: str
    replaced_attention: int = 0
    skipped_attention: int = 0
    replaced_norm: int = 0
    replaced_functional_norm: int = 0

    def format(self) -> str:
        return (
            "[TurboT2AV][accel] "
            f"attention_type={self.attention_type} "
            f"attention_scope={self.attention_scope} "
            f"replaced_attention={self.replaced_attention} "
            f"skipped_attention={self.skipped_attention} "
            f"replaced_norm={self.replaced_norm} "
            f"replaced_functional_norm={self.replaced_functional_norm}"
        )


def _ensure_turbodiffusion_path() -> None:
    """Make TurboDiffusion's local ops importable when TurboT2AV is a submodule."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        for candidate in (
            parent / "turbodiffusion",
            parent / "TurboDiffusion" / "turbodiffusion",
        ):
            if not (candidate / "ops").is_dir():
                continue
            for path in (candidate, candidate.parent):
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)


def _import_turbodiffusion_attr(module_names: tuple[str, ...], attr_name: str) -> object:
    _ensure_turbodiffusion_path()
    errors = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        f"Unable to import TurboDiffusion acceleration symbol {attr_name}. "
        "Install TurboDiffusion from source or place TurboT2AV inside the "
        "TurboDiffusion repository. Tried: " + "; ".join(errors)
    )


class SageAttentionCallable(torch.nn.Module):
    """Adapter from LTX attention tensors to SageAttention's HND layout."""

    def __init__(self) -> None:
        super().__init__()
        self._fn, self._fn_kwargs = self._load_sage_attention()

    @staticmethod
    def _load_sage_attention() -> tuple[Callable[..., torch.Tensor], dict[str, object]]:
        errors = []
        for module_name, attr_name in (
            ("sageattention", "sageattn"),
            ("sageattention.core", "sageattn"),
            ("sageattention", "sageattn_qk_int8_pv_fp16_cuda"),
            ("sageattention.core", "sageattn_qk_int8_pv_fp16_cuda"),
            ("sageattention", "sageattn_qk_int8_pv_fp16_triton"),
            ("sageattention.core", "sageattn_qk_int8_pv_fp16_triton"),
        ):
            try:
                module = importlib.import_module(module_name)
                fn = getattr(module, attr_name)
                kwargs: dict[str, object] = {}
                if attr_name == "sageattn_qk_int8_pv_fp16_cuda":
                    kwargs["qk_quant_gran"] = "per_warp"
                return fn, kwargs
            except (ImportError, AttributeError) as exc:
                errors.append(f"{module_name}.{attr_name}: {exc}")
        raise ImportError(
            "Unable to import SageAttention. Install `sageattention` or use "
            "--attention_type default. Tried: " + "; ".join(errors)
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise NotImplementedError(
                "SageAttention is enabled only for unmasked LTX attention. "
                "Use --attention_scope self or self_av."
            )

        batch, _, inner_dim = q.shape
        head_dim = inner_dim // heads
        q, k, v = (
            tensor.view(batch, -1, heads, head_dim).transpose(1, 2).contiguous()
            for tensor in (q, k, v)
        )
        out = self._fn(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=False,
            sm_scale=head_dim**-0.5,
            **self._fn_kwargs,
        )
        return out.transpose(1, 2).reshape(batch, -1, inner_dim)


class LTXSLAAttention(torch.nn.Module):
    """Adapter from LTX attention tensors to TurboDiffusion's vendored SLA."""

    def __init__(
        self,
        head_dim: int,
        topk: float,
        block_q: int,
        block_k: int,
        use_bf16: bool,
    ) -> None:
        super().__init__()
        _ensure_turbodiffusion_path()
        from SLA import SparseLinearAttention

        self.requested_topk = topk
        self.block_k = block_k
        self.local_attn = SparseLinearAttention(
            head_dim=head_dim,
            topk=topk,
            BLKQ=block_q,
            BLKK=block_k,
            use_bf16=use_bf16,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise NotImplementedError(
                "SLA is enabled only for unmasked LTX self-attention. "
                "Use --attention_scope self."
            )

        batch, _, inner_dim = q.shape
        head_dim = inner_dim // heads
        q, k, v = (tensor.view(batch, -1, heads, head_dim).contiguous() for tensor in (q, k, v))
        key_blocks = max(1, math.ceil(k.shape[1] / self.block_k))
        effective_topk = max(self.requested_topk, 1.0 / key_blocks)
        original_topk = self.local_attn.topk
        self.local_attn.topk = effective_topk
        try:
            out = self.local_attn(q, k, v)
        finally:
            self.local_attn.topk = original_topk
        return out.reshape(batch, -1, inner_dim)


class LTXSageSLAAttention(torch.nn.Module):
    """Adapter from LTX attention tensors to TurboDiffusion's SageSLA path."""

    def __init__(self, head_dim: int, topk: float, use_bf16: bool) -> None:
        super().__init__()
        _ensure_turbodiffusion_path()
        from SLA import SageSparseLinearAttention

        self.requested_topk = topk
        self.local_attn = SageSparseLinearAttention(
            head_dim=head_dim,
            topk=topk,
            use_bf16=use_bf16,
        )

    @staticmethod
    def _block_k_for_device(device: torch.device) -> int:
        if device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0):
            return 128
        return 64

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise NotImplementedError(
                "SageSLA is enabled only for unmasked LTX self-attention. "
                "Use --attention_scope self."
            )

        batch, _, inner_dim = q.shape
        head_dim = inner_dim // heads
        q, k, v = (tensor.view(batch, -1, heads, head_dim).contiguous() for tensor in (q, k, v))
        key_blocks = max(1, math.ceil(k.shape[1] / self._block_k_for_device(k.device)))
        effective_topk = max(self.requested_topk, 1.0 / key_blocks)
        original_topk = self.local_attn.topk
        self.local_attn.topk = effective_topk
        try:
            out = self.local_attn(q, k, v)
        finally:
            self.local_attn.topk = original_topk
        return out.reshape(batch, -1, inner_dim)


def _is_self_attention_name(name: str) -> bool:
    return name.endswith("attn1")


def _is_av_cross_attention_name(name: str) -> bool:
    return name.endswith(("audio_to_video_attn", "video_to_audio_attn"))


def _attention_name_in_scope(name: str, attention_scope: str) -> bool:
    if attention_scope == "self":
        return _is_self_attention_name(name)
    if attention_scope == "self_av":
        return _is_self_attention_name(name) or _is_av_cross_attention_name(name)
    raise ValueError(f"Unsupported attention_scope: {attention_scope}")


def _attention_supported_by_backend(name: str, attention_type: str) -> bool:
    if attention_type in {"sla", "sagesla"} and not _is_self_attention_name(name):
        return False
    return True


def replace_ltx_attention(
    model: torch.nn.Module,
    attention_type: str,
    attention_scope: str = "self",
    sla_topk: float = DEFAULT_SLA_TOPK,
    sla_block_q: int = 128,
    sla_block_k: int = 64,
) -> tuple[int, int]:
    if attention_type not in ATTENTION_TYPES:
        raise ValueError(f"--attention_type must be one of {ATTENTION_TYPES}")
    if attention_scope not in ATTENTION_SCOPES:
        raise ValueError(f"--attention_scope must be one of {ATTENTION_SCOPES}")
    if attention_type == "default":
        return 0, 0

    replaced = 0
    skipped = 0
    for name, module in model.named_modules():
        if not isinstance(module, Attention) or not _attention_name_in_scope(name, attention_scope):
            continue
        if not _attention_supported_by_backend(name, attention_type):
            skipped += 1
            continue
        if attention_type == "sageattn":
            attention_callable = SageAttentionCallable().to(device=module.to_q.weight.device)
        elif attention_type == "sla":
            attention_callable = LTXSLAAttention(
                head_dim=module.dim_head,
                topk=sla_topk,
                block_q=sla_block_q,
                block_k=sla_block_k,
                use_bf16=module.to_q.weight.dtype == torch.bfloat16,
            ).to(device=module.to_q.weight.device, dtype=module.to_q.weight.dtype)
        elif attention_type == "sagesla":
            attention_callable = LTXSageSLAAttention(
                head_dim=module.dim_head,
                topk=sla_topk,
                use_bf16=module.to_q.weight.dtype == torch.bfloat16,
            ).to(device=module.to_q.weight.device, dtype=module.to_q.weight.dtype)
        else:
            raise ValueError(f"Unsupported attention_type={attention_type!r}")
        module.attention_function = attention_callable
        replaced += 1
    return replaced, skipped


def _set_child_module(root: torch.nn.Module, qualified_name: str, new_module: torch.nn.Module) -> None:
    parent = root
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _fast_rmsnorm_from_torch(
    original_rmsnorm: torch.nn.RMSNorm,
    fast_rmsnorm_cls: Callable[..., torch.nn.Module],
) -> torch.nn.Module:
    normalized_shape = original_rmsnorm.normalized_shape
    dim = normalized_shape[0] if isinstance(normalized_shape, tuple) else normalized_shape
    fast_rmsnorm = fast_rmsnorm_cls(dim=dim, eps=original_rmsnorm.eps)
    device = original_rmsnorm.weight.device if original_rmsnorm.weight is not None else torch.device("cpu")
    if original_rmsnorm.weight is not None and original_rmsnorm.weight.device != torch.device("meta"):
        fast_rmsnorm.weight.data.copy_(original_rmsnorm.weight.float().data)
    return fast_rmsnorm.to(device=device)


def _fast_layernorm_from_torch(
    original_layernorm: torch.nn.LayerNorm,
    fast_layernorm_cls: type[torch.nn.Module],
) -> torch.nn.Module:
    normalized_shape = original_layernorm.normalized_shape
    if not isinstance(normalized_shape, tuple) or len(normalized_shape) != 1:
        raise ValueError(
            "TurboDiffusion FastLayerNorm only supports 1D normalized_shape; "
            f"got {normalized_shape!r}"
        )
    fast_layernorm = fast_layernorm_cls.from_layernorm(original_layernorm)
    if original_layernorm.weight is not None:
        device = original_layernorm.weight.device
    elif original_layernorm.bias is not None:
        device = original_layernorm.bias.device
    else:
        device = torch.device("cpu")
    return fast_layernorm.to(device=device)


def replace_ltx_norms(model: torch.nn.Module) -> int:
    fast_rmsnorm_cls = _import_turbodiffusion_attr(("ops", "turbodiffusion.ops"), "FastRMSNorm")
    fast_layernorm_cls = _import_turbodiffusion_attr(("ops", "turbodiffusion.ops"), "FastLayerNorm")

    replacements: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if not name or ".attention_function." in name or name.endswith(".attention_function"):
            continue
        if isinstance(module, torch.nn.RMSNorm):
            replacements[name] = _fast_rmsnorm_from_torch(module, fast_rmsnorm_cls)
        elif isinstance(module, torch.nn.LayerNorm):
            replacements[name] = _fast_layernorm_from_torch(module, fast_layernorm_cls)

    for name, new_module in replacements.items():
        _set_child_module(model, name, new_module)
    return len(replacements)


def enable_fast_functional_rms_norm() -> int:
    """Patch LTX's imported rms_norm helper to TurboDiffusion's fused kernel."""

    fused_rmsnorm = _import_turbodiffusion_attr(("ops", "turbodiffusion.ops"), "rmsnorm")
    utils_module = importlib.import_module("ltx_core.utils")
    original_rms_norm = utils_module.rms_norm
    if getattr(original_rms_norm, "_turbot2av_fast_functional_norm", False):
        return 0

    one_weight_cache: dict[tuple[str, int | None, int], torch.Tensor] = {}

    def fast_rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if (
            not x.is_cuda
            or x.dim() not in {2, 3}
            or not x.is_contiguous()
            or x.dtype not in {torch.float16, torch.bfloat16, torch.float32}
        ):
            return original_rms_norm(x, weight=weight, eps=eps)

        try:
            if weight is None:
                key = (x.device.type, x.device.index, x.shape[-1])
                fused_weight = one_weight_cache.get(key)
                if fused_weight is None or fused_weight.device != x.device:
                    fused_weight = torch.ones(x.shape[-1], device=x.device, dtype=torch.float32)
                    one_weight_cache[key] = fused_weight
            else:
                if weight.device == torch.device("meta") or tuple(weight.shape) != (x.shape[-1],):
                    return original_rms_norm(x, weight=weight, eps=eps)
                fused_weight = weight.to(device=x.device, dtype=torch.float32)

            return fused_rmsnorm(x.float(), fused_weight, eps).to(dtype=x.dtype)
        except (AssertionError, RuntimeError):
            return original_rms_norm(x, weight=weight, eps=eps)

    fast_rms_norm._turbot2av_fast_functional_norm = True
    fast_rms_norm._turbot2av_original_rms_norm = original_rms_norm

    patched = 0
    for module_name in (
        "ltx_core.utils",
        "ltx_core.model.transformer.transformer",
        "ltx_core.text_encoders.gemma.embeddings_connector",
    ):
        module = importlib.import_module(module_name)
        current = getattr(module, "rms_norm", None)
        if current is original_rms_norm:
            module.rms_norm = fast_rms_norm
            patched += 1
    return patched


def apply_turbodiffusion_acceleration(
    model: torch.nn.Module,
    attention_type: str = "default",
    attention_scope: str = "self",
    fast_norm: bool = False,
    sla_topk: float = DEFAULT_SLA_TOPK,
    sla_block_q: int = 128,
    sla_block_k: int = 64,
) -> AccelerationReport:
    if attention_type not in ATTENTION_TYPES:
        raise ValueError(f"Unsupported attention_type={attention_type!r}; expected one of {ATTENTION_TYPES}")
    replaced_attention, skipped_attention = replace_ltx_attention(
        model=model,
        attention_type=attention_type,
        attention_scope=attention_scope,
        sla_topk=sla_topk,
        sla_block_q=sla_block_q,
        sla_block_k=sla_block_k,
    )
    replaced_norm = replace_ltx_norms(model) if fast_norm else 0
    replaced_functional_norm = enable_fast_functional_rms_norm() if fast_norm else 0
    return AccelerationReport(
        attention_type=attention_type,
        attention_scope=attention_scope,
        replaced_attention=replaced_attention,
        skipped_attention=skipped_attention,
        replaced_norm=replaced_norm,
        replaced_functional_norm=replaced_functional_norm,
    )
