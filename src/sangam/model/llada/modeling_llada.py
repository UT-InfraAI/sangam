from __future__ import annotations

import importlib
import logging
import math
import sys
from abc import abstractmethod
from functools import lru_cache, partial
from typing import (
    Callable,
    Iterable,
    NamedTuple,
    Optional,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import Union

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel

from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")


@lru_cache(maxsize=1)
def _load_flashinfer():
    """Import flashinfer lazily so the module stays importable without a GPU."""
    return importlib.import_module("flashinfer")


def _fi_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _load_flashinfer().norm.rmsnorm(input, weight, eps)


def _fi_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    _load_flashinfer().norm.fused_add_rmsnorm(input, residual, weight, eps)


def _fi_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    return _load_flashinfer().activation.silu_and_mul(input)


__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "GemmaRMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "LLaDABlock",
    "LLaDASequentialBlock",
    "LLaDAModel",
    "LLaDAOutput",
    "LLaDAGenerateOutput",
]


log = logging.getLogger(__name__)


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value
            )
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(
                f"When using the {InitFnType.full_megatron} init, every module must have a type."
            )

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(
            module, "_is_residual", False
        ):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(
    x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False
):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0)
        and (cfg.embedding_dropout == 0.0)
        and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (
            elementwise_affine is None and self.config.layer_norm_with_affine
        ):
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=config.init_device)
            )
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.normalized_shape, device=config.init_device)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(
        cls, config: ModelConfig, size: Optional[int] = None, **kwargs
    ) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(
                f"Unknown LayerNorm type: '{config.layer_norm_type}'"
            )

    def _cast_if_autocast_enabled(
        self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(
                dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype()
            )
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(
                dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype()
            )
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(
            config, size=size, elementwise_affine=elementwise_affine, eps=eps
        )
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight)
                if self.weight is not None
                else self.weight
            )
            downcast_bias = (
                self._cast_if_autocast_enabled(self.bias)
                if self.bias is not None
                else self.bias
            )
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x,
                    self.normalized_shape,
                    weight=downcast_weight,
                    bias=downcast_bias,
                    eps=self.eps,
                )
        else:
            return F.layer_norm(
                x,
                self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(
            config,
            size=size,
            elementwise_affine=elementwise_affine,
            eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = F.rms_norm(x, self.normalized_shape, weight=self.weight, eps=self.eps)
        if self.bias is not None:
            normed = normed + self.bias
        return normed


class GemmaRMSLayerNorm(LayerNormBase):
    """
    Gemma RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(
            config,
            size=size,
            elementwise_affine=elementwise_affine,
            eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (1.0 + self.weight) if self.weight is not None else None
        normed = F.rms_norm(x, self.normalized_shape, weight=weight, eps=self.eps)
        if self.bias is not None:
            normed = normed + self.bias
        return normed


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(
            config.max_sequence_length, _non_meta_init_device(config)
        )

    def get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (
                self.rope_theta
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = (
                positions.sin()[None, None, :, :],
                positions.cos()[None, None, :, :],
            )
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        half = t.shape[-1] // 2
        cos = pos_cos[..., :half]
        sin = pos_sin[..., :half]
        t1 = t[..., :half]
        t2 = t[..., half:]
        out = torch.empty_like(t)
        # out[..., :half] = t1 * cos - t2 * sin
        torch.mul(t1, cos, out=out[..., :half])
        out[..., :half].addcmul_(t2, sin, value=-1)
        # out[..., half:] = t2 * cos + t1 * sin
        torch.mul(t2, cos, out=out[..., half:])
        out[..., half:].addcmul_(t1, sin)
        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        block_end_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)

            # Build tensor indices instead of using .item()
            if block_end_index is None:
                start = key_len - query_len
                end = key_len
            else:
                # block_end_index is a tensor; keep ops tensor-based
                start = block_end_index - query_len
                end = block_end_index

            # Make an index tensor [start, ..., end-1] on the right device/dtype
            idx = torch.arange(start, end, device=q_.device, dtype=torch.long)

            # Use index_select on the sequence dimension (dim=2)
            pos_sin_slice = pos_sin.index_select(2, idx)
            pos_cos_slice = pos_cos.index_select(2, idx)

            q_ = self.apply_rotary_pos_emb(pos_sin_slice, pos_cos_slice, q_)
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)

        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=True))  # To save memory in inference
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(
    cache: BufferCache, seq_len: int, device: torch.device
) -> torch.Tensor:
    if (
        causal_bias := cache.get("causal_attention_bias")
    ) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(
    seq_len: int, config: ModelConfig, device: torch.device
) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(
        1, 1, 1, seq_len
    )

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(
        1 - seq_len, 1, dtype=torch.float, device=device
    ).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class LLaDABlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size
            if config.mlp_hidden_size is not None
            else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(
                config, elementwise_affine=config.attention_layer_norm_with_affine
            )

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        # Set by mixed-batch execution context when operation metrics are enabled.
        self._op_metrics_context = None

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy]
    ):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    def _measure_operation_time(
        self,
        metric_name: str,
        fn: Callable[[], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        from sangam.model.model_runner import measure_operation_time

        return measure_operation_time(
            getattr(self, "_op_metrics_context", None),
            self.layer_id,
            metric_name,
            fn,
            device,
        )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype
        head_dim = C // self.config.n_heads

        # Optionally apply layer norm to keys and queries.
        if (
            self.q_norm is not None and self.k_norm is not None
        ):  # self.q_norm: None, self.k_norm: None
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Reshape projections directly into FlashInfer's NHD layout.
        q = q.reshape(B * T, self.config.n_heads, head_dim)
        k = k.reshape(B * T, self.config.effective_n_kv_heads, head_dim)
        v = v.reshape(B * T, self.config.effective_n_kv_heads, head_dim)

        # --- Paged KV + FlashInfer path (set by decode worker) ---
        paged_state = getattr(self, "_paged_attn_state", None)
        if paged_state is None:
            raise RuntimeError("LLaDA blocks require paged attention state")
        return self._paged_attention(q, k, v, B, T, C, paged_state)

    def _paged_attention(self, q, k, v, B, T, C, state):
        """FlashInfer paged attention: RoPE → scatter K,V to pages → attention."""
        layer_idx = state.next_layer_idx(self.layer_id)

        # Apply RoPE at the correct absolute positions (fused, in-place)
        if self.config.rope:
            state.apply_rope_inplace(self.rotary_emb, q, k, C // self.config.n_heads)
        q_nhd = q
        k_nhd = k

        # Write RoPE'd K and V to pages
        state.update_kv_pages(layer_idx, k_nhd, v)

        # FlashInfer batched attention against full paged KV. The "attn" metric
        # times only this kernel call (RoPE, KV write, and the output projection
        # are excluded).
        att_nhd = self._measure_operation_time(
            "attn", lambda: state.run_attention(q_nhd, layer_idx), q.device
        )

        # NHD → (B, T, C)
        att = att_nhd.reshape(B, T, C)

        state.finish_layer(layer_idx)

        # The "attn_post_proj" metric times only the attention output projection.
        return self._measure_operation_time(
            "attn_post_proj", lambda: self.attn_out(att), q.device
        )

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(
        cls, layer_id: int, config: ModelConfig, cache: BufferCache
    ) -> LLaDABlock:
        if config.block_type == BlockType.sequential:
            return LLaDASequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDASequentialBlock(LLaDABlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model,
            sum(self.fused_dims),
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config,
            self.att_proj,
            d=self.config.d_model,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.ff_proj,
            d=self.config.d_model,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        if self._activation_checkpoint_fn is not None:
            x_normed = self._activation_checkpoint_fn(self.attn_norm, x)
        else:
            x_normed = self.attn_norm(x)

        # The "attn_pre_proj" metric times only the qkv GEMM + split (the input
        # norm above runs untimed).
        q, k, v = self._measure_operation_time(
            "attn_pre_proj",
            lambda: self.att_proj(x_normed).split(self.fused_dims, dim=-1),
            x.device,
        )

        # Get attention scores. (The "attn" metric is recorded around the
        # FlashInfer kernel inside `attention`.)
        if self._activation_checkpoint_fn is not None:
            att = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
            )
        else:
            att = self.attention(q, k, v)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        def _run_mlp() -> torch.Tensor:
            mlp_x = x
            og_x = mlp_x
            if self._activation_checkpoint_fn is not None:
                mlp_x = self._activation_checkpoint_fn(self.ff_norm, mlp_x)  # type: ignore
            else:
                mlp_x = self.ff_norm(mlp_x)
            mlp_x = self.ff_proj(mlp_x)
            if self._activation_checkpoint_fn is not None:
                mlp_x = self._activation_checkpoint_fn(self.act, mlp_x)  # type: ignore
            else:
                mlp_x = self.act(mlp_x)
            mlp_x = self.ff_out(mlp_x)
            mlp_x = self.dropout(mlp_x)
            mlp_x = og_x + mlp_x
            return mlp_x

        x = self._measure_operation_time("mlp", _run_mlp, x.device)

        return x


class LLaDALlamaBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model,
            q_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            k_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            v_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        # qkv_proj is materialized lazily on the GPU by fuse_qkv() after the
        # checkpoint loads and the module is moved to device. Avoids a large
        # CPU allocation + kaiming_uniform_ init that bloats load time and a
        # wasted CPU->GPU transfer of placeholder weights.
        self.fused_dims = (q_proj_out_dim, k_proj_out_dim, v_proj_out_dim)
        self.qkv_proj: Optional[nn.Linear] = None
        self._qkv_fused = False

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )

        # Fused gate+up projection, materialized lazily by fuse_ff_gate_up()
        # after load (analogous to qkv_proj). When set, the fused-MLP forward
        # path replaces ff_proj/up_proj + SiLU + multiply with one wide GEMM
        # plus flashinfer.silu_and_mul.
        self.ff_gate_up_proj: Optional[nn.Linear] = None
        self._ff_fused = False

        # The fused norm/residual + silu_and_mul path is only numerically
        # equivalent for the served LLaDA-8B config: weight-only RMSNorm (no
        # bias), no attention QK-norm, and a plain SiLU activation (split
        # gate/up). Other configs fall through to the unfused forward.
        self._fused_mlp_norm = (
            isinstance(self.attn_norm, RMSLayerNorm)
            and isinstance(self.ff_norm, RMSLayerNorm)
            and self.attn_norm.bias is None
            and self.ff_norm.bias is None
            and self.q_norm is None
            and self.k_norm is None
            and isinstance(self.act, SiLU)
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(
            self.config, self.up_proj, d=self.config.d_model, layer_id=None
        )  # new add

    def fuse_qkv(self) -> None:
        if self._qkv_fused:
            return
        dtype = self.q_proj.weight.dtype
        in_features = self.q_proj.in_features
        out_features = sum(self.fused_dims)
        has_bias = self.q_proj.bias is not None

        qkv_proj = nn.Linear(
            in_features, out_features, bias=has_bias, device="meta", dtype=dtype
        )
        with torch.no_grad():
            weight = torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
            )
            qkv_proj.weight = nn.Parameter(weight)
            if has_bias:
                bias = torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
                )
                qkv_proj.bias = nn.Parameter(bias)
        self.qkv_proj = qkv_proj
        self._qkv_fused = True

    def fuse_ff_gate_up(self) -> None:
        # Merge the gate (ff_proj) and up (up_proj) projections into one wide
        # GEMM so the fused-MLP forward issues a single matmul feeding
        # flashinfer.silu_and_mul. silu_and_mul applies SiLU to the first half
        # of its input, and the unfused path applies SiLU to ff_proj, so ff_proj
        # must come first in the concatenation.
        if self._ff_fused:
            return
        dtype = self.ff_proj.weight.dtype
        in_features = self.ff_proj.in_features
        out_features = self.ff_proj.out_features + self.up_proj.out_features
        has_bias = self.ff_proj.bias is not None

        ff_gate_up_proj = nn.Linear(
            in_features, out_features, bias=has_bias, device="meta", dtype=dtype
        )
        with torch.no_grad():
            weight = torch.cat([self.ff_proj.weight, self.up_proj.weight], dim=0)
            ff_gate_up_proj.weight = nn.Parameter(weight)
            if has_bias:
                bias = torch.cat([self.ff_proj.bias, self.up_proj.bias], dim=0)
                ff_gate_up_proj.bias = nn.Parameter(bias)
        self.ff_gate_up_proj = ff_gate_up_proj
        self._ff_fused = True

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self._fused_mlp_norm and self._ff_fused:
            return self._forward_fused(x)

        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        assert self._qkv_fused, "fuse_qkv() must be called after model load"
        x_normed = self.attn_norm(x)
        # The "attn_pre_proj" metric times only the qkv GEMM + split (the input
        # norm above runs untimed).
        q, k, v = self._measure_operation_time(
            "attn_pre_proj",
            lambda: self.qkv_proj(x_normed).split(self.fused_dims, dim=-1),
            x.device,
        )

        # Get attention scores. (The "attn" metric is recorded around the
        # FlashInfer kernel inside `attention`.)
        if self._activation_checkpoint_fn is not None:
            att = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
            )
        else:
            att = self.attention(q, k, v)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        def _run_mlp() -> torch.Tensor:
            mlp_x = x
            og_x = mlp_x
            if self._activation_checkpoint_fn is not None:
                mlp_x = self._activation_checkpoint_fn(self.ff_norm, mlp_x)  # type: ignore
            else:
                mlp_x = self.ff_norm(mlp_x)
            x_up = self.up_proj(mlp_x)
            mlp_x = self.ff_proj(mlp_x)
            if self._activation_checkpoint_fn is not None:
                mlp_x = self._activation_checkpoint_fn(self.act, mlp_x)  # type: ignore
            else:
                mlp_x = self.act(mlp_x)
            mlp_x *= x_up
            mlp_x = self.ff_out(mlp_x)
            mlp_x = self.dropout(mlp_x)
            mlp_x = og_x + mlp_x
            return mlp_x

        x = self._measure_operation_time("mlp", _run_mlp, x.device)

        return x

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused inference path using FlashInfer norm/activation kernels.

        Equivalent to ``forward`` for the served LLaDA-8B config (guarded by
        ``self._fused_mlp_norm``). Dropout is identity at inference so it is
        dropped. FlashInfer norm kernels operate on 2D ``(num_tokens, hidden)``
        tensors, so activations are flattened and reshaped around them.
        """
        eps = self.attn_norm.eps
        C = x.shape[-1]

        # qkv: standalone RMSNorm then the fused QKV GEMM.
        assert self._qkv_fused, "fuse_qkv() must be called after model load"
        x_normed = _fi_rmsnorm(x.view(-1, C), self.attn_norm.weight, eps)
        x_normed = x_normed.view_as(x)
        # The "attn_pre_proj" metric times only the qkv GEMM + split (the input
        # norm above runs untimed).
        q, k, v = self._measure_operation_time(
            "attn_pre_proj",
            lambda: self.qkv_proj(x_normed).split(self.fused_dims, dim=-1),
            x.device,
        )

        # The "attn" metric is recorded around the FlashInfer kernel inside
        # `attention`.
        att = self.attention(q, k, v)

        # mlp: fold the attention residual add and ff_norm into one kernel, run
        # the merged gate+up GEMM through silu_and_mul, then the residual add.
        def _run_mlp() -> torch.Tensor:
            residual = x.view(-1, C)
            normed = att.reshape(-1, C)
            # In place: residual += normed; normed = rms_norm(residual).
            _fi_fused_add_rmsnorm(normed, residual, self.ff_norm.weight, eps)
            h = _fi_silu_and_mul(self.ff_gate_up_proj(normed))
            # residual is the flattened view of x, already written in place by
            # fused_add_rmsnorm; accumulate the MLP output into it to avoid a
            # fresh allocation + add kernel per layer.
            residual.add_(self.ff_out(h))
            return residual.view_as(x)

        return self._measure_operation_time("mlp", _run_mlp, x.device)


class LLaDABlockDiffBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model,
            q_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            k_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            v_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        # qkv_proj is materialized lazily on the GPU by fuse_qkv() after the
        # checkpoint loads and the module is moved to device. Avoids a large
        # CPU allocation + kaiming_uniform_ init that bloats load time and a
        # wasted CPU->GPU transfer of placeholder weights.
        self.fused_dims = (q_proj_out_dim, k_proj_out_dim, v_proj_out_dim)
        self.qkv_proj: Optional[nn.Linear] = None
        self._qkv_fused = False

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )

        # Fused gate+up projection, materialized lazily by fuse_ff_gate_up()
        # after load (analogous to qkv_proj). When set, the fused-MLP forward
        # path replaces ff_proj/up_proj + SiLU + multiply with one wide GEMM
        # plus flashinfer.silu_and_mul.
        self.ff_gate_up_proj: Optional[nn.Linear] = None
        self._ff_fused = False

        # The fused norm/residual + silu_and_mul path is only numerically
        # equivalent for the served LLaDA-8B config: weight-only RMSNorm (no
        # bias), no attention QK-norm, and a plain SiLU activation (split
        # gate/up). Other configs fall through to the unfused forward.
        self._fused_mlp_norm = (
            isinstance(self.attn_norm, RMSLayerNorm)
            and isinstance(self.ff_norm, RMSLayerNorm)
            and self.attn_norm.bias is None
            and self.ff_norm.bias is None
            and self.q_norm is None
            and self.k_norm is None
            and isinstance(self.act, SiLU)
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(
            self.config, self.up_proj, d=self.config.d_model, layer_id=None
        )  # new add

    def fuse_qkv(self) -> None:
        if self._qkv_fused:
            return
        dtype = self.q_proj.weight.dtype
        in_features = self.q_proj.in_features
        out_features = sum(self.fused_dims)
        has_bias = self.q_proj.bias is not None

        qkv_proj = nn.Linear(
            in_features, out_features, bias=has_bias, device="meta", dtype=dtype
        )
        with torch.no_grad():
            weight = torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
            )
            qkv_proj.weight = nn.Parameter(weight)
            if has_bias:
                bias = torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
                )
                qkv_proj.bias = nn.Parameter(bias)
        self.qkv_proj = qkv_proj
        self._qkv_fused = True

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        assert self._qkv_fused, "fuse_qkv() must be called after model load"
        x_normed = self.attn_norm(x)
        # The "attn_pre_proj" metric times only the qkv GEMM + split (the input
        # norm above runs untimed).
        q, k, v = self._measure_operation_time(
            "attn_pre_proj",
            lambda: self.qkv_proj(x_normed).split(self.fused_dims, dim=-1),
            x.device,
        )

        # Get attention scores. (The "attn" metric is recorded around the
        # FlashInfer kernel inside `attention`.)
        if self._activation_checkpoint_fn is not None:
            att = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
            )
        else:
            att = self.attention(q, k, v)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        def _run_mlp() -> torch.Tensor:
            mlp_x = x
            og_x = mlp_x
            if self._activation_checkpoint_fn is not None:
                mlp_x = self._activation_checkpoint_fn(self.ff_norm, mlp_x)  # type: ignore
            else:
                mlp_x = self.ff_norm(mlp_x)
            x_up = self.up_proj(mlp_x)
            mlp_x = self.ff_proj(mlp_x)
            if self._activation_checkpoint_fn is not None:
                mlp_x = self._activation_checkpoint_fn(self.act, mlp_x)  # type: ignore
            else:
                mlp_x = self.act(mlp_x)
            mlp_x *= x_up  # new add
            mlp_x = self.ff_out(mlp_x)
            mlp_x = self.dropout(mlp_x)
            mlp_x = og_x + mlp_x
            return mlp_x

        x = self._measure_operation_time("mlp", _run_mlp, x.device)

        return x


class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: None
    """
    Dense KV cache is not supported; this is always ``None``.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """


class LLaDAGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class LLaDABlockGroup(nn.ModuleList):
    def __init__(
        self,
        config: ModelConfig,
        layer_offset: int,
        modules: Optional[Iterable[nn.Module]] = None,
    ):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[
            ActivationCheckpointingStrategy
        ] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for block_idx, block in enumerate(self):
            block_idx += self.layer_offset
            if (
                (
                    self.activation_checkpointing_strategy
                    == ActivationCheckpointingStrategy.whole_layer
                )
                or (
                    self.activation_checkpointing_strategy
                    == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy
                    == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy
                    == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                # shape: (batch_size, seq_len, d_model)
                x = self._activation_checkpoint_fn(  # type: ignore
                    block,
                    x,
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x = block(x)
        return x

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy]
    ):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if (
            self.config.embedding_size is not None
            and self.config.embedding_size != self.config.vocab_size
        ):
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception(
                    "embedding size should be at least as big as vocab size"
                )
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.",
                    UserWarning,
                )

        self.activation_checkpointing_strategy: Optional[
            ActivationCheckpointingStrategy
        ] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(
            self.config
        )

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(
            False
        )  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size,
                    config.d_model,
                    device=config.init_device,
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [
            LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)
        ]
        if self.config.block_group_size > 1:
            block_groups = [
                LLaDABlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {
                    "wpe": nn.Embedding(
                        config.max_sequence_length,
                        config.d_model,
                        device=config.init_device,
                    )
                }
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(
                self.__cache, config.max_sequence_length, _non_meta_init_device(config)
            )
            self.get_alibi_attention_bias(
                config.max_sequence_length, _non_meta_init_device(config)
            )

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy]
    ):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model))
            if self.config.scale_logits
            else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(
                self.config, self.transformer.wpe, type_of_module=ModuleType.emb
            )  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(
                self.config,
                self.transformer.ff_out,
                type_of_module=ModuleType.final_out,
            )  # type: ignore

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def fuse_qkv(self) -> None:
        """Materialize fused qkv_proj weights on every block that supports it."""
        if hasattr(self.transformer, "blocks"):
            block_iter = self.transformer.blocks
        else:
            block_iter = (b for group in self.transformer.block_groups for b in group)
        for block in block_iter:
            if hasattr(block, "fuse_qkv"):
                block.fuse_qkv()

    def fuse_ff_gate_up(self) -> None:
        """Materialize fused gate+up MLP weights on every block that supports it."""
        if hasattr(self.transformer, "blocks"):
            block_iter = self.transformer.blocks
        else:
            block_iter = (b for group in self.transformer.block_groups for b in group)
        for block in block_iter:
            if hasattr(block, "fuse_ff_gate_up"):
                block.fuse_ff_gate_up()

    def get_alibi_attention_bias(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        if (
            alibi_bias := self.__cache.get("alibi_attention_bias")
        ) is not None and alibi_bias.shape[-1] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
    ) -> LLaDAOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        # Add Basic MDM Model config check
        assert not self.config.alibi, (
            "Alibi length extrapolation is not supported for MDM."
        )
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        if attention_mask is not None:
            raise NotImplementedError("Paged LLaDA does not support attention_mask")
        if attention_bias is not None:
            raise NotImplementedError("Paged LLaDA does not support attention_bias")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        batch_size, seq_len = (
            input_ids.size()
            if input_embeddings is None
            else input_embeddings.size()[:2]
        )
        past_length = 0

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = (
            self.transformer.wte(input_ids)
            if input_embeddings is None
            else input_embeddings
        )  # type: ignore

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=x.device
            ).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                if (
                    (
                        self.activation_checkpointing_strategy
                        == ActivationCheckpointingStrategy.whole_layer
                    )
                    or (
                        self.activation_checkpointing_strategy
                        == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy
                        == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy
                        == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                    )
                ):
                    # shape: (batch_size, seq_len, d_model)
                    x = self._activation_checkpoint_fn(
                        block,
                        x,
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x = block(x)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                x = block_group(x)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return LLaDAOutput(
            logits=logits,
            attn_key_values=None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )  # type: ignore[arg-type]


def create_model_config_from_pretrained_config(config: LLaDAConfig):
    """
    Utility function
    """

    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)

    model_config = ModelConfig(**kwargs)
    return model_config


class LLaDAModelLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock"]

    def __init__(
        self,
        config: LLaDAConfig,
        model: Optional[LLaDAModel] = None,
        init_params: bool = False,
    ):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = LLaDAModel(model_config, init_params=init_params)
        else:
            self.model = model

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # import pdb; pdb.set_trace()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            output_hidden_states=output_hidden_states,
        )
        # import pdb; pdb.set_trace()
        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            import warnings

            warnings.warn(
                "Note that for LLaDA, you cannot calculate the loss here.", UserWarning
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
        )

    @property
    def num_layers(self) -> int:
        return self.model.config.n_layers

    @property
    def num_kv_heads(self) -> int:
        return self.model.config.effective_n_kv_heads

    @property
    def num_q_heads(self) -> int:
        return self.model.config.n_heads

    @property
    def head_dim(self) -> int:
        return self.model.config.d_model // self.model.config.n_heads

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        model_inputs = {"input_ids": input_ids}
        model_inputs.update(kwargs)
        return model_inputs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte

    def fuse_qkv(self) -> None:
        """Materialize fused qkv_proj weights from q/k/v_proj on every block that
        supports it. Must be called after weights are loaded (e.g. after
        from_pretrained) and before any forward pass."""
        self.model.fuse_qkv()

    def fuse_ff_gate_up(self) -> None:
        """Materialize fused gate+up MLP weights from ff_proj/up_proj on every
        block that supports it. Must be called after weights are loaded (e.g.
        after from_pretrained) and before any forward pass."""
        self.model.fuse_ff_gate_up()


# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(LLaDAConfig, LLaDAModelLM)
