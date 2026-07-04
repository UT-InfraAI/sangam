"""PyTorch Dream model."""

import importlib
from functools import lru_cache
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    logging,
)
from .configuration_dream import DreamConfig
from sangam.model.model_runner import measure_operation_time


logger = logging.get_logger(__name__)


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


_CHECKPOINT_FOR_DOC = "Dream-7B"
_CONFIG_FOR_DOC = "DreamConfig"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Dream
class DreamRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DreamRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Dream
class DreamRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[DreamConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`DreamRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        # Cached fp32 sin/cos table built by `get_rotary_embedding`, so the RoPE
        # table is computed once per forward (via PagedAttentionState._rope_cache)
        # instead of recomputed in every attention layer.
        self._pos_sin: Optional[torch.Tensor] = None
        self._pos_cos: Optional[torch.Tensor] = None

    def reset_parameters(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, self.inv_freq.device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        # inv_freq changed, so the cached sin/cos table is stale.
        self._pos_sin = None
        self._pos_cos = None

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @torch.no_grad()
    def get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a cached fp32 (sin, cos) table covering positions [0, seq_len).

        Built once and reused for the lifetime of the cache (rebuilt only when a
        longer table is requested, the device changes, or `inv_freq` is reset),
        mirroring LLaDA's `RotaryEmbedding.get_rotary_embedding`. Shape of each
        returned tensor is `[1, 1, seq_len, head_dim]`.
        """
        if (
            self._pos_sin is not None
            and self._pos_cos is not None
            and self._pos_sin.shape[-2] >= seq_len
        ):
            if self._pos_sin.device != device:
                self._pos_sin = self._pos_sin.to(device)
                self._pos_cos = self._pos_cos.to(device)
            return self._pos_sin[:, :, :seq_len, :], self._pos_cos[:, :, :seq_len, :]

        device_type = device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            inv_freq = self.inv_freq.to(device=device, dtype=torch.float)
            positions = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            pos_sin = (emb.sin() * self.attention_scaling)[None, None, :, :]
            pos_cos = (emb.cos() * self.attention_scaling)[None, None, :, :]

        self._pos_sin = pos_sin
        self._pos_cos = pos_cos
        return pos_sin, pos_cos

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Fused in-place RoPE application matching the GPT-NeoX rotate-half layout.

        `pos_sin`/`pos_cos` are position-selected tensors broadcastable to `t`
        (e.g. `[T, 1, head_dim]` against `t` of shape `[T, num_heads, head_dim]`).
        Writes the rotation through `torch.mul(out=...)` + `addcmul_` to avoid
        allocating `rotate_half` intermediates.
        """
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


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Dream
class DreamMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # Fused gate+up projection, materialized lazily by fuse_ff_gate_up()
        # after load. When set, the forward issues one wide GEMM feeding
        # flashinfer.silu_and_mul instead of two narrow GEMMs + SiLU + multiply.
        self.gate_up_proj: Optional[nn.Linear] = None
        self._ff_fused = False

    def fuse_ff_gate_up(self) -> None:
        # silu_and_mul applies SiLU to the first half of its input and the
        # unfused path applies act_fn to gate_proj, so gate_proj must come first
        # in the concatenation. Both projections are bias-free.
        if self._ff_fused:
            return
        dtype = self.gate_proj.weight.dtype
        gate_up_proj = nn.Linear(
            self.hidden_size,
            self.gate_proj.out_features + self.up_proj.out_features,
            bias=False,
            device="meta",
            dtype=dtype,
        )
        with torch.no_grad():
            weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
            gate_up_proj.weight = nn.Parameter(weight)
        self.gate_up_proj = gate_up_proj
        self._ff_fused = True

    def forward(self, hidden_state):
        if self._ff_fused:
            return self.down_proj(_fi_silu_and_mul(self.gate_up_proj(hidden_state)))
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DreamAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: DreamConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Fused QKV projection, materialized lazily by fuse_qkv() after load.
        # When set, the forward issues one wide GEMM + split instead of three
        # narrow GEMMs. q/k/v all carry bias, so the fused linear does too.
        self.fused_dims = (
            self.num_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
        )
        self.qkv_proj: Optional[nn.Linear] = None
        self._qkv_fused = False

        self.rotary_emb = DreamRotaryEmbedding(config=self.config)

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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        op_metrics_context = getattr(self, "_op_metrics_context", None)

        # The "attn_pre_proj" metric times only the qkv GEMM + split
        # (hidden_states is already normed by the decoder layer; the reshapes
        # below stay untimed).
        if self._qkv_fused:
            q, k, v = measure_operation_time(
                op_metrics_context,
                self.layer_idx,
                "attn_pre_proj",
                lambda: self.qkv_proj(hidden_states).split(self.fused_dims, dim=-1),
                hidden_states.device,
            )
        else:
            q, k, v = measure_operation_time(
                op_metrics_context,
                self.layer_idx,
                "attn_pre_proj",
                lambda: (
                    self.q_proj(hidden_states),
                    self.k_proj(hidden_states),
                    self.v_proj(hidden_states),
                ),
                hidden_states.device,
            )
        q = q.reshape(bsz * q_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(bsz * q_len, self.num_key_value_heads, self.head_dim)

        state = getattr(self, "_paged_attn_state", None)
        if state is None:
            raise RuntimeError("Dream attention requires paged attention state")

        layer_idx = state.next_layer_idx(self.layer_idx)

        # Apply RoPE at the correct absolute positions (fused, in-place)
        state.apply_rope_inplace(self.rotary_emb, q, k, self.head_dim)
        state.update_kv_pages(layer_idx, k, v)

        # The "attn" metric times only the FlashInfer kernel call (RoPE, KV
        # update, and the output projection are excluded).
        attn_output = measure_operation_time(
            op_metrics_context,
            self.layer_idx,
            "attn",
            lambda: state.run_attention(q, layer_idx),
            hidden_states.device,
        )
        state.finish_layer(layer_idx)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # The "attn_post_proj" metric times only the attention output projection.
        attn_output = measure_operation_time(
            op_metrics_context,
            self.layer_idx,
            "attn_post_proj",
            lambda: self.o_proj(attn_output),
            hidden_states.device,
        )

        return attn_output, None, past_key_value


class DreamDecoderLayer(nn.Module):
    def __init__(self, config: DreamConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

        self.self_attn = DreamAttention(config, layer_idx)

        self.mlp = DreamMLP(config)
        self.input_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # The fused norm/residual + silu_and_mul path is only numerically
        # equivalent for the served Dream-7B config: weight-only RMSNorm (always
        # true for DreamRMSNorm) and a plain SiLU activation feeding the split
        # gate/up silu_and_mul. ACT2FN["silu"] is SiLUActivation (not nn.SiLU),
        # so guard on the config string. Other configs use the unfused forward.
        self._fused_mlp_norm = (
            isinstance(self.input_layernorm, DreamRMSNorm)
            and isinstance(self.post_attention_layernorm, DreamRMSNorm)
            and config.hidden_act == "silu"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        if self._fused_mlp_norm and self.mlp._ff_fused:
            return self._forward_fused(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = measure_operation_time(
            getattr(self, "_op_metrics_context", None),
            self.layer_idx,
            "mlp",
            lambda: self.mlp(hidden_states),
            hidden_states.device,
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def _forward_fused(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_value: Optional[Tuple[torch.Tensor]],
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.FloatTensor]:
        """Fused inference path using FlashInfer norm/activation kernels.

        Equivalent to ``forward`` for the served Dream-7B config (guarded by
        ``self._fused_mlp_norm``). FlashInfer norm kernels operate on 2D
        ``(num_tokens, hidden)`` tensors, so activations are flattened and
        reshaped around them. Only used in the serving path where
        output_attentions/use_cache are False.
        """
        eps = self.input_layernorm.variance_epsilon
        C = hidden_states.shape[-1]

        residual = hidden_states
        normed = _fi_rmsnorm(
            hidden_states.view(-1, C), self.input_layernorm.weight, eps
        ).view_as(hidden_states)

        attn = self.self_attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]

        # Fold the attention residual add and post_attention_layernorm into one
        # in-place kernel: res2 += attn2; attn2 = rms_norm(res2).
        res2 = residual.view(-1, C)
        attn2 = attn.reshape(-1, C)
        _fi_fused_add_rmsnorm(attn2, res2, self.post_attention_layernorm.weight, eps)
        mlp_in = attn2.view_as(hidden_states)
        mlp_out = measure_operation_time(
            getattr(self, "_op_metrics_context", None),
            self.layer_idx,
            "mlp",
            lambda: self.mlp(mlp_in),
            hidden_states.device,
        )
        hidden_states = (res2 + mlp_out.view(-1, C)).view_as(hidden_states)
        return (hidden_states,)


class DreamPreTrainedModel(PreTrainedModel):
    config_class = DreamConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DreamDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class DreamBaseModel(DreamPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DreamDecoderLayer`]

    Args:
        config: DreamConfig
    """

    def __init__(self, config: DreamConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DreamDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DreamRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # RoPE is applied inside DreamAttention via the per-forward cached table on
        # PagedAttentionState; the attention layers ignore `position_embeddings`.
        position_embeddings = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DreamModel(DreamPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # Dream's lm_head (inherited from Qwen2) predicts position i+1 from
    # hidden_states[i]; the runner shifts logits right by 1 within each query
    # span before sampling. See generation_utils.py:_sample in the Dream repo.
    requires_logit_shift = True

    def __init__(self, config):
        super().__init__(config)
        self.model = DreamBaseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def num_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def num_q_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.config.hidden_size // self.config.num_attention_heads

    def fuse_qkv(self) -> None:
        """Materialize fused QKV weights on every decoder layer. Must be called
        after weights are loaded and before any forward pass."""
        for layer in self.model.layers:
            layer.self_attn.fuse_qkv()

    def fuse_ff_gate_up(self) -> None:
        """Materialize fused gate+up MLP weights on every decoder layer. Must be
        called after weights are loaded and before any forward pass."""
        for layer in self.model.layers:
            layer.mlp.fuse_ff_gate_up()

    def reset_rope_parameters(self):
        self.model.rotary_emb.reset_parameters()
        for layer in self.model.layers:
            layer.self_attn.rotary_emb.reset_parameters()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, MaskedLMOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
