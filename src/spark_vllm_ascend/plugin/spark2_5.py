# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Spark model.

VENDORED FROM vLLM commit 81efe7883f30582696b69f9b9ea93c4819a8c608.
Keep this file aligned with the source commit; re-vendor it when the model
implementation changes upstream.
"""

from collections.abc import Iterable
from itertools import islice
import os
import functools

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)



# Keep cross-stream scheduling opaque to the graph compiler's buffer planner.
_SPARK_PARALLEL_LAYERS = {}

@torch.library.custom_op("spark_overlap::gated_attention", mutates_args=())
def _spark_gated_attention(hidden_states: torch.Tensor, positions: torch.Tensor, layer_key: str) -> torch.Tensor:
    return _SPARK_PARALLEL_LAYERS[layer_key]._parallel_attention(hidden_states, positions)

@_spark_gated_attention.register_fake
def _spark_gated_attention_fake(hidden_states, positions, layer_key):
    layer = _SPARK_PARALLEL_LAYERS[layer_key]
    return hidden_states.new_empty((hidden_states.shape[0], layer.q_size))

class Spark2_5MLP(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.activation_fn = GeluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states):
        intermediate_parallel, _ = self.gate_up_proj(hidden_states)
        intermediate_parallel = self.activation_fn(intermediate_parallel)
        output, _ = self.down_proj(intermediate_parallel)
        return output
    

class Spark2_5Attention(nn.Module):
    _spark_gate_stream = None

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = head_dim or self.hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5

        tp_size = get_tensor_model_parallel_world_size()
        self.spark_gate_overlap = os.environ.get("SPARK_GATE_OVERLAP", "0") == "1"
        self.spark_vector_fusion = os.environ.get("SPARK_VECTOR_FUSION", "0") == "1"
        if self.spark_vector_fusion and not self.spark_gate_overlap:
            raise ValueError("Vector fusion requires SPARK_GATE_OVERLAP=1 opaque path")
        self._spark_layer_key = prefix
        if self.spark_gate_overlap:
            if tp_size != 1 or quant_config is not None:
                raise ValueError("SPARK_GATE_OVERLAP currently requires unquantized TP=1")
            if Spark2_5Attention._spark_gate_stream is None:
                Spark2_5Attention._spark_gate_stream = torch.npu.Stream()
            _SPARK_PARALLEL_LAYERS[prefix] = self

        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        layer_type = config.layer_types[self.layer_idx]
        rope_parameters = dict(config.rope_parameters[layer_type])

        self.headwise_attn_output_gate = config.headwise_attn_output_gate
        if self.headwise_attn_output_gate:
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=None,  # g_proj keeps bf16
                prefix=f"{prefix}.g_proj",
            )
        self.q_k_v_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_k_v_proj",
        )

        self.core_attention = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=config.sliding_window if layer_type == "sliding_attention" else None,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.core_attention",
        )

        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rope_parameters,
        )

        if self.spark_vector_fusion:
            from .spark_vector_attention import initialize
            initialize(self)

    def _parallel_attention(self, hidden_states, positions):
        if self.spark_vector_fusion:
            from .spark_vector_attention import forward
            return forward(self, hidden_states, positions)
        main_stream = torch.npu.current_stream()
        gate_stream = Spark2_5Attention._spark_gate_stream
        gate_stream.wait_stream(main_stream)
        hidden_states.record_stream(gate_stream)
        with torch.npu.stream(gate_stream):
            g, _ = self.g_proj(hidden_states)
            g = torch.sigmoid(g.float()).to(hidden_states.dtype)
        qkv, _ = self.q_k_v_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.core_attention(q, k, v.contiguous())
        main_stream.wait_stream(gate_stream)
        g.record_stream(main_stream)
        return (attn_output.view(-1, self.num_heads, self.head_dim)
                * g.unsqueeze(-1)).reshape(-1, self.q_size)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.spark_gate_overlap and self.headwise_attn_output_gate and hidden_states.shape[0]>=128:
            attn_output = _spark_gated_attention(hidden_states, positions, self._spark_layer_key)
            output, _ = self.out_proj(attn_output)
            return output
        qkv, _ = self.q_k_v_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # CANN non-paged attention requires contiguous value tensors.
        attn_output = self.core_attention(q, k, v.contiguous())

        if self.headwise_attn_output_gate:
            g, _ = self.g_proj(hidden_states)
            g = torch.sigmoid(g.float()).to(attn_output.dtype)
            gate_output = (
                attn_output.view(*attn_output.shape[:-1], self.num_heads, self.head_dim)
                * g.unsqueeze(-1)
            )
            attn_output = gate_output.view(*attn_output.shape)

        output, _ = self.out_proj(attn_output)
        return output
    

class Spark2_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(hidden_size=self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size=self.hidden_size, eps=config.rms_norm_eps)

        self.mlp = Spark2_5MLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.self_attn = Spark2_5Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Spark2_5Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        if get_pp_group().is_first_rank:
            self.embedding = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embedding",
            )
        else:
            self.embedding = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Spark2_5DecoderLayer(
                vllm_config=vllm_config,
                config=config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        if os.environ.get("SPARK_V43", "0") == "1":
            from spark_vllm_ascend.integration.model import layer_run
            assert residual is None
            residual = hidden_states
            hidden_states = self.layers[0].input_layernorm(hidden_states)
            for layer in islice(self.layers, self.start_layer, self.end_layer):
                hidden_states, residual = layer_run(hidden_states, residual, positions, layer.self_attn.core_attention.layer_name)
            return hidden_states

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if os.environ.get("SPARK_V43", "0") == "1":
            from spark_vllm_ascend.integration.model import refresh_caches
            hidden_states = refresh_caches(hidden_states)
        return hidden_states


class Spark2_5ForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "q_k_v_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # vLLM 0.23 has no orig_to_new_stacked. Pack the same gate/up
    # checkpoint tensors before passing them to AutoWeightsLoader.
    hf_to_vllm_mapper = WeightsMapper()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = Spark2_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if self.config.tie_word_embeddings:
                self.lm_head = self.model.embedding
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=maybe_prefix(prefix, "lm_head")
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from spark_vllm_ascend.integration.weights import weights as gguf_weights
        weights = gguf_weights()
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        def packed_weights():
            pending = {}
            for name, weight in weights:
                shard = None
                for suffix, index in ((".gate_proj.weight", 0), (".up_proj.weight", 1)):
                    if name.endswith(suffix):
                        target = name[:-len(suffix)] + ".gate_up_proj.weight"
                        shard = index
                        break
                if shard is None:
                    yield name, weight
                    continue
                pair = pending.setdefault(target, {})
                if shard in pair:
                    raise ValueError(f"Duplicate checkpoint shard: {name}")
                pair[shard] = weight
                if len(pair) == 2:
                    yield target, torch.cat((pair[0], pair[1]), dim=0)
                    del pending[target]
            if pending:
                raise ValueError(f"Incomplete gate/up checkpoint pairs: {list(pending)}")
        loaded = loader.load_weights(packed_weights(), mapper=self.hf_to_vllm_mapper)
        from spark_vllm_ascend.integration.weights import install as install_gguf
        install_gguf(self)
        if os.environ.get('CANNBOT_CAPTURE')=='1':
            from dsl_probe import install as install_probe
            install_probe(self)
        if os.environ.get("SPARK_V43", "0") == "1":
            from spark_vllm_ascend.integration.model import packed
            for layer in self.model.layers:
                packed(layer, layer.self_attn.q_k_v_proj.weight)
            print("V43_WEIGHTS_PREPACKED", len(self.model.layers), flush=True)
        return loaded


if os.environ.get("SPARK_V43", "0") == "1":
    _init_original = Spark2_5ForCausalLM.__init__
    @functools.wraps(_init_original)
    def _init_v43(self, *args, **kwargs):
        _init_original(self, *args, **kwargs)
        from spark_vllm_ascend.integration.model import install
        install(self.model)
    Spark2_5ForCausalLM.__init__ = _init_v43

# Private compatibility override: identical upstream math, smaller head tile.
_rope_original_init = Spark2_5ForCausalLM.__init__
@functools.wraps(_rope_original_init)
def _rope_init(self, *args, **kwargs):
    import vllm_ascend.ops.rotary_embedding as rope_impl
    from spark_vllm_ascend.integration.rope import rope_forward_triton
    rope_impl.rope_forward_triton = rope_forward_triton
    _rope_original_init(self, *args, **kwargs)
Spark2_5ForCausalLM.__init__ = _rope_init
