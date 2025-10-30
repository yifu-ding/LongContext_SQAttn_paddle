import gc
import math
import re

import paddle
from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from cache_hub import flash_attn_cache
from .utils import *
from .LLM import LLM
from sqattn.sdpa_attention import sdpa_attention_forward
from sqattn.attn_replacer import cal_attn_weight


class QwenLayer(paddle.nn.Layer):
    """
    A class representing the Qwen layer.
    """

    def __init__(self, layer_idx, device, attn_type, config, max_length) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device
        self.attention_type = attn_type

        self.config = config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.max_length = max_length
        self.position_ids = paddle.arange(0, self.max_length).to("gpu:0")
        self.num_gpus = paddle.device.cuda.device_count()
    
    def prefill_attention(self, query_states, key_states, value_states, layer_idx=None):
        if layer_idx == 0 or layer_idx == self.num_layers - 1: # skip the first and last layer
            attn_weights = cal_attn_weight(self, query_states, key_states)
            setattr(self, "current_attention", attn_weights)
        if self.attention_type == "SQAttn" or self.attention_type == "SDPA":
            return sdpa_attention_forward(self, query_states, key_states, value_states, attention_mask=None, dropout=0.0, scaling=1.0, is_causal=True, tensor_layout="NHD")
        elif self.attention_type == "Full_Flash_Attn":
            return paddle.nn.functional.flash_attention.flash_attention(query_states, key_states, value_states, causal=True)[0]
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

    def decode_attention(self, query_states, key_states, value_states, layer_idx=None):
        if self.attention_type == "SDPA":
            return sdpa_attention_forward(self, query_states, key_states, value_states, attention_mask=None, dropout=0.0, scaling=1.0, is_causal=True, tensor_layout="NHD")
        elif self.attention_type == "Full_Flash_Attn" or self.attention_type == "SQAttn":
            return paddle.nn.functional.flash_attention.flash_attention(query_states, key_states, value_states, causal=True)[0]
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        
    def init_layer(self, hf_qwen_layer):
        self.wq = hf_qwen_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_qwen_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_qwen_layer.self_attn.v_proj.weight.detach()
        self.bq = hf_qwen_layer.self_attn.q_proj.bias.detach()
        self.bk = hf_qwen_layer.self_attn.k_proj.bias.detach()
        self.bv = hf_qwen_layer.self_attn.v_proj.bias.detach()
        self.wqkv = paddle.cat((self.wq, self.wk, self.wv), dim=0).to(
            self.device, 
        )
        self.bqkv = paddle.cat((self.bq, self.bk, self.bv), dim=0).to(
            self.device, 
        )
        self.wo = hf_qwen_layer.self_attn.o_proj.weight.detach().to(
            self.device, 
        )
        self.gate_proj = hf_qwen_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_qwen_layer.mlp.up_proj.weight.detach()
        self.gate_up_proj = paddle.cat((self.gate_proj, self.up_proj), dim=0).to(
            self.device, 
        )
        self.down_proj = hf_qwen_layer.mlp.down_proj.weight.detach().to(
            self.device, 
        )
        self.input_layernorm_weight = hf_qwen_layer.input_layernorm.weight.detach().to(
            self.device, 
        )
        self.input_layernorm_variance_epsilon = (
            hf_qwen_layer.input_layernorm.variance_epsilon
        )
        self.post_attention_layernorm_weight = (
            hf_qwen_layer.post_attention_layernorm.weight.detach().to(
                self.device, 
            )
        )
        self.post_attention_layernorm_variance_epsilon = (
            hf_qwen_layer.post_attention_layernorm.variance_epsilon
        )
        del (
            self.wq,
            self.wk,
            self.wv,
            self.bq,
            self.bk,
            self.bv,
            self.gate_proj,
            self.up_proj,
        )



class QwenModel(LLM):
    """
    A class representing the Qwen model.
    """

    def __init__(
        self, model_name: str, max_length: int, dtype: paddle.dtype, device_map: str, attention_type: str
    ) -> None:
        super().__init__(model_name, max_length, dtype, device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.base = self.config.rope_theta
        self.max_position_embeddings = self.config.max_position_embeddings
        self.yarn_factor = 4
        self.vocab_size = self.config.vocab_size
        self.eos_tokens = [self.config.eos_token_id]
        self.attention_type = attention_type
        self.init_model()

    def _set_cos_sin_cache(self):
        if self.max_length > 32768:

            def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
                """Inverse dimension formula to find the dimension based on the number of rotations"""
                return (
                    dim
                    * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            def find_correction_range(
                low_rot, high_rot, dim, base, max_position_embeddings
            ):
                """Find dimension range bounds based on rotations"""
                low = math.floor(
                    find_correction_dim(low_rot, dim, base, max_position_embeddings)
                )
                high = math.ceil(
                    find_correction_dim(high_rot, dim, base, max_position_embeddings)
                )
                return max(low, 0), min(high, dim - 1)

            def linear_ramp_factor(min, max, dim):
                if min == max:
                    max += 0.001
                linear_func = (paddle.arange(dim, dtype=paddle.float32) - min) / (
                    max - min
                )
                ramp_func = paddle.clamp(linear_func, 0, 1)
                return ramp_func

            attention_factor = 0.1 * math.log(self.yarn_factor) + 1.0
            beta_fast = 32
            beta_slow = 1
            pos_freqs = self.base ** (
                paddle.arange(0, self.head_dim, 2).float().to(self.inv_freq.place)
                / self.head_dim
            )
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (self.yarn_factor * pos_freqs)
            low, high = find_correction_range(
                beta_fast,
                beta_slow,
                self.head_dim,
                self.base,
                self.max_position_embeddings,
            )
            inv_freq_extrapolation_factor = 1 - linear_ramp_factor(
                low, high, self.head_dim // 2
            ).float().to(self.inv_freq.place)
            inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
                + inv_freq_extrapolation * inv_freq_extrapolation_factor
            )
            self.inv_freq = inv_freq
            self.attention_scaling = attention_factor
        t = paddle.arange(
            self.max_length, device=self.inv_freq.place, dtype=self.inv_freq.dtype
        )
        freqs = paddle.outer(t, self.inv_freq)
        return (
            freqs.cos() * self.attention_scaling,
            freqs.sin() * self.attention_scaling,
        )

    def init_model(self):
        hf_qwen = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=self.dtype, low_cpu_mem_usage=True
        )
        self.num_gpus = paddle.device.cuda.device_count() if self.device_map == "auto" else 1
        if self.device_map == "auto" and self.num_gpus == 1:
            self.device_map = "cuda:0"
        if self.device_map != "auto":
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): self.device_map})
            self.embed_tokens = hf_qwen.model.embed_tokens.weight.detach().to(
                self.device_map, 
            )
            self.lm_head = hf_qwen.lm_head.weight.detach().to(
                self.device_map, 
            )
            self.norm_weight = hf_qwen.model.norm.weight.detach().to(
                self.device_map, 
            )
            self.norm_variance_epsilon = hf_qwen.model.norm.variance_epsilon
            self.position_ids = paddle.arange(0, self.max_length).to(
                self.device_map, 
            )
            self.inv_freq = hf_qwen.model.rotary_emb.inv_freq.detach().to(
                self.device_map, 
            )
            self.attention_scaling = hf_qwen.model.rotary_emb.attention_scaling
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = paddle.cat((self.cos_cache, self.sin_cache), dim=-1)
            self.layers = []
            for idx, hf_qwen_layer in enumerate(hf_qwen.model.layers):
                qwen_layer = QwenLayer(idx, device=self.device_map, attn_type=self.attention_type, config=self.config, max_length=self.max_length)
                qwen_layer.init_layer(hf_qwen_layer)
                self.layers.append(qwen_layer)
                hf_qwen.model.layers[idx] = None
        # else:
        #     self.gpu_ids = list(range(self.num_gpus))
        #     self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
        #     self.layer_mapping = {}
        #     for ldx in range(0, self.num_layers):
        #         self.layer_mapping.update(
        #             {str(ldx): f"cuda:{ldx // self.layer_interval}"}
        #         )
        #     self.embed_tokens = hf_qwen.model.embed_tokens.weight.detach().to(
        #         f"cuda:{self.gpu_ids[0]}", 
        #     )
        #     self.lm_head = hf_qwen.lm_head.weight.detach().to(
        #         f"cuda:{self.gpu_ids[0]}", 
        #     )
        #     self.norm_weight = hf_qwen.model.norm.weight.detach().to(
        #         f"cuda:{self.gpu_ids[0]}", 
        #     )
        #     self.norm_variance_epsilon = hf_qwen.model.norm.variance_epsilon
        #     self.position_ids = paddle.arange(0, self.max_length).to(
        #         f"cuda:{self.gpu_ids[0]}", 
        #     )
        #     self.inv_freq = hf_qwen.model.rotary_emb.inv_freq.detach().to(
        #         f"cuda:{self.gpu_ids[0]}", 
        #     )
        #     self.attention_scaling = hf_qwen.model.rotary_emb.attention_scaling
        #     self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
        #     self.cos_sin_cache = paddle.cat((self.cos_cache, self.sin_cache), dim=-1)
        #     self.layers = []
        #     for ldx, hf_qwen_layer in enumerate(hf_qwen.model.layers):
        #         qwen_layer = QwenLayer(ldx, device=self.layer_mapping[str(ldx)], attn_type=self.attention_type, config=self.config, max_length=self.max_length)
        #         qwen_layer.init_layer(hf_qwen_layer)
        #         self.layers.append(qwen_layer)
        #         hf_qwen.model.layers[ldx] = None
        del self.inv_freq, self.cos_cache, self.sin_cache
        gc.collect()
        paddle.device.cuda.empty_cache()

    def init_kv_cache(self, real_input_length, valid_start):
        self.kv_cache = flash_attn_cache(
            valid_start=valid_start,
            layer_num=self.num_layers,
            batch_size=self.batch_size,
            max_length=self.max_new_length + real_input_length,
            num_key_value_heads=self.num_key_value_heads,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            layer_mapping=self.layer_mapping,
            num_gpus=self.num_gpus,
            model_size=int(re.search("(\\d+)[B]", self.model_name).group(1)),
        )
      
    def move(self):
        paddle.device.cuda.empty_cache()
        self.kv_cache.move_gpu()
        paddle.device.cuda.empty_cache()

    def word_embedding(self, inputs_id):
        hidden_states = paddle.nn.functional.embedding(inputs_id, self.embed_tokens)
        return hidden_states

    def lm(self, hidden_states):
        logits = paddle.nn.functional.linear(
            x=hidden_states, weight=self.lm_head.T
        ).float()
        return logits

    def wqkv(self, hidden_states, layer):
        qkv = paddle.nn.functional.linear( x=hidden_states, weight=layer.wqkv.T, bias=layer.bqkv   )
        query_states, key_states, value_states = qkv.split(
            [
                self.hidden_size,
                self.hidden_size // self.num_key_value_groups,
                self.hidden_size // self.num_key_value_groups,
            ],
            dim=-1,
        )
        return query_states, key_states, value_states

    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        hidden_states = paddle.nn.functional.linear(x=hidden_states, weight=layer.wo.T)
        return hidden_states

    def mlp(self, hidden_states, layer):
        hidden_states = paddle.nn.functional.linear(
            x=hidden_states, weight=layer.gate_up_proj.T
        )
        # flashinfer.activation.silu_and_mul(hidden_states, out)  
        out = silu_and_mul(hidden_states)
        hidden_states = paddle.nn.functional.linear(x=out, weight=layer.down_proj.T)   
        return hidden_states

    def parameter_move(self, hidden_states, ldx):
        next_device = (
            self.layer_mapping[str(ldx + 1)]
            if str(ldx + 1) in self.layer_mapping
            else self.layer_mapping[str(0)]
        )
        paddle.device.cuda.set_device(next_device)
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.cos_sin_cache = self.cos_sin_cache.to(next_device)
        if hidden_states.shape[1] == 1:
            self.kv_cache.batch_indices = self.kv_cache.batch_indices.to(
                next_device
            )
            self.kv_cache.valid_length = self.kv_cache.valid_length.to(next_device)
        return hidden_states

    def layernorm(self, hidden_states, epsilon, weight):
        bsz, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.reshape(bsz * seq_len, dim)
        hidden_states = rmsnorm(hidden_states, weight, epsilon)
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        return hidden_states

    def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
        bsz, _, hidden_dim = query_states.shape
        _, _, kv_dim = key_states.shape
        query_states = query_states.view(-1, hidden_dim)
        key_states = key_states.view(-1, kv_dim)
        query_states, key_states = apply_rope_with_cos_sin_cache_inplace(
            position_ids,
            query_states,
            key_states,
            self.head_dim,
            self.cos_sin_cache,
            True,
        )  
        query_states = query_states.reshape([bsz, -1, hidden_dim])   
        key_states = key_states.reshape([bsz, -1, kv_dim])
        return query_states, key_states   

    def position_embedd(self, query_states, key_states):
        bsz, seq_len, _ = key_states.shape
        position_ids = (
            self.position_ids[self.kv_cache.context : self.kv_cache.context + seq_len]
            .unsqueeze(0)
            .repeat(bsz, 1)
        )
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, position_ids
        )
        return query_states, key_states

