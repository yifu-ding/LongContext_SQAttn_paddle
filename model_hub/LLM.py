import time
import gc
import paddle
from termcolor import colored
paddle.set_grad_enabled(False)

def print_gpu_mem(tag=""):
    device = paddle.device.get_device()
    if device.startswith("gpu"):
        gpu_id = int(device.split(":")[-1])
        alloc = paddle.device.cuda.memory_allocated(gpu_id) / 1024**3
        resv = paddle.device.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"[{tag}] allocated={alloc:.2f} GB, reserved={resv:.2f} GB on {device}")
    else:
        print(f"[{tag}] running on {device}, no GPU memory info.")
    return alloc, resv

class LLM(paddle.nn.Layer):
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, model_name: str, max_length: int, dtype: paddle.dtype, device_map: str
    ) -> None:
        """Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map
        self.capture_layer_inps = None

    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        if self.capture_layer_inps is not None:
            self.capture_layer_inps.append(hidden_states.detach())

        bsz, seq_len, dim = hidden_states.shape 
        layer = self.layers[layer_idx]
        temp_hidden_states = hidden_states.clone()  
        for start_idx in range(0, seq_len, 8192 // bsz):
            end_idx = min(seq_len, start_idx + 8192 // bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(
                temp_hidden_states[:, start_idx:end_idx, :],
                layer.input_layernorm_variance_epsilon,
                layer.input_layernorm_weight,
            )
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        query_states, key_states = self.position_embedd(query_states, key_states)
        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        )  
        key_states, value_states = self.kv_cache.prefill_update_kv_cache(
            query_states, key_states, value_states, layer_idx, start_bdx
        )  
        temp_attn_out = layer.prefill_attention(query_states, key_states, value_states, None)
        self.kv_cache.sync(layer_idx, start_bdx)
        hidden_states += self.wo(temp_attn_out, layer, bsz, seq_len, dim)
        residual = hidden_states.clone()
        for start_idx in range(0, seq_len, 8192 // bsz):
            end_idx = min(seq_len, start_idx + 8192 // bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(
                hidden_states[:, start_idx:end_idx, :],
                layer.post_attention_layernorm_variance_epsilon,
                layer.post_attention_layernorm_weight,
            )
            hidden_states[:, start_idx:end_idx, :] = self.mlp(
                hidden_states[:, start_idx:end_idx, :], layer
            )
        hidden_states += residual
        del residual, query_states, key_states, value_states, temp_attn_out
        paddle.device.cuda.empty_cache()
        return hidden_states

    def layer_decode(self, layer_idx, hidden_states):
        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        hidden_states = self.layernorm(
            hidden_states,
            layer.input_layernorm_variance_epsilon,
            layer.input_layernorm_weight,
        )
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(
            bsz, -1, self.num_key_value_heads, self.head_dim
        )
        key_states, value_states = self.kv_cache.decode_update_kv_cache(
            key_states, value_states, layer_idx
        )
        attn_out = layer.decode_attention(query_states, key_states, value_states, None)[0]
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layernorm(
            hidden_states,
            layer.post_attention_layernorm_variance_epsilon,
            layer.post_attention_layernorm_weight,
        )
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states
        return hidden_states

    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.place
        last_hidden_states = paddle.empty(
            (bsz, 1, self.hidden_size), dtype=self.dtype, device=device
        )
        paddle.device.cuda.empty_cache()
        gc.collect()
        for start_bdx in range(0, bsz, 1):
            print(f"Prefill batch {start_bdx} of {bsz}", flush=True)
            end_bdx = min(bsz, start_bdx + 1)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])
            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                    paddle.device.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(
                    self.layers[0].place
                )
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    paddle.device.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        last_hidden_states = self.layernorm(
            last_hidden_states.contiguous(),
            self.norm_variance_epsilon,
            self.norm_weight,
        )
        logits = self.lm(last_hidden_states)
        return logits

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)
        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].place)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
        hidden_states = self.layernorm(
            hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight
        )
        logits = self.lm(hidden_states)
        return logits

    def inference(self, inputs_ids):
        outputs_ids = []
        output_ids = []
        print("Start prefilling ...")
        
        paddle.device.synchronize()
        prefill_start = time.time()
        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)
        self.move()
        paddle.device.synchronize()
        prefill_end = time.time()
        print(
            colored(
                f"Prefilling latency: {round(prefill_end - prefill_start, 4)} s\n",
                "green",
            )
        )
        if self.max_new_length > 0:
            print("Start decoding ...")
            decode_start = time.time()
            for _ in range(self.max_new_length - 1):
                logits = self.decode_forward(inputs_ids=output_ids)
                output_ids = logits.argmax(dim=-1)
                outputs_ids.append(output_ids)
            decode_end = time.time()
            print(
                colored(
                    f"""Decoding latency: {round((decode_end - decode_start) * 1000 / (self.max_new_length - 1), 2)} ms/step, Throughput: {round(self.batch_size * (self.max_new_length - 1) / (decode_end - decode_start), 2)} tokens/s
                    """,
                    "green",
                )
            )
        outputs_ids = paddle.cat(outputs_ids, dim=-1).tolist()
        return outputs_ids

    def generate(
        self,
        attention_type,
        inputs_ids,
        attention_masks,
        max_new_length,
    ):
        """LLM Inference.
        Args:
            attention_type: str,
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
        """
        bs, input_length = inputs_ids.shape
        assert (
            input_length + max_new_length <= self.max_length
        ), f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"
        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type
        if attention_masks is None:
            valid_start = (0, )
        else:
            valid_start = (attention_masks.shape[1] - paddle.sum(attention_masks).detach().cpu().numpy())
        del attention_masks
        paddle.device.cuda.empty_cache()
        print("Allocate GPU buffers and CPU pin memory ...\n")
        self.init_kv_cache(input_length, valid_start)
        outputs = self.inference(inputs_ids)
        return outputs
