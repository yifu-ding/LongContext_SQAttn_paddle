import paddle
from paddle_utils import *

from .cache import KV_Cache


def safe_copy_chunked(dst, src, max_chunk=1024):
    """
    将 src 分块 copy_ 到 dst 中（支持 Paddle GPU Tensor)。
    dst, src 必须形状一致。
    max_chunk: 每次复制的 seq_len 块大小。
    """
    assert dst.shape == src.shape, \
        f"Shape mismatch: dst={dst.shape}, src={src.shape}"
    total_len = dst.shape[1]
    for i in range(0, total_len, max_chunk):
        end = min(i + max_chunk, total_len)
        dst[:, i:end, :, :].copy_(src[:, i:end, :, :], blocking=True)
        paddle.device.synchronize()


class flash_attn_cache(KV_Cache):
    """
    A class representing the KV Cache of Full flash-attn.
    """

    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: paddle.dtype,
        layer_mapping: dict,
        num_gpus: int,
        model_size: int,
    ) -> None:
        super().__init__(
            layer_num,
            batch_size,
            max_length,
            num_key_value_heads,
            num_heads,
            head_dim,
            dtype,
            layer_mapping,
            num_gpus,
            model_size,
        )
        self.valid_start = valid_start
        self.valid_length = None
        self.batch_indices = paddle.arange(
            self.batch_size, dtype=paddle.int32, device=self.layer_mapping[str(0)]
        )
        self.allocated = self.pre_allocate_decision()
        if self.allocated:
            self.key_cache = [
                paddle.empty(
                    self.batch_size,
                    self.max_length,
                    self.kv_head,
                    self.head_dim,
                    device=self.layer_mapping[str(ldx)],
                    dtype=self.dtype,
                )
                for ldx in range(self.layer_num)
            ]
            self.value_cache = [
                paddle.empty(
                    self.batch_size,
                    self.max_length,
                    self.kv_head,
                    self.head_dim,
                    device=self.layer_mapping[str(ldx)],
                    dtype=self.dtype,
                )
                for ldx in range(self.layer_num)
            ]
        else:
            self.key_cache = [
                paddle.empty(
                    self.batch_size,
                    self.max_length,
                    self.kv_head,
                    self.head_dim,
                    device="cpu",
                    # pin_memory=True,
                    dtype=self.dtype,
                )
                for _ in range(self.layer_num)
            ]
            self.value_cache = [
                paddle.empty(
                    self.batch_size,
                    self.max_length,
                    self.kv_head,
                    self.head_dim,
                    device="cpu",
                    # pin_memory=True,
                    dtype=self.dtype,
                )
                for _ in range(self.layer_num)
            ]
        self.copystream = paddle.device.Stream()
        self.copyevents = {}
        self.KVreadyevents = {}
        # device_list = sorted(
        #     set(self.layer_mapping.values()), key=lambda x: int(x.split(":")[-1])
        # )
        # for device_idx in device_list:
        #     with paddle.device._convert_to_place(device=device2str(device_idx)):
        #         self.copyevents[device_idx] = paddle.device.cuda.Event()
        #         self.KVreadyevents[device_idx] = paddle.device.cuda.Event()
        device_list = sorted(
            set(self.layer_mapping.values()), key=lambda x: int(x.split(":")[-1])
        )
        for device_idx in device_list:
            # 1️⃣ 切换全局设备
            gpu_idx = int(device_idx.split(":")[-1])
            paddle.device.set_device(f"gpu:{gpu_idx}")
            # 2️⃣ 创建 CUDA Events
            self.copyevents[f"cuda:{gpu_idx}"] = paddle.device.cuda.Event()
            self.KVreadyevents[f"cuda:{gpu_idx}"] = paddle.device.cuda.Event()

    def pre_allocate_decision(self):
        kv_consumption = (
            2
            * self.layer_num
            * self.batch_size
            * self.max_length
            * self.kv_head
            * self.head_dim
            * 2
            / 1024
            / 1024
            / 1024
        )
        return self.free_memory > kv_consumption * 1.3

    def move_gpu(self):
        if not self.allocated:
            for ldx in range(self.layer_num):
                self.key_cache[ldx] = self.key_cache[ldx].to(
                    self.layer_mapping[str(ldx)]
                )
                self.value_cache[ldx] = self.value_cache[ldx].to(
                    self.layer_mapping[str(ldx)]
                )

    def prefill_update_kv_cache(
        self, query_states, key_states, value_states, layer_idx, start_bdx
    ):
        """
        update part of batches keys and values, start from start_bdx
        Args:
            query_states: (bsz, seq_len, num_heads, head_dim)
            key_states: (bsz, seq_len, kv_head, head_dim)
            value_states: (bsz, seq_len, kv_head, head_dim)
            layer_idx: the index of the layer
            start_bdx: the start index of the batch (=batch_idx)
        """
        paddle.device.cuda.empty_cache()
        bsz, seq_len, _, _ = key_states.shape
        assert (
            bsz == 1
        ), f"Multi-batch prefilling only support prefill single batch one by one."
        assert (
            seq_len <= self.max_length
        ), f"Prefilling sequence length {seq_len} exceeds max length {self.max_length}."
        self.KVreadyevents[self.layer_mapping[str(layer_idx)]].record()
        if self.valid_length is None:
            # self.valid_length = (
            #     paddle.from_numpy(seq_len - self.valid_start)
            #     .to(paddle.int32)
            #     .to(self.layer_mapping[str(0)])
            # )
            # self.valid_length = (
            #     paddle.to_tensor(seq_len - self.valid_start, dtype='int32')
            #     .to(self.layer_mapping[str(0)])
            # )
            if isinstance(self.valid_start, (tuple, list)):
                self.valid_length = (
                    paddle.to_tensor(seq_len - self.valid_start[0], dtype='int32')
                    .unsqueeze(0)
                    .to(self.layer_mapping[str(0)])
                )
            else:
                self.valid_length = (
                    paddle.to_tensor(seq_len - self.valid_start, dtype='int32')
                    .unsqueeze(0)
                    .to(self.layer_mapping[str(0)])
                )
                self.valid_start = paddle.to_tensor([self.valid_start])

        _valid_start = self.valid_start[start_bdx]
        _valid_length = seq_len - _valid_start
        with paddle.device.stream_guard(self.copystream):
            # self.KVreadyevents[self.layer_mapping[str(layer_idx)]].wait()
            
            # paddle.device.current_stream().wait_event(
            self.KVreadyevents[self.layer_mapping[str(layer_idx)]]
            # self.key_cache[layer_idx][start_bdx : start_bdx + bsz, :_valid_length, :, :].copy_(
            #     key_states[:, _valid_start:, :, :].astype(self.key_cache[layer_idx].dtype),)            

            # self.value_cache[layer_idx][start_bdx : start_bdx + bsz, :_valid_length, :, :].copy_(
            #     value_states[:, _valid_start:, :, :].astype(self.value_cache[layer_idx].dtype),)

            lhs = self.key_cache[layer_idx][start_bdx:start_bdx + bsz, :_valid_length, :, :]
            if self.allocated:
                rhs = key_states[:, _valid_start:_valid_start + _valid_length, :, :].astype(lhs.dtype)
            else:
                rhs = key_states[:, _valid_start:_valid_start + _valid_length, :, :].astype(lhs.dtype).cpu()

            safe_copy_chunked(lhs, rhs, max_chunk=131072)
            
            lhs = self.value_cache[layer_idx][start_bdx:start_bdx + bsz, :_valid_length, :, :]
            if self.allocated:
                rhs = value_states[:, _valid_start:_valid_start + _valid_length, :, :].astype(lhs.dtype)
            else:
                rhs = value_states[:, _valid_start:_valid_start + _valid_length, :, :].astype(lhs.dtype).cpu()

            safe_copy_chunked(lhs, rhs, max_chunk=131072)
            
            self.copyevents[self.layer_mapping[str(layer_idx)]].record()
        
        if layer_idx == self.layer_num - 1 and start_bdx + bsz == self.batch_size:
            self.context += seq_len
        return key_states[:, _valid_start:, :, :], value_states[:, _valid_start:, :, :]

    def sync(self, layer_idx, start_bdx):
        # self.copyevents[self.layer_mapping[str(layer_idx)]].wait()
        # paddle.device.current_stream().wait_event(
        self.copyevents[self.layer_mapping[str(layer_idx)]]
        # )

    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        update all batch of the key and value cache for decoding
        Args:
            key_states: (bsz, seq_len(=1), kv_head, head_dim)
            value_states: (bsz, seq_len(=1), kv_head, head_dim)
            layer_idx: the index of the layer
        """
        self.key_cache[layer_idx][
            self.batch_indices, self.valid_length, :, :
        ] = key_states[:, 0, :, :]
        self.value_cache[layer_idx][
            self.batch_indices, self.valid_length, :, :
        ] = value_states[:, 0, :, :]
        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.valid_length += 1
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
