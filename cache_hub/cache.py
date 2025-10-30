import paddle


class KV_Cache:
    """
    A class representing the KV Cache.
    """

    def __init__(
        self,
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
        """Initializes the KV Cache.
        Args:
            layer_num (int)
            batch_size (int)
            num_key_value_heads (int)
            num_heads (int)
            max_length (int)
            head_dim (int)
            dtype (torch.dtype)
            layer_mapping (dict)
            num_gpus (int)
            model_size (int)
        """
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.max_length = max_length
        self.kv_head = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.layer_mapping = layer_mapping
        self.context = 0
        self.num_gpus = num_gpus
        self.model_size = model_size
        total_gpu_memory = (
            self.num_gpus
            * paddle.device.cuda.get_device_properties(0).total_memory
            / 1024
            / 1024
            / 1024
        )
        model_weight_consumption = self.model_size * 2
        prefill_consumption = (
            self.num_heads * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024
        )
        prefill_consumption += (
            self.num_heads * self.max_length * self.head_dim * 2 / 1024 / 1024 / 1024
        )
        prefill_consumption += (
            (self.num_heads + 2 * self.kv_head)
            * self.max_length
            * self.head_dim
            * 2
            / 1024
            / 1024
            / 1024
        )
        prefill_consumption += (
            4
            * self.num_heads
            * self.max_length
            * self.head_dim
            * 2
            / 1024
            / 1024
            / 1024
        )
        self.free_memory = (
            total_gpu_memory
            - model_weight_consumption
            - prefill_consumption * self.num_gpus
        )
