import paddle


def silu_and_mul(hidden_states: paddle.Tensor) -> paddle.Tensor:
    """
    Paddle 等价实现 flashinfer.activation.silu_and_mul
    输入: [B, T, 2*D]
    输出: [B, T, D]
    计算: silu(x1) * x2
    """
    # 切分为两半
    x1, x2 = paddle.split(hidden_states, 2, axis=-1)
    return x2 * (x1 * paddle.nn.functional.sigmoid(x1))

def rmsnorm(x: paddle.Tensor, weight: paddle.Tensor, epsilon: float = 1e-6):
    # 计算均方根归一化
    variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
    norm_x = x * paddle.rsqrt(variance + epsilon)
    return norm_x * weight

def apply_rope_with_cos_sin_cache_inplace(
    position_ids: paddle.Tensor,
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    head_dim: int,
    cos_sin_cache: paddle.Tensor,
    use_cache: bool = True,
):
    """
    Args:
        position_ids: [batch_size, seq_len]
        query_states: [batch * seq_len, num_heads * head_dim]
        key_states:   [batch * seq_len, num_heads * head_dim]
        cos_sin_cache: [max_seq_len, head_dim] (cos|sin cache)
    """
    dtype = query_states.dtype  # 32.09GB
    bsz, seq_len = position_ids.shape
    _, num_heads_x_head_dim = query_states.shape
    num_heads = num_heads_x_head_dim // head_dim
    kv_heads = key_states.shape[1] // head_dim

    # 拆出 cos / sin
    cos = cos_sin_cache[..., : head_dim // 2]
    sin = cos_sin_cache[..., head_dim // 2 :]   # 32.84GB
    cos = cos.detach()
    sin = sin.detach()

    # 按 position_ids 取缓存（注意 axis=0）
    cos = paddle.index_select(cos, position_ids.flatten(), axis=0).reshape([bsz, seq_len, head_dim // 2])  
    sin = paddle.index_select(sin, position_ids.flatten(), axis=0).reshape([bsz, seq_len, head_dim // 2])   # 32.61GB

    # broadcast: [B, num_heads, S, D/2]
    # cos_q = cos.unsqueeze(1).expand([bsz, num_heads, seq_len, head_dim // 2])
    # sin_q = sin.unsqueeze(1).expand([bsz, num_heads, seq_len, head_dim // 2])
    cos_q = cos[:, None, :, :].tile([1, num_heads, 1, 1])  # 53.34GB
    sin_q = sin[:, None, :, :].tile([1, num_heads, 1, 1])
    
    # cos_kv = cos.unsqueeze(1).expand([bsz, kv_heads, seq_len, head_dim // 2])
    # sin_kv = sin.unsqueeze(1).expand([bsz, kv_heads, seq_len, head_dim // 2])   # 48.29GB
    cos_kv = cos[:, None, :, :].tile([1, kv_heads, 1, 1])
    sin_kv = sin[:, None, :, :].tile([1, kv_heads, 1, 1])
    
    del cos, sin, cos_sin_cache, position_ids
    paddle.device.cuda.empty_cache()   # 52.56GB
    
    # 调整 query/key 形状以匹配
    query_states = query_states.reshape([bsz, seq_len, num_heads, head_dim]).permute([0, 2, 1, 3])  
    key_states   = key_states.reshape([bsz, seq_len, kv_heads, head_dim]).permute([0, 2, 1, 3])
    
    q_out = paddle.empty_like(query_states)   # 59GB
    k_out = paddle.empty_like(key_states)

    # 拆分奇偶维
    # q1, q2 = paddle.split(query_states, 2, axis=-1)
    # k1, k2 = paddle.split(key_states, 2, axis=-1)
    half = head_dim // 2
    q1 = query_states[..., :half]
    q2 = query_states[..., half:]
    k1 = key_states[..., :half]
    k2 = key_states[..., half:]
    
    q_out[..., :half] = q1 * cos_q - q2 * sin_q
    q_out[..., half:] = q1 * sin_q + q2 * cos_q   # 81GB
    k_out[..., :half] = k1 * cos_kv - k2 * sin_kv
    k_out[..., half:] = k1 * sin_kv + k2 * cos_kv # 81GB
    
    query_states[..., :half] = q1 * cos_q - q2 * sin_q
    query_states[..., half:] = q1 * sin_q + q2 * cos_q   # 81GB
    key_states[..., :half] = k1 * cos_kv - k2 * sin_kv
    key_states[..., half:] = k1 * sin_kv + k2 * cos_kv # 81GB

    q_out = q_out.transpose([0, 2, 1, 3])
    k_out = k_out.transpose([0, 2, 1, 3]) # 81GB
    
    
    # 显存清理
    del q1, q2, k1, k2, cos_q, sin_q, cos_kv, sin_kv, query_states, key_states  
    # gc.collect()
    paddle.device.cuda.empty_cache()  # 45.66GB
    
    if q_out.dtype != dtype:
        q_out = q_out.astype(dtype)
    if k_out.dtype != dtype:
        k_out = k_out.astype(dtype)
    
    return q_out, k_out
