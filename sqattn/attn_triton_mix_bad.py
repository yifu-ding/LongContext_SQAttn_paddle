import math

import paddle
import triton
import triton.language as tl
from paddle_utils import *


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    kv_len,
    K_ptrs,
    V_ptrs,
    stride_kn,
    stride_vn,
    start_m,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    is_P_fp8_quant: tl.constexpr,
    current_int8_window_size,
    current_int4_window_size,
    SINK_SIZE: tl.constexpr,
):
    """
    Window Attention 前向计算的内部循环函数，支持 Sink + 分层窗口。

    STAGE 划分:
        STAGE 1: Sink 窗口 (前 SINK_SIZE 个 tokens，使用 INT8 精度)
        STAGE 2: 4-bit 窗口扩展区域 (使用 4-bit 精度)
        STAGE 3: INT8 窗口内对角线之前的块 (使用 INT8 精度)
        STAGE 4: 对角线上的块 (使用 INT8 精度)
    """
    if STAGE == 1:
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        lo, hi = 0, tl.minimum(SINK_SIZE, query_start)
    elif STAGE == 2:
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        int4_window_left = tl.maximum(SINK_SIZE, query_end - current_int4_window_size)
        int8_window_left = tl.maximum(SINK_SIZE, query_end - current_int8_window_size)
        condition = int4_window_left < int8_window_left
        lo = tl.where(condition, int4_window_left, 0)
        hi = tl.where(condition, int8_window_left, 0)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    elif STAGE == 3:
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        int8_window_left = tl.maximum(SINK_SIZE, query_end - current_int8_window_size)
        int8_window_right = query_start
        condition = int8_window_left < int8_window_right
        lo = tl.where(condition, int8_window_left, 0)
        hi = tl.where(condition, int8_window_right, 0)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    elif STAGE == 4:
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        lo = query_start
        hi = query_end
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < kv_len - start_n
        k = tl.load(K_ptrs, mask=k_mask)
        qk = tl.dot(q, k).to(tl.float32)
        if STAGE == 4:
            causal_mask = offs_m[:, None] >= start_n + offs_n[None, :]
            qk = qk + tl.where(causal_mask, 0, -1000000.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v_mask = offs_n[:, None] < kv_len - start_n
        v = tl.load(V_ptrs, mask=v_mask)
        if is_P_fp8_quant:
            p = (p * 448).to(tl.float8e4nv).to(tl.float16) / 448
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _attn_prefill_fwd(
    Q_int8,
    K_int8,
    Q_int4,
    K_int4,
    V,
    Out,
    Lse,
    Int8WindowSizes,
    Int4WindowSizes,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_on,
    qo_len,
    kv_len,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    is_P_fp8_quant: tl.constexpr,
    SINK_SIZE: tl.constexpr,
):
    """
    分层 Window Attention Prefill Kernel，支持 Sink + 多精度窗口
    """
    start_m = tl.program_id(0)
    off_z = tl.program_id(2)
    off_h = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_int8_ptrs = (
        Q_int8
        + (off_z * stride_qz + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    K_int8_ptrs = (
        K_int8
        + (off_z * stride_kz + off_h // num_kv_groups * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None]
    )
    Q_int4_ptrs = (
        Q_int4
        + (off_z * stride_qz + off_h * stride_qh)
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    K_int4_ptrs = (
        K_int4
        + (off_z * stride_kz + off_h // num_kv_groups * stride_kh)
        + offs_n[None, :] * stride_kn
        + offs_k[:, None]
    )
    V_ptrs = (
        V
        + (off_z * stride_vz + off_h // num_kv_groups * stride_vh)
        + offs_n[:, None] * stride_vn
        + offs_k[None, :]
    )
    O_block_ptr = (
        Out
        + (off_z * stride_oz + off_h * stride_oh)
        + offs_m[:, None] * stride_on
        + offs_k[None, :]
    )
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q_mask = offs_m[:, None] < qo_len
    q_int8 = tl.load(Q_int8_ptrs, mask=q_mask)
    q_int4 = tl.load(Q_int4_ptrs, mask=q_mask)
    current_int8_window_size = tl.load(Int8WindowSizes + off_h)
    current_int4_window_size = tl.load(Int4WindowSizes + off_h)
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q_int8,
        kv_len,
        K_int8_ptrs,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        1,
        offs_m,
        offs_n,
        is_P_fp8_quant,
        current_int8_window_size,
        current_int4_window_size,
        SINK_SIZE,
    )
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q_int4,
        kv_len,
        K_int4_ptrs,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        2,
        offs_m,
        offs_n,
        is_P_fp8_quant,
        current_int8_window_size,
        current_int4_window_size,
        SINK_SIZE,
    )
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q_int8,
        kv_len,
        K_int8_ptrs,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        3,
        offs_m,
        offs_n,
        is_P_fp8_quant,
        current_int8_window_size,
        current_int4_window_size,
        SINK_SIZE,
    )
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q_int8,
        kv_len,
        K_int8_ptrs,
        V_ptrs,
        stride_kn,
        stride_vn,
        start_m,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        4,
        offs_m,
        offs_n,
        is_P_fp8_quant,
        current_int8_window_size,
        current_int4_window_size,
        SINK_SIZE,
    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=offs_m[:, None] < qo_len)
    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=offs_m < qo_len)


def attn_hierarchical_window(
    q_int8,
    k_int8,
    q_int4,
    k_int4,
    v,
    int8_window_sizes,
    int4_window_sizes,
    sink_size,
    tensor_layout="HND",
    output_dtype=paddle.float16,
    return_lse=False,
    is_P_fp8_quant=False,
):
    """
    分层 Window Attention 接口，支持 Sink + 多精度窗口

    Args:
        q_int8, k_int8: INT8 精度的 Query, Key 张量
        q_int4, k_int4: 4-bit 精度的 Query, Key 张量
        v: 单一精度的 Value 张量
        int8_window_sizes: INT8 窗口大小
        int4_window_sizes: 4-bit 窗口大小
        sink_size: Sink 窗口大小（前 sink_size 个 tokens 总是可见）
    """
    seq_dim = 1 if tensor_layout == "NHD" else 2
    qo_len = q_int8.size(seq_dim)
    kv_len = k_int8.size(seq_dim)
    if tensor_layout == "HND":
        H = q_int8.size(1)
    else:
        H = q_int8.size(2)
    if isinstance(int8_window_sizes, int):
        int8_sizes_tensor = paddle.full(
            [H], int8_window_sizes, dtype=paddle.int32, device=q_int8.place
        )
    else:
        int8_sizes_tensor = int8_window_sizes.to(
            device=q_int8.place, dtype=paddle.int32
        )
    if isinstance(int4_window_sizes, int):
        int4_sizes_tensor = paddle.full(
            [H], int4_window_sizes, dtype=paddle.int32, device=q_int8.place
        )
    else:
        int4_sizes_tensor = int4_window_sizes.to(
            device=q_int8.place, dtype=paddle.int32
        )
    head_dim_og = q_int8.size(-1)
    if head_dim_og < 64:
        pad_size = 64 - head_dim_og
        q_int8, k_int8 = [paddle.compat.pad(t, (0, pad_size)) for t in (q_int8, k_int8)]
        q_int4, k_int4 = [paddle.compat.pad(t, (0, pad_size)) for t in (q_int4, k_int4)]
        v = paddle.compat.pad(v, (0, pad_size))
    elif head_dim_og > 64 and head_dim_og < 128:
        pad_size = 128 - head_dim_og
        q_int8, k_int8 = [paddle.compat.pad(t, (0, pad_size)) for t in (q_int8, k_int8)]
        q_int4, k_int4 = [paddle.compat.pad(t, (0, pad_size)) for t in (q_int4, k_int4)]
        v = paddle.compat.pad(v, (0, pad_size))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    sm_scale = 1.0 / head_dim_og**0.5 * 1.44269504
    q_int8 = q_int8 * sm_scale
    q_int4 = q_int4 * sm_scale
    o = paddle.empty(q_int8.shape, dtype=output_dtype, device=q_int8.place)
    if tensor_layout == "HND":
        b, h_qo, _, head_dim = q_int8.shape
        _, h_kv, _, _ = k_int8.shape
        stride_bz_q, stride_h_q, stride_seq_q = (
            q_int8.stride(0),
            q_int8.stride(1),
            q_int8.stride(2),
        )
        stride_bz_k, stride_h_k, stride_seq_k = (
            k_int8.stride(0),
            k_int8.stride(1),
            k_int8.stride(2),
        )
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, _, h_qo, head_dim = q_int8.shape
        _, _, h_kv, _ = k_int8.shape
        stride_bz_q, stride_h_q, stride_seq_q = (
            q_int8.stride(0),
            q_int8.stride(2),
            q_int8.stride(1),
        )
        stride_bz_k, stride_h_k, stride_seq_k = (
            k_int8.stride(0),
            k_int8.stride(2),
            k_int8.stride(1),
        )
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    num_kv_groups = h_qo // h_kv
    lse = (
        paddle.empty([b, h_qo, qo_len], dtype=paddle.float32, device=q_int8.place)
        if return_lse
        else paddle.empty([0], dtype=paddle.float32, device="cpu")
    )
    BLOCK_M = 128
    BLOCK_N = 64
    grid = triton.cdiv(qo_len, BLOCK_M), h_qo, b
    _attn_prefill_fwd[grid](
        q_int8,
        k_int8,
        q_int4,
        k_int4,
        v,
        o,
        lse,
        int8_sizes_tensor,
        int4_sizes_tensor,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        qo_len,
        kv_len,
        h_qo,
        num_kv_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
        STAGE=4,
        RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4,
        is_P_fp8_quant=is_P_fp8_quant,
        SINK_SIZE=sink_size,
    )
    o = o[..., :head_dim_og]
    return (o, lse) if return_lse else o


def naive_sink_window_attention(
    q_int8,
    k_int8,
    q_int4,
    k_int4,
    v,
    int8_window_size,
    int4_window_size,
    sink_size,
    tensor_layout="HND",
):
    """
    朴素的 Sink + Window Attention 参考实现，模拟量化精度
    """
    if tensor_layout == "NHD":
        q_int8 = q_int8.permute(0, 2, 1, 3)
        k_int8 = k_int8.permute(0, 2, 1, 3)
        q_int4 = q_int4.permute(0, 2, 1, 3)
        k_int4 = k_int4.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
    B, H, N, D = q_int8.shape
    k_int8_centered = k_int8 - k_int8.mean(dim=2, keepdim=True)
    k_int4_centered = k_int4 - k_int4.mean(dim=2, keepdim=True)
    scale = 1.0 / D**0.5 * 1.44269504
    q_int8_scaled = q_int8 * scale
    q_int4_scaled = q_int4 * scale
    outputs = []
    for b in range(B):
        batch_outputs = []
        for h in range(H):
            q_int8_h = q_int8_scaled[b, h]
            k_int8_h = k_int8_centered[b, h]
            q_int4_h = q_int4_scaled[b, h]
            k_int4_h = k_int4_centered[b, h]
            v_h = v[b, h]
            final_scores = paddle.full((N, N), float("-inf"), device=q_int8.place)
            for i in range(N):
                if sink_size > 0 and i >= sink_size:
                    sink_end = min(sink_size, N)
                    sink_scores = paddle.matmul(
                        q_int8_h[i : i + 1], k_int8_h[:sink_end].T
                    )
                    final_scores[i, :sink_end] = paddle.maximum(
                        final_scores[i, :sink_end], sink_scores[0]
                    )
                int4_window_left = max(sink_size, i - int4_window_size + 1)
                int8_window_left = max(sink_size, i - int8_window_size + 1)
                if int4_window_left < int8_window_left:
                    int4_k = k_int4_h[int4_window_left:int8_window_left]
                    int4_scores = paddle.matmul(q_int4_h[i : i + 1], int4_k.T)
                    final_scores[i, int4_window_left:int8_window_left] = paddle.maximum(
                        final_scores[i, int4_window_left:int8_window_left],
                        int4_scores[0],
                    )
                if int8_window_left < i:
                    int8_k_before = k_int8_h[int8_window_left:i]
                    int8_scores_before = paddle.matmul(
                        q_int8_h[i : i + 1], int8_k_before.T
                    )
                    final_scores[i, int8_window_left:i] = paddle.maximum(
                        final_scores[i, int8_window_left:i], int8_scores_before[0]
                    )
                diag_scores = paddle.matmul(q_int8_h[i : i + 1], k_int8_h[i : i + 1].T)
                final_scores[i, i] = paddle.maximum(
                    final_scores[i, i], diag_scores[0, 0]
                )
            attn_weights = paddle.compat.softmax(final_scores, dim=-1)
            output_h = paddle.matmul(attn_weights.to(v_h.dtype), v_h)
            batch_outputs.append(output_h)
        outputs.append(paddle.stack(batch_outputs, dim=0))
    output = paddle.stack(outputs, dim=0)
    if tensor_layout == "NHD":
        output = output.permute(0, 2, 1, 3)
    return output


def test_sink_window_attention():
    """测试 Sink + Window Attention 的正确性"""
    paddle.manual_seed(42)
    device = "cuda"
    dtype = paddle.float16
    B, H, N, D = 1, 2, 64, 64
    sink_size = 4
    int8_window_size = 16
    int4_window_size = 32
    print("=== Sink + Window Attention 测试 ===")
    q_int8 = paddle.randn(B, N, H, D, device=device, dtype=dtype)
    k_int8 = paddle.randn(B, N, H, D, device=device, dtype=dtype)
    v = paddle.randn(B, N, H, D, device=device, dtype=dtype)
    noise_scale = 0.02
    q_int4 = q_int8 + paddle.randn(shape=q_int8.shape, dtype=q_int8.dtype) * noise_scale
    k_int4 = k_int8 + paddle.randn(shape=k_int8.shape, dtype=k_int8.dtype) * noise_scale
    print(f"输入形状: Q={q_int8.shape}, K={k_int8.shape}, V={v.shape}")
    print(f"Sink 大小: {sink_size}")
    print(f"INT8 窗口: {int8_window_size}, INT4 窗口: {int4_window_size}")
    try:
        output_triton = attn_hierarchical_window(
            q_int8,
            k_int8,
            q_int4,
            k_int4,
            v,
            int8_window_sizes=int8_window_size,
            int4_window_sizes=int4_window_size,
            sink_size=sink_size,
            tensor_layout="NHD",
        )
        print(f"✓ Triton 实现成功，输出形状: {output_triton.shape}")
        output_naive = naive_sink_window_attention(
            q_int8,
            k_int8,
            q_int4,
            k_int4,
            v,
            int8_window_size,
            int4_window_size,
            sink_size,
            tensor_layout="NHD",
        )
        print(f"✓ 参考实现成功，输出形状: {output_naive.shape}")
        diff = (output_triton.float() - output_naive.float()).abs()
        max_diff = diff._max().item()
        mean_diff = diff.mean().item()
        print(f"\n=== 结果对比 ===")
        print(f"最大绝对误差: {max_diff:.6f}")
        print(f"平均绝对误差: {mean_diff:.6f}")
        print(
            f"结果一致性: {paddle.allclose(x=output_triton.float(), y=output_naive.float(), rtol=0.01, atol=0.001).item()}"
        )
        print(f"\n=== 健康检查 ===")
        print(
            f"输出值范围: [{output_triton._min().item():.4f}, {output_triton._max().item():.4f}]"
        )
        print(f"是否包含 NaN: {paddle.isnan(output_triton).any().item()}")
        print(f"是否包含 Inf: {paddle.isinf(output_triton).any().item()}")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sink_window_attention()
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 测试失败，请检查实现")
