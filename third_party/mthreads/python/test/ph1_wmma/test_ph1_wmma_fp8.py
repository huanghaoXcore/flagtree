import os

os.environ["DISABLE_SQMMA"] = "1"

import pytest
import torch
import triton
import triton.language as tl
import torch_musa


def get_resolution(dtype):
    atol_resolution_map = {
        torch.float16: 2.1e-3,
        torch.bfloat16: 1.4e-3,
        torch.float8_e4m3fn: 1.25e-1,
        torch.float8_e5m2: 2.5e-1,
    }
    rtol_resolution_map = {
        torch.float16: 1e-3,
        torch.bfloat16: 7.9e-3,
        torch.float8_e4m3fn: 1.25e-1,
        torch.float8_e5m2: 2.5e-1,
    }
    return atol_resolution_map.get(dtype, None), rtol_resolution_map.get(dtype, None)


def torch_matmul(A: torch.Tensor, B: torch.Tensor):
    return torch.mm(A, B)


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=A_ptr.dtype.element_ty)
    B_block = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=B_ptr.dtype.element_ty)
    C_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        mask_A = tl.where(offs_m[:, None] < M, k + offs_k[None, :] < K, False)
        mask_B = tl.where(k + offs_k[:, None] < K, offs_n[None, :] < N, False)
        A_block = tl.load(
            A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=mask_A,
            other=0.0,
        )
        B_block = tl.load(
            B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_B,
            other=0.0,
        )
        C_block += tl.dot(A_block, B_block)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        C_block,
        mask=mask,
    )


def triton_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    block_size_m,
    block_size_n,
    block_size_k,
    num_warps,
):
    M, K = A.shape
    K, N = B.shape
    C = torch.zeros((M, N), device="musa", dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        num_warps=num_warps,
        num_stages=1,
        en_wmma=True,
    )
    return C


DTYE_CONFIG = [torch.float8_e4m3fn, torch.float8_e5m2]
MNK = [
    (4096, 4096, 4096),
]
BLOCK_CONFIG = [
    (8, 16, 16),
    (8, 16, 32),
    (8, 16, 64),
    (8, 32, 16),
    (8, 32, 32),
    (8, 32, 64),
    (8, 64, 16),
    (8, 64, 32),
    (8, 64, 64),
    (16, 8, 16),
    (16, 8, 32),
    (16, 8, 64),
    (16, 16, 16),
    (16, 16, 32),
    (16, 16, 64),
    (16, 32, 64),
    (16, 64, 64),
    (32, 8, 16),
    (32, 8, 32),
    (32, 8, 64),
    (32, 16, 16),
    (32, 16, 32),
    (32, 16, 64),
    (32, 32, 16),
    (32, 32, 32),
    (32, 32, 64),
    (32, 64, 64),
    (64, 8, 16),
    (64, 8, 32),
    (64, 8, 64),
    (64, 16, 16),
    (64, 16, 32),
    (64, 16, 64),
    (64, 32, 16),
    (64, 32, 32),
    (64, 32, 64),
    (64, 64, 16),
    (64, 64, 32),
    (64, 64, 64),
]
WARP_CONFIG = [
    1,
]
CONFIG = [(dtype, M, N, K, block_m, block_n, block_k, num_warps)
          for dtype in DTYE_CONFIG
          for M, N, K in MNK
          for block_m, block_n, block_k in BLOCK_CONFIG
          for num_warps in WARP_CONFIG]


@pytest.mark.parametrize("dtype,M,N,K,block_m,block_n,block_k,num_warps", CONFIG)
def test_matmul(dtype, M, N, K, block_m, block_n, block_k, num_warps):
    atol, rtol = get_resolution(dtype)
    A = torch.randn(M, K, device="musa", dtype=torch.float16).to(dtype)
    B = torch.randn(K, N, device="musa", dtype=torch.float16).to(dtype)
    C_triton = triton_matmul(A, B, block_m, block_n, block_k, num_warps)
    C_torch = torch_matmul(A, B).to(torch.float32)
    torch.testing.assert_close(C_torch, C_triton, atol=atol, rtol=rtol)
