import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from triton.tools.experimental_descriptor import create_2d_tma_descriptor


@triton.jit
def matmul_kernel_tma(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float8e4nv)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(tl.float16)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


@pytest.mark.parametrize(
    "M, N, K",
    [
        (4096, 3072, 2048),
        (4096, 576, 2048),
        (4096, 4096, 512),
        (4096, 2048, 2048),
        (4096, 5632, 2048),
        (4096, 2048, 2816),
        (4096, 2816, 2048),
        (2048, 2816, 4096),
        (4096, 2048, 5632),
        (5632, 2048, 4096),
        (2048, 4096, 2048),
        (4096, 512, 4096),
        (4096, 2048, 576),
        (4096, 2048, 3072),
        (3072, 2048, 4096),
        (2048, 2048, 4096),
    ],
)
@pytest.mark.parametrize("BLOCK_M", [16, 32, 64, 128])
@pytest.mark.parametrize("BLOCK_N", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("BLOCK_K", [32, 64, 128])
@pytest.mark.parametrize("num_warps", [4, 8, 16])
@pytest.mark.parametrize("num_stages", [1, 2])
def test_experimental_tma_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages):
    device = "musa"
    A = torch.randn((M, K), dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    B = torch.randn((K, N), dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    C = torch.zeros((M, N), dtype=torch.float16, device=device)
    desc_a = create_2d_tma_descriptor(A.data_ptr(), M, K, BLOCK_M, BLOCK_K, A.element_size())
    desc_b = create_2d_tma_descriptor(B.data_ptr(), K, N, BLOCK_K, BLOCK_N, B.element_size())
    desc_c = create_2d_tma_descriptor(C.data_ptr(), M, N, BLOCK_M, BLOCK_N, C.element_size())
    kernel = matmul_kernel_tma[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1,
                                1)](desc_a, desc_b, desc_c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=num_warps,
                                    num_stages=num_stages)
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-2, atol=1e-2)
