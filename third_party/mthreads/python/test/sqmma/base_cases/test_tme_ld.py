import torch
import triton
import triton.language as tl
import pytest

from triton.tools.experimental_descriptor import (
    create_1d_tma_descriptor,
    create_2d_tma_descriptor,
    create_3d_tma_descriptor,
)


@pytest.mark.parametrize("is_byval", [True, False])
def test_tme_1d_ld(is_byval):
    device = "musa"
    SIZE = 64

    @triton.jit
    def kernel(Z, desc, SIZE: tl.constexpr):
        off_desc = 0
        off = tl.arange(0, SIZE)
        x = tl._experimental_descriptor_load(desc, [off_desc], [SIZE], Z.dtype.element_ty)
        tl.store(Z + off, x)

    x = torch.ones(SIZE, dtype=torch.float32, device=device)
    if is_byval:
        desc = create_1d_tma_descriptor(x.data_ptr(), SIZE, SIZE, x.element_size())
    else:
        desc = torch.empty(64, dtype=torch.int8, device="cpu")
        triton.runtime.driver.active.utils.fill_1d_tma_descriptor(x.data_ptr(), SIZE, SIZE, x.element_size(),
                                                                  desc.data_ptr())
        desc = desc.to(device)
    z_tri = torch.zeros(SIZE, device=device)
    kernel[(1, )](z_tri, desc, SIZE=SIZE, num_warps=4)
    assert torch.equal(x, z_tri), "TME 1D load test failed"


@pytest.mark.parametrize("is_byval", [True, False])
def test_tme_2d_ld(is_byval):
    device = "musa"
    M, N = 128, 128

    @triton.jit
    def kernel(desc_a, desc_b, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        off_desc = 0
        a = tl._experimental_descriptor_load(desc_a, [off_desc, off_desc], [BLOCK_M, BLOCK_N], tl.float16)
        tl._experimental_descriptor_store(desc_b, a, [off_desc, off_desc])

    a = torch.ones((M, N), dtype=torch.float16, device=device)
    b = torch.zeros((M, N), dtype=torch.float16, device=device)
    if is_byval:
        desc_a = create_2d_tma_descriptor(a.data_ptr(), M, N, M, N, a.element_size())
        desc_b = create_2d_tma_descriptor(b.data_ptr(), M, N, M, N, b.element_size())
    else:
        desc_a = torch.empty(64, dtype=torch.int8, device="cpu")
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), M, N, M, N, a.element_size(),
                                                                  desc_a.data_ptr())
        desc_a = desc_a.to(device)
        desc_b = torch.empty(64, dtype=torch.int8, device="cpu")
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), M, N, M, N, b.element_size(),
                                                                  desc_b.data_ptr())
        desc_b = desc_b.to(device)
    kernel[(1, )](desc_a, desc_b, BLOCK_M=M, BLOCK_N=N, num_warps=4)
    assert torch.equal(a, b), "TME 2D load test failed"


@pytest.mark.parametrize("is_byval", [True, False])
def test_tme_3d_ld(is_byval):
    device = "musa"
    M, N, K = 4, 32, 32

    @triton.jit
    def kernel(desc_a, desc_b, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        off_desc = 0
        a = tl._experimental_descriptor_load(desc_a, [off_desc, off_desc, off_desc], [BLOCK_M, BLOCK_N, BLOCK_K],
                                             tl.float16)
        tl._experimental_descriptor_store(desc_b, a, [off_desc, off_desc, off_desc])

    a = torch.randn((M, N, K), dtype=torch.float16, device=device)
    b = torch.zeros((M, N, K), dtype=torch.float16, device=device)
    if is_byval:
        desc_a = create_3d_tma_descriptor(a.data_ptr(), M, N, K, M, N, K, a.element_size())
        desc_b = create_3d_tma_descriptor(b.data_ptr(), M, N, K, M, N, K, b.element_size())
    else:
        desc_a = torch.empty(64, dtype=torch.int8, device="cpu")
        triton.runtime.driver.active.utils.fill_3d_tma_descriptor(a.data_ptr(), M, N, K, M, N, K, a.element_size(),
                                                                  desc_a.data_ptr())
        desc_a = desc_a.to(device)
        desc_b = torch.empty(64, dtype=torch.int8, device="cpu")
        triton.runtime.driver.active.utils.fill_3d_tma_descriptor(b.data_ptr(), M, N, K, M, N, K, b.element_size(),
                                                                  desc_b.data_ptr())
        desc_b = desc_b.to(device)
    kernel[(1, )](desc_a, desc_b, BLOCK_M=M, BLOCK_N=N, BLOCK_K=K, num_warps=1)
    assert torch.equal(a, b), "TME 3D load test failed"
