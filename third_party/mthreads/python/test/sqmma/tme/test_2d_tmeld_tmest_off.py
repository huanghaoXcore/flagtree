import torch
import triton
import triton.language as tl
import numpy as np
from triton.tools.experimental_descriptor import create_2d_tma_descriptor


def test_2d_tmeld_tmest_off():
    device = "musa"
    M, N = 4096, 4096
    block_m, block_n = 128, 128

    @triton.jit
    def kernel(src_desc, dst_desc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        off_r = pid_m * BLOCK_M
        off_c = pid_n * BLOCK_N
        src = tl._experimental_descriptor_load(src_desc, [off_r, off_c], [BLOCK_M, BLOCK_N], tl.float16)
        tl._experimental_descriptor_store(dst_desc, src, [off_r, off_c])

    src = torch.randn((M, N), dtype=torch.float16, device=device)
    dst = torch.zeros((M, N), dtype=torch.float16, device=device)

    src_desc = create_2d_tma_descriptor(src.data_ptr(), M, N, block_m, block_n, src.element_size())
    dst_desc = create_2d_tma_descriptor(dst.data_ptr(), M, N, block_m, block_n, dst.element_size())

    kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](src_desc, dst_desc, BLOCK_M=block_m, BLOCK_N=block_n,
                                                               num_warps=4)
    assert torch.equal(src, dst), "The output tensor does not match the input tensor."


if __name__ == "__main__":
    test_2d_tmeld_tmest_off()
