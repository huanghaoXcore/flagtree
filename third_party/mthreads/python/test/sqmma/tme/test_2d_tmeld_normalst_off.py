import torch
import triton
import triton.language as tl
from triton.tools.experimental_descriptor import create_2d_tma_descriptor


def test_2d_tmeld_normalst_off():
    device = "musa"
    M, N = 512, 32
    block_m, block_n = 128, 32

    @triton.jit
    def kernel(src_desc, dst_ptr, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        off_r = pid_m * BLOCK_M
        off_c = pid_n * BLOCK_N
        src = tl._experimental_descriptor_load(src_desc, [off_r, off_c], [BLOCK_M, BLOCK_N], tl.float16)
        off_rv = off_r + tl.arange(0, BLOCK_M)
        off_cv = off_c + tl.arange(0, BLOCK_N)
        dst_ptrs = dst_ptr + stride_m * off_rv[:, None] + stride_n * off_cv[None, :]
        tl.store(dst_ptrs, src)

    @triton.jit
    def kernel_golden(src, dst, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        off_r = pid_m * BLOCK_M
        off_c = pid_n * BLOCK_N
        off_rv = off_r + tl.arange(0, BLOCK_M)
        off_cv = off_c + tl.arange(0, BLOCK_N)
        src_ptrs = src + stride_m * off_rv[:, None] + stride_n * off_cv[None, :]
        src_data = tl.load(src_ptrs)
        dst_ptrs = dst + stride_m * off_rv[:, None] + stride_n * off_cv[None, :]
        tl.store(dst_ptrs, src_data)

    src = torch.randn((M, N), dtype=torch.float16, device=device)
    dst = torch.zeros((M, N), dtype=torch.float16, device=device)

    src_desc = create_2d_tma_descriptor(src.data_ptr(), M, N, block_m, block_n, src.element_size())

    kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](src_desc, dst, stride_m=src.stride(0),
                                                               stride_n=src.stride(1), BLOCK_M=block_m, BLOCK_N=block_n,
                                                               num_warps=4)
    assert torch.equal(src, dst), "The output tensor does not match the input tensor."


if __name__ == "__main__":
    test_2d_tmeld_normalst_off()
