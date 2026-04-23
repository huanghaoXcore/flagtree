import torch
import triton
import triton.language as tl
from triton.tools.experimental_descriptor import create_1d_tma_descriptor


def test_1d_normalld_tmest_off():
    device = "musa"
    TENSOR_SIZE = 4096 * 4096
    block_size = 128

    @triton.jit
    def kernel(src, dst_desc, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        off_desc = pid * BLOCK_SIZE
        off = off_desc + tl.arange(0, BLOCK_SIZE)
        src = tl.load(src + off)
        tl._experimental_descriptor_store(dst_desc, src, [off_desc])

    src = torch.randn(TENSOR_SIZE, dtype=torch.float16, device=device)
    dst = torch.zeros(TENSOR_SIZE, dtype=torch.float16, device=device)
    dst_desc = create_1d_tma_descriptor(dst.data_ptr(), TENSOR_SIZE, block_size, dst.element_size())
    kernel[(triton.cdiv(TENSOR_SIZE, block_size), )](src, dst_desc, BLOCK_SIZE=block_size, num_warps=4)
    assert torch.equal(dst, src), "The output tensor does not match the input tensor."


if __name__ == "__main__":
    test_1d_normalld_tmest_off()
