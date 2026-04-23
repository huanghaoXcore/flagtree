import torch
import triton
import triton.language as tl
from triton.tools.experimental_descriptor import create_1d_tma_descriptor


def test_1d_tmeld_normalst_off():
    device = "musa"
    TENSOR_SIZE = 4096 * 4096
    block_size = 128

    @triton.jit
    def kernel(Z, desc, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        off_desc = pid * BLOCK_SIZE
        off = off_desc + tl.arange(0, BLOCK_SIZE)
        x = tl._experimental_descriptor_load(desc, [off_desc], [BLOCK_SIZE], Z.dtype.element_ty)
        tl.store(Z + off, x)

    x = torch.randn(TENSOR_SIZE, dtype=torch.float32, device=device)
    desc = create_1d_tma_descriptor(x.data_ptr(), TENSOR_SIZE, block_size, x.element_size())
    z_tri = torch.zeros(TENSOR_SIZE, device=device)
    kernel[(triton.cdiv(TENSOR_SIZE, block_size), )](z_tri, desc, BLOCK_SIZE=block_size, num_warps=4)
    assert torch.equal(x, z_tri), "The output tensor does not match the input tensor."


if __name__ == "__main__":
    test_1d_tmeld_normalst_off()
