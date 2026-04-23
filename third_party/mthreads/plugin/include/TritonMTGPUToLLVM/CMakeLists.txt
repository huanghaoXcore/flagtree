# this folder includes TritonGPU transformed to LLVM specifically
# for mtgpu target, not only MTGPU dialect to llvm transformation
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonMTGPUToLLVM)
add_public_tablegen_target(TritonMTGPUConversionPassIncGen)
