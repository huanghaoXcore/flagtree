#ifndef TRITONMTGPU_CONVERSION_PASSES
#define TRITONMTGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def DecomposeUnsupportedMTGPUConversions : Pass<"decompose-unsupported-mtgpu-conversions", "mlir::ModuleOp"> {
    let summary = "Decompose conversions that are not supported by TritonGPU -> LLVM";
    let constructor = "mlir::triton::MUSA::createDecomposeUnsupportedConversionsPass()";
}

def ConvertTritonMTGPUToLLVM : Pass<"convert-triton-mtgpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonMTGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::MTGPU::MTGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
    ];
}

def TritonMTGPUAccelerateSQMMA : Pass<"tritonmtgpu-accelerate-sqmma", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., MT tce)
  }];

  let constructor = "mlir::triton::createTritonMTGPUAccelerateSQMMAPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::mthreads_gpu::TritonMthreadsGPUDialect",
                           "mlir::triton::TritonDialect"];
}


def TritonMTGPUMarkInplaceLoads : Pass<"tritonmtgpu-mark-inplace-loads", "mlir::ModuleOp"> {
  let summary = "mark tt.load values that eventually feed a tt.store";
  let constructor = "mlir::triton::createTritonMTGPUMarkInplaceLoadsPass()";

  let dependentDialects = ["mlir::triton::TritonDialect"];
}


def ConvertMTGPUBuiltinFuncToLLVM : Pass<"convert-mtgpu-builtin-func-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert MTGPU Builtin Func to LLVM";
    let constructor = "mlir::triton::createConvertMTGPUBuiltinFuncToLLVMPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def ConvertMTGPUInplaceLoadToLLVM : Pass<"convert-mtgpu-inplace-load-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert MTGPU inplace load builtin calls to LLVM cache-hint intrinsics";
    let constructor = "mlir::triton::createConvertMTGPUInplaceLoadToLLVMPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // TRITONMTGPU_CONVERSION_PASSES
