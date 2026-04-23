#ifndef TRITONMTHREADSGPU_DIALECT
#define TRITONMTHREADSGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonMthreadsGPU_Dialect : Dialect {
  let name = "triton_mthreads_gpu";

  let cppNamespace = "::mlir::triton::mthreads_gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton Mthreads GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
    "mlir::gpu::GPUDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "triton_gpu.num-warps"; }
    static int getNumWarps(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-warps attribute");
      return cast<IntegerAttr>(mod->getAttr("triton_gpu.num-warps")).getInt();
    }
    static int getNumCTAs(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu.num-ctas"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-ctas attribute");
      return cast<IntegerAttr>(mod->getAttr("triton_gpu.num-ctas")).getInt();
    }
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
}

include "Dialect/TritonMthreadsGPU/IR/TritonMthreadsGPUTypes.td"

#endif
