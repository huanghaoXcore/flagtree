#include "triton/Dialect/Triton/IR/Dialect.h"
#include "Dialect/TritonMthreadsGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Dialect/TritonMthreadsGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton::mthreads_gpu;

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonMthreadsGPU/IR/TritonMthreadsGPUAttrDefs.cpp.inc"

void TritonMthreadsGPUDialect::initialize() {
  registerTypes();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/TritonMthreadsGPU/IR/TritonMthreadsGPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Dialect/TritonMthreadsGPU/IR/Ops.cpp.inc"
#include "Dialect/TritonMthreadsGPU/IR/OpsEnums.cpp.inc"
      >();
}

// verify TritonMthreadsGPU ops
LogicalResult
TritonMthreadsGPUDialect::verifyOperationAttribute(Operation *op,
                                                   NamedAttribute attr) {
  return success();
}
