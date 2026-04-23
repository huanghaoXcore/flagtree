#include "Dialect/TritonMthreadsGPU/IR/Types.h"
#include "Dialect/TritonMthreadsGPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "llvm/ADT/TypeSwitch.h"           // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton::mthreads_gpu;

#define GET_TYPEDEF_CLASSES
#include "Dialect/TritonMthreadsGPU/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void ::mlir::triton::mthreads_gpu::TritonMthreadsGPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/TritonMthreadsGPU/IR/Types.cpp.inc"
      >();
}
