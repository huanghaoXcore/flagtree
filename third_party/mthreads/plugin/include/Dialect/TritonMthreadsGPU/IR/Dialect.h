#ifndef TRITON_DIALECT_TRITONMTHREADSGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONMTHREADSGPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonMthreadsGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Dialect/TritonMthreadsGPU/IR/Dialect.h.inc"
#include "Dialect/TritonMthreadsGPU/IR/Types.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonMthreadsGPU/IR/TritonMthreadsGPUAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "Dialect/TritonMthreadsGPU/IR/Ops.h.inc"

#endif // TRITON_DIALECT_TRITONMTHREADSGPU_IR_DIALECT_H_
