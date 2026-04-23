#ifndef TRITON_CONVERSION_TRITONMTGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONMTGPU_TO_LLVM_UTILITY_H

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "Dialect/TritonMthreadsGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "third_party/mthreads/plugin/include/Dialect/MTGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace LLVM {

namespace MUSA {
const char Predicated_AsyncLoad[] = "__predicated_asyncLoad";
const char Predicated_Load[] = "__predicated_load";
const char Predicated_Store[] = "__predicated_store";
const char Predicated_InplaceLoad[] = "__predicated_inplace_load";

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 unsigned width);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                unsigned width);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 unsigned width);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 unsigned width);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis);

// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
Value llAsyncLoad(RewriterBase &rewriter, Location loc,
                  SmallVector<Value> asyncLoadOperands);

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal);
Value llInplaceLoad(RewriterBase &rewriter, Location loc, Value ptr,
                    Type elemTy, Value pred, Value falseVal);

// Stores to shared or global memory with predication.
void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred);

// Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
//               Value mask);

/// Create a predicate with just single active thread.
Value createElectPredicate(Location loc, PatternRewriter &rewriter);

LLVM::LLVMFuncOp getLibdeviceFuncCall(RewriterBase &rewriter, Operation *op,
                                      StringRef funcName, Type retType,
                                      ValueRange ins);

} // namespace MUSA
} // namespace LLVM

namespace musa_util {

bool supportMMA(Value value, int version);
bool supportMMA(triton::DotOp op, int version);
bool isMmaToMmaShortcut(RankedTensorType srcTy, RankedTensorType dstTy);
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout);
// Return true if the src and dst layout match.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy);
SmallVector<unsigned, 3> mmaVersionToInstrShape(int version,
                                                const ArrayRef<int64_t> &shape,
                                                RankedTensorType type,
                                                int numWarps);
Value createElectPredicate(Location loc, PatternRewriter &rewriter);

} // namespace musa_util

} // namespace mlir

#endif
