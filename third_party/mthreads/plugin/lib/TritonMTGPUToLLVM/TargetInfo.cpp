#include "TargetInfo.h"
#include "Dialect/MTGPU/IR/Dialect.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace mlir::triton::MUSA {

bool TargetInfo::supportMaximumMinimum() const { return false; }

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  auto int32Ty = rewriter.getI32Type();
  return rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
                                           rewriter.getI32IntegerAttr(0));
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  mlir::LLVM::MUSA::llStore(rewriter, loc, ptr, val, pred);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred) const {
  Value falseVal = rewriter.create<arith::ConstantOp>(
      loc, elemTy, rewriter.getZeroAttr(elemTy));
  return mlir::LLVM::MUSA::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal);
}

void TargetInfo::storeMatrixShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value val) const {
  SmallVector<Value> elems;
  if (isa<VectorType>(val.getType()))
    elems = unpackLLVector(loc, val, rewriter);
  else
    elems.push_back(val);

  if (elems.empty())
    return;

  auto elemTy = elems.front().getType();
  assert(elemTy.getIntOrFloatBitWidth() == 16 &&
         "stmatrix requires 16-bit elements");
  assert(elems.size() == 8 && "stmatrix expects exactly 8 elements per store");

  SmallVector<Value> inputs;
  Type pairTy = vec_ty(elemTy, 2);
  Type i32Ty = i32_ty;
  for (int i = 0; i < elems.size(); i += 2) {
    Value packed = undef(pairTy);
    packed = insert_element(pairTy, packed, elems[i], i32_val(0));
    packed = insert_element(pairTy, packed, elems[i + 1], i32_val(1));
    inputs.push_back(bitcast(packed, i32Ty));
  }

  rewriter.create<triton::mtgpu::StoreMatrixOp>(loc, ptr, inputs);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return mlir::LLVM::MUSA::shuffleXor(loc, rewriter, val, i, warpSize);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return mlir::LLVM::MUSA::shuffleUp(loc, rewriter, val, i, warpSize);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return mlir::LLVM::MUSA::shuffleIdx(loc, rewriter, val, i, warpSize);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return mlir::LLVM::MUSA::shuffleIdx(loc, rewriter, val, i, warpSize);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::MUSA::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__mt_umulhi" : "__mt_umul64hi";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args) const {
  return;
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg,
                        ValueRange args) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  return;
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

bool TargetInfo::supportVectorizedAtomics() const { return false; }
} // namespace mlir::triton::MUSA
