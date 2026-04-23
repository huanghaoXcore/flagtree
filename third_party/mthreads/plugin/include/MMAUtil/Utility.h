#ifndef MTGPU_TRITONMTGPUTOLLVM_MMAUTIL_H
#define MTGPU_TRITONMTGPUTOLLVM_MMAUTIL_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace mma_util {

bool supportMMA(Value value, int version);
bool supportMMA(triton::DotOp op, int version);

constexpr StringRef kNextBarIdAttr = "mma.next_bar_id";
constexpr StringRef kMaxBarIdAttr = "mma.max_bar_id";

inline ModuleOp getEnclosingModule(Operation *op) {
  if (auto mod = dyn_cast<ModuleOp>(op))
    return mod;
  return op->getParentOfType<ModuleOp>();
}

inline FunctionOpInterface getEnclosingFunction(Operation *op) {
  while (op) {
    if (auto func = dyn_cast<FunctionOpInterface>(op))
      return func;
    op = op->getParentOp();
  }
  return nullptr;
}

inline int getNewBarId(Operation *anchorOp) {
  ModuleOp module = getEnclosingModule(anchorOp);
  assert(module && "getNewBarId requires an op inside a ModuleOp");

  MLIRContext *ctx = module.getContext();
  auto i32 = IntegerType::get(ctx, 32);
  auto attr = module->getAttrOfType<IntegerAttr>(kNextBarIdAttr);

  int cur = attr ? static_cast<int>(attr.getInt()) : 0;
  int next = cur + 1;
  module->setAttr(kNextBarIdAttr, IntegerAttr::get(i32, next));

  if (auto func = getEnclosingFunction(anchorOp)) {
    auto funcAttr = func->getAttrOfType<IntegerAttr>(kMaxBarIdAttr);
    int funcCur = funcAttr ? static_cast<int>(funcAttr.getInt()) : 0;
    if (next > funcCur)
      func->setAttr(kMaxBarIdAttr, IntegerAttr::get(i32, next));
  }
  return next;
}
} // namespace mma_util
} // namespace mlir

#endif // MTGPU_TRITONMTGPUTOLLVM_MMAUTIL_H
