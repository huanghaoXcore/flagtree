#include "TritonMTGPUToLLVM/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DEF_TRITONMTGPUMARKINPLACELOADS
#include "TritonMTGPUToLLVM/Passes.h.inc"

namespace mlir::triton {
namespace {

constexpr llvm::StringLiteral kInplaceLoadAttr = "mtgpu.inplace_load_candidate";

static bool
areEquivalentValues(Value lhs, Value rhs,
                    llvm::DenseMap<std::pair<Value, Value>, bool> &cache);

static bool
areEquivalentOps(Operation *lhs, Operation *rhs,
                 llvm::DenseMap<std::pair<Value, Value>, bool> &cache) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs)
    return false;

  auto checkEquivalent = [&](Value left, Value right) -> LogicalResult {
    return success(areEquivalentValues(left, right, cache));
  };

  return OperationEquivalence::isEquivalentTo(
      lhs, rhs, checkEquivalent, /*markEquivalent=*/nullptr,
      OperationEquivalence::Flags::IgnoreLocations);
}

static bool
areEquivalentValues(Value lhs, Value rhs,
                    llvm::DenseMap<std::pair<Value, Value>, bool> &cache) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs || lhs.getType() != rhs.getType())
    return false;

  auto key = std::make_pair(lhs, rhs);
  auto reverseKey = std::make_pair(rhs, lhs);
  if (auto it = cache.find(key); it != cache.end())
    return it->second;
  if (auto it = cache.find(reverseKey); it != cache.end())
    return it->second;

  // Break recursive cycles conservatively.
  cache[key] = false;
  cache[reverseKey] = false;

  if (auto lhsArg = dyn_cast<BlockArgument>(lhs)) {
    auto rhsArg = dyn_cast<BlockArgument>(rhs);
    bool equivalent = rhsArg && lhsArg.getOwner() == rhsArg.getOwner() &&
                      lhsArg.getArgNumber() == rhsArg.getArgNumber();
    cache[key] = equivalent;
    cache[reverseKey] = equivalent;
    return equivalent;
  }

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  bool equivalent = areEquivalentOps(lhsDef, rhsDef, cache);
  cache[key] = equivalent;
  cache[reverseKey] = equivalent;
  return equivalent;
}

static bool
hasSameAddressStoreInFunc(triton::LoadOp loadOp,
                          llvm::ArrayRef<triton::StoreOp> storeOps) {
  llvm::DenseMap<std::pair<Value, Value>, bool> cache;
  Value loadPtr = loadOp.getPtr();

  for (triton::StoreOp storeOp : storeOps) {
    if (areEquivalentValues(loadPtr, storeOp.getPtr(), cache))
      return true;
  }
  return false;
}

struct TritonMTGPUMarkInplaceLoadsPass
    : public ::impl::TritonMTGPUMarkInplaceLoadsBase<
          TritonMTGPUMarkInplaceLoadsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    getOperation()->walk([&](triton::FuncOp funcOp) {
      llvm::SmallVector<triton::StoreOp> storeOps;
      funcOp.walk(
          [&](triton::StoreOp storeOp) { storeOps.push_back(storeOp); });

      if (storeOps.empty())
        return;

      funcOp.walk([&](triton::LoadOp loadOp) {
        if (hasSameAddressStoreInFunc(loadOp, storeOps))
          loadOp->setAttr(kInplaceLoadAttr, UnitAttr::get(ctx));
      });
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTritonMTGPUMarkInplaceLoadsPass() {
  return std::make_unique<TritonMTGPUMarkInplaceLoadsPass>();
}

} // namespace mlir::triton
