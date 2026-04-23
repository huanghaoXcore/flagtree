#include "TritonMTGPUToLLVM/Passes.h"

#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

#define GEN_PASS_DEF_CONVERTMTGPUINPLACELOADTOLLVM
#include "TritonMTGPUToLLVM/Passes.h.inc"

namespace mlir {
namespace triton {

namespace {

enum class InplaceLoadDataKind {
  Unsupported,
  Integer,
  Float,
};

static std::optional<unsigned> getTypeBitWidth(Type type) {
  if (auto vecTy = dyn_cast<VectorType>(type)) {
    Type elemTy = vecTy.getElementType();
    if (elemTy.isIntOrFloat())
      return vecTy.getNumElements() * elemTy.getIntOrFloatBitWidth();
    return std::nullopt;
  }
  if (auto vecTy = dyn_cast<LLVM::LLVMFixedVectorType>(type)) {
    Type elemTy = vecTy.getElementType();
    if (elemTy.isIntOrFloat())
      return vecTy.getNumElements() * elemTy.getIntOrFloatBitWidth();
    return std::nullopt;
  }
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();
  return std::nullopt;
}

static InplaceLoadDataKind getInplaceLoadDataKind(Type type) {
  Type elemTy = type;
  if (auto vecTy = dyn_cast<VectorType>(type))
    elemTy = vecTy.getElementType();
  else if (auto vecTy = dyn_cast<LLVM::LLVMFixedVectorType>(type))
    elemTy = vecTy.getElementType();

  if (elemTy.isIntOrIndex())
    return InplaceLoadDataKind::Integer;
  if (isa<FloatType>(elemTy))
    return InplaceLoadDataKind::Float;
  return InplaceLoadDataKind::Unsupported;
}

class InplaceLoadCallConversion : public RewritePattern {
public:
  InplaceLoadCallConversion(MLIRContext *context)
      : RewritePattern(LLVM::CallOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto callOp = dyn_cast<LLVM::CallOp>(op);
    if (!callOp || !callOp.getCallee() ||
        callOp.getCallee().value().find(
            mlir::LLVM::MUSA::Predicated_InplaceLoad) ==
            llvm::StringRef::npos) {
      return failure();
    }

    if (callOp.getNumOperands() != 3 || callOp.getNumResults() != 1)
      return failure();

    auto loc = callOp.getLoc();
    Type resultTy = callOp.getResult().getType();
    Value ptr = callOp.getOperand(0);
    Value pred = callOp.getOperand(1);
    Value falseVal = callOp.getOperand(2);
    std::optional<unsigned> typeBits = getTypeBitWidth(resultTy);
    InplaceLoadDataKind dataKind = getInplaceLoadDataKind(resultTy);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({resultTy}, {loc});
    Block *trueBlock = rewriter.createBlock(afterLoad);
    Block *falseBlock =
        rewriter.splitBlock(trueBlock, rewriter.getInsertionPoint());

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, falseBlock);

    rewriter.setInsertionPointToStart(trueBlock);
    Value trueVal;
    if (typeBits && *typeBits == 128 &&
        dataKind != InplaceLoadDataKind::Unsupported) {
      auto i32Ty = rewriter.getI32Type();
      Value innerPersist = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(0));
      Value outerPersist = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(2));
      Value chrnt = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(1));
      Value slc = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(1));
      llvm::StringRef intrinsicName = dataKind == InplaceLoadDataKind::Float
                                          ? "llvm.musa.lsu.ld.cache.hint.f"
                                          : "llvm.musa.lsu.ld.cache.hint.i";
      trueVal = LLVM::createLLVMIntrinsicCallOp(
                    rewriter, loc, intrinsicName, TypeRange{resultTy},
                    ValueRange{ptr, innerPersist, outerPersist, chrnt, slc})
                    .getResult(0);
    } else {
      callOp->emitWarning()
          << "skip inplace cache-hint load lowering because the load type is "
          << (dataKind == InplaceLoadDataKind::Unsupported
                  ? std::string("not integer/float")
                  : "not 128 bits");
      trueVal = rewriter.create<LLVM::LoadOp>(loc, resultTy, ptr);
    }
    rewriter.create<LLVM::BrOp>(loc, trueVal, afterLoad);

    rewriter.setInsertionPointToStart(falseBlock);
    rewriter.create<LLVM::BrOp>(loc, falseVal, afterLoad);

    rewriter.setInsertionPointToStart(afterLoad);
    rewriter.replaceOp(callOp, afterLoad->getArgument(0));
    return success();
  }
};

struct ConvertMTGPUInplaceLoadToLLVM
    : public ::impl::ConvertMTGPUInplaceLoadToLLVMBase<
          ConvertMTGPUInplaceLoadToLLVM> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;

    RewritePatternSet patterns(context);
    patterns.add<InplaceLoadCallConversion>(context);

    if (applyPatternsAndFoldGreedily(mod, std::move(patterns), config)
            .failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMTGPUInplaceLoadToLLVMPass() {
  return std::make_unique<ConvertMTGPUInplaceLoadToLLVM>();
}

} // namespace triton
} // namespace mlir
