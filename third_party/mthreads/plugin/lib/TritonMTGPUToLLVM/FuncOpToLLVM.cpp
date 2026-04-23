#include <cstdint>

#include "MMAUtil/Utility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir {
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

namespace {

using namespace mlir;
using namespace mlir::triton;

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   int threadsPerWarp, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), numWarps(numWarps),
        threadsPerWarp(threadsPerWarp) {}

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {

    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  triton::FuncOp amendFuncOp(triton::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const {
    // Push back a variable that indicates the current stack pointer of shared
    // memory to the function arguments.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    // 1. Modify the function type to add the new argument.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    amendedInputTy.push_back(ptrTy);
    auto amendedFuncTy = FunctionType::get(funcTy.getContext(), amendedInputTy,
                                           funcTy.getResults());
    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    auto amendedArgAttrs = llvm::to_vector<4>(funcOp.getAllArgAttrs());
    amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    amendedAttrs.push_back(rewriter.getNamedAttr(
        funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(amendedArgAttrs)));
    // 3. Add a new argument to the region
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();
    region.addArgument(ptrTy, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  // Map the MLIR attribute `tt.mt_tma_desc` to the appropriate LLVM and NVVM
  // attributes.
  static void handleByvalTmaDescArgs(LLVM::LLVMFuncOp &llvmFuncOp) {
    const bool isKernel = LLVM::isKernel(llvmFuncOp);
    for (unsigned i = 0; i < llvmFuncOp.getNumArguments(); ++i) {
      const auto attrs = llvmFuncOp.getArgAttrDict(i);
      if (!attrs) {
        continue;
      }

      for (const auto &attr : attrs) {
        if (attr.getName() == "tt.mt_tma_desc") {
          const auto i32_type =
              mlir::IntegerType::get(llvmFuncOp.getContext(), 32);
          assert(attr.getValue() == mlir::IntegerAttr::get(i32_type, 1));
          assert(isKernel &&
                 "tt.mt_tma_desc is not supported for device functions");

          mlir::BlockArgument arg = llvmFuncOp.getArgument(i);
          auto *ctx = llvmFuncOp.getContext();
          const auto byteType = mlir::IntegerType::get(ctx, 8);
          const auto arrayType =
              mlir::LLVM::LLVMArrayType::get(ctx, byteType, 64);
          // TODO: Support more by-value ArgType in backend
          const auto structType = mlir::LLVM::LLVMStructType::getLiteral(
              llvmFuncOp.getContext(), {arrayType}, /*isPacked=*/false);
          llvmFuncOp.setArgAttr(i, "llvm.byval",
                                mlir::TypeAttr::get(structType));
          llvmFuncOp.setArgAttr(i, "mtgpu.grid_constant",
                                mlir::UnitAttr::get(llvmFuncOp.getContext()));
          llvmFuncOp.setArgAttr(i, "llvm.align",
                                mlir::IntegerAttr::get(i32_type, 64));
        }
      }
    }
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto maxBarAttr =
        funcOp->getAttrOfType<IntegerAttr>(mlir::mma_util::kMaxBarIdAttr);
    auto amendedFuncOp = funcOp;
    if (!LLVM::isKernel(funcOp))
      amendedFuncOp = amendFuncOp(funcOp, rewriter);

    LLVM::LLVMFuncOp newFuncOp = *mlir::convertFuncOpToLLVMFuncOp(
        amendedFuncOp, rewriter, *getTypeConverter());
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();
    if (maxBarAttr && LLVM::isKernel(funcOp)) {
      OpBuilder::InsertionGuard guard(rewriter);
      auto &entryBlock = newFuncOp.getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      auto i32Ty = rewriter.getI32Type();
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      Value maxBarValue = rewriter.create<LLVM::ConstantOp>(
          funcOp.getLoc(), i32Ty,
          rewriter.getI32IntegerAttr(
              static_cast<int32_t>(maxBarAttr.getInt())));
      rewriter.create<LLVM::CallIntrinsicOp>(
          funcOp.getLoc(), voidTy,
          rewriter.getStringAttr("llvm.musa.async.bar.record"),
          ValueRange{maxBarValue});
    }

    if (LLVM::isKernel(funcOp)) {
      // Set an attribute to indicate this function is a kernel entry.
      newFuncOp->setAttr("mtgpu.kernel",
                         rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
      newFuncOp.setLinkage(LLVM::Linkage::External);
    } else {
      // The noinline attribute will be used by the LLVM codegen to prevent
      // inlining.
      // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/LLVMIR/IR/LLVMInlining.cpp#L267
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      rewriter.eraseOp(amendedFuncOp);
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
    }
    // Set an attribute for reqntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr("mtgpu.maxntid", rewriter.getDenseI32ArrayAttr(
                                            threadsPerWarp * numWarps));
    rewriter.eraseOp(funcOp);
    handleByvalTmaDescArgs(newFuncOp);
    return success();
  }

private:
  int numWarps{0};
  int threadsPerWarp{0};
};

} // namespace

void mlir::triton::MUSA::populateFuncOpConversionPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, int numWarps,
    int threadsPerWarp, PatternBenefit benefit) {
  patterns.add<FuncOpConversion>(typeConverter, numWarps, threadsPerWarp,
                                 benefit);
}
