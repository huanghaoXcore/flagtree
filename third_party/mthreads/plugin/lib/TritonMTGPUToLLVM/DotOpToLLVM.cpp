#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include <cstdlib>

#include "DotOpToLLVM/DotOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::musa_util;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MthreadsSqmmaEncodingAttr;

LogicalResult convertSQMMA(triton::mthreads_gpu::SquadDotOp op,
                           triton::mthreads_gpu::SquadDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread);

LogicalResult convertMTFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter);

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();
    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    MthreadsSqmmaEncodingAttr mmaLayout = dyn_cast<MthreadsSqmmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());
    bool disableSQMMA = ::triton::tools::getBoolEnv("DISABLE_SQMMA");
    if (!disableSQMMA && !isOuter && mmaLayout && mmaLayout.isPH1() &&
        musa_util::supportMMA(op, mmaLayout.getVersionMajor())) {
      return convertMTFMADot(op, adaptor, getTypeConverter(), rewriter);
    }

    if (MusaDotOpConversion::supportMusaMma(op)) {
      auto dot = MusaDotOpConversion(op, adaptor, rewriter, getTypeConverter());
      if (dot.isQY2()) {
        return dot.convertIntoMMA323216();
      } else if (dot.isPH1()) {
        return dot.convertIntoPH1WMMA();
      }
      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(op.getResult().getType()).getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return convertMTFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct SquadDotOpConversion
    : public ConvertOpToLLVMPattern<triton::mthreads_gpu::SquadDotOp> {
  using ConvertOpToLLVMPattern<
      triton::mthreads_gpu::SquadDotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mthreads_gpu::SquadDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    MthreadsSqmmaEncodingAttr mmaLayout = dyn_cast<MthreadsSqmmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());
    if (!::triton::tools::getBoolEnv("DISABLE_SQMMA") && !isOuter &&
        mmaLayout &&
        musa_util::supportMMA(op.getOperand(0), mmaLayout.getVersionMajor())) {
      if (mmaLayout.isPH1()) {
        return convertSQMMA(op, adaptor, getTypeConverter(), rewriter,
                            getThreadId(rewriter, loc));
      }

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotAsyncOp to LLVM.");
    }

    llvm::report_fatal_error(
        "Unsupported DotAsyncOp found when converting TritonGPU to LLVM.");
  }
};

struct SquadDotWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::mthreads_gpu::SquadDotWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::mthreads_gpu::SquadDotWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mthreads_gpu::SquadDotWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pendings = op.getPendings();
    Location loc = op.getLoc();
    if (adaptor.getInputs().size() <= 1) {
      rewriter.create<triton::mtgpu::SQMMAWaitGroupOp>(loc);
      rewriter.eraseOp(op);

      return success();
    }

    assert(0 && "mtgpu dot_wait has no arguments and results");
    return failure();
  }
};
} // namespace

void mlir::triton::MUSA::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
  patterns.add<SquadDotOpConversion>(typeConverter, benefit);
  patterns.add<SquadDotWaitOpConversion>(typeConverter, benefit);
}
