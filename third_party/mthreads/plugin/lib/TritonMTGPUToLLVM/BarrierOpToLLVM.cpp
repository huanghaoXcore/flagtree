#include "PatternTritonGPUOpToLLVM.h"
// #include "TritonMTGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct FenceAsyncSharedOpConversion
    : public ConvertOpToLLVMPattern<triton::mthreads_gpu::FenceAsyncSharedOp> {
  using ConvertOpToLLVMPattern<
      triton::mthreads_gpu::FenceAsyncSharedOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mthreads_gpu::FenceAsyncSharedOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::mtgpu::FenceAsyncSharedOp>(
        op, adaptor.getBCluster());
    return success();
  }
};

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::mthreads_gpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mthreads_gpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto operands = op.getOperands();
    Value barId = operands[0];
    Value warpCnt = operands[1];
    Value phase = operands[2];
    auto voidTy = void_ty(op->getContext());
    MLIRContext *ctx = op->getContext();

    // generate if threadIdx == 0: async_init_arrival()
    auto id = getThreadId(rewriter, loc);
    auto pred = icmp_eq(id, i32_val(0));
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBarrier =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterBarrier);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, afterBarrier);
    rewriter.setInsertionPointToStart(trueBlock);
    SmallVector<Value> initArrivOps = {barId, warpCnt, phase};
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                    "llvm.musa.async.init.arrival", TypeRange{},
                                    initArrivOps);
    rewriter.create<LLVM::BrOp>(loc, afterBarrier);
    rewriter.setInsertionPointToStart(afterBarrier);
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierExpectConversion
    : public ConvertOpToLLVMPattern<triton::mthreads_gpu::BarrierExpectOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mthreads_gpu::BarrierExpectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto barId = op.getBarId();
    SmallVector<Value> ops = {barId};
    MLIRContext *ctx = op->getContext();
    SmallVector<Type, 1> resultTypes{i32_ty};
    auto newOp = LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.musa.async.arrive", resultTypes, ops);
    rewriter.replaceOp(op, newOp.getResult(0));
    return success();
  }
};

struct WaitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::mthreads_gpu::WaitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mthreads_gpu::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value barId = op->getOperand(0);
    Value phaseId = op->getOperand(1);
    Type voidTy = void_ty(op->getContext());
    SmallVector<Value> ops = {barId, phaseId};
    MLIRContext *ctx = op->getContext();
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.async.wait",
                                    TypeRange{}, ops);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::triton::MUSA::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, benefit);
  patterns.add<InitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<BarrierExpectConversion>(typeConverter, benefit);
}
