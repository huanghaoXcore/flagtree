#include <memory>

#include "TritonMTGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace gpu {

namespace {

static SmallVector<int64_t> getSqmmaPaddedShape(RankedTensorType argType,
                                                ArrayRef<unsigned> order) {
  auto shape = argType.getShape();
  SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
  int64_t elemBytes = argType.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t leadingBytes = shape[order[0]] * elemBytes;
  int64_t paddedLeadingBytes = leadingBytes;
  if (leadingBytes <= 256) {
    if (!llvm::isPowerOf2_64(leadingBytes))
      paddedLeadingBytes = llvm::PowerOf2Ceil(leadingBytes);
  } else {
    paddedLeadingBytes = llvm::alignTo(leadingBytes, int64_t{256});
  }
  if (paddedLeadingBytes != leadingBytes) {
    paddedShape[order[0]] = paddedLeadingBytes / elemBytes;
  }
  return paddedShape;
}

SmallVector<unsigned, 2>
warpsPerTilePH1(DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
                const SmallVector<unsigned, 3> &instrShape) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getResult(), &slices);
  if (llvm::find_if(slices, [](Operation *op) { return isa<DotOp>(op); }) !=
      slices.end())
    return {(unsigned)numWarps, 1};

  // For MMAv3, the smallest indivisible unit of warp shape is (4, 1).
  SmallVector<unsigned, 2> ret = {4, 1};
  SmallVector<int64_t, 2> shapePerWarp = {instrShape[0], instrShape[1]};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] > shapePerWarp[0] * ret[0]) {
      ret[0] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

class BlockedToMMA : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;
  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding
  mutable llvm::DenseMap<Operation *, unsigned> dotOpInstNs;

  static bool bwdFilter(Operation *op) {
    return op->getNumOperands() == 1 &&
           (isa<FpToFpOp, BitcastOp, ConvertLayoutOp>(op) ||
            isPureUnaryInlineAsm(op) ||
            op->getDialect()->getTypeID() ==
                mlir::TypeID::get<arith::ArithDialect>());
  }

  // Finds the first different bitwidth in the chain of shape-preserving
  // unary ops that x depends on.
  // There are two primary scenarios:
  // (1) Upcasting: A sequence such as loading an fp16, followed by arithmetic
  // operations, then bitcasting to fp32, and finally computing in fp32.
  // (2) Downcasting: This might involve loading an fp32, performing arithmetic
  // operations, bitcasting to fp16, and finally computing in fp16.
  // In the upcasting scenario, element reordering converts the original
  // elements distribution to the order of higher precision primitives. As a
  // result, kwidth can be the bitwidth of the lower precision primitive.
  // Conversely, in the downcasting scenario, no reordering is performed,
  // making it directory use the lower precision primitive.
  static int computeOrigBitWidth(Value x) {
    int finalBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
    int origBitWidth = finalBitWidth;
    SetVector<Operation *> slice;
    mlir::BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = bwdFilter;
    getBackwardSlice(x, &slice, opt);
    for (auto op : slice) {
      if (Value arg = op->getOperand(0))
        if (auto argTy = dyn_cast<RankedTensorType>(arg.getType())) {
          auto argBitWidth = argTy.getElementType().getIntOrFloatBitWidth();
          if (argBitWidth != origBitWidth) {
            origBitWidth = std::min<int>(origBitWidth, argBitWidth);
            break;
          }
        }
    }
    return origBitWidth;
  }

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability)
      : OpRewritePattern<DotOp>(context), computeCapability(computeCapability) {
  }

  static SmallVector<unsigned, 3>
  getWarpsPerTile(DotOp dotOp, const ArrayRef<int64_t> shape, int version,
                  int numWarps, const SmallVector<unsigned, 3> &instrShape) {
    switch (version) {
    case 3:
      return warpsPerTilePH1(dotOp, shape, numWarps, instrShape);
    default:
      assert(false && "not supported version");
      return {0, 0};
    }
  }

  static Value getSQMMAOperand(Value v, mlir::PatternRewriter &rewriter,
                               int opIdx, MLIRContext *ctx) {
    OpBuilder::InsertionGuard g(rewriter);
    Value arg = v;
    while (auto cvtOp = arg.getDefiningOp<ConvertLayoutOp>()) {
      arg = cvtOp.getSrc();
    }
    auto argType = cast<RankedTensorType>(arg.getType());
    assert(argType.getEncoding() && "unexpected tensor type");
    auto newOrder = getOrder(argType.getEncoding());

    Attribute SharedMemorySpace =
        SharedMemorySpaceAttr::get(argType.getContext());
    auto CTALayout = getCTALayout(argType.getEncoding());
    auto newLayout = SharedEncodingAttr::get(argType.getContext(), 1, 1, 1,
                                             newOrder, CTALayout, true);
    auto paddedShape = getSqmmaPaddedShape(argType, newOrder);
    auto newType = MemDescType::get(paddedShape, argType.getElementType(),
                                    newLayout, SharedMemorySpace);
    rewriter.setInsertionPointAfterValue(arg);
    auto localAllocOp =
        rewriter.create<LocalAllocOp>(arg.getLoc(), newType, arg);

    auto tmeOp = arg.getDefiningOp<ExperimentalDescriptorLoadOp>();
    if (tmeOp && !tmeOp->hasAttr("sqmma.opIdx")) {
      // add check
      // tme + sqmma
      tmeOp->setAttr("sqmma.opIdx",
                     IntegerAttr::get(IntegerType::get(ctx, 32), opIdx));
    } else {
      // tl.load + sqmma
      localAllocOp->setAttr("sqmma.opIdx",
                            IntegerAttr::get(IntegerType::get(ctx, 32), opIdx));
    }

    return localAllocOp;
  }

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 31)
      return failure();
    if (::triton::tools::getBoolEnv("DISABLE_SQMMA"))
      return failure();
    auto ctx = dotOp->getContext();
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<MthreadsSqmmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    if (!musa_util::supportMMA(dotOp, /*version=*/3))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    auto mod = dotOp->getParentOfType<mlir::ModuleOp>();
    int numWarps = TritonGPUDialect::getNumWarps(mod);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());

    int versionMajor = 3;
    if (!versionMajor)
      return failure();

    auto instrShape = musa_util::mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, dotOp.getA().getType(), numWarps);
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = dotOp.getA().getType();
    auto oldBType = dotOp.getB().getType();

    MthreadsSqmmaEncodingAttr mmaEnc;
    if (versionMajor == 3) {
      int versionMinor = computeCapability % 10;
      auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                          numWarps, instrShape);
      mmaEnc = MthreadsSqmmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor, versionMinor, warpsPerTile,
          CTALayout, instrShape);
    } else {
      return failure();
    }
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    Operation *newDot = nullptr;
    if (versionMajor == 3) {
      a = getSQMMAOperand(a, rewriter, 0, ctx);
      b = getSQMMAOperand(b, rewriter, 1, ctx);
      newDot = rewriter.create<triton::mthreads_gpu::SquadDotOp>(
          dotOp.getLoc(), newRetType, a, b, newAcc, nullptr,
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc(), false);
    } else {
      return failure();
    }
    // convert dot instruction
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType,
                                                 newDot->getResult(0));
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  return builder.create<FpToFpOp>(loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod, int computeCapability) {
  mod.walk([=](DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    MthreadsSqmmaEncodingAttr mmaLayout =
        dyn_cast<MthreadsSqmmaEncodingAttr>(D.getType().getEncoding());
    if (mmaLayout) {
      bool isNativeFP8 = AElType.isFloat8E5M2() || AElType.isFloat8E4M3FNUZ();
      // promote operands for sm < 31 since fp8 mma is not natively supported
      // promote operands for sm >= 31  when mma is not v3
      if (!isNativeFP8 ||
          (isNativeFP8 && (computeCapability == 31 || mmaLayout.isPH1())))
        return;
      promoteType = builder.getF16Type();
    } else {
      // FMA case.
      Type AElType = dotOp.getA().getType().getElementType();
      Type DElType = D.getType().getElementType();
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

#define GEN_PASS_DEF_TRITONMTGPUACCELERATESQMMA
#include "TritonMTGPUToLLVM/Passes.h.inc"

class TritonMTGPUAccelerateSQMMAPass
    : public impl::TritonMTGPUAccelerateSQMMABase<
          TritonMTGPUAccelerateSQMMAPass> {
public:
  using impl::TritonMTGPUAccelerateSQMMABase<
      TritonMTGPUAccelerateSQMMAPass>::TritonMTGPUAccelerateSQMMABase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    auto computeCapability = getMthreadsComputeCapability(m);

    mlir::RewritePatternSet patterns(context);
    patterns.add<BlockedToMMA>(context, computeCapability);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};
} // namespace gpu

std::unique_ptr<Pass> createTritonMTGPUAccelerateSQMMAPass() {
  return std::make_unique<gpu::TritonMTGPUAccelerateSQMMAPass>();
}

} // namespace triton
} // namespace mlir
