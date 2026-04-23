#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// blocked -> shared.
// Swizzling in shared memory to avoid bank conflict. Normally used for
// A/B operands of dots.
void lowerDistributedToShared(Location loc, Value src, Value dst,
                              Value adaptorSrc,
                              const SharedMemoryObject &smemObj,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto dstTy = cast<MemDescType>(dst.getType());
  auto outOrd = mlir::cast<SharedEncodingAttr>(dstTy.getEncoding()).getOrder();
  assert(srcTy.getShape().size() <= 2 ||
         (srcTy.getShape().size() == 3 && outOrd[2] == 0) &&
             "Unexpected rank of ConvertLayout(blocked->shared)");
  auto elemTy = typeConverter->convertType(srcTy.getElementType());

  auto smemBase = smemObj.getBase();
  auto dstStrides = smemObj.getStrides();
  auto inVals = unpackLLElements(loc, adaptorSrc, rewriter);
  // storeDistributedToShared(dstTy, srcTy, elemTy, inVals, smemBase,
  // dstStrides,
  //                          loc, rewriter, targetInfo);
  storeDistributedToShared(src, inVals, dstStrides, dst, smemBase, elemTy, loc,
                           rewriter, targetInfo);
}

static arith::CmpIPredicate swapCmpPredicate(arith::CmpIPredicate pred) {
  // Re-orient predicates when we swap compare operands, e.g. C > x => x < C.
  switch (pred) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
    return pred;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  return pred;
}

static std::optional<int64_t> getConstantIntValue(Value v) {
  // Peel through shape-preserving wrappers to recover scalar integer constants
  // used in address/mask arithmetic.
  while (auto *op = v.getDefiningOp()) {
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      v = splat.getSrc();
      continue;
    }
    if (auto broadcast = dyn_cast<triton::BroadcastOp>(op)) {
      v = broadcast.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
      v = expand.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      v = reshape.getSrc();
      continue;
    }
    if (auto bitcast = dyn_cast<triton::BitcastOp>(op)) {
      v = bitcast.getSrc();
      continue;
    }
    if (auto fpCast = dyn_cast<triton::FpToFpOp>(op)) {
      v = fpCast.getSrc();
      continue;
    }
    if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
      v = cast.getIn();
      continue;
    }
    if (auto ext = dyn_cast<arith::ExtSIOp>(op)) {
      v = ext.getIn();
      continue;
    }
    if (auto ext = dyn_cast<arith::ExtUIOp>(op)) {
      v = ext.getIn();
      continue;
    }
    break;
  }

  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
      return intAttr.getInt();
    }
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(cst.getValue())) {
      if (!denseAttr.isSplat())
        return std::nullopt;
      if (denseAttr.getElementType().isIntOrIndex()) {
        APInt splat = denseAttr.getSplatValue<APInt>();
        return splat.getSExtValue();
      }
    }
  }

  return std::nullopt;
}

struct RangeWithOffset {
  triton::MakeRangeOp range;
  int64_t offset;
};

static std::optional<RangeWithOffset> getMakeRangeWithOffset(Value v) {
  // Match a canonical "make_range + const" (or "- const") expression and
  // return both the base range op and accumulated constant offset.
  int64_t offset = 0;
  while (auto *op = v.getDefiningOp()) {
    if (auto add = dyn_cast<arith::AddIOp>(op)) {
      if (auto rhsConst = getConstantIntValue(add.getRhs())) {
        offset += *rhsConst;
        v = add.getLhs();
        continue;
      }
      if (auto lhsConst = getConstantIntValue(add.getLhs())) {
        offset += *lhsConst;
        v = add.getRhs();
        continue;
      }
    }
    if (auto sub = dyn_cast<arith::SubIOp>(op)) {
      if (auto rhsConst = getConstantIntValue(sub.getRhs())) {
        offset -= *rhsConst;
        v = sub.getLhs();
        continue;
      }
    }
    if (auto broadcast = dyn_cast<triton::BroadcastOp>(op)) {
      v = broadcast.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
      v = expand.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      v = reshape.getSrc();
      continue;
    }
    if (auto bitcast = dyn_cast<triton::BitcastOp>(op)) {
      v = bitcast.getSrc();
      continue;
    }
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      v = splat.getSrc();
      continue;
    }
    if (auto trans = dyn_cast<triton::TransOp>(op)) {
      v = trans.getSrc();
      continue;
    }
    break;
  }

  if (auto range = v.getDefiningOp<triton::MakeRangeOp>())
    return RangeWithOffset{range, offset};

  return std::nullopt;
}

static std::optional<int64_t> getLeadingDimBoundFromCmp(arith::CmpIOp cmp,
                                                        int64_t leadingExtent) {
  // Try to infer the valid leading extent from predicates such as
  // (make_range + offset) < bound. Only upper-bound forms are used here.
  if (leadingExtent <= 0)
    return std::nullopt;
  auto lhsRange = getMakeRangeWithOffset(cmp.getLhs());
  auto rhsRange = getMakeRangeWithOffset(cmp.getRhs());
  auto lhsConst = getConstantIntValue(cmp.getLhs());
  auto rhsConst = getConstantIntValue(cmp.getRhs());
  arith::CmpIPredicate pred = cmp.getPredicate();

  if (!lhsRange && rhsRange && lhsConst) {
    pred = swapCmpPredicate(pred);
    lhsRange = rhsRange;
    rhsConst = lhsConst;
    rhsRange = std::nullopt;
  }

  if (!lhsRange || !rhsConst)
    return std::nullopt;

  int64_t rangeStart = lhsRange->range.getStart();
  int64_t rangeEnd = lhsRange->range.getEnd();
  int64_t rangeExtent = rangeEnd - rangeStart;
  if (rangeExtent != leadingExtent)
    return std::nullopt;

  int64_t bound = *rhsConst;
  int64_t effectiveBound = 0;
  switch (pred) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    effectiveBound = bound - lhsRange->offset;
    break;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    effectiveBound = bound - lhsRange->offset + 1;
    break;
  default:
    return std::nullopt;
  }

  int64_t actualExtent = effectiveBound - rangeStart;
  if (actualExtent <= 0 || actualExtent > rangeExtent)
    return std::nullopt;
  return actualExtent;
}

static std::optional<int64_t>
getLeadingDimBoundFromMask(Value mask, int64_t leadingExtent) {
  // Walk the mask expression DAG and look for a compare pattern that can
  // recover the runtime-valid leading extent.
  if (!mask)
    return std::nullopt;

  SmallVector<Value> worklist;
  worklist.push_back(mask);
  llvm::SmallPtrSet<Operation *, 16> visited;

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *op = v.getDefiningOp();
    if (!op || !visited.insert(op).second)
      continue;

    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      if (auto bound = getLeadingDimBoundFromCmp(cmp, leadingExtent))
        return bound;
      for (Value operand : cmp.getOperands())
        worklist.push_back(operand);
      continue;
    }
    if (auto andOp = dyn_cast<arith::AndIOp>(op)) {
      for (Value operand : andOp.getOperands())
        worklist.push_back(operand);
      continue;
    }
    if (auto orOp = dyn_cast<arith::OrIOp>(op)) {
      for (Value operand : orOp.getOperands())
        worklist.push_back(operand);
      continue;
    }
    if (auto broadcast = dyn_cast<triton::BroadcastOp>(op)) {
      worklist.push_back(broadcast.getSrc());
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
      worklist.push_back(expand.getSrc());
      continue;
    }
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      worklist.push_back(splat.getSrc());
      continue;
    }
    if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      worklist.push_back(reshape.getSrc());
      continue;
    }
    if (auto bitcast = dyn_cast<triton::BitcastOp>(op)) {
      worklist.push_back(bitcast.getSrc());
      continue;
    }
    if (auto trans = dyn_cast<triton::TransOp>(op)) {
      worklist.push_back(trans.getSrc());
      continue;
    }
  }

  return std::nullopt;
}

static Value findLoadMask(Value src) {
  // Follow layout/cast/view transforms and recover the source LoadOp mask,
  // if any, that guards the tensor materialized into shared memory.
  while (src) {
    if (auto load = src.getDefiningOp<triton::LoadOp>())
      return load.getMask();
    if (auto trans = src.getDefiningOp<triton::TransOp>()) {
      src = trans.getSrc();
      continue;
    }
    if (auto cvt = src.getDefiningOp<triton::gpu::ConvertLayoutOp>()) {
      src = cvt.getSrc();
      continue;
    }
    if (auto reshape = src.getDefiningOp<triton::ReshapeOp>()) {
      src = reshape.getSrc();
      continue;
    }
    if (auto bitcast = src.getDefiningOp<triton::BitcastOp>()) {
      src = bitcast.getSrc();
      continue;
    }
    if (auto fpCast = src.getDefiningOp<triton::FpToFpOp>()) {
      src = fpCast.getSrc();
      continue;
    }
    if (auto broadcast = src.getDefiningOp<triton::BroadcastOp>()) {
      src = broadcast.getSrc();
      continue;
    }
    if (auto expand = src.getDefiningOp<triton::ExpandDimsOp>()) {
      src = expand.getSrc();
      continue;
    }
    if (auto splat = src.getDefiningOp<triton::SplatOp>()) {
      src = splat.getSrc();
      continue;
    }
    break;
  }
  return Value();
}

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto resultTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();
    auto sharedLayout =
        cast<triton::gpu::SharedEncodingAttr>(resultTy.getEncoding());
    auto order = sharedLayout.getOrder();
    // Workaround for 3D tensors
    // TODO: we need to modify the pipeline pass to give a proper shared
    // encoding to 3D tensors
    SmallVector<unsigned> newOrder;
    if (resultTy.getShape().size() != order.size()) {
      for (auto i = 0; i < order.size(); ++i)
        newOrder.push_back(order[i] + 1);
      newOrder.push_back(0);
    } else {
      newOrder = SmallVector<unsigned>(order.begin(), order.end());
    }

    auto llvmElemTy = typeConverter->convertType(resultTy.getElementType());
    auto shapePerCTA = getShapePerCTA(sharedLayout, resultTy.getShape());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, shapePerCTA,
                                      newOrder, loc, rewriter);
    // If there is an initial tensor, store it into the shared memory.
    if (op.getSrc()) {
      // SQMMA allocs can be tagged either on local_alloc itself or propagated
      // from the producer op feeding local_alloc.
      bool isSqmma = op->hasAttr("sqmma.opIdx");
      if (!isSqmma) {
        if (auto srcOp = op.getSrc().getDefiningOp())
          isSqmma = srcOp->hasAttr("sqmma.opIdx");
      }
      bool needsPadding = false;
      if (isSqmma && resultTy.getShape().size() == newOrder.size()) {
        auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
        auto srcShape = srcTy.getShape();
        auto dstShape = resultTy.getShape();
        if (srcShape.size() != dstShape.size()) {
          needsPadding = true;
        } else {
          for (int64_t i = 0; i < static_cast<int64_t>(srcShape.size()); ++i) {
            if (dstShape[i] > srcShape[i]) {
              needsPadding = true;
              break;
            }
          }
        }
        int64_t elemBytes = srcTy.getElementType().getIntOrFloatBitWidth() / 8;
        int64_t leadingExtent = 0;
        if (!newOrder.empty() && newOrder[0] < srcShape.size())
          leadingExtent = srcShape[newOrder[0]];
        int64_t actualLeadingExtent = leadingExtent;
        std::optional<int64_t> maskBound;
        Value mask = findLoadMask(op.getSrc());
        if (leadingExtent > 0) {
          maskBound = getLeadingDimBoundFromMask(mask, leadingExtent);
          if (maskBound)
            actualLeadingExtent = *maskBound;
        }
        // SQMMA descriptor encoding only supports discrete leading strides:
        // <=256B must be power-of-two; >256B must be a multiple of 256B.
        if (!needsPadding && elemBytes > 0 && actualLeadingExtent > 0) {
          int64_t leadingBytes = actualLeadingExtent * elemBytes;
          if (leadingBytes <= 256) {
            if (!llvm::isPowerOf2_64(static_cast<uint64_t>(leadingBytes))) {
              needsPadding = true;
            }
          } else if ((leadingBytes % 256) != 0) {
            needsPadding = true;
          }
        }
      }
      if (isSqmma && resultTy.getShape().size() == newOrder.size()) {
        int64_t totalElems = product<int64_t>(shapePerCTA);
        if (needsPadding && totalElems > 0) {
          // Zero-initialize the whole shared tile so padded elements stay
          // deterministic after later stores of only the valid region.
          auto mod = op->getParentOfType<ModuleOp>();
          unsigned numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
          unsigned warpSize =
              triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
          int64_t numThreads = static_cast<int64_t>(numWarps) * warpSize;
          int64_t numIters = (totalElems + numThreads - 1) / numThreads;
          Value threadId = getThreadId(rewriter, loc);
          Value totalElemsVal = i32_val(static_cast<int32_t>(totalElems));
          Value zero = rewriter.create<LLVM::ConstantOp>(
              loc, llvmElemTy, rewriter.getZeroAttr(llvmElemTy));
          for (int64_t iter = 0; iter < numIters; ++iter) {
            int32_t iterOffset = static_cast<int32_t>(iter * numThreads);
            Value offset = add(threadId, i32_val(iterOffset));
            Value pred = icmp_slt(offset, totalElemsVal);
            Value ptr = gep(smemBase.getType(), llvmElemTy, smemBase, offset);
            targetInfo.storeShared(rewriter, loc, ptr, zero, pred);
          }
          // Synchronize before cooperative tensor->shared stores.
          barrier();
        }
      }
      lowerDistributedToShared(loc, op.getSrc(), op.getResult(),
                               adaptor.getSrc(), smemObj, typeConverter,
                               rewriter, targetInfo);
    }
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalDeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalDeallocOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalDeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  // FIXME [Dot LL]
  // Do for all DotOperandEncodingAttr once we have LLs for all of them
  static bool isSupportedDotOpLayout(Attribute layout) {
    if (auto dot = dyn_cast<DotOperandEncodingAttr>(layout)) {
      if (auto mma = dyn_cast<NvidiaMmaEncodingAttr>(dot.getParent())) {
        return mma.isAmpere() && dot.getKWidth() == 8;
      }
      if (isa<AMDMfmaEncodingAttr>(dot.getParent()))
        return true;
    }
    return false;
  };

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<SharedEncodingAttr>(srcLayout) &&
        (isa<BlockedEncodingAttr, MmaEncodingTrait, SliceEncodingAttr>(
             dstLayout) ||
         isSupportedDotOpLayout(dstLayout))) {
      return lowerSharedToDistributed(op, adaptor, getTypeConverter(),
                                      rewriter);
    }
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<BlockedEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOpFMA(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  LogicalResult
  lowerSharedToDotOpFMA(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                        const LLVMTypeConverter *typeConverter,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType dstTy = op.getType();
    Attribute dstLayout = dstTy.getEncoding();
    auto dotLayout = cast<DotOperandEncodingAttr>(dstLayout);
    auto blockedLayout = cast<BlockedEncodingAttr>(
        cast<DotOperandEncodingAttr>(dstLayout).getParent());
    auto thread = getThreadId(rewriter, loc);
    Value res = SharedToDotOperandFMA::convertLayout(
        dotLayout.getOpIdx(), op.getSrc(), adaptor.getSrc(), blockedLayout,
        thread, loc, getTypeConverter(), rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
  LogicalResult
  lowerSharedToDistributed(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto dstShape = dstTy.getShape();
    auto srcSharedLayout = cast<SharedEncodingAttr>(srcTy.getEncoding());
    auto dstLayout = dstTy.getEncoding();
    assert((dstShape.size() <= 2 || isSupportedDotOpLayout(dstLayout)) &&
           "Unexpected rank of ConvertLayout(shared->distributed)");
    auto inOrd = getOrder(srcSharedLayout);

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemLlvmTy = typeConverter->convertType(dstTy.getElementType());

    // SmallVector<Value> outVals = loadSharedToDistributed(
    //     dstTy, srcTy, elemLlvmTy, smemObj, loc, rewriter, targetInfo);

    SmallVector<Value> outVals =
        loadSharedToDistributed(op.getResult(), op.getSrc(), smemObj,
                                elemLlvmTy, loc, rewriter, targetInfo);

    // FIXME [Dot LL]
    // Ampere case
    // In this case, we need to pack the outputs into i32
    if (auto dotOp = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding())) {
      if (auto parent = dyn_cast<NvidiaMmaEncodingAttr>(dotOp.getParent())) {
        if (parent.isAmpere()) {
          if (elemLlvmTy.isInteger(8)) {
            auto concat = [&](Value a1, Value a2, Value a3, Value a4) {
              return or_(
                  or_(zext(i32_ty, a1), shl(zext(i32_ty, a2), i32_val(8))),
                  or_(shl(zext(i32_ty, a3), i32_val(16)),
                      shl(zext(i32_ty, a4), i32_val(24))));
            };
            SmallVector<Value> outVals32(outVals.size() / 4);
            for (int i = 0; i < outVals32.size(); ++i) {
              outVals32[i] = concat(outVals[4 * i], outVals[4 * i + 1],
                                    outVals[4 * i + 2], outVals[4 * i + 3]);
            }
            outVals = outVals32;
          } else {
            assert(elemLlvmTy.isBF16() && "Unexpected element type");
            auto concat = [&](Value a, Value b) {
              return or_(zext(i32_ty, bitcast(a, i16_ty)),
                         shl(zext(i32_ty, bitcast(b, i16_ty)), i32_val(16)));
            };

            SmallVector<Value> outVals32(outVals.size() / 2);
            for (int i = 0; i < outVals32.size(); ++i) {
              outVals32[i] = concat(outVals[2 * i], outVals[2 * i + 1]);
            }
            outVals = outVals32;
          }
        }
      }
    }

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertOpToLLVMPattern;

  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value memDescVal = op.getDst();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);
    lowerDistributedToShared(op.getLoc(), op.getSrc(), op.getDst(),
                             adaptor.getSrc(), smemObj, getTypeConverter(),
                             rewriter, targetInfo);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalDeallocOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo, benefit);
}
