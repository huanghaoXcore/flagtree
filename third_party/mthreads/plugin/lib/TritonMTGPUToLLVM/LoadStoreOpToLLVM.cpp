#include "TargetInfo.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/IR/Value.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::MUSA;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::MUSA::llAsyncLoad;
using ::mlir::LLVM::MUSA::llLoad;
using ::mlir::LLVM::MUSA::llStore;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

// Return the mask for the unique data accessed by given tensor type.
// Used to mask out the redundant data accessed by threads.
Value redundantDataMask(Type valueTy, ConversionPatternRewriter &rewriter,
                        Location loc, const MUSA::TargetInfo &targetInfo) {
  auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
  Value mask = int_val(1, 1);
  auto tid = tid_val();
  auto clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
  if (tensorTy) {
    auto layout = tensorTy.getEncoding();
    auto shape = tensorTy.getShape();
    unsigned rank = shape.size();
    auto sizePerThread = triton::gpu::getSizePerThread(layout);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
    auto order = triton::gpu::getOrder(layout);
    auto warpOrder = triton::gpu::getWarpOrder(layout);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout, shape);
    auto wrapSizeInt = product<unsigned>(threadsPerWarp);
    Value warpSize = i32_val(wrapSizeInt);
    Value laneId = urem(tid, warpSize);
    Value warpId = udiv(tid, warpSize);
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, warpOrder);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
    for (unsigned dim = 0; dim < rank; ++dim) {
      // if there is no data replication across threads on this dimension
      if (shape[dim] >= shapePerCTATile[dim])
        continue;
      // Otherwise, we need to mask threads that will replicate data on this
      // dimension. Calculate the thread index on this dimension for the CTA
      Value threadDim =
          add(mul(multiDimWarpId[dim], i32_val(threadsPerWarp[dim])),
              multiDimThreadId[dim]);
      mask = and_(mask, icmp_slt(mul(threadDim, i32_val(sizePerThread[dim])),
                                 i32_val(shape[dim])));
    }
    // Do not write duplicated data when multicast is enabled
    if (triton::gpu::getNumCTAs(layout) > 1) {
      auto _0 = i32_val(0);
      auto CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
      auto CTASplitNum = triton::gpu::getCTASplitNum(layout);
      auto CTAOrder = triton::gpu::getCTAOrder(layout);

      LLVM_DEBUG(DBGS() << "[pattern storeOpConversion] "
                        << " numCTAS = " << triton::gpu::getNumCTAs(layout));
      auto multiDimClusterCTAId =
          delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

      for (unsigned dim = 0; dim < rank; ++dim) {
        // Skip when multicast is not enabled in this dimension
        if (CTAsPerCGA[dim] == CTASplitNum[dim])
          continue;
        // This wrapping rule must be consistent with emitCTAOffsetForLayout
        unsigned splitNum = std::min<unsigned>(shape[dim], CTASplitNum[dim]);
        Value repId = udiv(multiDimClusterCTAId[dim], i32_val(splitNum));
        // Consider the example where CTAsPerCGA = [4] and CTASplitNum = [2]:
        //     CTA0 and CTA2 holds data of block0,
        //     CTA1 and CTA3 holds data of block1.
        // Only CTA0 and CTA1 are expected to write while CTA2 and CTA3 should
        // be masked. We add the following mask:
        //     multiDimClusterCTAId[dim] / splitNum == 0
        // Actually in all existing cases of multicast, splitNum is always 1.
        // The mask is equivalent to:
        //     multiDimClusterCTAId[dim] == 0
        mask = and_(mask, icmp_eq(repId, _0));
      }
    }
  } else {
    // If the tensor is not ranked, then it is a scalar and only thread 0 of
    // CTA0 can write
    mask = and_(mask, icmp_eq(clusterCTAId, i32_val(0)));
    mask = and_(mask, icmp_eq(tid, i32_val(0)));
  }
  return mask;
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const MUSA::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);
    // The maximum vector size is 128 bits on MTGPU GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const MUSA::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr llvm::StringLiteral kInplaceLoadAttr =
        "mtgpu.inplace_load_candidate";
    auto loc = op->getLoc();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    SmallVector<Value> loadedVals;
    bool useInplaceLoadAsm = op->hasAttr(kInplaceLoadAttr);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is GEP with constant offset

      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);
      auto vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
      Value ptr = addrspacecast(ptr_ty(getContext()), ptrElems[vecStart]);

      mlir::Attribute zeroAttr = rewriter.getZeroAttr(valueElemTy);
      auto denseValue =
          DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
      Value zeroVal = rewriter.create<LLVM::ConstantOp>(loc, vecTy, denseValue);

      Value falseVal = zeroVal;
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0) {
        Value v = undef(vecTy);
        for (size_t s = 0; s < vec; ++s) {
          Value otherElem = otherElems[vecStart + s];
          Value indexVal = createIndexAttrConstant(
              rewriter, loc, this->getTypeConverter()->getIndexType(), s);
          v = insert_element(vecTy, v, otherElem, indexVal);
        }
        falseVal = v;
      }

      Value loadVal = useInplaceLoadAsm
                          ? mlir::LLVM::MUSA::llInplaceLoad(
                                rewriter, loc, ptr, vecTy, pred, falseVal)
                          : llLoad(rewriter, loc, ptr, vecTy, pred, falseVal);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, this->getTypeConverter()->getIndexType(), ii % vec);
        Value loaded = extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const MUSA::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (llMask) {
      Value mask = op.getMask();
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);

    auto vecTy = vec_ty(valueElemTy, vec);
    auto normalizeElem = [&](Value elem) -> Value {
      if (elem.getType().isInteger(1))
        elem = sext(i8_ty, elem);
      return bitcast(elem, valueElemTy);
    };
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      // TODO: optimization when ptr is AddPtr with constant offset
      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.

      Value storeVal;
      if (vec == 1) {
        storeVal = normalizeElem(valueElems[vecStart]);
      } else {
        storeVal = undef(vecTy);
        for (size_t elemIdx = 0; elemIdx < vec; ++elemIdx) {
          Value elem = normalizeElem(valueElems[vecStart + elemIdx]);
          storeVal = insert_element(vecTy, storeVal, elem, i32_val(elemIdx));
        }
      }
      Value maskVal = llMask ? and_(mask, maskElems[vecStart]) : mask;
      auto address = ptrElems[vecStart];
      llStore(rewriter, loc, address, storeVal, maskVal);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void createBarrier(ConversionPatternRewriter &rewriter, Location loc,
                   int numCTAs) {
  if (numCTAs == 1) {
    barrier();
  } else {
    return;
  }
}

static LLVM::AtomicOrdering getMemoryOrdering(MemSemantic memOrdering) {
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    return LLVM::AtomicOrdering::monotonic;
  case MemSemantic::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case MemSemantic::RELEASE:
    return LLVM::AtomicOrdering::release;
  case MemSemantic::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    return LLVM::AtomicOrdering::acq_rel;
  }
}

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::AtomicCASOp>::ConvertOpToLLVMPattern;

  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const MUSA::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  // Helper function to convert value to integer type
  Value convertToAtomicType(Location loc, Value val, Type elemTy,
                            ConversionPatternRewriter &rewriter) const {
    if (elemTy.isIntOrIndex()) {
      return val;
    }
    unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
    Type intTy = rewriter.getIntegerType(bitWidth);
    return rewriter.create<LLVM::BitcastOp>(loc, intTy, val);
  }

  // Helper function to convert back from integer type
  Value convertFromAtomicType(Location loc, Value val, Type elemTy,
                              ConversionPatternRewriter &rewriter) const {
    if (elemTy.isIntOrIndex()) {
      return val;
    }
    return rewriter.create<LLVM::BitcastOp>(loc, elemTy, val);
  }

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto ptrElements = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto cmpElements = unpackLLElements(loc, adaptor.getCmp(), rewriter);
    auto valElements = unpackLLElements(loc, adaptor.getVal(), rewriter);
    auto atomicMemOrdering = getMemoryOrdering(op.getSem());
    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    auto valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr());
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(op.getVal().getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    }
    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);

    for (size_t i = 0; i < elemsPerThread; i += vec) {
      auto casPtr = ptrElements[i];
      auto casCmp = cmpElements[i];
      auto casVal = valElements[i];
      auto atomicType = valueElemTy.isIntOrIndex()
                            ? valueElemTy
                            : rewriter.getIntegerType(valueElemNBits);
      auto atomicCmp = convertToAtomicType(loc, casCmp, valueElemTy, rewriter);
      auto atomicVal = convertToAtomicType(loc, casVal, valueElemTy, rewriter);

      if (tensorTy) {
        auto retType = vec == 1 ? atomicType : vec_ty(atomicType, vec);
        auto successOrdering = atomicMemOrdering;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, atomicCmp, atomicVal, successOrdering, failureOrdering,
            StringRef("musa_device"));

        Value ret = extract_val(atomicType, cmpxchg, 0);
        ret = convertFromAtomicType(loc, ret, valueElemTy, rewriter);

        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
        }
      } else {
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));
        // Determine the thread that performs the atomic operation
        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = tid_val();
        auto pred = icmp_eq(tid, i32_val(i));
        rewriter.create<LLVM::CondBrOp>(loc, pred, atomicBlock, endBlock);
        // Performs the atomic operation
        rewriter.setInsertionPointToEnd(atomicBlock);
        auto successOrdering = LLVM::AtomicOrdering::acq_rel;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, atomicCmp, atomicVal, successOrdering, failureOrdering,
            StringRef("musa_device"));
        if (atomicNeedsSharedMemory(op.getResult())) {
          Value newLoaded = extract_val(valueElemTy, cmpxchg, 0);
          Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                    op.getOperation());
          store(newLoaded, atomPtr);
        }
        rewriter.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

        if (!atomicNeedsSharedMemory(op.getResult())) {
          rewriter.eraseOp(op);
          return success();
        }
        // Synced load from shared memory
        rewriter.setInsertionPointToStart(endBlock);
        barrier();
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        Value ret = load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
      }
    }

    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const MUSA::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  /// Try to match the mlir::triton::RMWOp to LLVM::AtomicBinOp.
  static std::optional<LLVM::AtomicBinOp> matchAtomicOp(RMWOp atomicOp) {
    switch (atomicOp) {
    case RMWOp::AND:
      return LLVM::AtomicBinOp::_and;
    case RMWOp::OR:
      return LLVM::AtomicBinOp::_or;
    case RMWOp::XOR:
      return LLVM::AtomicBinOp::_xor;
    case RMWOp::ADD:
      return LLVM::AtomicBinOp::add;
    case RMWOp::FADD:
      return LLVM::AtomicBinOp::fadd;
    case RMWOp::MAX:
      return LLVM::AtomicBinOp::max;
    case RMWOp::MIN:
      return LLVM::AtomicBinOp::min;
    case RMWOp::UMAX:
      return LLVM::AtomicBinOp::umax;
    case RMWOp::UMIN:
      return LLVM::AtomicBinOp::umin;
    case RMWOp::XCHG:
      return LLVM::AtomicBinOp::xchg;
    default:
      return std::nullopt;
    }
    llvm_unreachable("Invalid RMWOp");
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto atomicRmwAttr = op.getAtomicRmwOp();
    Value ptr = op.getPtr();
    Value val = op.getVal();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    Value opResult = op.getResult();
    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();
    const size_t valueElemNbits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // vec = 1, numElements = 1 for scalar
    auto vec = getVectorSize(ptr);
    int numElems = 1;
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(val.getType());
      // NV for the f16v2 case generates one packed instruction.
      if (funcOp->hasAttr("nvvm.kernel"))
        vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
      else
        vec = std::min<unsigned>(vec, 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = int_val(1, 1);
    auto tid = tid_val();
    mask = and_(mask,
                icmp_slt(mul(tid, i32_val(elemsPerThread)), i32_val(numElems)));

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);

    auto vecTy = vec_ty(valueElemTy, vec);
    auto retType = vec == 1 ? valueElemTy : vecTy;
    SmallVector<Value> resultVals(elemsPerThread);
    const bool f16v2 = vec == 2 && valueElemTy.isF16();
    const bool scalarResultUsed = !tensorTy && !op.getResult().use_empty();
    const bool hasSharedMemoryAlloc = op->hasAttr("allocation.offset");
    if (scalarResultUsed && !hasSharedMemoryAlloc) {
      return rewriter.notifyMatchFailure(
          op, "missing allocation.offset for scalar atomic result");
    }
    const bool needScalarResult = !tensorTy && hasSharedMemoryAlloc;
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwPtr = ptrElements[i];
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? and_(mask, maskElements[i]) : mask;

      Value undefVal = undef(retType);
      // Build blocks to bypass the atomic instruction for ~rmwMask.
      auto *curBlock = rewriter.getInsertionBlock();
      auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(
          curBlock->getParent(), std::next(Region::iterator(curBlock)));
      endBlock->addArgument({retType}, {loc});

      rewriter.setInsertionPointToEnd(curBlock);
      rewriter.create<LLVM::CondBrOp>(loc, rmwMask, atomicBlock, endBlock,
                                      undefVal);

      rewriter.setInsertionPointToEnd(atomicBlock);
      auto maybeKind = matchAtomicOp(atomicRmwAttr);
      // TODO: use rocdl.raw.buffer.atomic from ROCDL dialect to use efficient
      // atomics for MI-* series of AMD GPU.
      Value atom;
      if (*maybeKind == LLVM::AtomicBinOp::fadd) {
        StringRef funcName;
        Type fpType;
        if (valueElemTy.isF16()) {
          funcName = "__mt_atomicAdd_f16";
          fpType = rewriter.getF16Type();
        } else if (valueElemTy.isF32()) {
          funcName = "__mt_atomicAdd_f32";
          fpType = rewriter.getF32Type();
        } else if (valueElemTy.isF64()) {
          funcName = "__mt_atomicAdd_f64";
          fpType = rewriter.getF64Type();
        } else {
          llvm_unreachable("Invalid value element type.");
          return failure();
        }
        auto moduleOp = op->getParentOfType<ModuleOp>();
        auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
        LLVM::LLVMFuncOp calleeFuncOp =
            moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
        if (!calleeFuncOp) {
          rewriter.setInsertionPointToStart(moduleOp.getBody());
          auto type = LLVM::LLVMFunctionType::get(fpType, {ptrTy, fpType});
          calleeFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
              loc, funcName, type, LLVM::Linkage::External);
        }
        rewriter.setInsertionPointToEnd(atomicBlock);
        Value addressCast =
            rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrTy, rmwPtr);
        atom =
            rewriter
                .create<LLVM::CallOp>(loc, calleeFuncOp,
                                      ValueRange{addressCast, valElements[i]})
                .getResult();
      } else {
        atom = rewriter
                   .create<LLVM::AtomicRMWOp>(loc, *maybeKind, rmwPtr,
                                              valElements[i], atomicMemOrdering)
                   .getResult();
      }

      // NV for the f16v2 case generates one packed instruction. We have to
      // create two separate instructions since LLVM::AtomicRMWOp doesn't
      // support this. Can be optimized out with rocdl.raw.buffer.atomic.
      if (f16v2 && funcOp->hasAttr("nvvm.kernel")) {
        Value atom2 =
            rewriter
                .create<LLVM::AtomicRMWOp>(
                    loc, *maybeKind, ptrElements[i + 1], valElements[i + 1],
                    LLVM::AtomicOrdering::monotonic, StringRef("agent"))
                .getResult();
        auto tmp = insert_element(vecTy, undef(vecTy), atom, i32_val(0));
        atom = insert_element(vecTy, tmp, atom2, i32_val(1)).getResult();
      }
      if (needScalarResult) {
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        store(atom, atomPtr);
      }

      rewriter.create<LLVM::BrOp>(loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      Value retVal = endBlock->getArgument(0);
      if (tensorTy) {
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? retVal
                       : extract_element(valueElemTy, retVal, i32_val(ii));
        }
      } else {
        if (!needScalarResult) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        barrier();
        Value ret = load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
      }
    }
    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const MUSA::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value res = op.getResult();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto srcLayout = srcTy.getEncoding();
    assert((isa<BlockedEncodingAttr, SliceEncodingAttr>(srcLayout) &&
            "Unexpected srcLayout in AsyncCopyGlobalToLocalOpConversion"));
    auto resSharedLayout = cast<SharedEncodingAttr>(dstTy.getEncoding());
    auto srcShape = srcTy.getShape();
    assert((srcShape.size() <= 3) &&
           "insert_slice_async: Unexpected rank of %src");

    Value llDst = adaptor.getResult();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // %src
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);

    // %dst
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }

    // %other
    SmallVector<Value> otherElems;
    if (llOther) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
      assert(srcElems.size() == otherElems.size());
    }
    unsigned inVec = getContiguity(op.getSrc());
    unsigned outVec = resSharedLayout.getVec();
    unsigned minVec = inVec;
    if (outVec > 1)
      minVec = std::min(outVec, inVec);
    unsigned numElems = getTotalElemsPerThread(srcTy);
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    SmallVector<Value> offsetVals = {smemObj.strides.size(), i32_val(0)};

    // get sharedPtrs for all elements.
    DenseMap<unsigned, Value> sharedPtrs = getSwizzledSharedPtrs(
        loc, targetInfo, inVec, srcTy, resSharedLayout, resElemTy, smemObj,
        rewriter, offsetVals, smemObj.strides);

    // do SQMMA swizzle if it has the 'sqmma.opIdx' attribute.
    triton::gpu::MemDescSubviewOp memDescOp =
        cast<triton::gpu::MemDescSubviewOp>(res.getDefiningOp());
    auto localAllocOp =
        dyn_cast<triton::gpu::LocalAllocOp>(memDescOp.getSrc().getDefiningOp());
    if (localAllocOp && localAllocOp->hasAttr("sqmma.opIdx")) {
      unsigned opIdx = cast<IntegerAttr>(localAllocOp->getAttr("sqmma.opIdx"))
                           .getValue()
                           .getZExtValue();
      size_t smemOffset =
          cast<IntegerAttr>(localAllocOp->getAttr("allocation.offset"))
              .getValue()
              .getZExtValue();
      Value smemIndex0 = memDescOp.getOffsets()[0];
      auto memDescType = cast<MemDescType>(memDescOp->getResult(0).getType());
      unsigned size = product(memDescType.getShape()) *
                      memDescType.getElementTypeBitWidth() / 8;
      // %0 = localAlloc() {allocation.offset = offset}
      // %1 = memdesc_subview %47[x, 0, 0] <shape[0]*shape[1]*shape[2]> ->
      // <shape[1]*shape[2]> async_copy_global_to_local(dst, %1) the allocation
      // offset of memdesc_subview euqal to: offset + x * (shape[1]*shape[2]) *
      // BytesOfElem
      Value smemOffsetVal =
          add(mul(smemIndex0, i32_val(size)), i32_val(smemOffset));
      sharedPtrs = getSqmmaSwizzledSharedPtrs(
          loc, opIdx, 0, targetInfo, inVec, srcTy, srcShape, resSharedLayout,
          resElemTy, smemObj, rewriter, offsetVals, smemObj.strides,
          smemOffsetVal);
    }

    // Predication may introduce control flow in the pipelined loop and hurt
    // perf. If mask is a constant splat (all-true), emit unpredicated async
    // copies; otherwise keep predication for correctness.
    bool usePred = false;
    if (mask) {
      Operation *maskOp = mask.getDefiningOp();
      usePred = !isa_and_nonnull<triton::SplatOp>(maskOp);
    }

    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      // 16 * 8 = 128bits
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto numWords = vecBitWidth / bitWidth;
      auto numWordElems = bitWidth / resElemTy.getIntOrFloatBitWidth();
      auto byteWidth = bitWidth / 8;
      auto resByteWidth = resElemTy.getIntOrFloatBitWidth() / 8;

      Value basePtr = sharedPtrs[elemIdx];
      for (size_t wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        StringRef cpAsyncFucName = "llvm.musa.memcpy.g2s";
        auto wordElemIdx = wordIdx * numWordElems;
        unsigned offset = wordElemIdx * resByteWidth;
        Value src = srcElems[elemIdx + wordElemIdx];
        Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
        Value dst = gep(elemPtrTy, resElemTy, basePtr, i32_val(offset));
        Value cpSize = i32_val(byteWidth);
        Value prefetchSize = i32_val(0);

        // When 'other != 0' is supported, we will need to fold the op.getMask()
        // and redundantDataMask() into the same predicate, the way it is done
        // for LoadOp.
        // TODO(lingfeng.qiu): hanle shape < shapePerTile
        // Value maskVal = redundantDataMask(srcTy, rewriter, loc, targetInfo);

        if (usePred) {
          Value pred = mask ? maskElems[elemIdx + wordElemIdx] : int_val(1, 1);
          // async g2s with pred does not write masked-off lanes. Materialize
          // tl.load(..., other=0) semantics by explicitly zero-filling those
          // destinations before issuing predicated async copies.
          Value notPred = xor_(pred, int_val(1, 1));
          Value zeroElem = null(resElemTy);
          for (unsigned elem = 0; elem < numWordElems; ++elem) {
            Value zeroDst = gep(elemPtrTy, resElemTy, dst, i32_val(elem));
            llStore(rewriter, loc, zeroDst, zeroElem, notPred);
          }
          SmallVector<Value> cpAsyncOps = {dst, src, cpSize, prefetchSize,
                                           pred};
          llAsyncLoad(rewriter, loc, cpAsyncOps);
        } else {
          SmallVector<Value> cpAsyncOps = {dst, src, cpSize, prefetchSize};
          LLVM::createLLVMIntrinsicCallOp(rewriter, loc, cpAsyncFucName,
                                          TypeRange{}, cpAsyncOps);
        }
      }
    }

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

enum SG { SG_NONE = 0, SG_16B, SG_32B, SG_64B, SG_128B };

enum SS { SS_32B = 0, SS_64B, SS_128B, SS_256B };

enum SL { SL_128B = 0, SL_256B };

struct TMEConfig {
  Value sg;
  Value ss;
  Value sl;
  Value prefetchSize;
  Value innerPersist;
  Value outerPersist;
  Value cachePolicy;
};

/*
  SL = 256Bytes
  8bit element:
  r-major: SG = 16Bytes
  c-major: SG = 16Bytes
  16bit element:
  r-major: SG = 16Bytes
  c-major: SG = 32Bytes
  32bit element:
  r-major: SG = 16Bytes
  c-major: SG = 64Bytes

  SS = 256B (HW constraint)
  thus:
  vec = SG / elementSizeInByte
  perPhase = 1
  maxPhase = SS / SG
*/
// same to sqmma createDescriptor
// fp8
//   k-major sg16
//   mn-major sg16
// fp16
//   k-major sg16
//   mn-major sg32

// A(opIdx==0)
// rowmajor m*k which also means k-major
// B(opIdx==1)
// rowmajor k*n which also means mn-major
TMEConfig getTMEConfig(Operation *op, int elementSizeInBytes,
                       SharedEncodingAttr sharedLayout,
                       ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  TMEConfig config;

  config.sg = i32_val(SG_NONE);
  config.ss = i32_val(SS_256B);
  config.sl = i32_val(SL_256B);
  config.prefetchSize = i32_val(0);
  config.innerPersist = i32_val(2);
  config.outerPersist = i32_val(2);
  config.cachePolicy = i32_val(0);

  if (op->hasAttr("sqmma.opIdx")) {
    unsigned opIdx =
        cast<IntegerAttr>(op->getAttr("sqmma.opIdx")).getValue().getZExtValue();
    auto order = sharedLayout.getOrder();
    bool isRowMajor = (order[0] != 0);
    bool isMNMajor =
        ((opIdx == 0) && !isRowMajor) || ((opIdx == 1) && isRowMajor);

    if (elementSizeInBytes == 2) {
      // handle fp16
      config.sg = isMNMajor ? i32_val(SG_32B) : i32_val(SG_16B);
    } else if (elementSizeInBytes == 4) {
      config.sg = isMNMajor ? i32_val(SG_64B) : i32_val(SG_16B);
    } else {
      // handle fp8
      config.sg = i32_val(SG_16B);
    }
  }

  return config;
}

struct AsyncTMECopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::mthreads_gpu::AsyncTMECopyGlobalToLocalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult splitTmeLoadByLeadingDim(
      triton::mthreads_gpu::AsyncTMECopyGlobalToLocalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, unsigned opIdx) const {
    auto loc = op.getLoc();
    Type llvmElemTy =
        typeConverter->convertType(op.getResult().getType().getElementType());
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), llvmElemTy, rewriter);
    Value barId = adaptor.getBarId();
    auto id = getThreadId(rewriter, loc);

    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value warpID = udiv(id, i32_val(warpSize));
    Value pred = adaptor.getPred();
    // We enable squad 1 and thread 0 to do the tma load B, and load the whole
    // tensor data to the shared memory
    // FIXME: failed when numWarps = 4 and N = 256, f16
    // Value isWarp0 = icmp_eq(id, i32_val(128));
    Value isWarp0 = icmp_ult(id, i32_val(1));

    Value predWithWarp0 = and_(pred, isWarp0);

    int elementSizeInBytes =
        op.getResult().getType().getElementType().getIntOrFloatBitWidth() / 8;
    auto shape = op.getResult().getType().getShape();
    int totalNumElements = product(shape);
    int64_t transactionSize = totalNumElements * elementSizeInBytes;
    Value transCnt = i32_val(transactionSize);

    Type resultTy = op.getResult().getType();
    auto sharedEncoding =
        cast<SharedEncodingAttr>(cast<MemDescType>(resultTy).getEncoding());
    SmallVector<int64_t> shapePerCTA = getShapePerCTA(resultTy);
    auto order = sharedEncoding.getOrder();

    int swizzlingLineBytes = 256;
    int elemsPerSwizzlingRow = swizzlingLineBytes / elementSizeInBytes;
    assert(shapePerCTA[order[0]] % elemsPerSwizzlingRow == 0 &&
           "The leading dimension of shared memory must be multiple of "
           "swizzling row");
    int numRepLeadingDim = shapePerCTA[order[0]] / elemsPerSwizzlingRow;

    Type v2i32Ty = vec_ty(i32_ty, 2);
    Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    SmallVector<Value> BlockPos(numRepLeadingDim);
    SmallVector<Value> shMemPtrs(numRepLeadingDim);
    Value BlockDim = undef(v2i32Ty);
    BlockDim = insert_element(v2i32Ty, BlockDim, i32_val(elemsPerSwizzlingRow),
                              i32_val(0));
    BlockDim =
        insert_element(v2i32Ty, BlockDim, i32_val(shape[order[1]]), i32_val(1));
    for (size_t i = 0; i < numRepLeadingDim; i++) {
      Value blockPos = undef(v2i32Ty);
      blockPos = insert_element(
          v2i32Ty, blockPos,
          add(adaptor.getCoord()[1], i32_val(i * elemsPerSwizzlingRow)),
          i32_val(0));
      blockPos =
          insert_element(v2i32Ty, blockPos, adaptor.getCoord()[0], i32_val(1));
      BlockPos[i] = blockPos;
      Value shMemOffset = i32_val(i * shape[order[1]] * elemsPerSwizzlingRow);
      Value shMemPtr =
          gep(elemPtrTy, llvmElemTy, dstMemObj.getBase(), shMemOffset);
      shMemPtrs[i] = shMemPtr;
    }

    auto config = getTMEConfig(op.getOperation(), elementSizeInBytes,
                               sharedEncoding, rewriter);
    Value sg = config.sg;
    Value ss = config.ss;
    Value sl = config.sl;
    Value prefetchSize = config.prefetchSize;
    Value innerPersist = config.innerPersist;
    Value outerPersist = config.outerPersist;
    Value cachePolicy = config.cachePolicy;

    StringRef tmeLdFuncName = "llvm.musa.tme.ld.tile.2d";
    StringRef asyncTransCntFuncName = "llvm.musa.async.add.trans";
    Value descPtr = ptrtoint(i64_ty, adaptor.getDescPtr());
    SmallVector<Value> asyncOps = {barId, transCnt};
    SmallVector<SmallVector<Value>> tmeLdOps(numRepLeadingDim);
    for (size_t i = 0; i < numRepLeadingDim; i++) {
      tmeLdOps[i] = {barId,        shMemPtrs[i], descPtr,      BlockDim,
                     BlockPos[i],  sg,           ss,           sl,
                     prefetchSize, innerPersist, outerPersist, cachePolicy};
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterLoad);
    rewriter.setInsertionPointToEnd(currentBlock);
    SmallVector<Value> barrierOperands = {};
    barrier();
    rewriter.create<LLVM::CondBrOp>(loc, predWithWarp0, trueBlock, afterLoad);
    rewriter.setInsertionPointToStart(trueBlock);
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, asyncTransCntFuncName,
                                    TypeRange{}, asyncOps);
    for (auto &tmeLdOp : tmeLdOps) {
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc, tmeLdFuncName, TypeRange{},
                                      tmeLdOp);
    }
    rewriter.create<LLVM::BrOp>(loc, afterLoad);
    rewriter.setInsertionPointToStart(afterLoad);
    rewriter.eraseOp(op);
    return success();
  }

  Value fillBlockDim(ConversionPatternRewriter &rewriter, Location loc,
                     int rank, ArrayRef<int64_t> shape,
                     bool isContiguous) const {
    bool isReshape = (rank != (int)shape.size());
    if (isReshape) {
      assert(shape.size() == 2 && "only support reshape to 2d");
      assert(rank >= 2 && "reshape requires rank >= 2");
    }

    auto i32VecTy = vec_ty(i32_ty, rank);
    Value blockDim = undef(i32VecTy);

    // ---- Fill BlockDim ---- //
    if (!isReshape) {
      // normal: dim follows shape (with contiguous mapping)
      for (int i = 0; i < rank; ++i) {
        int idx = isContiguous ? (rank - i - 1) : i;
        blockDim =
            insert_element(i32VecTy, blockDim, i32_val(shape[idx]), i32_val(i));
      }
    } else {
      // reshape-to-2d: only two dims take shape, others are 1
      // contiguous: put shape reversed into the first 2 lanes
      // non-contiguous: keep order for first 2 lanes
      for (int i = 0; i < rank; ++i) {
        Value dimV = i32_val(1);
        if (i < 2) {
          int sidx = isContiguous ? (1 - i) : i;
          dimV = i32_val(shape[sidx]);
        }
        blockDim = insert_element(i32VecTy, blockDim, dimV, i32_val(i));
      }
    }

    return blockDim;
  }

  Value fillBlockPos(ConversionPatternRewriter &rewriter, Location loc,
                     int rank, ArrayRef<int64_t> shape, bool isContiguous,
                     ValueRange coord) const {
    bool isReshape = (rank != (int)shape.size());
    if (isReshape) {
      assert(shape.size() == 2 && "only support reshape to 2d");
      assert(rank >= 2 && "reshape requires rank >= 2");
    }

    auto i32VecTy = vec_ty(i32_ty, rank);
    Value blockPos = undef(i32VecTy);

    // ---- Fill BlockPos ---- //
    for (int i = 0; i < rank; ++i) {
      int idx;
      if (isReshape && !isContiguous && i < 2) {
        // reshape: swap last two dims
        idx = (i == 0) ? (rank - 2) : (rank - 1);
      } else {
        idx = isContiguous ? (rank - i - 1) : i;
      }
      blockPos = insert_element(i32VecTy, blockPos, coord[idx], i32_val(i));
    }

    return blockPos;
  }

  LogicalResult
  matchAndRewrite(triton::mthreads_gpu::AsyncTMECopyGlobalToLocalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = op.getResult().getType();
    auto shape = ty.getShape();
    auto sharedLayout = cast<SharedEncodingAttr>(ty.getEncoding());
    auto ord = sharedLayout.getOrder();
    bool isContiguous = (ord[0] == (shape.size() - 1));
    // SQMMA: loadB is split when the N-major dimension width of operand B
    // exceeds 256B.
    if (op->hasAttr("sqmma.opIdx")) {
      unsigned opIdx = cast<IntegerAttr>(op->getAttr("sqmma.opIdx"))
                           .getValue()
                           .getZExtValue();
      auto ty = op.getResult().getType();
      auto sharedLayout = cast<SharedEncodingAttr>(ty.getEncoding());
      auto shape = ty.getShape();
      auto ord = sharedLayout.getOrder();
      uint32_t elemBytes = ty.getElementTypeBitWidth() / 8;
      uint32_t leadingWidthInByte = shape[ord[0]] * elemBytes;

      auto trySplitByLeading =
          [&](unsigned idx) -> std::optional<LogicalResult> {
        uint32_t elemBytes = ty.getElementTypeBitWidth() / 8;
        uint32_t leadingWidthInByte = shape[ord[0]] * elemBytes;
        int ord0 = ord[0];

        const char *opName = (idx == 0) ? "A" : "B";
        // A: ord0==0->M, ord0==1->K
        // B: ord0==0->K, ord0==1->N
        const char *majorName =
            (idx == 0) ? (ord0 == 0 ? "M" : "K") : (ord0 == 0 ? "K" : "N");

        if (ord0 == 0) {
          assert(leadingWidthInByte <= 256 &&
                 (std::string("The maximum width of the ") + majorName +
                  "-major dimension of SQMMA operand " + opName + " is 256B")
                     .c_str());
          return std::nullopt;
        }

        if (leadingWidthInByte > 256) {
          assert((leadingWidthInByte % 256) == 0 &&
                 (std::string("The width of the ") + majorName +
                  "-major dimension of SQMMA operand " + opName +
                  " should be multiple of 512B when it is larger than 256B")
                     .c_str());
          return splitTmeLoadByLeadingDim(op, adaptor, rewriter, idx);
        }

        return std::nullopt;
      };

      if (opIdx == 0 || opIdx == 1) {
        if (auto res = trySplitByLeading(opIdx)) {
          return *res;
        }
      }
    }

    auto loc = op.getLoc();
    Type llvmElemTy =
        typeConverter->convertType(op.getResult().getType().getElementType());
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), llvmElemTy, rewriter);
    Value barId = adaptor.getBarId();
    auto id = getThreadId(rewriter, loc);

    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value warpID = udiv(id, i32_val(warpSize));
    Value pred = adaptor.getPred();
    // We enable warp 0 and thread 0 to do the tma load, and load the whole
    // tensor data to the shared memory
    Value isWarp0 = icmp_ult(id, i32_val(1));
    Value predWithWarp0 = and_(pred, isWarp0);

    int elementSizeInBytes =
        op.getResult().getType().getElementType().getIntOrFloatBitWidth() / 8;
    int totalNumElements = product(shape);
    int64_t transactionSize = totalNumElements * elementSizeInBytes;
    Value transCnt = i32_val(transactionSize);
    int rank = op.getCoord().size();

    Value blockDim;
    Value blockPos;
    switch (rank) {
    case 1: {
      blockDim = i32_val(shape[0]);
      blockPos = adaptor.getCoord()[0];
      break;
    }
    case 2:
    case 3: {
      blockDim = fillBlockDim(rewriter, loc, rank, shape, isContiguous);
      blockPos = fillBlockPos(rewriter, loc, rank, shape, isContiguous,
                              adaptor.getCoord());
      break;
    }
    default:
      llvm_unreachable("not supported rank");
      break;
    }

    auto resultTy = op.getResult().getType();
    TMEConfig config = getTMEConfig(op.getOperation(), elementSizeInBytes,
                                    sharedLayout, rewriter);
    Value sg = config.sg;
    Value ss = config.ss;
    Value sl = config.sl;
    Value prefetchSize = config.prefetchSize;
    Value innerPersist = config.innerPersist;
    Value outerPersist = config.outerPersist;
    Value cachePolicy = config.cachePolicy;

    Value shMemOffset = i32_val(0);
    Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    Value shMemPtr =
        gep(elemPtrTy, llvmElemTy, dstMemObj.getBase(), shMemOffset);

    auto getTmeIntrName = [&](int rank) {
      switch (rank) {
      case 1:
        return "llvm.musa.tme.ld.tile.1d";
      case 2:
        return "llvm.musa.tme.ld.tile.2d";
      case 3:
        return "llvm.musa.tme.ld.tile.3d";
      default:
        llvm_unreachable("not supported rank");
        return "";
      }
      return "";
    };

    StringRef tmeLdFuncName = getTmeIntrName(rank);
    StringRef asyncTransCntFuncName = "llvm.musa.async.add.trans";
    Value descPtr = ptrtoint(i64_ty, adaptor.getDescPtr());
    SmallVector<Value> asyncOps = {barId, transCnt};
    SmallVector<Value> tmeLdOps = {
        barId, shMemPtr, descPtr,      blockDim,     blockPos,     sg,
        ss,    sl,       prefetchSize, innerPersist, outerPersist, cachePolicy};

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterLoad);
    rewriter.setInsertionPointToEnd(currentBlock);
    // barrier();
    SmallVector<Value> barrierOperands = {};
    barrier();
    rewriter.create<LLVM::CondBrOp>(loc, predWithWarp0, trueBlock, afterLoad);
    rewriter.setInsertionPointToStart(trueBlock);
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, asyncTransCntFuncName,
                                    TypeRange{}, asyncOps);
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, tmeLdFuncName, TypeRange{},
                                    tmeLdOps);
    rewriter.create<LLVM::BrOp>(loc, afterLoad);
    rewriter.setInsertionPointToStart(afterLoad);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncTMECopyLocalToGlobalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::mthreads_gpu::AsyncTMECopyLocalToGlobalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mthreads_gpu::AsyncTMECopyLocalToGlobalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type llvmElemTy =
        typeConverter->convertType(op.getSrc().getType().getElementType());
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), llvmElemTy, rewriter);
    auto id = getThreadId(rewriter, loc);
    int elementSizeInBytes =
        op.getSrc().getType().getElementType().getIntOrFloatBitWidth() / 8;
    int totalNumElements = product(op.getSrc().getType().getShape());
    int64_t size = totalNumElements * elementSizeInBytes;

    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    auto shape = op.getSrc().getType().getShape();
    Value warpID = udiv(id, i32_val(warpSize));
    int rank = op.getCoord().size();
    // We enable warp 0 and thread 0 to do the tma store, and load the whole
    // tensor data to the shared memory
    Value predWithWarp0 = icmp_ult(id, i32_val(1));

    Value blockDim;
    Value blockPos;
    switch (rank) {
    case 1: {
      blockDim = i32_val(shape[0]);
      blockPos = adaptor.getCoord()[0];
      break;
    }
    case 2: {
      auto v2i32Ty = vec_ty(i32_ty, 2);
      blockDim = undef(v2i32Ty);
      blockPos = undef(v2i32Ty);
      for (int i = 0; i < rank; ++i) {
        blockDim = insert_element(v2i32Ty, blockDim,
                                  i32_val(shape[rank - i - 1]), i32_val(i));
        blockPos = insert_element(v2i32Ty, blockPos,
                                  adaptor.getCoord()[rank - i - 1], i32_val(i));
      }
      break;
    }
    case 3: {
      auto v3i32Ty = vec_ty(i32_ty, 3);
      blockDim = undef(v3i32Ty);
      blockPos = undef(v3i32Ty);
      for (int i = 0; i < rank; ++i) {
        blockDim = insert_element(v3i32Ty, blockDim,
                                  i32_val(shape[rank - i - 1]), i32_val(i));
        blockPos = insert_element(v3i32Ty, blockPos,
                                  adaptor.getCoord()[rank - i - 1], i32_val(i));
      }
      break;
    }
    default:
      llvm_unreachable("not supported rank");
      break;
    }

    Value sg = i32_val(SG_NONE);
    Value ss = i32_val(SS_256B);
    Value sl = i32_val(SL_256B);
    Value InnerPersist = i32_val(2);
    Value OuterPersist = i32_val(2);
    Value CachePolicy = i32_val(0);

    Value shMemOffset = i32_val(0);
    Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    Value shMemPtr =
        gep(elemPtrTy, llvmElemTy, dstMemObj.getBase(), shMemOffset);

    auto getTmeIntrName = [&](int rank) {
      switch (rank) {
      case 1:
        return "llvm.musa.tme.st.1d";
      case 2:
        return "llvm.musa.tme.st.2d";
      case 3:
        return "llvm.musa.tme.st.3d";
      default:
        llvm_unreachable("not supported rank");
        return "";
      }
      return "";
    };

    auto ctx = op->getContext();
    StringRef tmeStFuncName = getTmeIntrName(rank);
    Value descPtr = ptrtoint(i64_ty, adaptor.getDescPtr());
    SmallVector<Value> tmeStOperands = {
        shMemPtr, descPtr, blockDim,     blockPos,     sg,
        ss,       sl,      InnerPersist, OuterPersist, CachePolicy};
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterStore =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterStore);
    rewriter.setInsertionPointToEnd(currentBlock);
    barrier();
    rewriter.create<LLVM::CondBrOp>(loc, predWithWarp0, trueBlock, afterStore);
    rewriter.setInsertionPointToStart(trueBlock);
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, tmeStFuncName, TypeRange{},
                                    tmeStOperands);
    rewriter.create<LLVM::BrOp>(loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    StringRef asyncWaitFuncName = "llvm.musa.memcpy.g2s.wait";
    LLVM::createLLVMIntrinsicCallOp(rewriter, op.getLoc(), asyncWaitFuncName,
                                    TypeRange{}, ValueRange{});
    // llvm.musa.memcpy.g2s.wait only guarantees completion for the current
    // thread's outstanding g2s operations. Make the wait visible CTA-wide
    // before any shared-memory consumer executes.
    barrier();

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct TMEStoreWaitConversion
    : public ConvertOpToLLVMPattern<triton::mthreads_gpu::TMEStoreWait> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::mthreads_gpu::TMEStoreWait op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value> operands = {};
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.tme.store.commit",
                                    TypeRange{}, operands);
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.musa.tme.store.read.wait", TypeRange{}, operands);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<LoadOpConversion, StoreOpConversion>(typeConverter, targetInfo,
                                                    axisInfoAnalysis, benefit);
  patterns.add<AtomicCASOpConversion>(typeConverter, targetInfo,
                                      axisInfoAnalysis, benefit);
  patterns.add<AtomicRMWOpConversion>(typeConverter, targetInfo,
                                      axisInfoAnalysis, benefit);
  patterns.add<AsyncCopyGlobalToLocalOpConversion>(typeConverter, targetInfo,
                                                   axisInfoAnalysis, benefit);

  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, benefit);
  patterns.add<AsyncTMECopyGlobalToLocalOpConversion,
               AsyncTMECopyLocalToGlobalOpConversion, TMEStoreWaitConversion>(
      typeConverter, benefit);
}
