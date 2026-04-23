#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::MthreadsSqmmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

static triton::mtgpu::SQMMAEltType getMmaRetType(Value d) {
  auto dTy = cast<RankedTensorType>(d.getType()).getElementType();
  if (dTy.isF32()) {
    return triton::mtgpu::SQMMAEltType::f32;
  } else if (dTy.isF16()) {
    return triton::mtgpu::SQMMAEltType::f16;
  } else if (dTy.isInteger(32)) {
    return triton::mtgpu::SQMMAEltType::s32;
  } else {
    llvm::report_fatal_error("Unsupported mma result type found");
  }
}

static triton::mtgpu::SQMMAEltType getMmaOperandType(Value a, bool allowTF32) {
  auto aTy = cast<TensorOrMemDesc>(a.getType()).getElementType();
  if (aTy.isF16()) {
    return triton::mtgpu::SQMMAEltType::f16;
  } else if (aTy.isBF16()) {
    return triton::mtgpu::SQMMAEltType::bf16;
  } else if (aTy.isF32() && allowTF32) {
    return triton::mtgpu::SQMMAEltType::tf32;
  } else if (aTy.isInteger(8)) {
    return triton::mtgpu::SQMMAEltType::s8;
  } else if (aTy.isFloat8E5M2()) {
    return triton::mtgpu::SQMMAEltType::e5m2;
  } else if (aTy.isFloat8E4M3FNUZ() || aTy.isFloat8E4M3FN()) {
    return triton::mtgpu::SQMMAEltType::e4m3;
  } else {
    llvm::report_fatal_error("Unsupported mma operand type found");
  }
}

namespace {
int64_t getSwizzlingFromLayout(const SharedEncodingAttr &layout,
                               uint32_t widthInByte) {
  int perPhase = layout.getPerPhase();
  int maxPhase = layout.getMaxPhase();
  uint32_t swizzlingByteWidth = 0;
  if (perPhase == 4 && maxPhase == 2) {
    swizzlingByteWidth = 32;
  } else if (perPhase == 2 && maxPhase == 4) {
    swizzlingByteWidth = 64;
  } else if (perPhase == 1 && maxPhase == 8) {
    swizzlingByteWidth = 128;
  } else if (perPhase == 1 && maxPhase == 1) {
    // FIXME: no swizzle is not supported on ph1
    swizzlingByteWidth = 0;
  } else {
    llvm::report_fatal_error("Unsupported shared layout.");
  }

  // TODO[biaow]: remove it once we support swizzling size larger than matrix
  // width, which requires padding the matrix width to the swizzling size when
  // allocating shared memory.
  assert(swizzlingByteWidth <= widthInByte &&
         "swizzling size larger than matrix width is not supported.");
  return swizzlingByteWidth;
}

enum { SG_NONE = 0, SG_16B = 1, SG_32B = 2, SG_64B = 3 };
enum {
  SS_8B = 1,
  SS_16B = 2,
  SS_32B = 3,
  SS_64B = 4,
  SS_128B = 5,
  SS_256B = 6
};
union SqmmaDescriptor {
  int32_t descriptor;
  struct {
    // start_address, bit [0, 18)
    uint32_t start_address_ : 18;
    // leading dimension byte offset type, bit [19, 21)
    // 8B:1,16B:2,32B:3,64B:4,128B:5,256B:6
    uint32_t leading_stride_type_ : 3;
    // swizzle granularity type, bit [22, 23)
    // SG_NONE=0,SG_16B=1,SG_32B=2,SG_64B=3
    uint32_t swizzle_granularity_type_ : 2;
  };
  void dump() {
    printf("create desc: addr=0x%x leading_stride=%s sg=%s\n", start_address_,
           get_sqmma_desc_name_ls(leading_stride_type_).c_str(),
           get_sqmma_desc_name_sg(swizzle_granularity_type_).c_str());
  }

  static std::string get_sqmma_desc_name_sg(uint32_t val) {
    switch (val) {
    case 0:
      return "SG_NONE";
    case 1:
      return "SG_16B";
    case 2:
      return "SG_32B";
    case 3:
      return "SG_64B";
    default:
      return "bad value";
    }
  }

  static std::string get_sqmma_desc_name_ls(uint32_t val) {
    switch (val) {
    case 1:
      return "8B";
    case 2:
      return "16B";
    case 3:
      return "32B";
    case 4:
      return "64B";
    case 5:
      return "128B";
    case 6:
      return "256B";
    case 0:
    default:
      assert(0 && "not support leading_stride_type");
      return "bad value";
    }
  }
};

static Value createDescriptor(ConversionPatternRewriter &rewriter, Location loc,
                              uint32_t widthInByte, int elemBytes,
                              bool isRowMajor, unsigned int opIdx) {
  static_assert(sizeof(SqmmaDescriptor) == 4,
                "Descriptor size should be 32 bits.");
  SqmmaDescriptor desc;
  desc.descriptor = 0;
  // for sqmma
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
  bool isMNMajor =
      ((opIdx == 0) && !isRowMajor) || ((opIdx == 1) && isRowMajor);
  if (isMNMajor) {
    // mn-major
    if (elemBytes == 2) {
      // handle fp16
      desc.swizzle_granularity_type_ = SG_32B;
    } else if (elemBytes == 4) {
      desc.swizzle_granularity_type_ = SG_64B;
    } else {
      // handle fp8
      desc.swizzle_granularity_type_ = SG_16B;
    }
  } else {
    // k-major
    desc.swizzle_granularity_type_ = SG_16B;
  }

  uint32_t leading_stride_type = (int)(std::log2(widthInByte)) - 2;
  // loadB is split when the N-major dimension width of operand B exceeds 256B
  desc.leading_stride_type_ = leading_stride_type > 6 ? 6 : leading_stride_type;
  return int_val(32, desc.descriptor);
}

class DotOpMmaV3SmemLoader {
public:
  DotOpMmaV3SmemLoader() {}
  DotOpMmaV3SmemLoader(Value tensor, Value base, SmallVector<int64_t> shape,
                       Value warpId, unsigned int dimWpt, bool trans,
                       SmallVector<unsigned int> instrShape, unsigned int opIdx,
                       ConversionPatternRewriter &rewriter, Location loc)
      : base(base), shape(shape), warpId(warpId), dimWpt(dimWpt), trans(trans),
        instrShape(instrShape), opIdx(opIdx) {
    auto ty = cast<MemDescType>(tensor.getType());
    auto sharedLayout = cast<SharedEncodingAttr>(ty.getEncoding());
    ord = sharedLayout.getOrder();
    const int perPhase = sharedLayout.getPerPhase();
    const int maxPhase = sharedLayout.getMaxPhase();
    elemBytes = ty.getElementTypeBitWidth() / 8;
    uint32_t widthInByte = shape[ord[0]] * elemBytes;
    if (widthInByte > 256) {
      elemsPerSwizzlingRow = 256 / 1 /*perPhase*/ / elemBytes;
    } else {
      elemsPerSwizzlingRow = shape[ord[0]];
    }
    elemsPerSwizzlingRowVal = i32_val(elemsPerSwizzlingRow);
    descriptor = createDescriptor(rewriter, loc, widthInByte, elemBytes,
                                  ord[0] == 1, opIdx);
  }

  Value smemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                 Location loc) {
    Value k = i32_val(b * instrShape[1]);
    Value m = add(i32_val(a * dimWpt * instrShape[0]),
                  mul(warpId, i32_val(instrShape[0])));
    if (trans) {
      std::swap(k, m);
    }
    Value leading_offset = mul(udiv(k, elemsPerSwizzlingRowVal),
                               i32_val(shape[ord[1]] * elemsPerSwizzlingRow));
    Value stride_offset = mul(m, elemsPerSwizzlingRowVal);
    Value offset = add(add(leading_offset, stride_offset),
                       urem(k, elemsPerSwizzlingRowVal));
    Value off1 = mul(i32_val(elemBytes), offset);
    Value loadDesc = add(descriptor, off1);
    loadDesc = add(loadDesc, ptrtoint(i32_ty, base));
    return loadDesc;
  }

private:
  Value base;
  SmallVector<int64_t> shape;
  Value warpId;
  int dimWpt;
  bool trans;
  Value elemsPerSwizzlingRowVal;
  SmallVector<unsigned int> instrShape;
  ArrayRef<unsigned> ord;
  int elemsPerSwizzlingRow;
  int elemBytes;
  Value descriptor;
  unsigned int opIdx;
};

DotOpMmaV3SmemLoader loadA(const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const MthreadsSqmmaEncodingAttr &mmaEncoding,
                           Value tensor, Value smemObjBase, Value thread) {
  auto aTy = cast<TensorOrMemDesc>(tensor.getType());
  auto aSharedLayout = dyn_cast<SharedEncodingAttr>(aTy.getEncoding());
  assert(aSharedLayout && "only support load dot operand from shared.");
  auto instrShape = mmaEncoding.getInstrShape();
  auto wpt = mmaEncoding.getWarpsPerCTA();
  auto aOrd = aSharedLayout.getOrder();
  bool transA = aOrd[0] == 0;
  auto shapePerCTA = getShapePerCTA(aTy);

  int numRepM = ceil<unsigned>(shapePerCTA[0], instrShape[0] * wpt[0]);
  int numRepK = ceil<unsigned>(shapePerCTA[1], instrShape[2]);

  // The descriptor should be calculated based on the first warp of the
  // warpgroup.
  Value warp = and_(udiv(thread, i32_val(32)), i32_val(0xFFFFFFFC));
  Value warpM = urem(warp, i32_val(wpt[0]));
  Value warpId = urem(warpM, i32_val(shapePerCTA[0] / instrShape[0]));

  return {tensor,
          smemObjBase,
          shapePerCTA,
          warpId,
          wpt[0],
          transA,
          {instrShape[0], instrShape[2]},
          0, // opIdx
          rewriter,
          loc};
}

DotOpMmaV3SmemLoader loadB(const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Location loc,
                           MthreadsSqmmaEncodingAttr &mmaEncoding, Value tensor,
                           Value base, Value thread) {
  auto bTy = cast<MemDescType>(tensor.getType());
  auto bSharedLayout = cast<SharedEncodingAttr>(bTy.getEncoding());
  assert(bSharedLayout && "only support load B from shared.");
  auto instrShape = mmaEncoding.getInstrShape();
  auto wpt = mmaEncoding.getWarpsPerCTA();
  auto bOrd = bSharedLayout.getOrder();
  bool transB = bOrd[0] == 1;
  auto shapePerCTA = triton::gpu::getShapePerCTA(bTy);

  int numRepK = ceil<unsigned>(shapePerCTA[0], instrShape[2]);
  int numRepN = ceil<unsigned>(shapePerCTA[1], instrShape[1] * wpt[1]);

  Value warp = and_(udiv(thread, i32_val(32)), i32_val(0xFFFFFFFC));
  Value warpMN = udiv(warp, i32_val(wpt[0]));
  Value warpN = urem(warpMN, i32_val(wpt[1]));
  Value warpId = urem(warpN, i32_val(shapePerCTA[1] / instrShape[1]));

  return {tensor,
          base,
          shapePerCTA,
          warpId,
          wpt[1],
          transB,
          {instrShape[1], instrShape[2]},
          1, // opIdx
          rewriter,
          loc};
}

// Return a vector of Value of the accumulator start at startIndex and pack the
// values into 32bits in case the accumulator is fp16.
llvm::SmallVector<Value> loadReg(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 const SmallVector<Value> &elements,
                                 int startIndex, int numElements,
                                 Operation *insertBefore) {
  OpBuilder::InsertionGuard g(rewriter);
  if (insertBefore) {
    rewriter.setInsertionPoint(insertBefore);
  }

  if (!elements[0].getType().isIntOrFloat() ||
      elements[0].getType().getIntOrFloatBitWidth() >= 32) {
    llvm::SmallVector<Value> mmaOut(numElements);
    for (int i = 0; i < numElements; ++i)
      mmaOut[i] = elements[startIndex + i];
    return mmaOut;
  }
  Type elementType = elements[0].getType();
  int numElemsPer32Bits = 32 / elementType.getIntOrFloatBitWidth();

  // For FP16 and BF16 we need to pack accumulator into 32-bit integers.
  int num32BitValues = numElements / numElemsPer32Bits;
  llvm::SmallVector<Value> mmaOut(num32BitValues);
  Type packTy = vec_ty(elementType, numElemsPer32Bits);
  for (int i = 0; i < num32BitValues; ++i) {
    Value pack = rewriter.create<LLVM::UndefOp>(loc, packTy);
    for (int j = 0; j < numElemsPer32Bits; ++j) {
      Value element = elements[startIndex + i * numElemsPer32Bits + j];
      pack = insert_element(packTy, pack, element, i32_val(j));
    }
    pack = bitcast(pack, rewriter.getIntegerType(32));
    mmaOut[i] = pack;
  }
  return mmaOut;
}

// If the accumulator is fp16 unpack it from 32-bit integers.
SmallVector<Value> unpackAccumulator(ConversionPatternRewriter &rewriter,
                                     Location loc,
                                     const SmallVector<Value> &packed,
                                     RankedTensorType tensorTy) {
  if (!tensorTy.getElementType().isF16())
    return packed;
  // For fp16 the accumulator is pack into 32-bit integers so we need to unpack
  // it.
  SmallVector<Value> results;
  for (Value elem : packed) {
    elem = bitcast(elem, vec_ty(rewriter.getF16Type(), 2));
    results.push_back(extract_element(rewriter.getF16Type(), elem, i32_val(0)));
    results.push_back(extract_element(rewriter.getF16Type(), elem, i32_val(1)));
  }
  return results;
}

static bool isFP8(triton::mtgpu::SQMMAEltType eltType) {
  return eltType == triton::mtgpu::SQMMAEltType::e5m2 ||
         eltType == triton::mtgpu::SQMMAEltType::e4m3;
}

static VectorType getVectorType(Type elemTy, unsigned length) {
  SmallVector<int64_t, 1> shape{static_cast<int64_t>(length)};
  return VectorType::get(shape, elemTy);
}

static Value packVectorElements(ConversionPatternRewriter &rewriter,
                                Location loc, Type vectorTy,
                                ArrayRef<Value> elements) {
  auto pack = [&](auto vecTy) -> Value {
    Type elemTy = vecTy.getElementType();
    unsigned count = vecTy.getNumElements();
    assert(elements.size() == static_cast<size_t>(count) &&
           "element count mismatch when packing vector");
    Value result = rewriter.create<LLVM::UndefOp>(loc, vecTy);
    for (const auto &it : llvm::enumerate(elements))
      result = insert_element(vecTy, result, it.value(), i32_val(it.index()));
    return result;
  };
  if (auto vecTy = dyn_cast<LLVM::LLVMFixedVectorType>(vectorTy))
    return pack(vecTy);
  if (auto vecTy = dyn_cast<VectorType>(vectorTy))
    return pack(vecTy);
  llvm_unreachable("expected vector accumulator type");
}

static Value createZeroVector(ConversionPatternRewriter &rewriter, Location loc,
                              Type vectorTy) {
  if (auto vecTy = dyn_cast<VectorType>(vectorTy)) {
    auto zeroAttr = rewriter.getZeroAttr(vecTy.getElementType());
    auto denseAttr = SplatElementsAttr::get(vecTy, zeroAttr);
    return rewriter.create<LLVM::ConstantOp>(loc, vecTy, denseAttr);
  }
  auto build = [&](auto vecTy) -> Value {
    Type elemTy = vecTy.getElementType();
    Value zeroElem = null(elemTy);
    SmallVector<Value> zeros(vecTy.getNumElements(), zeroElem);
    return packVectorElements(rewriter, loc, vecTy, zeros);
  };
  if (auto vecTy = dyn_cast<LLVM::LLVMFixedVectorType>(vectorTy))
    return build(vecTy);
  if (auto vecTy = dyn_cast<VectorType>(vectorTy))
    return build(vecTy);
  llvm_unreachable("expected vector type");
}

static Value faddAccumulate(ConversionPatternRewriter &rewriter, Location loc,
                            Value a, Value b) {
  auto accumulate = [&](auto vectorTy) -> Value {
    Type elemTy = vectorTy.getElementType();
    auto count = static_cast<unsigned>(vectorTy.getNumElements());
    Value result = rewriter.create<LLVM::UndefOp>(loc, vectorTy);
    for (unsigned i = 0; i < count; ++i) {
      Value lhs = extract_element(elemTy, a, i32_val(i));
      Value rhs = extract_element(elemTy, b, i32_val(i));
      Value add = rewriter.create<LLVM::FAddOp>(loc, elemTy, lhs, rhs);
      result = insert_element(vectorTy, result, add, i32_val(i));
    }
    return result;
  };
  if (auto vecTy = dyn_cast<VectorType>(a.getType()))
    return accumulate(vecTy);
  if (auto vecTy = dyn_cast<LLVM::LLVMFixedVectorType>(a.getType()))
    return accumulate(vecTy);
  llvm_unreachable("expected vector accumulator type");
}

static bool isFromTmeLoad(Value v) {
  auto allocOp = v.getDefiningOp();
  if (!isa<mlir::triton::gpu::LocalAllocOp>(allocOp)) {
    return false;
  }
  // if v is from tl.load, the allocOp has attr sqmma.opIdx.
  if (allocOp->hasAttr("sqmma.opIdx")) {
    return false;
  }
  return true;
}

static SmallVector<Value> emitWait(ConversionPatternRewriter &rewriter,
                                   Location loc, SmallVector<Value> acc,
                                   int pendings) {
  return acc;
}

LogicalResult convertDot(const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Operation *op, Value a, Value b, Value c, Value d,
                         Value useCOperand, Value loadedA, Value loadedB,
                         Value loadedC, bool allowTF32,
                         bool needsPartialAccumulator,
                         uint32_t maxNumImpreciseAcc, bool sync, Value thread) {
  auto aTensorTy = cast<TensorOrMemDesc>(a.getType());
  auto bTensorTy = cast<TensorOrMemDesc>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  auto aSharedLayout = dyn_cast<SharedEncodingAttr>(aTensorTy.getEncoding());
  auto bSharedLayout = cast<SharedEncodingAttr>(bTensorTy.getEncoding());
  auto mmaEncoding = cast<MthreadsSqmmaEncodingAttr>(dTensorTy.getEncoding());
  auto bOrd = bSharedLayout.getOrder();
  bool transA = false;
  Value baseA;
  Value baseB;
  if (aSharedLayout)
    baseA =
        getSharedMemoryObjectFromStruct(
            loc, loadedA,
            typeConverter->convertType(aTensorTy.getElementType()), rewriter)
            .base;
  baseB = getSharedMemoryObjectFromStruct(
              loc, loadedB,
              typeConverter->convertType(bTensorTy.getElementType()), rewriter)
              .base;
  if (aSharedLayout) {
    auto aOrd = aSharedLayout.getOrder();
    transA = aOrd[0] == 0;
  }
  bool transB = bOrd[0] == 1;
  auto dShapePerCTA = getShapePerCTA(dTensorTy);
  auto instrShape = mmaEncoding.getInstrShape();
  auto accSize = instrShape[0] * instrShape[1] /
                 32; // C/D matrix layout, each regiter contain 4x8 sub-matrix
  // wgmma instr will use 4 warp to mma, so multiply 4
  int M = 4 * instrShape[0];
  int N = instrShape[1];
  int K = instrShape[2];
  bool zeroAcc = isZeroConst(c);
  auto shapePerCTATile = getShapePerCTATile(mmaEncoding);
  int numRepM = ceil<unsigned>(dShapePerCTA[0], shapePerCTATile[0]);
  int numRepN = ceil<unsigned>(dShapePerCTA[1], shapePerCTATile[1]);
  int numRepK = ceil<unsigned>(aTensorTy.getShape()[1], instrShape[2]);

  if (zeroAcc) {
    if (auto bitcastOp = loadedC.getDefiningOp<LLVM::BitcastOp>()) {
      if (auto vecTy = dyn_cast<VectorType>(bitcastOp.getType())) {
        auto zeroAttr = rewriter.getZeroAttr(vecTy.getElementType());
        auto denseAttr = SplatElementsAttr::get(vecTy, zeroAttr);
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(bitcastOp);
        Value zeroVec =
            rewriter.create<LLVM::ConstantOp>(loc, vecTy, denseAttr);
        bitcastOp.replaceAllUsesWith(zeroVec);
        rewriter.eraseOp(bitcastOp);
        loadedC = zeroVec;
      }
    }
  }
  DotOpMmaV3SmemLoader aLoader;
  SmallVector<Value> structA;
  if (aSharedLayout) {
    aLoader =
        loadA(typeConverter, rewriter, loc, mmaEncoding, a, baseA, thread);
  } else {
    structA = unpackLLElements(loc, loadedA, rewriter);
  }
  DotOpMmaV3SmemLoader bLoader =
      loadB(typeConverter, rewriter, loc, mmaEncoding, b, baseB, thread);

  SmallVector<Value> fc;
  if (zeroAcc) {
    Type accElemTy = typeConverter->convertType(dTensorTy.getElementType());
    Value zero = null(accElemTy);
    size_t totalAcc = static_cast<size_t>(accSize) * numRepM * numRepN;
    fc.assign(totalAcc, zero);
  } else {
    fc = unpackLLElements(loc, loadedC, rewriter);
  }

  triton::mtgpu::SQMMAEltType eltTypeC = getMmaRetType(d);
  triton::mtgpu::SQMMAEltType eltTypeA = getMmaOperandType(a, allowTF32);
  triton::mtgpu::SQMMAEltType eltTypeB = getMmaOperandType(b, allowTF32);

  triton::mtgpu::SQMMALayout layoutA = transA ? triton::mtgpu::SQMMALayout::col
                                              : triton::mtgpu::SQMMALayout::row;
  triton::mtgpu::SQMMALayout layoutB = transB ? triton::mtgpu::SQMMALayout::row
                                              : triton::mtgpu::SQMMALayout::col;

  auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
  Operation *startSequence = nullptr;
  if (!isFromTmeLoad(a) || !isFromTmeLoad(b)) {
    startSequence = rewriter.create<triton::mtgpu::SQMMAFenceOp>(loc);
  }

  SmallVector<Value> mmaResults;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      llvm::SmallVector<Value> mmaOut =
          loadReg(rewriter, loc, fc, (m * numRepN + n) * accSize, accSize,
                  startSequence);
      Type accElemTy =
          mmaOut.empty()
              ? typeConverter->convertType(dTensorTy.getElementType())
              : mmaOut.front().getType();
      auto accTy = getVectorType(accElemTy, accSize);
      Value d;
      Value useC = i1_val(0);
      if (!zeroAcc) {
        d = packVectorElements(rewriter, loc, accTy, mmaOut);
        useC = i1_val(1);
      }
      if (useCOperand)
        useC = and_(useC, useCOperand);
      uint32_t numLowPrecisionAcc = 0;
      Value partialAcc;
      for (int k = 0; k < numRepK; ++k) {
        Value a;
        if (aSharedLayout) {
          a = aLoader.smemLoad(m, k, rewriter, loc);
        } else {
          unsigned regASize = (instrShape[0] * instrShape[2]) / 32;
          llvm::SmallVector<Value> regA =
              loadReg(rewriter, loc, structA, (m * numRepK + k) * regASize,
                      regASize, startSequence);
          auto regATy = LLVM::LLVMStructType::getLiteral(
              rewriter.getContext(),
              SmallVector<Type>(regA.size(), regA[0].getType()));
          a = packLLElements(loc, typeConverter, regA, rewriter, regATy);
        }
        auto b = bLoader.smemLoad(n, k, rewriter, loc);
        numLowPrecisionAcc += K;
        // If using native accumulation would cause use to do more low precion
        // accumulation than allowed do a separate allocation.
        bool requireAddAccumulator =
            needsPartialAccumulator &&
            (numLowPrecisionAcc >= maxNumImpreciseAcc || k == numRepK - 1);
        Value mmaAcc = needsPartialAccumulator ? partialAcc : d;
        auto elemTy = dTensorTy.getElementType();
        Type llvmElemTy = typeConverter->convertType(elemTy);
        auto ivecTy =
            getVectorType(IntegerType::get(rewriter.getContext(), 32), accSize);
        auto vecType = getVectorType(llvmElemTy, accSize);
        Value vecAcc =
            mmaAcc ? mmaAcc : createZeroVector(rewriter, loc, vecType);
        vecAcc = bitcast(vecAcc, ivecTy);
        vecAcc = rewriter.create<triton::mtgpu::SQMMAOp>(
            loc, ivecTy, a, b, useC, vecAcc, M, N, K, eltTypeC, eltTypeA,
            eltTypeB, layoutA, layoutB);
        vecAcc = bitcast(vecAcc, vecType);
        mmaAcc = vecAcc;
        useC = i1_val(1);

        if (needsPartialAccumulator)
          partialAcc = mmaAcc;
        else
          d = mmaAcc;
        // If we need accumulate separately to have higher precision, insert
        // adds.
        if (requireAddAccumulator) {
          d = d ? faddAccumulate(rewriter, loc, d, partialAcc) : partialAcc;
          numLowPrecisionAcc = 0;
          partialAcc = Value();
        }
      }
      auto acc = unpackLLElements(loc, d, rewriter);
      for (int i = 0; i < acc.size(); ++i) {
        mmaResults.push_back(acc[i]);
      }
    }
  }

  if (sync) {
    rewriter.create<triton::mtgpu::SQMMAWaitGroupOp>(loc);
  }

  SmallVector<Value> results =
      unpackAccumulator(rewriter, loc, mmaResults, dTensorTy);

  // replace with new packed result
  Type loweredDTensorTy = typeConverter->convertType(dTensorTy);
  assert((isa<VectorType>(loweredDTensorTy) ||
          isa<LLVM::LLVMFixedVectorType>(loweredDTensorTy)) &&
         "expected vector accumulator type after lowering");
  auto res = packVectorElements(rewriter, loc, loweredDTensorTy, results);
  rewriter.replaceOp(op, res);
  return success();
}
} // namespace

LogicalResult convertSQMMA(triton::mthreads_gpu::SquadDotOp op,
                           triton::mthreads_gpu::SquadDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread) {
  auto AEnc = op.getA().getType().getEncoding();
  auto BEnc = op.getB().getType().getEncoding();
  assert(mlir::isa<SharedEncodingAttr>(AEnc) ||
         mlir::isa<DotOperandEncodingAttr>(AEnc));
  assert(mlir::isa<SharedEncodingAttr>(BEnc) &&
         "Operand B should use Shared layout.");
  return convertDot(typeConverter, rewriter, op.getLoc(), op.getOperation(),  //
                    op.getA(), op.getB(), op.getC(), op.getD(), op.getUseC(), //
                    adaptor.getA(), adaptor.getB(), adaptor.getC(),           //
                    op.getInputPrecision() == InputPrecision::TF32,
                    op.needsPartialAccumulator(), op.getMaxNumImpreciseAcc(),
                    !op.getIsAsync(), thread);
}
