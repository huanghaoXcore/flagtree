#include "DotOpToLLVM/DotOpToLLVM.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::MthreadsWmmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

std::map<unsigned, std::string> mmas = {
    // QY2 WMMA FP16
    {MMASID::QY2_MMA_M32_N32_K16_FP32_FP16_FP16_FP32_WAVE128,
     "llvm.musa.ffmma.m32n32k16.mma"},

    // QY2 WMMA BF16
    {MMASID::QY2_MMA_M32_N32_K16_FP32_BF16_BF16_FP32_WAVE128,
     "llvm.musa.bfmma.m32n32k16.mma"},

    // PH1 WMMA E4M3
    {MMASID::PH1_MMA_M8_N16_K16_FP32_E4M3_E4M3_FP32_WAVE32,
     "llvm.musa.e4m3.m8n16k16.mma"},
    {MMASID::PH1_MMA_M16_N8_K16_FP32_E4M3_E4M3_FP32_WAVE32,
     "llvm.musa.e4m3.m16n8k16.mma"},
    {MMASID::PH1_MMA_M16_N16_K16_FP32_E4M3_E4M3_FP32_WAVE32,
     "llvm.musa.e4m3.m16n16k16.mma"},
    {MMASID::PH1_MMA_M16_N16_K32_FP32_E4M3_E4M3_FP32_WAVE32,
     "llvm.musa.e4m3.m16n16k32.mma"},
    {MMASID::PH1_MMA_M16_N16_K64_FP32_E4M3_E4M3_FP32_WAVE32,
     "llvm.musa.e4m3.m16n16k64.mma"},

    // PH1 WMMA E5M2
    {MMASID::PH1_MMA_M8_N16_K16_FP32_E5M2_E5M2_FP32_WAVE32,
     "llvm.musa.e5m2.m8n16k16.mma"},
    {MMASID::PH1_MMA_M16_N8_K16_FP32_E5M2_E5M2_FP32_WAVE32,
     "llvm.musa.e5m2.m16n8k16.mma"},
    {MMASID::PH1_MMA_M16_N16_K16_FP32_E5M2_E5M2_FP32_WAVE32,
     "llvm.musa.e5m2.m16n16k16.mma"},
    {MMASID::PH1_MMA_M16_N16_K32_FP32_E5M2_E5M2_FP32_WAVE32,
     "llvm.musa.e5m2.m16n16k32.mma"},
    {MMASID::PH1_MMA_M16_N16_K64_FP32_E5M2_E5M2_FP32_WAVE32,
     "llvm.musa.e5m2.m16n16k64.mma"},

    // PH1 WMMA FP16
    {MMASID::PH1_MMA_M16_N8_K8_FP32_FP16_FP16_FP32_WAVE32,
     "llvm.musa.ffmma.m16n8k8.mma"},
    {MMASID::PH1_MMA_M16_N8_K16_FP32_FP16_FP16_FP32_WAVE32,
     "llvm.musa.ffmma.m16n8k16.mma"},
    {MMASID::PH1_MMA_M8_N16_K16_FP32_FP16_FP16_FP32_WAVE32,
     "llvm.musa.ffmma.m8n16k16.mma"},
    {MMASID::PH1_MMA_M16_N16_K16_FP32_FP16_FP16_FP32_WAVE32,
     "llvm.musa.ffmma.m16n16k16.mma"},
    {MMASID::PH1_MMA_M16_N16_K32_FP32_FP16_FP16_FP32_WAVE32,
     "llvm.musa.ffmma.m16n16k32.mma"},

    // PH1 WMMA BF16
    {MMASID::PH1_MMA_M16_N8_K8_FP32_BF16_BF16_FP32_WAVE32,
     "llvm.musa.bfmma.m16n8k8.mma"},
    {MMASID::PH1_MMA_M16_N8_K16_FP32_BF16_BF16_FP32_WAVE32,
     "llvm.musa.bfmma.m16n8k16.mma"},
    {MMASID::PH1_MMA_M8_N16_K16_FP32_BF16_BF16_FP32_WAVE32,
     "llvm.musa.bfmma.m8n16k16.mma"},
    {MMASID::PH1_MMA_M16_N16_K16_FP32_BF16_BF16_FP32_WAVE32,
     "llvm.musa.bfmma.m16n16k16.mma"},
    {MMASID::PH1_MMA_M16_N16_K32_FP32_BF16_BF16_FP32_WAVE32,
     "llvm.musa.bfmma.m16n16k32.mma"},
};

unsigned getSatf() { return 1; }
unsigned getDWordBitWidth() { return 32; }
} // namespace

MusaDotOpConversion::MusaDotOpConversion(triton::DotOp &op,
                                         triton::DotOp::Adaptor &adaptor,
                                         ConversionPatternRewriter &rewriter,
                                         const LLVMTypeConverter *typeConverter)
    : op(op), typeConverter(typeConverter), rewriter(rewriter),
      adaptor(adaptor), loc(op.getLoc()), ctx(rewriter.getContext()) {

  Value A = this->op.getA();
  Value B = this->op.getB();
  Value C = this->op.getC();
  Value D = this->op.getResult();
  this->aTensorTy = cast<RankedTensorType>(A.getType());
  this->aTensorEleTy = this->aTensorTy.getElementType();
  this->bTensorTy = cast<RankedTensorType>(B.getType());
  this->bTensorEleTy = this->bTensorTy.getElementType();
  this->cTensorTy = cast<RankedTensorType>(C.getType());
  this->cTensorEleTy = this->cTensorTy.getElementType();
  this->dTensorTy = cast<RankedTensorType>(D.getType());
  this->dTensorEleTy = this->dTensorTy.getElementType();

  this->mmaLayout =
      dyn_cast<MthreadsWmmaEncodingAttr>(this->dTensorTy.getEncoding());
  auto warpsPerCta = this->mmaLayout.getWarpsPerCTA();
  assert(warpsPerCta.size() == 2 && "Unexpected condition.");
  auto mmaInstShape = this->mmaLayout.getInstrShape();
  assert(mmaInstShape.size() == 3 && "Unexpected mthreads mma version.");

  this->MMA_M = mmaInstShape[0];
  this->MMA_N = mmaInstShape[1];
  this->MMA_K = mmaInstShape[2];

  this->acc = unpackLLElements(this->loc, this->adaptor.getC(), this->rewriter);
  auto cTotalElemsPerThread =
      this->mmaLayout.getTotalElemsPerThread(this->cTensorTy.getShape());
  assert(acc.size() == cTotalElemsPerThread && "Unexpected condition.");
  this->aBlock =
      unpackLLElements(this->loc, this->adaptor.getA(), this->rewriter);
  auto aTotalElemsPerThread = this->mmaLayout.getTotalElemsPerThreadForOperands(
      this->aTensorTy.getShape(), 0 /*opIdx*/);
  assert(aBlock.size() == aTotalElemsPerThread && "Unexpected condition.");
  this->bBlock =
      unpackLLElements(this->loc, this->adaptor.getB(), this->rewriter);
  auto bTotalElemsPerThread = this->mmaLayout.getTotalElemsPerThreadForOperands(
      this->bTensorTy.getShape(), 1 /*opIdx*/);
  assert(bBlock.size() == bTotalElemsPerThread && "Unexpected condition.");
  this->aSizePerThread =
      product(this->mmaLayout.getSizePerThreadForOperands(0 /*opIdx*/));
  this->bSizePerThread =
      product(this->mmaLayout.getSizePerThreadForOperands(1 /*opIdx*/));
  this->cSizePerThread = product(this->mmaLayout.getSizePerThread());

  this->BLOCK_M =
      getShapePerCTA(this->mmaLayout, this->cTensorTy.getShape())[0];
  this->BLOCK_N =
      getShapePerCTA(this->mmaLayout, this->cTensorTy.getShape())[1];
  this->BLOCK_K =
      getShapePerCTA(this->mmaLayout, this->aTensorTy.getShape())[1];
  // split a block into warps
  this->WARPS_PER_CTA_M = warpsPerCta[0];
  this->WARPS_PER_CTA_N = warpsPerCta[1];
  this->WARPS_PER_CTA_K = 1; // donot split on K dimension

  this->WARP_M = this->mmaLayout.getShapePerWarp(this->cTensorTy.getShape())[0];
  assert((this->BLOCK_M / this->WARPS_PER_CTA_M) == this->WARP_M &&
         "Unexpected condition.");
  this->WARP_N = this->mmaLayout.getShapePerWarp(this->cTensorTy.getShape())[1];
  assert((this->BLOCK_N / this->WARPS_PER_CTA_N == this->WARP_N) &&
         "Unexpected condition.");
  this->WARP_K = this->mmaLayout.getShapePerWarpForOperands(
      0 /*opIdx*/, this->aTensorTy.getShape())[1];
  assert((this->BLOCK_K / this->WARPS_PER_CTA_K == this->WARP_K) &&
         "Unexpected condition.");

  this->REP_M =
      this->mmaLayout.getNumReplications(this->cTensorTy.getShape())[0];
  this->REP_N =
      this->mmaLayout.getNumReplications(this->cTensorTy.getShape())[1];
  this->REP_K = this->mmaLayout.getNumReplicationsForDotOperands(
      0 /*opIdx*/, this->aTensorTy.getShape())[1];
  assert(mlir::ceil<unsigned>(this->WARP_K, this->MMA_K) == this->REP_K);
}

bool MusaDotOpConversion::supportMusaMma(triton::DotOp &op) {

  // D = A * B + C
  auto AType = op.getA().getType();
  auto DType = op.getResult().getType();
  MthreadsWmmaEncodingAttr mmaLayout = dyn_cast<MthreadsWmmaEncodingAttr>(
      cast<RankedTensorType>(DType).getEncoding());
  if (!mmaLayout) {
    return false;
  }
  auto dTensorType = cast<RankedTensorType>(DType);
  // for 1-d tensor, use fma but not mma.
  if (dTensorType.getShape().size() <= 1) {
    return false;
  }
  return true;
}

bool MusaDotOpConversion::isQY2() {

  if (mmaLayout) {
    return mmaLayout.isQY2();
  }
  return false;
}

bool MusaDotOpConversion::isPH1() {

  if (mmaLayout) {
    return mmaLayout.isPH1();
  }
  return false;
}

Type MusaDotOpConversion::getMmaOperandsType(Type tensorEleType,
                                             unsigned tensorEleNum) {

  return vec_ty(tensorEleType, tensorEleNum);
}

mmaResultST MusaDotOpConversion::getMmaResultsType() {
  if (isQY2()) {
    return {vec_ty(i32_ty, 8), 8, i32_ty};
  } else if (isPH1()) {
    auto mmaInstShape = this->mmaLayout.getInstrShape();
    unsigned INST_M = mmaInstShape[0];
    unsigned INST_N = mmaInstShape[1];
    unsigned numElems = INST_M * INST_N;
    return {vec_ty(i32_ty, numElems / 32), numElems / 32, i32_ty};
  }
  llvm::report_fatal_error("Unsupported arch.");
  return {};
}

SmallVector<Value>
MusaDotOpConversion::packMmaOperands(SmallVector<Value> &tensorElems,
                                     Type tensorEleType) {

  // TODO: handle more types
  unsigned dWordBitWidth = getDWordBitWidth();
  unsigned elemsPerThread = tensorElems.size();
  unsigned dWordsPerThread = elemsPerThread;

  bool packFP8IntoInt32 = tensorEleType.isFloat8E4M3() ||
                          tensorEleType.isFloat8E4M3FN() ||
                          tensorEleType.isFloat8E5M2();
  bool packFP16IntoInt32 = tensorEleType.isF16() || tensorEleType.isBF16();
  auto packFP32IntoInt32 = tensorEleType.isF32() || tensorEleType.isTF32();

  if (packFP8IntoInt32) {
    // fp8 + fp8 + fp8 + fp8 -> int32
    auto llElemTy = typeConverter->convertType(tensorEleType);
    unsigned elemBitWidth = llElemTy.getIntOrFloatBitWidth();
    if (elemBitWidth != 8) {
      llvm::report_fatal_error("Unexpected float8 bitwidth (expected 8).");
    }
    if (dWordBitWidth != 32) {
      llvm::report_fatal_error(
          "Unexpected dword bitwidth for float8 packing (expected 32).");
    }
    if (elemsPerThread % 4 != 0) {
      llvm::report_fatal_error("Unexpected condition: float8 elemsPerThread is "
                               "not a multiple of 4.");
    }
    dWordsPerThread = elemsPerThread / 4;
    SmallVector<Value> packedElems(dWordsPerThread);
    for (unsigned i = 0, j = 0; i < dWordsPerThread; ++i) {
      Type packedType = vec_ty(llElemTy, 4);
      Value packedValue = undef(packedType);
      packedValue =
          insert_element(packedType, packedValue, tensorElems[j++], i32_val(0));
      packedValue =
          insert_element(packedType, packedValue, tensorElems[j++], i32_val(1));
      packedValue =
          insert_element(packedType, packedValue, tensorElems[j++], i32_val(2));
      packedValue =
          insert_element(packedType, packedValue, tensorElems[j++], i32_val(3));
      packedElems[i] = bitcast(packedValue, i32_ty);
    }
    return packedElems;
  } else if (packFP16IntoInt32) {
    // fp16 + fp16 -> int32
    dWordsPerThread =
        elemsPerThread * tensorEleType.getIntOrFloatBitWidth() / dWordBitWidth;
    if (elemsPerThread != dWordsPerThread * dWordBitWidth /
                              tensorEleType.getIntOrFloatBitWidth()) {
      llvm::report_fatal_error("Unexpected condition.");
    }
    SmallVector<Value> packedElems(dWordsPerThread);
    for (unsigned i = 0, j = 0; i < dWordsPerThread; ++i) {
      Type packedType = vec_ty(
          tensorEleType, dWordBitWidth / tensorEleType.getIntOrFloatBitWidth());
      Value packedValue = undef(packedType);
      packedValue =
          insert_element(packedType, packedValue, tensorElems[j++], i32_val(0));
      packedValue =
          insert_element(packedType, packedValue, tensorElems[j++], i32_val(1));
      packedElems[i] = bitcast(packedValue, i32_ty);
      if (j >= tensorElems.size()) {
        break;
      }
    }
    return packedElems;
  } else if (packFP32IntoInt32) {
    // fp32 -> int32
    dWordsPerThread =
        elemsPerThread * tensorEleType.getIntOrFloatBitWidth() / dWordBitWidth;
    if (elemsPerThread != dWordsPerThread * dWordBitWidth /
                              tensorEleType.getIntOrFloatBitWidth()) {
      llvm::report_fatal_error("Unexpected condition.");
    }
    SmallVector<Value> packedElems(dWordsPerThread);
    for (unsigned i = 0, j = 0; i < dWordsPerThread; ++i) {
      packedElems[i] = bitcast(tensorElems[i], i32_ty);
    }
    return packedElems;
  }
  return tensorElems;
}

SmallVector<Value>
MusaDotOpConversion::unpackMmaResults(SmallVector<Value> &elems,
                                      Type accEleType) {

  // unpack elements in elems and transforms the element into accEleType
  auto resEleType = getMmaResultsType().eleType;
  bool unpackInt32IntoFP16 =
      (resEleType == i32_ty) && (accEleType.isF16() || accEleType.isBF16());
  bool unpackInt32IntoFP32 =
      (resEleType == i32_ty) && (accEleType.isF32() || accEleType.isTF32());

  if (unpackInt32IntoFP16) {
    SmallVector<Value> results;
    auto et =
        accEleType.isF16() ? rewriter.getF16Type() : rewriter.getBF16Type();
    for (Value elem : elems) {
      elem = bitcast(elem, vec_ty(et, 2));
      results.push_back(extract_element(et, elem, i32_val(0)));
      results.push_back(extract_element(et, elem, i32_val(1)));
    }
    return results;
  } else if (unpackInt32IntoFP32) {
    SmallVector<Value> results;
    auto et =
        accEleType.isF32() ? rewriter.getF32Type() : rewriter.getTF32Type();
    for (Value elem : elems) {
      results.push_back(bitcast(elem, et));
    }
    return results;
  }
  return elems;
}

Value MusaDotOpConversion::extractMmaOperands(unsigned mmaIdx,
                                              SmallVector<Value> &blockData,
                                              Type tensorEleType,
                                              unsigned sizePerThread) {

  assert(((mmaIdx + 1) * sizePerThread <= blockData.size()) &&
         "Unexpected condition.");
  // extract data to construct mma inst operands
  SmallVector<Value> tensorElems(sizePerThread);
  // collect all data for the same thread
  for (unsigned i = 0; i < sizePerThread; ++i) {
    tensorElems[i] = blockData[mmaIdx * sizePerThread + i];
  }
  auto packedOperands = packMmaOperands(tensorElems, tensorEleType);

  if (packedOperands.size() == 1) {
    return packedOperands[0];
  }

  // insert elems in a vec
  Type opsType =
      getMmaOperandsType(packedOperands[0].getType(), packedOperands.size());
  Value packedVecOps = undef(opsType);
  for (unsigned i = 0; i < packedOperands.size(); ++i) {
    packedVecOps =
        insert_element(opsType, packedVecOps, packedOperands[i], i32_val(i));
  }
  return packedVecOps;
}

void MusaDotOpConversion::extractMmaResults(Value result,
                                            SmallVector<Value> &acc,
                                            unsigned cSizePerThread,
                                            unsigned accIdx) {

  // extract mma inst results into accmulator
  unsigned mmaResSize = getMmaResultsType().size;
  Type mmaResEleType = getMmaResultsType().eleType;
  SmallVector<Value> elems(mmaResSize);
  for (unsigned i = 0; i < elems.size(); ++i) {
    elems[i] = extract_element(mmaResEleType, result, i32_val(i));
  }
  auto unpackedResults = unpackMmaResults(elems, cTensorEleTy);
  assert((cSizePerThread == unpackedResults.size()) && "Unexpected condition.");
  assert(((accIdx + 1) * cSizePerThread <= acc.size()) &&
         "Unexpected condition.");
  for (unsigned i = 0; i < unpackedResults.size(); ++i) {
    acc[accIdx * cSizePerThread + i] = unpackedResults[i];
  }
}

void MusaDotOpConversion::callMMA(unsigned mmaIdxM, unsigned mmaIdxN,
                                  unsigned mmaIdxK) {

  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();

  unsigned aOpIdx = 0;
  auto aNumRep =
      mmaLayout.getNumReplicationsForDotOperands(aOpIdx, aTensorTy.getShape());
  unsigned aMmaIdx = mmaIdxM * aNumRep[1] + mmaIdxK;
  auto AOps = extractMmaOperands(aMmaIdx, aBlock, aTensorEleTy, aSizePerThread);

  unsigned bOpIdx = 1;
  auto bNumRep =
      mmaLayout.getNumReplicationsForDotOperands(bOpIdx, bTensorTy.getShape());
  unsigned bMmaIdx = mmaIdxK * bNumRep[1] + mmaIdxN;
  auto BOps = extractMmaOperands(bMmaIdx, bBlock, bTensorEleTy, bSizePerThread);

  unsigned cMmaIdx =
      mmaIdxM * mmaLayout.getNumReplications(cTensorTy.getShape())[1] + mmaIdxN;
  auto COps = extractMmaOperands(cMmaIdx, acc, cTensorEleTy, cSizePerThread);

  // Block *curBlock = rewriter.getInsertionBlock();
  Value satf = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(i32_ty, getSatf()));

  auto mmaResTy = getMmaResultsType().resultType;

  StringRef mmaFuncName = mmas[mmaID];
  rewriter.setInsertionPoint(op);
  // A, B, C, SatFlag, DstFmtFlag, ShapeFlag
  SmallVector<Value> mmaOps = {AOps, BOps, COps, satf, i32_val(1), i32_val(1)};
  if (aTensorEleTy.isFloat8E4M3() || aTensorEleTy.isFloat8E4M3FN() ||
      aTensorEleTy.isFloat8E5M2()) {
    // ScaleA, ScaleB, ScaleInfos
    for (size_t i = 0; i < 3; ++i) {
      mmaOps.push_back(i32_val(0));
    }
  }
  SmallVector<Type, 1> resultTypes{mmaResTy};
  auto intrinsic = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, mmas[mmaID],
                                                   resultTypes, mmaOps);
  Value result = intrinsic->getResult(0);
  // update results into accmulator
  assert((cSizePerThread == (MMA_M * MMA_N / mmaLayout.getWaveNum())) &&
         "Unexpected condition.");
  extractMmaResults(result, acc, cSizePerThread, cMmaIdx);
}

// split a block into mmas
LogicalResult MusaDotOpConversion::convertMMA(unsigned mmaID) {
  this->mmaID = mmaID;

  for (unsigned mmaIdxK = 0; mmaIdxK < REP_K; ++mmaIdxK) {
    for (unsigned mmaIdxM = 0; mmaIdxM < REP_M; ++mmaIdxM) {
      for (unsigned mmaIdxN = 0; mmaIdxN < REP_N; ++mmaIdxN) {
        callMMA(mmaIdxM, mmaIdxN, mmaIdxK);
      }
    }
  }

  // ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  Value res = packLLElements(loc, typeConverter, acc, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult MusaDotOpConversion::convertIntoMMA323216() {
  // used on QY2
  if (aTensorEleTy.isBF16()) {
    assert(bTensorEleTy.isBF16() && "Unexpected condition.");
    return convertMMA(MMASID::QY2_MMA_M32_N32_K16_FP32_BF16_BF16_FP32_WAVE128);
  } else if (aTensorEleTy.isF16()) {
    assert(bTensorEleTy.isF16() && "Unexpected condition.");
    return convertMMA(MMASID::QY2_MMA_M32_N32_K16_FP32_FP16_FP16_FP32_WAVE128);
  }
  assert(false && "Unhandled tensor element type.");
  return convertMMA(MMASID::QY2_MMA_M32_N32_K16_FP32_BF16_BF16_FP32_WAVE128);
}

LogicalResult MusaDotOpConversion::convertIntoPH1WMMA() {
  // used on PH1
  auto mmaInstShape = this->mmaLayout.getInstrShape();
  unsigned INST_M = mmaInstShape[0];
  unsigned INST_N = mmaInstShape[1];
  unsigned INST_K = mmaInstShape[2];

  unsigned mmasid = 0;
  if (aTensorEleTy.isBF16()) {
    assert(bTensorEleTy.isBF16() && "Unexpected condition.");
    if (INST_M == 16 && INST_N == 8 && INST_K == 8) {
      mmasid = MMASID::PH1_MMA_M16_N8_K8_FP32_BF16_BF16_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 8 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M16_N8_K16_FP32_BF16_BF16_FP32_WAVE32;
    } else if (INST_M == 8 && INST_N == 16 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M8_N16_K16_FP32_BF16_BF16_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M16_N16_K16_FP32_BF16_BF16_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 32) {
      mmasid = MMASID::PH1_MMA_M16_N16_K32_FP32_BF16_BF16_FP32_WAVE32;
    } else {
      assert(false && "Unhandled tensor element type and shape.");
    }
  } else if (aTensorEleTy.isF16()) {
    assert(bTensorEleTy.isF16() && "Unexpected condition.");
    if (INST_M == 16 && INST_N == 8 && INST_K == 8) {
      mmasid = MMASID::PH1_MMA_M16_N8_K8_FP32_FP16_FP16_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 8 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M16_N8_K16_FP32_FP16_FP16_FP32_WAVE32;
    } else if (INST_M == 8 && INST_N == 16 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M8_N16_K16_FP32_FP16_FP16_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M16_N16_K16_FP32_FP16_FP16_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 32) {
      mmasid = MMASID::PH1_MMA_M16_N16_K32_FP32_FP16_FP16_FP32_WAVE32;
    } else {
      assert(false && "Unhandled tensor element type and shape.");
    }
  } else if (aTensorEleTy.isFloat8E4M3FN() || aTensorEleTy.isFloat8E4M3()) {
    assert(((aTensorEleTy.isFloat8E4M3FN() && bTensorEleTy.isFloat8E4M3FN()) ||
            (aTensorEleTy.isFloat8E4M3() && bTensorEleTy.isFloat8E4M3())) &&
           "Unexpected condition.");
    if (INST_M == 16 && INST_N == 8 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M16_N8_K16_FP32_E4M3_E4M3_FP32_WAVE32;
    } else if (INST_M == 8 && INST_N == 16 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M8_N16_K16_FP32_E4M3_E4M3_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M16_N16_K16_FP32_E4M3_E4M3_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 32) {
      mmasid = MMASID::PH1_MMA_M16_N16_K32_FP32_E4M3_E4M3_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 64) {
      mmasid = MMASID::PH1_MMA_M16_N16_K64_FP32_E4M3_E4M3_FP32_WAVE32;
    } else {
      assert(false && "Unhandled tensor element type and shape.");
    }
  } else if (aTensorEleTy.isFloat8E5M2()) {
    assert(bTensorEleTy.isFloat8E5M2() && "Unexpected condition.");
    if (INST_M == 16 && INST_N == 8 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M16_N8_K16_FP32_E5M2_E5M2_FP32_WAVE32;
    } else if (INST_M == 8 && INST_N == 16 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M8_N16_K16_FP32_E5M2_E5M2_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 16) {
      mmasid = MMASID::PH1_MMA_M16_N16_K16_FP32_E5M2_E5M2_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 32) {
      mmasid = MMASID::PH1_MMA_M16_N16_K32_FP32_E5M2_E5M2_FP32_WAVE32;
    } else if (INST_M == 16 && INST_N == 16 && INST_K == 64) {
      mmasid = MMASID::PH1_MMA_M16_N16_K64_FP32_E5M2_E5M2_FP32_WAVE32;
    } else {
      assert(false && "Unhandled tensor element type and shape.");
    }
  } else {
    assert(false && "Unhandled tensor element type and shape.");
  }
  return convertMMA(mmasid);
}
