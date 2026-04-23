#ifndef TRITON_CONVERSION_TRITONMTGPU_TO_LLVM_DOT_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONMTGPU_TO_LLVM_DOT_OP_TO_LLVM_H

#include "Utility.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::MthreadsWmmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

enum MMASID {
  // ARCH_MMA_M_N_K_D_A_B_C_WAVE
  QY2_MMA_M32_N32_K16_FP32_FP16_FP16_FP32_WAVE128,
  QY2_MMA_M32_N32_K16_FP32_BF16_BF16_FP32_WAVE128,

  // PH1 WMMA E4M3
  PH1_MMA_M8_N16_K16_FP32_E4M3_E4M3_FP32_WAVE32,
  PH1_MMA_M16_N8_K16_FP32_E4M3_E4M3_FP32_WAVE32,
  PH1_MMA_M16_N16_K16_FP32_E4M3_E4M3_FP32_WAVE32,
  PH1_MMA_M16_N16_K32_FP32_E4M3_E4M3_FP32_WAVE32,
  PH1_MMA_M16_N16_K64_FP32_E4M3_E4M3_FP32_WAVE32,

  // PH1 WMMA E5M2
  PH1_MMA_M8_N16_K16_FP32_E5M2_E5M2_FP32_WAVE32,
  PH1_MMA_M16_N8_K16_FP32_E5M2_E5M2_FP32_WAVE32,
  PH1_MMA_M16_N16_K16_FP32_E5M2_E5M2_FP32_WAVE32,
  PH1_MMA_M16_N16_K32_FP32_E5M2_E5M2_FP32_WAVE32,
  PH1_MMA_M16_N16_K64_FP32_E5M2_E5M2_FP32_WAVE32,

  // PH1 WMMA FP16
  PH1_MMA_M16_N8_K8_FP32_FP16_FP16_FP32_WAVE32,
  PH1_MMA_M16_N8_K16_FP32_FP16_FP16_FP32_WAVE32,
  PH1_MMA_M8_N16_K16_FP32_FP16_FP16_FP32_WAVE32,
  PH1_MMA_M16_N16_K16_FP32_FP16_FP16_FP32_WAVE32,
  PH1_MMA_M16_N16_K32_FP32_FP16_FP16_FP32_WAVE32,

  // PH1 WMMA BF16
  PH1_MMA_M16_N8_K8_FP32_BF16_BF16_FP32_WAVE32,
  PH1_MMA_M16_N8_K16_FP32_BF16_BF16_FP32_WAVE32,
  PH1_MMA_M8_N16_K16_FP32_BF16_BF16_FP32_WAVE32,
  PH1_MMA_M16_N16_K16_FP32_BF16_BF16_FP32_WAVE32,
  PH1_MMA_M16_N16_K32_FP32_BF16_BF16_FP32_WAVE32,
  INVALID = unsigned(-1),
};

struct mmaResultST {
  Type resultType;
  unsigned size;
  Type eleType;
};

class MusaDotOpConversion {
public:
  MusaDotOpConversion(triton::DotOp &op, triton::DotOp::Adaptor &adaptor,
                      ConversionPatternRewriter &rewriter,
                      const LLVMTypeConverter *typeConverter);

  LogicalResult convertIntoMMA323216();
  LogicalResult convertIntoPH1WMMA();
  static bool supportMusaMma(triton::DotOp &op);
  bool isQY2();
  bool isPH1();

private:
  const LLVMTypeConverter *typeConverter;
  Location loc;
  ConversionPatternRewriter &rewriter;
  mlir::MLIRContext *ctx;
  triton::DotOp::Adaptor &adaptor;
  triton::DotOp &op;
  unsigned mmaID{MMASID::INVALID};
  unsigned BLOCK_M{0};
  unsigned BLOCK_N{0};
  unsigned BLOCK_K{0};
  unsigned WARP_M{0};
  unsigned WARP_N{0};
  unsigned WARP_K{0};
  unsigned MMA_M{0};
  unsigned MMA_N{0};
  unsigned MMA_K{0};
  unsigned WARPS_PER_CTA_M{0};
  unsigned WARPS_PER_CTA_N{0};
  unsigned WARPS_PER_CTA_K{0};
  unsigned REP_M{0};
  unsigned REP_N{0};
  unsigned REP_K{0};

  RankedTensorType aTensorTy;
  Type aTensorEleTy;
  RankedTensorType bTensorTy;
  Type bTensorEleTy;
  RankedTensorType cTensorTy;
  Type cTensorEleTy;
  RankedTensorType dTensorTy;
  Type dTensorEleTy;

  SmallVector<Value> acc;
  SmallVector<Value> aBlock;
  SmallVector<Value> bBlock;

  MthreadsWmmaEncodingAttr mmaLayout;
  unsigned aSizePerThread{0};
  unsigned bSizePerThread{0};
  unsigned cSizePerThread{0};

  LogicalResult convertMMA(unsigned mmaID);
  SmallVector<Value> unpackMmaResults(SmallVector<Value> &elems,
                                      Type accEleType);
  SmallVector<Value> packMmaOperands(SmallVector<Value> &tensorElems,
                                     Type tensorEleType);
  mmaResultST getMmaResultsType();
  Type getMmaOperandsType(Type tensorEleType, unsigned tensorEleNum);
  Value extractMmaOperands(unsigned mmaIdx, SmallVector<Value> &blockData,
                           Type tensorEleType, unsigned elemsPerThread);
  void extractMmaResults(Value result, SmallVector<Value> &acc,
                         unsigned mmaResEleNum, unsigned accIdx);
  void callMMA(unsigned mmaIdxM, unsigned mmaIdxN, unsigned mmaIdxK);
};

#endif
