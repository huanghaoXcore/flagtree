#include "Utility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/MTGPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <optional>

using namespace mlir::triton::gpu;

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;

namespace {
std::string getTypeString(Type ty) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  ty.print(rso);
  rso.flush();
  return str;
}

std::string mangleFunc(std::string name, Type type) {
  auto funcType = dyn_cast<LLVM::LLVMFunctionType>(type);
  assert(funcType && "Expecting an LLVMFunctionType");
  std::string mangled = name + "_";
  auto retTy = funcType.getReturnType();
  mangled += getTypeString(retTy) + "_";
  auto params = funcType.getParams();
  for (auto paramType : params) {
    mangled += getTypeString(paramType) + "_";
  }
  return mangled;
}
} // anonymous namespace

namespace mlir {
namespace LLVM {
namespace MUSA {

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value value,
                           Value i, const MTGPU::ShflKind &mode, int widthInt) {
  auto valueTy = value.getType();
  unsigned bits = valueTy.getIntOrFloatBitWidth();

  auto int8Ty = rewriter.getI8Type();
  auto int32Ty = rewriter.getI32Type();
  auto nullPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 5);
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
                                                rewriter.getI32IntegerAttr(1));
  Value seven = rewriter.create<LLVM::ConstantOp>(
      loc, int32Ty, rewriter.getI32IntegerAttr(7));
  Value num_128 = rewriter.create<LLVM::ConstantOp>(
      loc, int32Ty, rewriter.getI32IntegerAttr(128));
  Value width = rewriter.create<LLVM::ConstantOp>(
      loc, int32Ty, rewriter.getI32IntegerAttr(widthInt));
  Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, nullPtrTy);
  Value offset = i;

  Value maskAndClamp;

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(value, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shuffleCommon(loc, rewriter, val0, i, mode, widthInt);
    val1 = shuffleCommon(loc, rewriter, val1, i, mode, widthInt);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, value.getType());
  }
  if (valueTy != i32_ty) {
    value = bitcast(value, int_ty(bits));
    if (bits < 32)
      value = zext(i32_ty, value);
  }

  // maskAndClamp is set to 0 when in 'up' mode.
  if (mode == MTGPU::ShflKind::up) {
    maskAndClamp = zero;
  } else {
    Value Clamp = rewriter.create<LLVM::SubOp>(loc, int32Ty, width, one);
    Value SegMask = rewriter.create<LLVM::SubOp>(loc, int32Ty, num_128, width);
    SegMask = rewriter.create<LLVM::ShlOp>(loc, int32Ty, SegMask, seven);
    maskAndClamp = rewriter.create<LLVM::OrOp>(loc, int32Ty, SegMask, Clamp);
  }

  // shuffle argument pred is default nullptr if not given.
  Value result = rewriter.create<MTGPU::ShflOp>(loc, int32Ty, value, offset,
                                                maskAndClamp, mode, nullPtr);

  if (valueTy != i32_ty) {
    if (bits < 32)
      result = trunc(int_ty(bits), result);
    result = bitcast(result, valueTy);
  }

  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 unsigned width) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), MTGPU::ShflKind::bfly,
                       width);
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                unsigned width) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), MTGPU::ShflKind::up,
                       width);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 unsigned width) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), MTGPU::ShflKind::idx,
                       width);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 unsigned width) {
  return shuffleCommon(loc, rewriter, val, i, MTGPU::ShflKind::idx, width);
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
  Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Value llAsyncLoad(RewriterBase &rewriter, Location loc,
                  SmallVector<Value> asyncLoadOperands) {
  Type funcType = getFunctionType(void_ty(rewriter.getContext()),
                                  ValueRange(asyncLoadOperands));
  auto parent = asyncLoadOperands[0]
                    .getParentRegion()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
  auto funcName = mangleFunc(mlir::LLVM::MUSA::Predicated_AsyncLoad, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  auto asyncLoadVal =
      rewriter.create<LLVM::CallOp>(loc, funcOp, asyncLoadOperands).getResult();
  return asyncLoadVal;
}

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal) {
  Type funcType = getFunctionType(elemTy, ValueRange({ptr, pred, falseVal}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto funcName = mangleFunc(mlir::LLVM::MUSA::Predicated_Load, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  auto loadVal =
      rewriter
          .create<LLVM::CallOp>(loc, funcOp, ValueRange({ptr, pred, falseVal}))
          .getResult();
  return loadVal;
}

Value llInplaceLoad(RewriterBase &rewriter, Location loc, Value ptr,
                    Type elemTy, Value pred, Value falseVal) {
  Type funcType = getFunctionType(elemTy, ValueRange({ptr, pred, falseVal}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto funcName =
      mangleFunc(mlir::LLVM::MUSA::Predicated_InplaceLoad, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  auto loadVal =
      rewriter
          .create<LLVM::CallOp>(loc, funcOp, ValueRange({ptr, pred, falseVal}))
          .getResult();
  return loadVal;
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred) {
  auto ctx = ptr.getContext();
  Type funcType = getFunctionType(void_ty(ctx), ValueRange({ptr, val, pred}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto funcName = mangleFunc(mlir::LLVM::MUSA::Predicated_Store, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  rewriter.create<LLVM::CallOp>(loc, funcOp, ValueRange({ptr, val, pred}));
}

LLVM::LLVMFuncOp getLibdeviceFuncCall(RewriterBase &rewriter, Operation *op,
                                      StringRef funcName, Type retType,
                                      ValueRange ins = {}) {
  Type funcType = getFunctionType(retType, ins);
  return appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
}

} // namespace MUSA
} // namespace LLVM

namespace musa_util {

namespace {

struct SqmmaInstrShape {
  unsigned m;
  unsigned n;
  unsigned k;
};

std::optional<SqmmaInstrShape>
selectSqmmaInstrShape(const ArrayRef<int64_t> &shape, RankedTensorType type,
                      int numWarps) {
  // Warp-group level SQMMA: choose M/N from the backend-supported list.
  SmallVector<std::pair<unsigned, unsigned>> allowedMN = {
      {32, 32}, {32, 64},  {32, 128}, {16, 64},  {64, 16},  {64, 32},
      {64, 64}, {64, 128}, {128, 32}, {128, 64}, {128, 128}};

  SmallVector<unsigned> validK;
  auto bitWidth = type.getElementTypeBitWidth();
  if (bitWidth == 8) {
    validK = {128, 64, 32};
  } else if (bitWidth == 16) {
    validK = {64, 32, 16};
  } else if (bitWidth == 32) {
    validK = {32, 16, 8};
  } else {
    return std::nullopt;
  }

  // Select K: largest validK that divides BLOCK_K.
  auto shapeA = type.getShape();
  unsigned blockK = shapeA[1];
  unsigned selectedK = 0;
  for (auto valid_k : validK) {
    if (blockK >= valid_k && (blockK % valid_k == 0)) {
      selectedK = valid_k;
      break;
    }
  }
  if (selectedK == 0)
    return std::nullopt;

  // Warp-groups = numWarps / 4.
  if (numWarps <= 0 || numWarps % 4 != 0)
    return std::nullopt;
  int warpGroups = numWarps / 4;

  int64_t blockM = shape[0];
  int64_t blockN = shape[1];
  // Select M/N tile that evenly divides the block and distributes to warp
  // groups. Prefer the largest M, then largest N.
  bool found = false;
  unsigned bestM = 0, bestN = 0;
  for (auto mn : allowedMN) {
    unsigned mCand = mn.first;
    unsigned nCand = mn.second;
    if ((blockM % mCand) != 0 || (blockN % nCand) != 0)
      continue;
    int64_t tilesM = blockM / mCand;
    int64_t tilesN = blockN / nCand;
    int64_t tiles = tilesM * tilesN;
    if (tiles < warpGroups || (tiles % warpGroups) != 0)
      continue;
    if (!found || mCand > bestM || (mCand == bestM && nCand > bestN)) {
      found = true;
      bestM = mCand;
      bestN = nCand;
    }
  }

  if (!found)
    return std::nullopt;

  return SqmmaInstrShape{bestM, bestN, selectedK};
}

} // namespace

bool supportMMA(Value value, int version) {
  if (version != 3)
    return false;
  auto elemTy = cast<TensorOrMemDesc>(value.getType()).getElementType();
  // SQMMA supports wide set of formats (fp4/fp6/fp8/fp16/bf16/fp32/tf32/int8).
  bool isFP8 = elemTy.isFloat8E4M3FN() || elemTy.isFloat8E4M3FNUZ() ||
               elemTy.isFloat8E5M2();
  bool isFP6 = elemTy.isFloat6E2M3FN() || elemTy.isFloat6E3M2FN();
  bool isFP4 = elemTy.isFloat4E2M1FN();
  bool isTF32 = elemTy.isTF32();
  bool isInt8 = elemTy.isInteger(8);
  bool isInt4 = elemTy.isInteger(4);
  return isFP8 || isFP6 || isFP4 || elemTy.isF16() || elemTy.isBF16() ||
         elemTy.isF32() || isTF32 || isInt8 || isInt4;
}

bool supportMMA(triton::DotOp op, int version) {
  if (version != 3)
    return false;
  if (::triton::tools::getBoolEnv("DISABLE_SQMMA"))
    return false;
  auto mod = op->getParentOfType<ModuleOp>();
  int cc = getMthreadsComputeCapability(mod);
  if (cc < 31)
    return false;

  auto aElemTy = op.getA().getType().getElementType();
  auto bElemTy = op.getB().getType().getElementType();
  auto retType = op.getType();
  auto retShapePerCTA = getShapePerCTA(retType);
  auto rank = retShapePerCTA.size();
  int numWarps = TritonGPUDialect::getNumWarps(mod);
  if (rank != 2)
    return false;
  if (!(numWarps % 4 == 0 && retShapePerCTA[rank - 2] % 16 == 0 &&
        retShapePerCTA[rank - 1] % 16 == 0 &&
        musa_util::supportMMA(op.getA(), version) &&
        musa_util::supportMMA(op.getB(), version))) {
    return false;
  }
  if (!selectSqmmaInstrShape(retShapePerCTA,
                             cast<RankedTensorType>(op.getA().getType()),
                             numWarps)) {
    return false;
  }
  // TF32 requires explicit input precision.
  if (aElemTy.isF32() && bElemTy.isF32()) {
    if (op.getInputPrecision() != InputPrecision::TF32)
      return false;
    // SQMMA TF32 path requires >= 256B leading width for both operands.
    auto aTy = cast<RankedTensorType>(op.getA().getType());
    auto bTy = cast<RankedTensorType>(op.getB().getType());
    auto aOrder = getOrder(aTy.getEncoding());
    auto bOrder = getOrder(bTy.getEncoding());
    int64_t elemBytes = aTy.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t aLeading = aTy.getShape()[aOrder[0]];
    int64_t bLeading = bTy.getShape()[bOrder[0]];
    if (aLeading * elemBytes < 256 || bLeading * elemBytes < 256)
      return false;
    return true;
  }
  return true;
}

static bool isMmaToMmaShortcut(Attribute srcEncoding, Attribute dstEncoding) {
  auto src = dyn_cast<MthreadsSqmmaEncodingAttr>(srcEncoding);
  auto dst = dyn_cast<MthreadsSqmmaEncodingAttr>(dstEncoding);
  if (!src || !dst)
    return false;
  // when #mma = MmaEncoding<version=3, warpsPerCTA=[..., 1]>
  return src && dst && src.getVersionMajor() == 3 &&
         src.getWarpsPerCTA()[1] == 1 && dst.getVersionMajor() == 3 &&
         dst.getWarpsPerCTA()[1] == 1;
}

bool isMmaToMmaShortcut(RankedTensorType srcTy, RankedTensorType dstTy) {
  return isMmaToMmaShortcut(srcTy.getEncoding(), dstTy.getEncoding());
}

// Cases where distributed shared memory is not required in ConvertLayout:
// (1) numCTAs == 1
// (2) numCTAs > 1 but srcCTALayout == dstCTALayout
// TODO: Case with SliceLayout as srcLayout and numCTAs > 1 is to be implemented
// in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout) {
  unsigned numCTAs = getNumCTAs(srcLayout);
  assert(numCTAs == getNumCTAs(dstLayout) &&
         "Invalid layout conversion: the numbers of CTAs of src and dst "
         "layouts are different");

  // Case (1): Never use dsmem when numCTAs == 1
  if (numCTAs == 1)
    return false;

  // Case where CTAsPerCGA of srcLayout in the sliced dim is not 1 is not
  // implemented yet
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(srcLayout)) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      llvm::report_fatal_error("Layout conversion to be implemented");
  }

  // Case where CTAsPerCGA of dstLayout in the sliced dim is not 1 is supported
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(dstLayout)) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      return true;
  }

  // The above two branches make sure that it is legal to call getCTALayout of
  // srcLayout and dstLayout

  // Case (2): Do not use dsmem when srcCTALayout == dstCTALayout
  auto srcCTALayout = getCTALayout(srcLayout);
  auto dstCTALayout = getCTALayout(dstLayout);
  if (srcCTALayout == dstCTALayout)
    return false;

  // Dsmem access is required when srcCTALayout != dstCTALayout
  return true;
}

// For MMAV3 dotOperand layout matches mma operand for f16 and bf16 cases.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy) {
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  auto mmaLayout = cast<MthreadsSqmmaEncodingAttr>(srcLayout);
  auto dotOperandLayout = cast<DotOperandEncodingAttr>(dstLayout);
  int elementTypeSize = srcTy.getElementType().getIntOrFloatBitWidth();
  auto ans = mmaLayout.getVersionMajor() == 3 &&
             dotOperandLayout.getOpIdx() == 0 &&
             isMmaToMmaShortcut(dotOperandLayout.getParent(), srcLayout) &&
             (elementTypeSize == 16 || elementTypeSize == 8);
  return ans;
}

SmallVector<unsigned, 3> mmaVersionToInstrShape(int version,
                                                const ArrayRef<int64_t> &shape,
                                                RankedTensorType type,
                                                int numWarps) {
  if (version == 3) {
    auto instrShape = selectSqmmaInstrShape(shape, type, numWarps);
    if (!instrShape) {
      llvm::report_fatal_error(
          "No SQMMA instruction shape matches block_m/block_n/num_warps");
    }
    // instrShape stores M/4, N, K.
    return {instrShape->m / 4, instrShape->n, instrShape->k};
  } else {
    assert(false && "version not supported");
    return {0, 0};
  }
}

} // namespace musa_util
} // namespace mlir
