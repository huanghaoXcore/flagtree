
#include "Dialect/TritonMthreadsGPU/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#define GET_OP_CLASSES
#include "Dialect/TritonMthreadsGPU/IR/Ops.cpp.inc"

namespace mlir {
namespace triton {
namespace mthreads_gpu {

// -- SquadDotOp --
mlir::LogicalResult SquadDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc = cast<TensorOrMemDesc>(operands[0].getType()).getEncoding();
  auto bEnc = cast<TensorOrMemDesc>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = dyn_cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return mlir::failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return mlir::failure();
  }
  return mlir::success();
}

void SquadDotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &a = getAMutable();
  auto &b = getBMutable();
  if (isa<MemDescType>(a.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &a,
                         mlir::triton::gpu::SharedMemory::get());
  if (isa<MemDescType>(b.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &b,
                         mlir::triton::gpu::SharedMemory::get());
}

bool SquadDotOp::needsPartialAccumulator() {
  const auto &a = getA();
  const auto &d = getD();
  auto aTensorTy = cast<TensorOrMemDesc>(a.getType());
  auto aElTy = cast<TensorOrMemDesc>(a.getType()).getElementType();
  bool isFP8 = aElTy.isFloat8E5M2() || aElTy.isFloat8E4M3FN() ||
               aElTy.isFloat8E5M2FNUZ() || aElTy.isFloat8E4M3FNUZ();
  bool accFP32 = cast<TensorOrMemDesc>(d.getType()).getElementType().isF32();
  uint32_t maxNumImpreciseAcc = getMaxNumImpreciseAcc();
  return isFP8 && accFP32 && maxNumImpreciseAcc <= aTensorTy.getShape()[1];
}

// -- SquadDotWaitOp --
LogicalResult SquadDotWaitOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  for (Value operand : operands)
    inferredReturnTypes.push_back(operand.getType());
  return mlir::success();
}

static LogicalResult verifyBarrierType(Operation *op, MemDescType barrierType) {
  if (!barrierType.getElementType().isInteger(64) ||
      barrierType.getShape() != ArrayRef<int64_t>({1}))
    return op->emitOpError(
        "barrier allocation must be a descriptor of 1xi64 type");
  return success();
}

static LogicalResult verifyIntegerType(Operation *op, Type type,
                                       int sizeInBit) {
  if (!type.isInteger(sizeInBit))
    return failure();
  return success();
}

// -- InitBarrierOp --
LogicalResult InitBarrierOp::verify() {
  if (failed(verifyIntegerType(*this, getBarId().getType(), 32)))
    return failure();
  return success();
}

void InitBarrierOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- BarrierExpectOp --
LogicalResult BarrierExpectOp::verify() {
  if (failed(verifyIntegerType(*this, getPhaseId().getType(), 32)))
    return failure();
  return success();
}

void BarrierExpectOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- WaitBarrierOp --
LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyIntegerType(*this, getBarId().getType(), 32)))
    return failure();
  return success();
}

void WaitBarrierOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getBarIdMutable(),
                       mlir::triton::gpu::SharedMemory::get());
  // Need a side effect to prevent compiler from reordering and removing
  // the wait operation.
  effects.emplace_back(MemoryEffects::Write::get(),
                       mlir::SideEffects::DefaultResource::get());
}

// -- AsyncTMECopyGlobalToLocalOp --
LogicalResult AsyncTMECopyGlobalToLocalOp::verify() {
  if (failed(verifyIntegerType(*this, getBarId().getType(), 32)))
    return failure();
  if (getCoord().size() < 1 || getCoord().size() > 5)
    return emitOpError("TMA copies must have between 1 and 5 coordinates");
  return success();
}

void AsyncTMECopyGlobalToLocalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getDescPtrMutable(),
                       mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getResultMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// -- AsyncTMECopyLocalToGlobalOp --
void AsyncTMECopyLocalToGlobalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDescPtrMutable(),
                       mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

} // namespace mthreads_gpu
} // namespace triton
} // namespace mlir
