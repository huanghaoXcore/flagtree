#ifndef TRITONMTHREADSGPU_OPS
#define TRITONMTHREADSGPU_OPS

include "Dialect/TritonMthreadsGPU/IR/TritonMthreadsGPUDialect.td"
include "Dialect/TritonMthreadsGPU/IR/TritonMthreadsGPUTypes.td"
include "Dialect/TritonMthreadsGPU/IR/TritonMthreadsGPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonTypeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTMG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonMthreadsGPU_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def TTMG_FenceAsyncSharedOp : TTMG_Op<"fence_async_shared"> {
  let arguments = (ins BoolAttr:$bCluster);

  let summary = "fence proxy async";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 90;
    }
  }];
}

//
// SquadDot Op
//
def TTMG_SquadDotOp : TTMG_Op<"squad_dot", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                            DotLike,
                                            TypesMatchWith<"result's type matches accumulator's type",
                                                            "d", "c", "$_self">]> {
    let summary = "squad dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c. For docs on InputPrecisionAttr, see TT_DotOp
    }];

    let arguments = (ins TT_TensorOrMemDesc:$a,
                         TT_TensorOrMemDesc:$b,
                         TT_FpIntTensor:$c,
                         Optional<I1>:$useC,
                         DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
                         DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc,
                         DefaultValuedAttr<BoolAttr, "false">:$isAsync
                         );

    let results = (outs TT_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c (`,` $useC^)? attr-dict `:` type($a) `*` type($b) `->` type($d)";

    let extraClassDeclaration = [{
      bool needsPartialAccumulator();
    }];
}

def TTMG_SquadDotWaitOp : TTMG_Op<"squad_dot_wait", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                          AllTypesMatch<["inputs", "outputs"]>]> {
  let summary = "squad dot wait";
  let arguments = (ins Variadic<TT_TensorOrMemDesc>:$inputs, I32Attr:$pendings);
  let results = (outs Variadic<TT_TensorOrMemDesc>:$outputs);
  let description = [{
    Waits until there are $pendings or fewer outstanding async dot operations.

    $inputs must be the tensors corresponding to the async dot ops that we're
    waiting on.  For example, if there are N pending async dot ops and we call
    `squad_dot_wait 1`, then $inputs must be the result of the first dot op.
  }];

  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

def TTMG_InitBarrierOp : TTMG_Op<"init_arrival", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Initialize a barrier in the given shared memory allocation.";

    let description = [{
        Initializes a bar id create, and init_arrival the number of arrival warps,
        with the initialized phase id.

        This lowers to libdevice func calls:
         1. record_bar(barId)
         2. init_arrival(barId, arrive_warps, init_phase_id)
    }];

    let hasVerifier = 1;
    let arguments = (ins I32:$barId,
                         I32:$warpCnt,
                         I32:$phase,
                         I1:$pred);
    let assemblyFormat = "$barId `,` $warpCnt `,` $phase attr-dict `,` $pred `:` type($barId)";
}

def TTMG_BarrierExpectOp : TTMG_Op<"barrier_arrive", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Signal a barrier of warps arrived according to bar id.";

  let description = [{
    This signal the barrier that the expected warps are arrived. return the phase id cooesponding to bar id.
  }];

  let hasVerifier = 1;
  let arguments = (
    ins I32:$barId
  );

  let results = (outs I32:$phaseId);

  let assemblyFormat = [{
    $barId attr-dict `:` type($barId) `->` type($phaseId)
  }];
}

def TTMG_WaitBarrierOp : TTMG_Op<"wait_barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "wait until the mbarrier phase completes.";

    let description = [{
      Blocks the program progress until the task of bar id completes
      its current phase.
    }];

    let hasVerifier = 1;
    let arguments = (ins I32:$barId,
                         I32:$phaseId);
    let assemblyFormat = "$barId `,` $phaseId attr-dict `:` type($barId)";
}

def TTMG_AsyncTMECopyGlobalToLocalOp : TTMG_Op<"async_tme_copy_global_to_local", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy data based on descriptor from global memory to local memory asynchronously";

  let description = [{
    This operation copies data from global memory to local memory
    asynchronously.  This is analogue to tt.load except the data are copied to
    local memory pointed by the memory descriptor instread of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc_ptr`.
  }];

  let hasVerifier = 1;
  let arguments = (
    ins TT_PtrType:$desc_ptr,
    Variadic<I32>:$coord,
    I32:$barId,
    TT_MemDescType:$result,
    I1:$pred
  );

  let assemblyFormat = [{
    $desc_ptr `[` $coord `]` $barId `,` $result `,` $pred
    attr-dict `:` type($desc_ptr) `,` type($barId) `->` type($result)
  }];
}

def TTMG_AsyncTMECopyLocalToGlobalOp : TTMG_Op<"async_tme_copy_local_to_global", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy data based on descriptor from local memory to global memory asynchronously";

  let description = [{
    This operation copies data from local memory to global memory
    asynchronously.  This is analogue to tt.store except the data are copied from
    local memory pointed by the memory descriptor instread of a distributed
    tensor. The data copied depends on the global memory descriptor pointed to
    by `desc_ptr`.
  }];

  let arguments = (
    ins TT_PtrType:$desc_ptr,
    Variadic<I32>:$coord,
    TT_MemDescType:$src);

  let assemblyFormat = [{
    $desc_ptr `[` $coord `]` $src
    attr-dict `:` type($desc_ptr) `,` type($src)
  }];
}

def TTMG_TMEStoreWait : TTMG_Op<"async_tme_wait_store"> {
  let summary = "wait until all the data stored.";
  let arguments = (ins I32Attr:$pendings);
  let description = [{
    Wait until all the data store to global memory.
  }];

  let assemblyFormat = "attr-dict";
}


#endif
