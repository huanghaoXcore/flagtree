#ifndef TRITONMTHREADSGPU_TYPES
#define TRITONMTHREADSGPU_TYPES

include "Dialect/TritonMthreadsGPU/IR/TritonMthreadsGPUDialect.td"
include "mlir/IR/AttrTypeBase.td"

class TTNG_TypeDef<string name, string _mnemonic>
    : TypeDef<TritonMthreadsGPU_Dialect, name> {
  let mnemonic = _mnemonic;
}

def TTNG_TokenType : TTNG_TypeDef<"Token", "token">;

def TTNG_MutexType : TTNG_TypeDef<"Mutex", "mutex">;

#endif
