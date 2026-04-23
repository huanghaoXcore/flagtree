#include "TritonMTGPUToLLVM/Passes.h"

#define GEN_PASS_REGISTRATION
#include "TritonMTGPUToLLVM/Passes.h.inc"

namespace {
bool regAll = []() {
  registerConvertTritonMTGPUToLLVM();
  registerTritonMTGPUAccelerateSQMMA();
  registerTritonMTGPUMarkInplaceLoads();
  registerConvertMTGPUBuiltinFuncToLLVM();
  registerConvertMTGPUInplaceLoadToLLVM();
  return true;
}();
}
