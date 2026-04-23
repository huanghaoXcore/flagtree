add_triton_library(TritonMthreadsGPUIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonMthreadsGPUTableGen
  TritonMthreadsGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  )
