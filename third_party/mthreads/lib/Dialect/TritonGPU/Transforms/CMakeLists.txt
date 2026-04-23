add_triton_library(TritonGPUTransforms
  Coalesce.cpp
  CombineTensorSelectAndIf.cpp
  ReduceDataDuplication.cpp
  OptimizeAccumulatorInit.cpp
  OptimizeDotOperands.cpp
  OptimizeThreadLocality.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  Utility.cpp

  DEPENDS
  TritonGPUTransformsIncGen
  TritonMthreadsGPUTableGen
  TritonMthreadsGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)

target_include_directories(TritonGPUTransforms
    PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/mthreads/plugin/include
        ${CMAKE_BINARY_DIR}/third_party/mthreads/plugin/include
)
