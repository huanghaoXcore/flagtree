add_subdirectory(include)
add_subdirectory(lib)

if(EDITABLE_MODE)
  set(MTHREADS_PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  if(NOT DEFINED Python3_SITELIB)
    find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
  endif()
  set (MTHREADS_PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonMTHREADS
      SHARED_LIB mthreadsTritonPlugin
    )
  else()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin/include/CMakeLists.txt")
      # Generate plugin TableGen headers for compile-time consumers without
      # pulling plugin implementation sources into the main build.
      include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin/include)
      include_directories(${CMAKE_CURRENT_BINARY_DIR}/plugin/include)
      include_directories(${CMAKE_SOURCE_DIR}/third_party/mthreads/include)
      include_directories(${CMAKE_BINARY_DIR}/third_party/mthreads/include)
      add_subdirectory(plugin/include plugin/include)
    else()
      message(WARNING
        "plugin/include is missing; plugin TableGen headers will not be "
        "generated in the prebuilt-plugin flow.")
    endif()
    if(EDITABLE_MODE)
      find_library(mthreadsTritonPluginLib
      NAMES
        mthreadsTritonPlugin.so
      PATHS
      ${CMAKE_CURRENT_SOURCE_DIR}
      REQUIRED
      )
      add_triton_plugin(TritonMTHREADS
        SHARED_LIB ${mthreadsTritonPluginLib}
      )
    else()
      find_library(mthreadsTritonPluginLib
        NAMES
          mthreadsTritonPlugin.so
        PATHS
        ${MTHREADS_PLUGIN_DIR}
        REQUIRED
      )
      add_triton_plugin(TritonMTHREADS
        SHARED_LIB ${mthreadsTritonPluginLib}
      )
    endif()
  endif()
endif()

add_subdirectory(bin)
