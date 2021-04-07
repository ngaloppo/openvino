# Copyright (c) 2021 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =================================== Name / Output settings from parent ===============================

set(CLDNN_BUILD__PROJ             "cldnn_runtime")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}${CLDNN__OUT_CPU_SUFFIX}")

# ========================================== Compiler options ==========================================

intel_config_flag_apply_settings(
    CompilerOptions
    CMAKE_CXX_FLAGS
    ALL_PATTERN ""
    SET
      RttiEnabled
  )

if((CMAKE_C_COMPILER_ID MATCHES "^Clang$") OR (CMAKE_CXX_COMPILER_ID MATCHES "^Clang$"))
  intel_config_flag_apply_settings(
      CompilerOptions
      CMAKE_CXX_FLAGS
      ALL_PATTERN ""
      SET_RAW
        -stdlib=libstdc++
        StandardCxx14
    )
endif()

# ========================================= Source/Header files ========================================

set(__CLDNN_Directory__main           "${CMAKE_CURRENT_SOURCE_DIR}")
set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${__CLDNN_Directory__main}/*.h"
    "${__CLDNN_Directory__main}/*.hpp"
    "${__CLDNN_Directory__main}/*.cpp"
  )

file(GLOB __CLDNN_Sources__ocl
    "${__CLDNN_Directory__main}/ocl/*.h"
    "${__CLDNN_Directory__main}/ocl/*.hpp"
    "${__CLDNN_Directory__main}/ocl/*.cpp"
)

file(GLOB __CLDNN_Sources__sycl
    "${__CLDNN_Directory__main}/sycl/*.h"
    "${__CLDNN_Directory__main}/sycl/*.hpp"
    "${__CLDNN_Directory__main}/sycl/*.cpp"
)

set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__ocl}
  )

if (ENABLE_SYCL)
  list(APPEND __CLDNN_AllSources ${__CLDNN_Sources__sycl})
endif()

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"             FILES ${__CLDNN_Sources__main})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${__CLDNN_Directory__main}"
  )

# =================================== Link targets and dependencies ====================================

# Main static library.
add_library("${CLDNN_BUILD__PROJ}" STATIC
    ${__CLDNN_AllSources}
  )

set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")


target_link_libraries("${CLDNN_BUILD__PROJ}"
    clDNN_OpenCL
  )

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries("${CLDNN_BUILD__PROJ}" pthread)
endif()
target_link_libraries("${CLDNN_BUILD__PROJ}" ${CLDNN__SYSTEM_LINK_LIBRARIES})

# ======================================================================================================
