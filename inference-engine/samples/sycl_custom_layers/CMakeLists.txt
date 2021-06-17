# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(SYCL REQUIRED)

set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})
set(CMAKE_CXX_STANDARD 17)

intel_config_flag_apply_settings(
        CompilerOptions
        CMAKE_CXX_FLAGS
        ALL_PATTERN ""
        SET_RAW
        -stdlib=libstdc++
)

file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

ie_add_sample(NAME sycl_custom_layers
        SOURCES ${MAIN_SRC}
        HEADERS ${MAIN_HEADERS}
        DEPENDENCIES format_reader)

find_package(ngraph REQUIRED)
target_compile_options(sycl_custom_layers PRIVATE ${SYCL_FLAGS})
target_link_options(sycl_custom_layers PRIVATE ${SYCL_FLAGS})
target_link_libraries(sycl_custom_layers PRIVATE ${NGRAPH_LIBRARIES} ${SYCL_LIBRARY})
