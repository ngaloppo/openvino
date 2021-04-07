// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// we want exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <cl2_wrapper.h>

#include <vector>

namespace cldnn {
namespace gpu {

typedef cl::vector<cl::vector<unsigned char>> kernels_binaries_vector;
typedef cl::vector<kernels_binaries_vector> kernels_binaries_container;
typedef CL_API_ENTRY cl_command_queue(CL_API_CALL* pfn_clCreateCommandQueueWithPropertiesINTEL)(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcodeRet);

using queue_type = cl::CommandQueueIntel;
using kernel_type = cl::KernelIntel;
using kernel_id = std::string;

class ocl_error : public std::runtime_error {
public:
    explicit ocl_error(cl::Error const& err);
};

}  // namespace gpu
}  // namespace cldnn
