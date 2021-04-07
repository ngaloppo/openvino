// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_kernel.hpp"
#include "kernels_factory.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace gpu {

std::shared_ptr<kernel> create_ocl_kernel(engine& engine, cl_context context, cl_kernel kernel, gpu::kernel_id kernel_id) {
    // Retain kernel to keep it valid
    cl::Kernel k(kernel, true);
    return std::make_shared<gpu::ocl_kernel>(gpu::kernel_type(k, engine.get_device_info().supports_usm), kernel_id);
}

}  // namespace kernels_factory
}  // namespace cldnn
