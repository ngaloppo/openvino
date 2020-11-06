/*
// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "kernels_factory.hpp"

// #include "sycl/sycl_kernel.hpp"
#include "ocl/ocl_kernel.hpp"





namespace cldnn {
namespace sycl {
std::shared_ptr<kernel> create_sycl_kernel(engine& engine, cl_context context, cl_kernel kernel, gpu::kernel_id kernel_id);
}
namespace gpu {
std::shared_ptr<kernel> create_ocl_kernel(engine& engine, cl_context context, cl_kernel kernel, gpu::kernel_id kernel_id);
}  // namespace gpu

namespace kernels_factory {

std::shared_ptr<kernel> create(engine& engine, cl_context context, cl_kernel kernel, gpu::kernel_id kernel_id) {
    switch (engine.type()) {
        case engine_types::ocl: return gpu::create_ocl_kernel(engine, context, kernel, kernel_id);
        case engine_types::sycl: return sycl::create_sycl_kernel(engine, context, kernel, kernel_id);
    }
}

}  // namespace kernels_factory
}  // namespace cldnn
