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

#pragma once

#include "cldnn/runtime/kernel.hpp"
#include "cldnn/runtime/engine.hpp"
#include "ocl/ocl_common.hpp"

#include <memory>

namespace cldnn {
namespace kernels_factory {

// Creates instance of ocl_kernel/sycl_kernel for selected engine type.
// For ocl engine it creates a copy of kernel object
std::shared_ptr<kernel> create(engine& engine, cl_context context, cl_kernel kernel, gpu::kernel_id kernel_id);

}  // namespace kernels_factory
}  // namespace cldnn
