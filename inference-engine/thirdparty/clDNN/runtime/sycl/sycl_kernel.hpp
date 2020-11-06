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

#include "sycl_common.hpp"
#include "sycl_memory.hpp"
#include "cldnn/runtime/kernel_args.hpp"
#include "cldnn/runtime/kernel.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace sycl {

class sycl_kernel : public kernel {
    cl::sycl::kernel _compiled_kernel;
    std::string _kernel_id;

public:
    sycl_kernel(cl::sycl::kernel compiled_kernel, const std::string& kernel_id);

    // sycl_kernel(const sycl_kernel& other)
    //     : _compiled_kernel(other._compiled_kernel)
    //     , _kernel_id(other._kernel_id) {}

    // sycl_kernel& operator=(const sycl_kernel& other) {
    //     if (this == &other) {
    //         return *this;
    //     }

    //     _kernel_id = other._kernel_id;
    //     _compiled_kernel = other._compiled_kernel;

    //     return *this;
    // }

    std::shared_ptr<kernel> clone() const override { return std::make_shared<sycl_kernel>(get_handle(), _kernel_id); }
    const cl::sycl::kernel& get_handle() const { return _compiled_kernel; }
};

}  // namespace sycl
}  // namespace cldnn
