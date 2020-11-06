/*
// Copyright (c) 2016-2021 Intel Corporation
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

#include "ocl_common.hpp"
#include "ocl_memory.hpp"
#include "cldnn/runtime/kernel_args.hpp"
#include "cldnn/runtime/kernel.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace gpu {

class ocl_kernel : public kernel {
    kernel_type _compiled_kernel;
    std::string _kernel_id;

public:
    ocl_kernel(kernel_type compiled_kernel, const std::string& kernel_id)
        : _compiled_kernel(compiled_kernel)
        , _kernel_id(kernel_id) {}

    // ocl_kernel(const ocl_kernel& other)
    //     : _compiled_kernel(other._compiled_kernel)
    //     , _kernel_id(other._kernel_id) {}

    // ocl_kernel& operator=(const ocl_kernel& other) {
    //     if (this == &other) {
    //         return *this;
    //     }

    //     _kernel_id = other._kernel_id;
    //     _compiled_kernel = other._compiled_kernel;

    //     return *this;
    // }

    const kernel_type& get_handle() const { return _compiled_kernel; }
    std::shared_ptr<kernel> clone() const override { return std::make_shared<ocl_kernel>(get_handle().clone(), _kernel_id); }
};

}  // namespace gpu
}  // namespace cldnn
