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

#include "sycl_kernel.hpp"

#include <memory>
#include <vector>



namespace cldnn {
namespace sycl {


sycl_kernel::sycl_kernel(cl::sycl::kernel compiled_kernel, const std::string& kernel_id)
    : _compiled_kernel(compiled_kernel)
    , _kernel_id(kernel_id) {

}

}  // namespace sycl
}  // namespace cldnn
