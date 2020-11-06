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

#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/device.hpp"
#include "sycl_common.hpp"

#include <map>
#include <string>
#include <vector>
#include <algorithm>

namespace cldnn {
namespace sycl {

struct sycl_device : public device {
public:
    sycl_device(const cl::sycl::device& dev, const cl::sycl::context& ctx);

    device_info get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    const cl::sycl::device& get_device() const { return _device; }
    const cl::sycl::context& get_context() const { return _context; }

    cl_device_id get_ocl_device() const { return _device.get(); }

    ~sycl_device() = default;

private:
    cl::sycl::context _context;
    cl::sycl::device _device;
    device_info _info;
    memory_capabilities _mem_caps;
};

}  // namespace sycl
}  // namespace cldnn
