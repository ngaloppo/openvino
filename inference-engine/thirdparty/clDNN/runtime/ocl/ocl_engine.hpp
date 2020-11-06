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

#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/stream.hpp"
#include "ocl_device.hpp"

#include <memory>
#include <set>
#include <vector>
#include <utility>
#include <string>

namespace cldnn {
namespace gpu {

struct ocl_engine : public engine {
public:
    ocl_engine(const device::ptr dev, runtime_types runtime_type, const engine_configuration& conf);
    engine_types type() const override { return engine_types::ocl; };
    runtime_types runtime_type() const override { return runtime_types::ocl; };

    memory_ptr allocate_memory(const layout& layout, allocation_type type) override;
    memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) override;
    memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) override;
    bool is_the_same_buffer(const memory& mem1, const memory& mem2) override;

    void* get_user_context() const override;

    allocation_type get_default_allocation_type() const override { return allocation_type::cl_mem; }

    const cl::Context& get_cl_context() const;
    const cl::Device& get_cl_device() const;

    bool extension_supported(std::string extension) const;

    stream_ptr create_stream() const override;
    stream& get_program_stream() const override;

private:
    std::string _extensions;
    std::unique_ptr<stream> _program_stream;
};

}  // namespace gpu
}  // namespace cldnn
