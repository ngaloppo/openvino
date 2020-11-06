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

#include "sycl_event.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <list>
#include <map>

namespace cldnn {
namespace sycl {

void sycl_event::wait_impl() {
    if (_event.get() != nullptr) {
        _event.wait();
    }
}

bool sycl_event::is_set_impl() {
    return _event.get_info<cl::sycl::info::event::command_execution_status>() == cl::sycl::info::event_command_status::complete;
}

bool sycl_event::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    return true;
}

} // namespace sycl
} // namespace cldnn
