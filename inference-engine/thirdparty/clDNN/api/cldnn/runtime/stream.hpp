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

#include "event.hpp"
#include "kernel.hpp"
#include "kernel_args.hpp"

#include <memory>
#include <vector>

namespace cldnn {

class stream {
public:
    using ptr = std::shared_ptr<stream>;
    virtual ~stream() = default;

    virtual void sync_events(std::vector<event::ptr> const& deps, bool is_output_event = false) = 0;
    virtual void flush() = 0;
    virtual void finish() = 0;

    virtual void set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) = 0;
    virtual event::ptr enqueue_kernel(kernel& kernel,
                                      const kernel_arguments_desc& args_desc,
                                      const kernel_arguments_data& args,
                                      std::vector<event::ptr> const& deps,
                                      bool is_output_event = false) = 0;
    virtual event::ptr enqueue_marker(std::vector<event::ptr> const& deps, bool is_output_event = false) = 0;
    virtual void enqueue_barrier() = 0;
    virtual event::ptr group_events(std::vector<event::ptr> const& deps) = 0;
    virtual void wait_for_events(const std::vector<event::ptr>& events) = 0;
    virtual void reset_events() = 0;
    virtual event::ptr create_user_event(bool set) = 0;
    virtual event::ptr create_base_event() = 0;
    virtual void release_events_pool() = 0;
};

}  // namespace cldnn
