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

#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/stream.hpp"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

namespace cldnn {

memory::memory(engine* engine, const layout& layout,  allocation_type type, bool reused)
    : _engine(engine), _layout(layout), _bytes_count(_layout.bytes_count()), _type(type), _reused(reused) {}

memory::~memory() {
    if (!_reused && _engine) {
        // TODO: Make memory usage tracker static in memory class
        _engine->get_memory_pool().subtract_memory_used(_bytes_count);
    }
}

}  // namespace cldnn
