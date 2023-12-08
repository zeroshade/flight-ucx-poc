// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <ucp/api/ucp.h>
#include <atomic>
#include "arrow/memory_pool.h"

namespace arrow {
namespace ucx {

class UcxMappedPool : public MemoryPool {
 public:
  static Result<std::unique_ptr<UcxMappedPool>> Make(ucp_context_h ctx, size_t initial);

  virtual ~UcxMappedPool();

  Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override;
  Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                    uint8_t** ptr) override;
  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;
  void ReleaseUnused() override {}
  int64_t bytes_allocated() const override { return allocated_.load(); }
  int64_t total_bytes_allocated() const override { return allocated_.load(); }
  int64_t max_memory() const override { return initial_; }
  int64_t num_allocations() const override { return num_allocs_.load(); }
  std::string backend_name() const override { return "ucx_mapped_pool"; }

  ucp_mem_h get_mem_handle() const;
  void* get_exported_memh() const;


  UcxMappedPool(ucp_context_h ctx, ucp_mem_h handle, size_t length, void* address, void* exported_memh);

 private:
  virtual void copy_data(uint8_t* dest, uint8_t* src, size_t n);

  std::atomic<int64_t> allocated_{0};
  std::atomic<int64_t> num_allocs_{0};
  const size_t initial_;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace ucx
}  // namespace arrow