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

#include "arrow/memory_pool.h"
#include "arrow/record_batch.h"
#include "arrow/status.h"

namespace arrow {
Status MakeIntBatchSized(int length, std::shared_ptr<RecordBatch>* out, MemoryPool* pool,
                         uint32_t seed = 0);

inline Status MakeIntRecordBatch(std::shared_ptr<RecordBatch>* out, MemoryPool* pool) {
  return MakeIntBatchSized(10, out, pool);
}
}  // namespace arrow
