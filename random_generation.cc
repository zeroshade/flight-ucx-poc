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

#include "random_generation.h"

#include "arrow/array.h"
#include "arrow/record_batch.h"
#include "arrow/memory_pool.h"
#include "arrow/status.h"
#include "arrow/testing/random.h"

namespace arrow {

template <typename ArrayType>
Status MakeRandomArray(int64_t length, bool include_nulls, MemoryPool* pool,
                       std::shared_ptr<Array>* out, uint32_t seed) {
  random::RandomArrayGenerator rand(seed);
  const double null_probability = include_nulls ? 0.5 : 0.0;

  *out = rand.Numeric<ArrayType>(length, 0, 1000, null_probability);

  return Status::OK();
}

template <>
Status MakeRandomArray<Int8Type>(int64_t length, bool include_nulls, MemoryPool* pool,
                                 std::shared_ptr<Array>* out, uint32_t seed) {
  random::RandomArrayGenerator rand(seed);
  const double null_probability = include_nulls ? 0.5 : 0.0;

  *out = rand.Numeric<Int8Type>(length, 0, 127, null_probability);

  return Status::OK();
}

template <>
Status MakeRandomArray<UInt8Type>(int64_t length, bool include_nulls, MemoryPool* pool,
                                  std::shared_ptr<Array>* out, uint32_t seed) {
  random::RandomArrayGenerator rand(seed);
  const double null_probability = include_nulls ? 0.5 : 0.0;

  *out = rand.Numeric<UInt8Type>(length, 0, 127, null_probability);

  return Status::OK();
}

Status MakeIntBatchSized(int length, std::shared_ptr<RecordBatch>* out, MemoryPool* pool,
                         uint32_t seed) {
  // Make the schema
  auto f0 = field("f0", int8());
  auto f1 = field("f1", uint8());
  auto f2 = field("f2", int16());
  auto f3 = field("f3", uint16());
  auto f4 = field("f4", int32());
  auto f5 = field("f5", uint32());
  auto f6 = field("f6", int64());
  auto f7 = field("f7", uint64());
  auto schema = ::arrow::schema({f0, f1, f2, f3, f4, f5, f6, f7});

  // Example data
  std::shared_ptr<Array> a0, a1, a2, a3, a4, a5, a6, a7;  
  RETURN_NOT_OK(MakeRandomArray<Int8Type>(length, false, pool, &a0, seed));
  RETURN_NOT_OK(MakeRandomArray<UInt8Type>(length, true, pool, &a1, seed));
  RETURN_NOT_OK(MakeRandomArray<Int16Type>(length, true, pool, &a2, seed));
  RETURN_NOT_OK(MakeRandomArray<UInt16Type>(length, false, pool, &a3, seed));
  RETURN_NOT_OK(MakeRandomArray<Int32Type>(length, false, pool, &a4, seed));
  RETURN_NOT_OK(MakeRandomArray<UInt32Type>(length, true, pool, &a5, seed));
  RETURN_NOT_OK(MakeRandomArray<Int64Type>(length, true, pool, &a6, seed));
  RETURN_NOT_OK(MakeRandomArray<UInt64Type>(length, false, pool, &a7, seed));
  *out = RecordBatch::Make(schema, length, {a0, a1, a2, a3, a4, a5, a6, a7});
  return Status::OK();
}

}  // namespace arrow