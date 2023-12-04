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

#include "flight_ucx_utils.h"
#include "flight_ucx_conn.h"

#include <memory>
#include <mutex>
#include <deque>

#include "arrow/status.h"
#include "arrow/result.h"
#include "arrow/flight/types.h"
#include "arrow/util/uri.h"

namespace arrow {
namespace ucx {
class UcxClient {
 public:
  UcxClient() = default;
  ~UcxClient() {
    if (!ucp_context_) return;
    ARROW_WARN_NOT_OK(Close(), "UcxClient errored in Close() in destructor");
  }

  Status Init(const flight::Location& location);
  arrow::Result<Connection> Get();
  Status Put(Connection conn);
  Status Close();

 private:
  Status MakeConnection();

  std::shared_ptr<UcpContext> ucp_context_;
  arrow::internal::Uri uri_;
  std::mutex connections_mutex_;
  std::deque<Connection> connections_;
};
}  // namespace ucx
}  // namespace arrow