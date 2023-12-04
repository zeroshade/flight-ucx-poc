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

#include <netdb.h>
#include <ucp/api/ucp.h>
#include <array>

#include "arrow/buffer.h"
#include "arrow/result.h"
#include "arrow/util/logging.h"
#include "arrow/util/uri.h"

namespace arrow {

namespace ucx {

class UcpContext final {
 public:
  UcpContext() = default;
  explicit UcpContext(ucp_context_h context) : ucp_context_(context) {}
  ~UcpContext() {
    if (ucp_context_) ucp_cleanup(ucp_context_);
    ucp_context_ = nullptr;
  }
  ucp_context_h get() const {
    DCHECK(ucp_context_);
    return ucp_context_;
  }

 private:
  ucp_context_h ucp_context_{nullptr};
};

// small wrapper around ucp_worker_h
class UcpWorker final {
 public:
  UcpWorker() = default;
  UcpWorker(std::shared_ptr<UcpContext> context, ucp_worker_h worker)
      : ucp_context_(std::move(context)), ucp_worker_(worker) {}
  ~UcpWorker() {
    if (ucp_worker_) ucp_worker_destroy(ucp_worker_);
    ucp_worker_ = nullptr;
  }
  ucp_worker_h get() const { return ucp_worker_; }
  const UcpContext& context() const { return *ucp_context_; }

 private:
  ucp_worker_h ucp_worker_{nullptr};
  std::shared_ptr<UcpContext> ucp_context_;
};

arrow::Result<std::string> SockaddrToString(const struct sockaddr_storage& address);

class UcxStatusDetail : public StatusDetail {
 public:
  explicit UcxStatusDetail(ucs_status_t status) : status_(status) {}
  static constexpr char const kTypeId[] = "flight::transport::ucx::UcxStatusDetail";

  const char* type_id() const override { return kTypeId; }
  std::string ToString() const override;
  static ucs_status_t Unwrap(const Status& status);

 private:
  ucs_status_t status_;
};

Status FromUcsStatus(const std::string& context, ucs_status_t ucs_status);
arrow::Result<size_t> UriToSockaddr(const arrow::internal::Uri& uri,
                                    struct sockaddr_storage* addr);

static inline bool IsIgnorableDisconnectError(ucs_status_t ucs_status) {
  // Not connected, connection reset: we're already disconnected
  // Timeout: most likely disconnected, but we can't tell from our end
  return ucs_status == UCS_OK || ucs_status == UCS_ERR_ENDPOINT_TIMEOUT ||
         ucs_status == UCS_ERR_NOT_CONNECTED || ucs_status == UCS_ERR_CONNECTION_RESET;
}

class UcxDataBuffer : public Buffer {
 public:
  UcxDataBuffer(std::shared_ptr<UcpWorker> worker, void* data, const size_t size)
      : Buffer(reinterpret_cast<uint8_t*>(data), static_cast<int64_t>(size)),
        worker_(std::move(worker)) {}
  ~UcxDataBuffer() override {
    ucp_am_data_release(worker_->get(),
                        const_cast<void*>(reinterpret_cast<const void*>(data())));
  }

 private:
  std::shared_ptr<UcpWorker> worker_;
};

}  // namespace ucx

}  // namespace arrow