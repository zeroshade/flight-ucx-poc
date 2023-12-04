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
#include <array>
#include <future>

#include "flight_ucx_utils.h"

#include "arrow/device.h"
#include "arrow/util/logging.h"
#include "arrow/util/uri.h"

namespace arrow {
namespace ucx {
class Connection {
 public:
  Connection() = default;
  ARROW_DISALLOW_COPY_AND_ASSIGN(Connection);
  ARROW_DEFAULT_MOVE_AND_ASSIGN(Connection);
  ~Connection() { DCHECK(!ucp_worker_) << "Connection was not closed!"; }

  inline void set_memory_manager(std::shared_ptr<MemoryManager> mm) {
    if (mm) {
      memory_manager_ = std::move(mm);
    } else {
      memory_manager_ = CPUDevice::Instance()->default_memory_manager();
    }
  }

  Status Init(std::shared_ptr<UcpWorker> worker, ucp_ep_h endpoint);
  Status Init(std::shared_ptr<UcpContext> ucp_context, const internal::Uri& uri);
  Status Flush();
  Status Close();
  Status SendAM(unsigned int id, const void* data, const int64_t size);
  Status SendAMIov(unsigned int id, const void* header, const size_t header_length,
                   const ucp_dt_iov_t* iov, const size_t iov_cnt, void* user_data,
                   ucp_send_nbx_callback_t cb);
  Status SendStream(const void* data, const size_t length);
  ucs_status_t RecvAM(std::promise<std::unique_ptr<Buffer>> p, const void* header,
                      const size_t header_length, void* data, const size_t data_length,
                      const ucp_am_recv_param_t* param);
  Status RecvStream(const void* buffer, const size_t capacity, size_t* length);

  Status SetAMHandler(const ucp_am_handler_param_t* param) {
    return FromUcsStatus("ucp_worker_set_am_recv_handler",
                         ucp_worker_set_am_recv_handler(ucp_worker_->get(), param));
  }

  int MakeProgress() { return ucp_worker_progress(ucp_worker_->get()); }
  std::shared_ptr<UcpWorker> worker() { return ucp_worker_; }
 protected:
  inline Status CheckClosed() {
    if (!remote_endpoint_) {
      return Status::Invalid("Connection is closed");
    }
    return Status::OK();
  }

  Status CompleteRequestBlocking(const std::string& context, void* request);
  arrow::Result<ucs_status_t> RecvActiveMessageImpl(
      std::promise<std::unique_ptr<Buffer>> p, const void* header,
      const size_t header_length, void* data, const size_t data_length,
      const ucp_am_recv_param_t* param);

 private:
  std::shared_ptr<MemoryManager> memory_manager_;
  std::shared_ptr<UcpWorker> ucp_worker_;
  ucp_ep_h remote_endpoint_;

  std::string local_addr_, remote_addr_;
};

}  // namespace ucx
}  // namespace arrow