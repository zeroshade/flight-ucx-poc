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
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "arrow/buffer.h"
#include "arrow/flight/types.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/uri.h"

#include "flight_ucx_conn.h"
#include "flight_ucx_utils.h"

namespace arrow {
namespace ucx {
class UcxServer {
 public:
  Status Init(const internal::Uri& uri);
  Status Wait();
  inline flight::Location location() const { return location_; }
  Status Shutdown();

 protected:
  struct ClientWorker {
    std::shared_ptr<UcpWorker> worker_;
    std::unique_ptr<Connection> conn_;

    std::mutex queue_mutex_;
    std::queue<std::future<std::unique_ptr<Buffer>>> buffer_queue_;
  };

  virtual Status do_work(ClientWorker* worker) = 0;
  virtual Status setup_handlers(ClientWorker* worker) = 0;

 private:
  static void HandleIncomingConnection(ucp_conn_request_h connection_request,
                                       void* data) {
    UcxServer* server = reinterpret_cast<UcxServer*>(data);
    server->EnqueueClient(connection_request);
  }

  void EnqueueClient(ucp_conn_request_h connection_request);
  void DriveConnections();
  void WorkerLoop(ucp_conn_request_h request);
  arrow::Result<std::shared_ptr<ClientWorker>> CreateWorker();  

 protected:
  std::atomic<size_t> counter_;
  flight::Location location_;
  std::shared_ptr<UcpContext> ucp_context_;
  std::shared_ptr<UcpWorker> worker_conn_;
  ucp_listener_h listener_;

  std::shared_ptr<arrow::internal::ThreadPool> rpc_pool_;
  std::thread listener_thread_;
  std::atomic<bool> listening_;
  // std::thread::join cannot be called concurrently
  std::mutex join_mutex_;
  std::mutex pending_connections_mutex_;
  std::queue<ucp_conn_request_h> pending_connections_;
};
}  // namespace ucx
}  // namespace arrow
