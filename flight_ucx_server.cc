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

#include "flight_ucx_server.h"
#include "flight_ucx_utils.h"

#include <netdb.h>
#include "arrow/flight/types.h"
#include "arrow/util/string.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/uri.h"
#include "arrow/gpu/cuda_api.h"

namespace arrow {
using internal::ToChars;

namespace ucx {
Status UcxServer::Init(const internal::Uri& uri) {
  const auto max_threads = std::max<uint32_t>(8, std::thread::hardware_concurrency());
  ARROW_ASSIGN_OR_RAISE(rpc_pool_, arrow::internal::ThreadPool::Make(max_threads));

  struct sockaddr_storage listen_addr;
  ARROW_ASSIGN_OR_RAISE(auto addrlen, UriToSockaddr(uri, &listen_addr));

  // initialize ucx!
  {
    ucp_config_t* ucp_config;
    ucp_params_t ucp_params;
    ucs_status_t status = ucp_config_read(nullptr, nullptr, &ucp_config);
    RETURN_NOT_OK(FromUcsStatus("ucp_config_read", status));

    // if location is ipv6, adjust ucx config
    if (listen_addr.ss_family == AF_INET6) {
      status = ucp_config_modify(ucp_config, "AF_PRIO", "inet6");
      RETURN_NOT_OK(FromUcsStatus("ucp_config_modify", status));
    }

    std::memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_NAME |
                            UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    ucp_params.features =
        UCP_FEATURE_AM | UCP_FEATURE_STREAM | UCP_FEATURE_WAKEUP | UCP_FEATURE_RMA;
    ucp_params.mt_workers_shared = UCS_THREAD_MODE_MULTI;
    ucp_params.name = "ucx_flight_data_server";

    ucp_context_h ucp_context;
    status = ucp_init(&ucp_params, ucp_config, &ucp_context);
    ucp_config_release(ucp_config);
    RETURN_NOT_OK(FromUcsStatus("ucp_init", status));
    ucp_context_.reset(new UcpContext(ucp_context));
  }

  {
    // create one worker to listen for incoming connections
    ucp_worker_params_t worker_params;
    ucs_status_t status;

    std::memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    ucp_worker_h worker;
    status = ucp_worker_create(ucp_context_->get(), &worker_params, &worker);
    RETURN_NOT_OK(FromUcsStatus("ucp_worker_create", status));
    worker_conn_.reset(new UcpWorker(ucp_context_, worker));
  }

  // start listening
  {
    ucp_listener_params_t params;
    ucs_status_t status;

    params.field_mask =
        UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.sockaddr.addr = reinterpret_cast<const sockaddr*>(&listen_addr);
    params.sockaddr.addrlen = addrlen;
    params.conn_handler.cb = HandleIncomingConnection;
    params.conn_handler.arg = this;

    status = ucp_listener_create(worker_conn_->get(), &params, &listener_);
    RETURN_NOT_OK(FromUcsStatus("ucp_listener_create", status));

    // get real address/port
    ucp_listener_attr_t attr;
    attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    status = ucp_listener_query(listener_, &attr);
    RETURN_NOT_OK(FromUcsStatus("ucp_listener_query", status));

    std::string raw_uri = "ucx://";
    if (uri.host().find(":") != std::string::npos) {
      // IPv6 host
      raw_uri += '[';
      raw_uri += uri.host();
      raw_uri += ']';
    } else {
      raw_uri += uri.host();
    }
    raw_uri += ":";
    raw_uri +=
        ToChars(ntohs(reinterpret_cast<const sockaddr_in*>(&attr.sockaddr)->sin_port));

    std::string listen_str;
    ARROW_UNUSED(SockaddrToString(attr.sockaddr).Value(&listen_str));
    ARROW_LOG(DEBUG) << "Listening on " << listen_str;
    ARROW_ASSIGN_OR_RAISE(location_, flight::Location::Parse(raw_uri));
  }

  {
    listening_.store(true);
    std::thread listener_thread(&UcxServer::DriveConnections, this);
    listener_thread_.swap(listener_thread);
  }

  return Status::OK();
}

Status UcxServer::Wait() {
  std::lock_guard<std::mutex> guard(join_mutex_);
  try {
    listener_thread_.join();
  } catch (const std::system_error& e) {
    if (e.code() != std::errc::invalid_argument) {
      return Status::UnknownError("could not Wait(): ", e.what());
    }
    // else server wasn't running anyways
  }
  return Status::OK();
}

Status UcxServer::Shutdown() {
  if (!listening_.load()) return Status::OK();

  Status status;
  // wait for current RPCs to  finish
  listening_.store(false);
  RETURN_NOT_OK(
      FromUcsStatus("ucp_worker_signal", ucp_worker_signal(worker_conn_->get())));
  status &= Wait();

  {
    // reject all pending connections
    std::lock_guard<std::mutex> guard(pending_connections_mutex_);
    while (!pending_connections_.empty()) {
      status &=
          FromUcsStatus("ucp_listener_reject",
                        ucp_listener_reject(listener_, pending_connections_.front()));
      pending_connections_.pop();
    }
    ucp_listener_destroy(listener_);
    worker_conn_.reset();
  }

  status &= rpc_pool_->Shutdown();
  rpc_pool_.reset();
  ucp_context_.reset();
  return status;
}

void UcxServer::EnqueueClient(ucp_conn_request_h connection_request) {
  std::lock_guard<std::mutex> guard(pending_connections_mutex_);
  pending_connections_.push(connection_request);
}

void UcxServer::DriveConnections() {
  while (listening_.load()) {
    // wait for server to receive connection request from client
    while (ucp_worker_progress(worker_conn_->get())) {
    }
    {
      // check for requests in queue
      std::lock_guard<std::mutex> guard(pending_connections_mutex_);
      while (!pending_connections_.empty()) {
        ucp_conn_request_h request = pending_connections_.front();
        pending_connections_.pop();

        auto submitted = rpc_pool_->Submit([this, request]() { WorkerLoop(request); });
        ARROW_WARN_NOT_OK(submitted.status(), "failed to submit task to handle client");
      }
    }

    // check listening_ in case we're shutting down.
    // it's possible that shutdown was called while we were in
    // ucp_worker_progress above, in which case if we don't check
    // listening_ here, we'll enter ucp_worker_wait and get stuck.
    if (!listening_.load()) break;
    auto status = ucp_worker_wait(worker_conn_->get());
    if (status != UCS_OK) {
      ARROW_LOG(WARNING) << FromUcsStatus("ucp_worker_wait", status).ToString();
    }
  }
}

void UcxServer::WorkerLoop(ucp_conn_request_h request) {
  std::string peer = "unknown:" + ToChars(counter_++);
  {
    ucp_conn_request_attr_t request_attr;
    std::memset(&request_attr, 0, sizeof(request_attr));
    request_attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    if (ucp_conn_request_query(request, &request_attr) == UCS_OK) {
      ARROW_UNUSED(SockaddrToString(request_attr.client_address).Value(&peer));
    }
  }
  ARROW_LOG(DEBUG) << peer << ": Received connection request";

  auto maybe_worker = CreateWorker();
  if (!maybe_worker.ok()) {
    ARROW_LOG(WARNING) << peer << ": failed to create worker"
                       << maybe_worker.status().ToString();
    auto status = ucp_listener_reject(listener_, request);
    if (status != UCS_OK) {
      ARROW_LOG(WARNING) << peer << ": "
                         << FromUcsStatus("ucp_listener_reject", status).ToString();
    }
    return;
  }

  auto worker = maybe_worker.MoveValueUnsafe();
  // create an endpoint to the client using the data worker
  {
    ucs_status_t status;
    ucp_ep_params_t params;
    std::memset(&params, 0, sizeof(params));
    params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    params.conn_request = request;

    ucp_ep_h client_endpoint;
    status = ucp_ep_create(worker->worker_->get(), &params, &client_endpoint);
    if (status != UCS_OK) {
      ARROW_LOG(WARNING) << peer << ": failed to create endpoint: "
                         << FromUcsStatus("ucp_ep_create", status).ToString();
      return;
    }
    worker->conn_ = std::make_unique<Connection>();
    auto st = worker->conn_->Init(worker->worker_, client_endpoint);
    if (!st.ok()) {
      ARROW_LOG(WARNING) << peer
                         << ": failed to initialize connection: " << st.ToString();
      return;
    }
  }

  auto mgr = cuda::CudaDeviceManager::Instance().ValueOrDie();
  auto context = mgr->GetContext(0).ValueOrDie();
  cuCtxPushCurrent(reinterpret_cast<CUcontext>(context->handle()));

  while (listening_.load()) {
    ucp_worker_progress(worker->worker_->get());
    auto st = do_work(worker.get());
    if (!st.ok()) {
      ARROW_LOG(WARNING) << peer << ": error from do_work: " << st.ToString();
      break;
    }
  }

  // clean up
  auto status = worker->conn_->Close();
  if (!status.ok()) {
    ARROW_LOG(WARNING) << peer
                       << ": failed to close worker connection: " << status.ToString();
  }
  worker->worker_.reset();
  worker->conn_.reset();
  ARROW_LOG(DEBUG) << peer << ": disconnected";
}

arrow::Result<std::shared_ptr<UcxServer::ClientWorker>> UcxServer::CreateWorker() {
  auto worker = std::make_shared<ClientWorker>();

  ucp_worker_params_t worker_params;
  std::memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

  ucp_worker_h ucp_worker;
  auto status = ucp_worker_create(ucp_context_->get(), &worker_params, &ucp_worker);
  RETURN_NOT_OK(FromUcsStatus("ucp_worker_create", status));

  worker->worker_ = std::make_shared<UcpWorker>(ucp_context_, ucp_worker);
  RETURN_NOT_OK(setup_handlers(worker.get()));
  return worker;
}

}  // namespace ucx
}  // namespace arrow
