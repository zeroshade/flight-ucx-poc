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

#include "flight_ucx_conn.h"

#include <ucp/api/ucp.h>

#include "arrow/device.h"
#include "arrow/status.h"

namespace arrow {
namespace ucx {
Status Connection::Init(std::shared_ptr<UcpWorker> worker, ucp_ep_h endpoint) {
  ucp_worker_ = std::move(worker);
  remote_endpoint_ = endpoint;
  if (!memory_manager_) {
    memory_manager_ = CPUDevice::Instance()->default_memory_manager();
  }

  ucp_ep_attr_t attrs;
  std::memset(&attrs, 0, sizeof(attrs));
  attrs.field_mask = UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR | UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR;

  if (ucp_ep_query(remote_endpoint_, &attrs) == UCS_OK) {
    ARROW_UNUSED(SockaddrToString(attrs.local_sockaddr).Value(&local_addr_));
    ARROW_UNUSED(SockaddrToString(attrs.remote_sockaddr).Value(&remote_addr_));
  }
  return Status::OK();
}

Status Connection::Init(std::shared_ptr<UcpContext> ucp_context,
                        const internal::Uri& uri) {
  {
    ucp_worker_params_t worker_params;
    std::memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SERIALIZED;

    ucp_worker_h ucp_worker;
    ucs_status_t status =
        ucp_worker_create(ucp_context->get(), &worker_params, &ucp_worker);
    RETURN_NOT_OK(FromUcsStatus("ucp_worker_create", status));
    ucp_worker_.reset(new UcpWorker(std::move(ucp_context), ucp_worker));
  }
  {
    // create endpoint for remote worker
    struct sockaddr_storage connect_addr;
    ARROW_ASSIGN_OR_RAISE(auto addrlen, UriToSockaddr(uri, &connect_addr));
    std::string peer;
    ARROW_UNUSED(SockaddrToString(connect_addr).Value(&peer));
    ARROW_LOG(DEBUG) << "Connecting to " << peer;

    ucp_ep_params_t params;
    params.field_mask =
        UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_NAME | UCP_EP_PARAM_FIELD_SOCK_ADDR;
    params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    params.name = "UcxConn";
    params.sockaddr.addr = reinterpret_cast<const sockaddr*>(&connect_addr);
    params.sockaddr.addrlen = addrlen;

    auto status = ucp_ep_create(ucp_worker_->get(), &params, &remote_endpoint_);
    RETURN_NOT_OK(FromUcsStatus("ucp_ep_create", status));
  }

  return Status::OK();
}

Status Connection::Flush() {
  ucp_request_param_t param;
  param.op_attr_mask = 0;
  void* request = ucp_ep_flush_nbx(remote_endpoint_, &param);
  if (!request) {
    return Status::OK();
  }

  if (UCS_PTR_IS_ERR(request)) {
    return FromUcsStatus("ucp_ep_flush_nbx", UCS_PTR_STATUS(request));
  }

  ucs_status_t status;
  do {
    ucp_worker_progress(ucp_worker_->get());
    status = ucp_request_check_status(request);
  } while (status == UCS_INPROGRESS);
  ucp_request_free(request);
  return FromUcsStatus("ucp_request_check_status", status);
}

Status Connection::Close() {
  void* request = ucp_ep_close_nb(remote_endpoint_, UCP_EP_CLOSE_MODE_FLUSH);

  ucs_status_t status = UCS_OK;
  std::string origin = "ucp_ep_close_nb";
  if (UCS_PTR_IS_ERR(request)) {
    status = UCS_PTR_STATUS(request);
  } else if (UCS_PTR_IS_PTR(request)) {
    origin = "ucp_request_check_status";
    while ((status = ucp_request_check_status(request)) == UCS_INPROGRESS) {
      ucp_worker_progress(ucp_worker_->get());
    }
    ucp_request_free(request);
  } else {
    DCHECK(!request);
  }

  remote_endpoint_ = nullptr;
  ucp_worker_.reset();
  if (status != UCS_OK && !IsIgnorableDisconnectError(status)) {
    return FromUcsStatus(origin, status);
  }
  return Status::OK();
}

Status Connection::SendAM(unsigned int id, const void* data, const int64_t size) {
  RETURN_NOT_OK(CheckClosed());

  ucp_request_param_t request_param;
  request_param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
  request_param.flags = UCP_AM_SEND_FLAG_REPLY;

  void* request =
      ucp_am_send_nbx(remote_endpoint_, id, nullptr, 0, data, size, &request_param);
  return CompleteRequestBlocking("ucp_am_send_nbx", request);
}

Status Connection::SendAMIov(unsigned int id, const void* header,
                             const size_t header_length, const ucp_dt_iov_t* iov,
                             const size_t iov_cnt, void* user_data,
                             ucp_send_nbx_callback_t cb) {
  RETURN_NOT_OK(CheckClosed());

  ucp_request_param_t request_param;
  request_param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS | UCP_OP_ATTR_FIELD_DATATYPE |
                               UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  request_param.flags = UCP_AM_SEND_FLAG_COPY_HEADER | UCP_AM_SEND_FLAG_REPLY;
  request_param.datatype = UCP_DATATYPE_IOV;
  request_param.cb.send = cb;
  request_param.user_data = user_data;

  void* request = ucp_am_send_nbx(remote_endpoint_, id, header, header_length, iov,
                                  iov_cnt, &request_param);
  if (!request) {
    // request completed immediately, callback won't be called so we call it
    // ourselves manually.
    cb(request, UCS_OK, user_data);
  } else if (UCS_PTR_IS_ERR(request)) {
    cb(request, UCS_PTR_STATUS(request), user_data);
    return FromUcsStatus("ucp_am_send_nbx", UCS_PTR_STATUS(request));
  }

  return Status::OK();
}

Status Connection::SendStream(const void* data, const size_t length) {
  ucp_request_param_t request_param;
  request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
  request_param.datatype = ucp_dt_make_contig(1);

  void* request = ucp_stream_send_nbx(remote_endpoint_, data, length, &request_param);
  return CompleteRequestBlocking("ucp_stream_send_nbx", request);
}

ucs_status_t Connection::RecvAM(std::promise<std::unique_ptr<Buffer>> p,
                                const void* header, const size_t header_length,
                                void* data, const size_t data_length,
                                const ucp_am_recv_param_t* param) {
  auto maybe_status = RecvActiveMessageImpl(std::move(p), header, header_length, data,
                                            data_length, param);
  if (!maybe_status.ok()) {
    // handle bad status
    return UCS_OK;
  }
  return maybe_status.MoveValueUnsafe();
}

Status Connection::RecvStream(const void* buffer, const size_t capacity, size_t* length) {
  RETURN_NOT_OK(CheckClosed());

  ucp_request_param_t request_param;
  request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
  request_param.datatype = ucp_dt_make_contig(1);
  void* request = ucp_stream_recv_nbx(remote_endpoint_, const_cast<void*>(buffer),
                                      capacity, length, &request_param);

  if (request != nullptr) {
    ucs_status_t status;
    do {
      ucp_worker_progress(ucp_worker_->get());
      status = ucp_stream_recv_request_test(request, length);
    } while (status == UCS_INPROGRESS);
    ucp_request_free(request);
  }

  return Status::OK();
}

Status Connection::CompleteRequestBlocking(const std::string& context, void* request) {
  if (UCS_PTR_IS_ERR(request)) {
    return FromUcsStatus(context, UCS_PTR_STATUS(request));
  }

  if (UCS_PTR_IS_PTR(request)) {
    while (true) {
      auto status = ucp_request_check_status(request);
      if (status == UCS_OK) {
        break;
      } else if (status != UCS_INPROGRESS) {
        ucp_request_release(request);
        return FromUcsStatus("ucp_request_check_status", status);
      }
      ucp_worker_progress(ucp_worker_->get());
    }
    ucp_request_free(request);
  } else {
    DCHECK(!request);
  }
  return Status::OK();
}

Result<ucs_status_t> Connection::RecvActiveMessageImpl(
    std::promise<std::unique_ptr<Buffer>> p, const void* header,
    const size_t header_length, void* data, const size_t data_length,
    const ucp_am_recv_param_t* param) {
  DCHECK(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP);

  if (data_length > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return Status::Invalid("cannot allocate buffer greater than 2 GiB, requested: ",
                           data_length);
  }

  // do stuff with header

  if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
    // data provided can be held by us. return UCS_INPROGRESS to make the data persist
    // and we'll use ucp_am_data_release to release it.
    auto buffer = std::make_unique<UcxDataBuffer>(ucp_worker_, data, data_length);
    p.set_value(std::move(buffer));
    return UCS_INPROGRESS;
  }

  if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
    // rendezvous protocol
    ARROW_ASSIGN_OR_RAISE(auto buffer, memory_manager_->AllocateBuffer(data_length));

    ucp_request_param_t recv_param;
    recv_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                              UCP_OP_ATTR_FIELD_USER_DATA;
    recv_param.cb.recv_am = nullptr;
    recv_param.memory_type = UCS_MEMORY_TYPE_UNKNOWN;
    recv_param.user_data = nullptr;

    void* dest = reinterpret_cast<void*>(buffer->mutable_address());
    void* request =
        ucp_am_recv_data_nbx(ucp_worker_->get(), data, dest, data_length, &recv_param);

    if (UCS_PTR_IS_ERR(request)) {
      return FromUcsStatus("ucp_am_recv_data_nbx", UCS_PTR_STATUS(request));
    } else if (!request) {
      // request completed instantly
      // callback not called
      p.set_value(std::move(buffer));
    }
    return UCS_OK;
  }

  std::unique_ptr<Buffer> buffer;
  // data will be freed after callback returns - copy to buffer
  if (memory_manager_->is_cpu()) {
    ARROW_ASSIGN_OR_RAISE(buffer, memory_manager_->AllocateBuffer(data_length));
    std::memcpy(buffer->mutable_data(), data, data_length);
  } else {
    ARROW_ASSIGN_OR_RAISE(
        buffer, MemoryManager::CopyNonOwned(Buffer(reinterpret_cast<uint8_t*>(data),
                                                   static_cast<int64_t>(data_length)),
                                            memory_manager_));
  }

  p.set_value(std::move(buffer));
  return UCS_OK;
}

}  // namespace ucx
}  // namespace arrow