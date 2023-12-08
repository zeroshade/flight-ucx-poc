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

#include "flight_ucx_client.h"
#include "flight_ucx_utils.h"

#include <netdb.h>
#include <ucp/api/ucp.h>
#include <mutex>

namespace arrow {
namespace ucx {
Status UcxClient::Init(const flight::Location& location) {
  RETURN_NOT_OK(uri_.Parse(location.ToString()));
  {
    ucp_config_t* ucp_config;
    ucp_params_t ucp_params;
    ucs_status_t status;

    status = ucp_config_read(nullptr, nullptr, &ucp_config);
    RETURN_NOT_OK(FromUcsStatus("ucp_config_read", status));

    // if location is IPv6, must adjust UCX config
    // we assume locations always resolve to IPv6 or IPv4
    // but that's not necessarily true.
    {
      struct sockaddr_storage connect_addr;
      RETURN_NOT_OK(UriToSockaddr(uri_, &connect_addr));
      if (connect_addr.ss_family == AF_INET6) {
        status = ucp_config_modify(ucp_config, "AF_PRIO", "inet6");
        RETURN_NOT_OK(FromUcsStatus("ucp_config_modify", status));
      }
    }

    std::memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_WAKEUP | UCP_FEATURE_AM | UCP_FEATURE_RMA |
                          UCP_FEATURE_STREAM | UCP_FEATURE_TAG;

    ucp_context_h ucp_context;
    status = ucp_init(&ucp_params, ucp_config, &ucp_context);
    ucp_config_release(ucp_config);
    RETURN_NOT_OK(FromUcsStatus("ucp_init", status));
    ucp_context_.reset(new UcpContext(ucp_context));
  }

  RETURN_NOT_OK(MakeConnection());
  return Status::OK();
}

arrow::Result<Connection> UcxClient::Get() {
  std::lock_guard<std::mutex> lock(connections_mutex_);
  if (connections_.empty()) RETURN_NOT_OK(MakeConnection());
  Connection conn = std::move(connections_.front());
  connections_.pop_front();
  return conn;
}

Status UcxClient::Put(Connection conn) {
  std::lock_guard<std::mutex> lock(connections_mutex_);
  connections_.push_back(std::move(conn));
  return Status::OK();
}

Status UcxClient::Close() {
  std::lock_guard<std::mutex> lock(connections_mutex_);
  while (!connections_.empty()) {
    Connection conn = std::move(connections_.front());
    connections_.pop_front();
    RETURN_NOT_OK(conn.Close());
  }
  return Status::OK();
}

Status UcxClient::MakeConnection() {
  Connection conn;
  RETURN_NOT_OK(conn.Init(ucp_context_, uri_));
  std::lock_guard<std::mutex> lock(connections_mutex_);
  connections_.push_back(std::move(conn));
  return Status::OK();
}

}  // namespace ucx
}  // namespace arrow
