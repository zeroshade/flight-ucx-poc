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
#include "flight_ucx_conn.h"
#include "flight_ucx_server.h"

#include <signal.h>
#include <ucp/api/ucp.h>
#include <future>
#include <iostream>
#include <memory>

#include "arrow/buffer.h"
#include "arrow/flight/client.h"
#include "arrow/flight/server.h"
#include "arrow/ipc/test_common.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"
#include "arrow/util/uri.h"

static constexpr uint32_t kTicketAmHandlerId = 0xCAFE;
static constexpr uint32_t kEndStream = 0xDEADBEEF;

namespace arrow {

const std::shared_ptr<Schema> test_schema =
    ::arrow::schema({field("f0", int8()), field("f1", uint8()), field("f2", int16()),
                     field("f3", uint16()), field("f4", int32()), field("f5", uint32()),
                     field("f6", int64()), field("f7", uint64())});

namespace ucx {
class UcxStreamReader {
 public:
  explicit UcxStreamReader(Connection* cnxn)
      : cnxn_(cnxn), ipc_options_(ipc::IpcReadOptions::Defaults()) {}

  Status SetHandlers() {
    ucp_am_handler_param_t params;
    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ARG | UCP_AM_HANDLER_PARAM_FIELD_ID |
                        UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
    params.flags = UCP_AM_FLAG_WHOLE_MSG | UCP_AM_FLAG_PERSISTENT_DATA;
    params.arg = this;
    params.id = static_cast<unsigned int>(ipc::MessageType::SCHEMA);
    params.cb = RecvSchemaMsg;
    RETURN_NOT_OK(cnxn_->SetAMHandler(&params));

    params.id = static_cast<unsigned int>(ipc::MessageType::RECORD_BATCH);
    params.cb = RecvRecordBatchMsg;
    RETURN_NOT_OK(cnxn_->SetAMHandler(&params));

    params.id = kEndStream;
    params.cb = RecvEosMsg;
    RETURN_NOT_OK(cnxn_->SetAMHandler(&params));

    return Status::OK();
  }

  arrow::Result<std::shared_ptr<Schema>> GetSchema() {
    std::thread([this] {
      while (!finished_.load()) {
        if (cnxn_->MakeProgress()) {
          cv_progress_.notify_all();
        }
      }
    }).detach();

    std::unique_lock<std::mutex> lock(stream_lock_);
    if (schema_) {
      return schema_;
    }

    cv_progress_.wait(lock, [this] { return finished_.load() || schema_ != nullptr; });

    return schema_;
  }

  arrow::Result<std::shared_ptr<RecordBatch>> ReadNext() {
    std::unique_lock<std::mutex> lock(stream_lock_);
    if (queue_.empty()) {
      if (finished_.load()) {
        return Status::Cancelled("completed");
      }

      cv_progress_.wait(lock, [this] { return !queue_.empty() && !finished_.load(); });
    }

    auto out = std::move(queue_.front());
    queue_.pop();
    return out;
  }

 protected:
  static ucs_status_t RecvSchemaMsg(void* arg, const void* header, size_t header_length,
                                    void* data, size_t length,
                                    const ucp_am_recv_param_t* param) {
    UcxStreamReader* rdr = reinterpret_cast<UcxStreamReader*>(arg);
    DCHECK(!length);

    auto metadata =
        std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(header), header_length);
    auto msg = ipc::Message::Open(metadata, nullptr).ValueOrDie();

    std::lock_guard<std::mutex> lock(rdr->stream_lock_);
    rdr->schema_ = ipc::ReadSchema(*msg, &rdr->dictionary_memo_).ValueOrDie();
    return UCS_OK;
  }

  static ucs_status_t RecvRecordBatchMsg(void* arg, const void* header,
                                         size_t header_length, void* data, size_t length,
                                         const ucp_am_recv_param_t* param) {
    UcxStreamReader* rdr = reinterpret_cast<UcxStreamReader*>(arg);
    std::lock_guard<std::mutex> lock(rdr->stream_lock_);
    // TODO: check memory type and build buffers based on the memory type
    // so we can try passing data direct from GPU to GPU and avoid copies

    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
      auto metadata = std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(header),
                                               header_length);
      // UcxDataBuffer will free the data later, returning UCS_INPROGRESS
      // maintains the lifetime after the callback returns.
      auto body = std::make_shared<UcxDataBuffer>(rdr->cnxn_->worker(), data, length);
      // scatter gather IOV returns the buffers here as a single contiguous chunk
      // of data, which is perfect for processing IPC
      // could we get better performance by memh? is there a better way we should be doing this?
      auto msg = *ipc::Message::Open(metadata, body);
      auto rec = ipc::ReadRecordBatch(*msg, rdr->schema_, &rdr->dictionary_memo_, rdr->ipc_options_);      
      rdr->queue_.push(std::move(rec));
      return UCS_INPROGRESS;
    }

    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
      // handle rndv protocol
    }

    // handle 

    return UCS_ERR_IO_ERROR;
  }

  static ucs_status_t RecvEosMsg(void* arg, const void* header, size_t header_length,
                                 void* data, size_t length,
                                 const ucp_am_recv_param_t* param) {
    UcxStreamReader* rdr = reinterpret_cast<UcxStreamReader*>(arg);
    std::lock_guard<std::mutex> lock(rdr->stream_lock_);
    rdr->finished_.store(true);
    return UCS_OK;
  }

 private:
  Connection* cnxn_;
  std::atomic<bool> finished_{false};
  std::condition_variable cv_progress_;
  std::mutex stream_lock_;
  std::atomic<int64_t> counter_{0};
  std::queue<arrow::Result<std::shared_ptr<RecordBatch>>> queue_;

  ipc::IpcReadOptions ipc_options_;
  
  std::shared_ptr<Schema> schema_;    
  ipc::DictionaryMemo dictionary_memo_;
  bool swap_endian_;
};

class UcxStreamWriter {
 public:
  explicit UcxStreamWriter(Connection* cnxn)
      : cnxn_(cnxn), ipc_options_(ipc::IpcWriteOptions::Defaults()) {}

  Status Begin(const Schema& schema) {
    if (started_) {
      return Status::Invalid("writer has already been started");
    }
    started_ = true;

    RETURN_NOT_OK(mapper_.AddSchemaFields(schema));
    ipc::IpcPayload payload;
    RETURN_NOT_OK(ipc::GetSchemaPayload(schema, ipc_options_, mapper_, &payload));

    return WritePayload(payload);
  }

  Status WriteRecordBatch(const RecordBatch& batch) {
    if (!started_) {
      return Status::Invalid("writer has not been started yet");
    }

    RETURN_NOT_OK(EnsureDictsWritten(batch));
    ipc::IpcPayload payload;
    // the drawback here is that if there are any slices involved here
    // i.e. anything with a non-zero offset, then we're going to end up with
    // spurious allocations here to pass as IPC. is there a more optimal
    // way to handle this, particularly for non-cpu memory?
    RETURN_NOT_OK(ipc::GetRecordBatchPayload(batch, ipc_options_, &payload));
    RETURN_NOT_OK(WritePayload(payload));
    ++stats_.num_record_batches;
    return Status::OK();
  }

  ipc::WriteStats stats() const { return stats_; }

  Status Close() {
    if (!started_) {
      return Status::Invalid("close called on writer that was not started");
    }

    RETURN_NOT_OK(cnxn_->Flush());
    return cnxn_->SendAM(kEndStream, padding_bytes_.data(), 1);
  }

 private:
  Status EnsureDictsWritten(const RecordBatch& batch) {
    if (dicts_written_) {
      return Status::OK();
    }

    dicts_written_ = true;
    ARROW_ASSIGN_OR_RAISE(const auto dictionaries,
                          ipc::CollectDictionaries(batch, mapper_));
    for (const auto& pair : dictionaries) {
      ipc::IpcPayload payload;
      RETURN_NOT_OK(
          ipc::GetDictionaryPayload(pair.first, pair.second, ipc_options_, &payload));
      RETURN_NOT_OK(WritePayload(payload));
      ++stats_.num_dictionary_batches;
    }
    return Status::OK();
  }

  struct PendingIOV {
    std::vector<ucp_dt_iov_t> iovs;
    BufferVector body_buffers;
  };

  Status WritePayload(const ipc::IpcPayload& payload) {
    ++stats_.num_messages;
    unsigned int id = static_cast<unsigned int>(payload.type);
    const void* header = payload.metadata->data_as<void>();
    const size_t header_length = payload.metadata->size();

    auto pending = std::make_unique<PendingIOV>();

    int32_t total_buffers = 0;
    for (const auto& buffer : payload.body_buffers) {
      if (!buffer || buffer->size() == 0) continue;
      total_buffers++;
      // arrow ipc requires aligning buffers to 8 byte boundary
      const auto remainder = static_cast<int>(
          bit_util::RoundUpToMultipleOf8(buffer->size()) - buffer->size());
      if (remainder) total_buffers++;
    }
    pending->iovs.resize(total_buffers);
    ucp_dt_iov_t* iov = pending->iovs.data();
    pending->body_buffers = payload.body_buffers;

    for (const auto& buffer : payload.body_buffers) {
      if (!buffer || buffer->size() == 0) continue;

      iov->buffer = const_cast<void*>(reinterpret_cast<const void*>(buffer->address()));
      iov->length = buffer->size();
      ++iov;

      const auto remainder = static_cast<int>(
          bit_util::RoundUpToMultipleOf8(buffer->size()) - buffer->size());
      if (remainder) {
        iov->buffer =
            const_cast<void*>(reinterpret_cast<const void*>(padding_bytes_.data()));
        iov->length = remainder;
        ++iov;
      }
    }

    auto* pending_iov = pending.get();
    void* user_data = pending.release();
    // is IOV the optimal way to do this? is there a better way to do this
    // if we're dealing with other memory types?
    return cnxn_->SendAMIov(
        id, header, header_length, pending_iov->iovs.data(), pending_iov->iovs.size(),
        pending_iov, [](void* request, ucs_status_t status, void* user_data) {
          auto pending_iov = reinterpret_cast<PendingIOV*>(user_data);
          if (status != UCS_OK) {
            ARROW_LOG(WARNING)
                << FromUcsStatus("ucp_am_send_nbx callback", status).ToString();
          }
          delete pending_iov;
          ucp_request_free(request);
        });
  }

  Connection* cnxn_;

  ipc::IpcWriteOptions ipc_options_;
  ipc::DictionaryFieldMapper mapper_;
  ipc::WriteStats stats_;
  bool started_{false};
  bool dicts_written_{false};

  const std::array<uint8_t, 8> padding_bytes_{0, 0, 0, 0, 0, 0, 0, 0};
};

class Requester {
 public:
  explicit Requester(Connection* conn) : cnxn_{conn} {}

  arrow::Result<std::unique_ptr<UcxStreamReader>> GetData(flight::Ticket& tkt) {
    auto reader = std::make_unique<UcxStreamReader>(cnxn_);
    RETURN_NOT_OK(reader->SetHandlers());

    RETURN_NOT_OK(
        cnxn_->SendAM(kTicketAmHandlerId, tkt.ticket.data(), tkt.ticket.size()));
    return reader;
  }

 private:
  Connection* cnxn_;
};

class DataServer : public UcxServer {
 public:
  DataServer() : UcxServer() {
    auto st = ipc::test::MakeIntRecordBatch(&sample_record_batch_);
    if (!st.ok()) {
      st.Abort();
    }
  }

 protected:
  Status do_work(UcxServer::ClientWorker* worker) override {
    std::future<std::unique_ptr<Buffer>> fut;
    {
      std::lock_guard<std::mutex> lock(worker->queue_mutex_);
      if (worker->buffer_queue_.empty()) {
        return Status::OK();
      }
      fut = std::move(worker->buffer_queue_.front());
      worker->buffer_queue_.pop();
    }

    while (fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      ucp_worker_progress(worker->worker_->get());
    }

    auto buf = fut.get();
    std::cout << buf->ToString() << std::endl;

    UcxStreamWriter wr(worker->conn_.get());
    RETURN_NOT_OK(wr.Begin(*sample_record_batch_->schema()));
    RETURN_NOT_OK(wr.WriteRecordBatch(*sample_record_batch_));
    RETURN_NOT_OK(wr.Close());

    std::cout << "num messages sent: " << wr.stats().num_messages << std::endl;

    return Status::OK();
  }

  Status setup_handlers(UcxServer::ClientWorker* worker) override {
    ucp_am_handler_param_t handler_params;
    std::memset(&handler_params, 0, sizeof(handler_params));
    handler_params.field_mask =
        UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_CB |
        UCP_AM_HANDLER_PARAM_FIELD_FLAGS | UCP_AM_HANDLER_PARAM_FIELD_ARG;
    handler_params.id = kTicketAmHandlerId;
    handler_params.flags = UCP_AM_FLAG_PERSISTENT_DATA;
    handler_params.cb = HandleIncomingTicketMessage;
    handler_params.arg = worker;

    auto status = ucp_worker_set_am_recv_handler(worker->worker_->get(), &handler_params);
    RETURN_NOT_OK(FromUcsStatus("ucp_worker_set_am_recv_handler", status));
    return Status::OK();
  }

 private:
  std::shared_ptr<RecordBatch> sample_record_batch_;

  static ucs_status_t HandleIncomingTicketMessage(void* self, const void* header,
                                                  size_t header_length, void* data,
                                                  size_t data_length,
                                                  const ucp_am_recv_param_t* param) {
    ClientWorker* worker = reinterpret_cast<ClientWorker*>(self);
    std::promise<std::unique_ptr<Buffer>> p;
    {
      std::lock_guard<std::mutex> lock(worker->queue_mutex_);
      worker->buffer_queue_.push(p.get_future());
    }

    auto status = worker->conn_->RecvAM(std::move(p), header, header_length, data,
                                        data_length, param);
    return status;
  }
};
}  // namespace ucx

namespace flight {
class FlightServerWithUCXData : public FlightServerBase {
 public:
  explicit FlightServerWithUCXData(const std::string& data_server_loc)
      : FlightServerBase() {
    arrow::internal::Uri uri;
    auto st = uri.Parse(data_server_loc);
    if (!st.ok()) {
      st.Abort();
    }

    st = srv.Init(uri);
    if (!st.ok()) {
      st.Abort();
    }
  }

  ~FlightServerWithUCXData() override {
    auto st = srv.Shutdown();
    if (!st.ok()) {
      ARROW_LOG(WARNING) << "failed shutting down ucx data server: " << st.ToString();
    }
  }

  Status GetFlightInfo(const ServerCallContext& context, const FlightDescriptor& request,
                       std::unique_ptr<FlightInfo>* info) override {
    FlightEndpoint endpoint{{"ticket-ints"}, {srv.location()}, std::nullopt, {}};

    ARROW_ASSIGN_OR_RAISE(auto flightinfo,
                          FlightInfo::Make(*test_schema, request, {endpoint}, 10, -1));
    *info = std::make_unique<FlightInfo>(std::move(flightinfo));
    return Status::OK();
  }

  ucx::DataServer srv;
};

}  // namespace flight
}  // namespace arrow

int main(int argc, char** argv) {
  arrow::util::ArrowLog::StartArrowLog("ucxpoc", arrow::util::ArrowLogLevel::ARROW_DEBUG);

  auto flight_server =
      std::make_shared<arrow::flight::FlightServerWithUCXData>("ucx://127.0.0.1:0");
  auto location = *arrow::flight::Location::ForGrpcTcp("0.0.0.0", 0);
  arrow::flight::FlightServerOptions options(location);

  ARROW_CHECK_OK(flight_server->Init(options));
  ARROW_CHECK_OK(flight_server->SetShutdownOnSignals({SIGTERM}));

  std::thread serving(&arrow::flight::FlightServerWithUCXData::Serve,
                      flight_server.get());

  auto client = *arrow::flight::FlightClient::Connect(flight_server->location());

  auto info = *client->GetFlightInfo(arrow::flight::FlightDescriptor::Command("foobar"));
  std::cout << info->endpoints()[0].locations[0].ToString() << std::endl;

  arrow::ucx::UcxClient ucx_client;
  ARROW_CHECK_OK(ucx_client.Init(info->endpoints()[0].locations[0]));
  auto conn = *ucx_client.Get();

  auto tkt = info->endpoints()[0].ticket;
  arrow::ucx::Requester req{&conn};

  auto rdr = req.GetData(tkt).ValueOrDie();
  auto sc = *rdr->GetSchema();
  std::cout << sc->ToString() << std::endl;

  auto rec = rdr->ReadNext().ValueOrDie();
  std::cout << rec->ToString() << std::endl;

  // lots of "error during flush: Connection reset by remote peer"
  // i'm sure it's probably something i've overlooked but not worrying about it
  // for the purposes of this POC
  ARROW_CHECK_OK(ucx_client.Put(std::move(conn)));
  ARROW_CHECK_OK(ucx_client.Close());
  ARROW_CHECK_OK(flight_server->Shutdown());
  serving.join();
}