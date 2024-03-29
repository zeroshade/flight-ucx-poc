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
#include "random_generation.h"
#include "ucx_mmap_alloc.h"

#include <gflags/gflags.h>
#include <signal.h>
#include <ucp/api/ucp.h>
#include <future>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "arrow/buffer.h"
#include "arrow/flight/client.h"
#include "arrow/flight/server.h"
#include "arrow/gpu/cuda_api.h"
#include "arrow/ipc/test_common.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"
#include "arrow/util/uri.h"

static constexpr uint32_t kTicketAmHandlerId = 0xCAFE;
static constexpr uint32_t kEndStream = 0xDEADBEEF;
static constexpr ucp_tag_t kWantDataTag = 0xEBBEDDEADBA0BAB0;
static constexpr ucp_tag_t kFreeDataTag = 0x0000F4EEDA7A0000;

namespace arrow {

const std::shared_ptr<Schema> test_schema =
    ::arrow::schema({field("f0", int8()), field("f1", uint8()), field("f2", int16()),
                     field("f3", uint16()), field("f4", int32()), field("f5", uint32()),
                     field("f6", int64()), field("f7", uint64())});

namespace ucx {

static constexpr char kCPUData[] = "ticket-ints-cpu";
static constexpr char kGPUData[] = "ticket-ints-gpu";
static constexpr unsigned int RMA_RECORD_BATCH_ID = 42;

class UcxStreamReader {
 public:
  explicit UcxStreamReader(Connection* cnxn, std::string rkey)
      : cnxn_(cnxn), rkey_buf_{rkey}, ipc_options_(ipc::IpcReadOptions::Defaults()) {}

  void set_memory_mgr(std::shared_ptr<MemoryManager> mm) {
    if (!mm) {
      rndv_mem_mgr_ = CPUDevice::Instance()->default_memory_manager();
    } else {
      rndv_mem_mgr_ = std::move(mm);
    }
  }

  Status Start(flight::Ticket& tkt) {
    ucp_am_handler_param_t params;
    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ARG | UCP_AM_HANDLER_PARAM_FIELD_CB |
                        UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
    params.arg = this;
    params.id = 0;
    params.cb = RecvMsg;
    RETURN_NOT_OK(cnxn_->SetAMHandler(&params));
    RETURN_NOT_OK(cnxn_->SendTagSync(kWantDataTag, tkt.ticket.data(), tkt.ticket.size()));

    std::thread([this] {
      if (arrow::cuda::IsCudaMemoryManager(*rndv_mem_mgr_)) {
        auto ctx = *(*arrow::cuda::AsCudaMemoryManager(rndv_mem_mgr_))
                        ->cuda_device()
                        ->GetContext();
        cuCtxPushCurrent(reinterpret_cast<CUcontext>(ctx->handle()));
      }

      while (true) {
        if (cnxn_->MakeProgress()) {
          cv_progress_.notify_all();
        }
        if (finished_.load() && outstanding_rndv_.load() == 0) {
          // std::lock_guard lock(stream_lock_);
          break;
        }
        if (ucp_worker_wait(cnxn_->worker()->get()) != UCS_OK) {
          break;
        }
      }
    }).detach();

    std::thread([this] {
      if (arrow::cuda::IsCudaMemoryManager(*rndv_mem_mgr_)) {
        auto ctx = *(*arrow::cuda::AsCudaMemoryManager(rndv_mem_mgr_))
                        ->cuda_device()
                        ->GetContext();
        cuCtxPushCurrent(reinterpret_cast<CUcontext>(ctx->handle()));
      }

      while (true) {
        std::future<std::unique_ptr<Buffer>> buf;
        {
          std::unique_lock<std::mutex> lock(queue_mutex_);
          if (msg_queue_.empty()) {
            if (finished_.load()) {
              return;
            }
            cv_progress_.wait(lock,
                              [this] { return !msg_queue_.empty() || finished_.load(); });
          }
          buf = std::move(msg_queue_.front());
          msg_queue_.pop();
        }
        std::shared_ptr<Buffer> buffer = buf.get();
        auto metadata = SliceBuffer(buffer, 0, buffer->size() - 4);
        if (metadata->data_as<uint32_t>()[0] == 0xFFFFFFFF) {
          finished_.store(true);
          continue;
        }

        uint32_t sequence_number = BytesToUint32LE(buffer->data() + metadata->size());
        std::promise<std::unique_ptr<ipc::Message>> p;
        {
          std::lock_guard<std::mutex> lock(msg_mutex_);
          msg_map_.insert({sequence_number, p.get_future()});
        }
        cv_progress_.notify_all();

        auto msg = ipc::Message::Open(metadata, nullptr).ValueOrDie();
        if (!ipc::Message::HasBody(msg->type())) {
          p.set_value(std::move(msg));
          continue;
        }

        ucp_tag_t tag = (static_cast<uint64_t>(msg->type()) << 56) |
                        bit_util::ToLittleEndian(sequence_number);
        {
          std::lock_guard<std::mutex> lock(tag_polling_mutex_);
          tags_to_poll_.insert({tag, PendingMsg{std::move(p), std::move(metadata)}});
        }
        cv_progress_.notify_all();
      }
    }).detach();

    std::thread([this] {
      if (arrow::cuda::IsCudaMemoryManager(*rndv_mem_mgr_)) {
        auto ctx = *(*arrow::cuda::AsCudaMemoryManager(rndv_mem_mgr_))
                        ->cuda_device()
                        ->GetContext();
        cuCtxPushCurrent(reinterpret_cast<CUcontext>(ctx->handle()));
      }

      while (true) {
        {
          std::unique_lock<std::mutex> lock(tag_polling_mutex_);
          if (tags_to_poll_.empty()) {
            if (finished_.load()) {
              return;
            }
            cv_progress_.wait(
                lock, [this] { return !tags_to_poll_.empty() || finished_.load(); });
          }

          ucp_tag_recv_info_t info_tag;
          ucp_tag_message_h msg_tag;
          for (auto it = tags_to_poll_.begin(); it != tags_to_poll_.end();) {
            msg_tag = ucp_tag_probe_nb(cnxn_->worker()->get(), it->first,
                                       0xFF000000000000FF, 1, &info_tag);
            if (msg_tag != nullptr) {
              // got it! do something
              auto st = RecvTag(msg_tag, info_tag, std::move(it->second));
              if (!st.ok()) {
                st.Abort();
              }
              it = tags_to_poll_.erase(it);
            } else {
              ++it;
            }
          }
        }
        // cnxn_->MakeProgress();
      }
    }).detach();

    return Status::OK();
  }

  Status SetHandlers() {
    // ucp_am_handler_param_t params;
    // params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ARG | UCP_AM_HANDLER_PARAM_FIELD_ID
    // |
    //                     UCP_AM_HANDLER_PARAM_FIELD_CB |
    //                     UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
    // // params.flags = UCP_AM_FLAG_PERSISTENT_DATA;
    // params.arg = this;
    // params.id = static_cast<unsigned int>(ipc::MessageType::SCHEMA);
    // params.cb = RecvSchemaMsg;
    // RETURN_NOT_OK(cnxn_->SetAMHandler(&params));

    // params.id = static_cast<unsigned int>(ipc::MessageType::RECORD_BATCH);
    // params.cb = RecvRecordBatchMsg;
    // RETURN_NOT_OK(cnxn_->SetAMHandler(&params));

    // params.id = RMA_RECORD_BATCH_ID;
    // params.cb = RecvMappedRecordBatchMsg;
    // RETURN_NOT_OK(cnxn_->SetAMHandler(&params));

    // params.id = kEndStream;
    // params.cb = RecvEosMsg;
    // RETURN_NOT_OK(cnxn_->SetAMHandler(&params));

    return Status::OK();
  }

  arrow::Result<std::shared_ptr<Schema>> GetSchema() {
    if (schema_) {
      return schema_;
    }

    ARROW_ASSIGN_OR_RAISE(auto msg, ReadNextMsg());
    ARROW_ASSIGN_OR_RAISE(schema_, ipc::ReadSchema(*msg, &dictionary_memo_));
    return schema_;

    // std::thread([this] {
    //   if (arrow::cuda::IsCudaMemoryManager(*rndv_mem_mgr_)) {
    //     auto ctx = *(*arrow::cuda::AsCudaMemoryManager(rndv_mem_mgr_))
    //                     ->cuda_device()
    //                     ->GetContext();
    //     cuCtxPushCurrent(reinterpret_cast<CUcontext>(ctx->handle()));
    //   }
    //   while (true) {
    //     if (cnxn_->MakeProgress()) {
    //       cv_progress_.notify_all();
    //     }
    //     if (finished_.load()) {
    //       std::lock_guard lock(stream_lock_);
    //       if (queue_.empty() && outstanding_rndv_.load() == 0) {
    //         break;
    //       }
    //     }
    //   }
    // }).detach();

    // std::unique_lock<std::mutex> lock(stream_lock_);
    // if (schema_) {
    //   return schema_;
    // }

    // cv_progress_.wait(lock, [this] { return finished_.load() || schema_ != nullptr; });

    // return schema_;
  }

  arrow::Result<std::shared_ptr<RecordBatch>> ReadNext() {
    ARROW_ASSIGN_OR_RAISE(auto msg, ReadNextMsg());
    return ipc::ReadRecordBatch(*msg, schema_, &dictionary_memo_, ipc_options_);
    // std::future<processed_rb> out;
    // {
    //   std::unique_lock<std::mutex> lock(stream_lock_);
    //   if (queue_.empty()) {
    //     if (finished_.load()) {
    //       return Status::Cancelled("completed");
    //     }

    //     cv_progress_.wait(lock, [this] { return !queue_.empty() || finished_.load();
    //     });
    //   }

    //   out = std::move(queue_.front());
    //   queue_.pop();
    // }
    // return out.get();
  }

 protected:
  Result<std::unique_ptr<ipc::Message>> ReadNextMsg() {
    std::future<std::unique_ptr<ipc::Message>> futr;
    {
      std::unique_lock<std::mutex> lock(msg_mutex_);

      const uint32_t counter = next_counter_++;
      auto it = msg_map_.find(counter);
      if (it == msg_map_.end()) {
        cv_progress_.wait(lock, [this, counter, &it] {
          it = msg_map_.find(counter);
          return it != msg_map_.end() || finished_.load();
        });
      }
      futr = std::move(it->second);
      msg_map_.erase(it);
    }

    // while (futr.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    //   cnxn_->MakeProgress();
    // }

    return futr.get();
  }

  using processed_rb = arrow::Result<std::shared_ptr<RecordBatch>>;
  struct PendingDataRecv {
    std::promise<processed_rb> promise;
    std::shared_ptr<Buffer> metadata;
    std::unique_ptr<Buffer> buffer;
    UcxStreamReader* rdr;
  };

  struct PendingMsg {
    std::promise<std::unique_ptr<ipc::Message>> p;
    std::shared_ptr<Buffer> metadata;
    std::shared_ptr<Buffer> body;
    UcxStreamReader* rdr;
  };

  static constexpr uint64_t kbody_mask_ = 0x80000000000000;

  Status RecvTag(ucp_tag_message_h msg, ucp_tag_recv_info_t info_tag,
                 PendingMsg pending) {
    std::unique_ptr<Buffer> buf;
    if (info_tag.sender_tag & kbody_mask_) {
      ARROW_ASSIGN_OR_RAISE(
          buf, CPUDevice::Instance()->default_memory_manager()->AllocateBuffer(
                   info_tag.length));
    } else {
      ARROW_ASSIGN_OR_RAISE(buf, rndv_mem_mgr_->AllocateBuffer(info_tag.length));
    }
    ++outstanding_rndv_;

    PendingMsg* new_pending = new PendingMsg(std::move(pending));
    new_pending->body = std::move(buf);
    new_pending->rdr = this;
    return cnxn_->RecvTagData(
        msg, reinterpret_cast<void*>(new_pending->body->address()), info_tag.length,
        new_pending,
        [](void* request, ucs_status_t status, const ucp_tag_recv_info_t* tag_info,
           void* user_data) {
          PendingMsg* pending = reinterpret_cast<PendingMsg*>(user_data);
          if (status != UCS_OK) {
            ARROW_LOG(WARNING)
                << FromUcsStatus("ucp_tag_recv_nbx_callback", status).ToString();
            pending->p.set_value(nullptr);
            ucp_request_free(request);
            delete pending;
            return;
          }

          ucp_request_free(request);

          if (tag_info->sender_tag & kbody_mask_) {
            std::thread(&UcxStreamReader::RecvMappedBody, pending->rdr,
                        std::move(pending->p), std::move(pending->metadata),
                        std::move(pending->body))
                .detach();
          } else {
            auto msg = *ipc::Message::Open(pending->metadata, pending->body);
            pending->p.set_value(std::move(msg));
            --pending->rdr->outstanding_rndv_;
          }

          delete pending;
        },
        (new_pending->body->is_cpu()) ? UCS_MEMORY_TYPE_HOST : UCS_MEMORY_TYPE_CUDA);
  }

  void RecvMappedBody(std::promise<std::unique_ptr<ipc::Message>> p,
                      std::shared_ptr<Buffer> metadata, std::shared_ptr<Buffer> in_buf) {
    ucp_rkey_h rkey;
    auto status = ucp_ep_rkey_unpack(cnxn_->endpoint(), rkey_buf_.data(), &rkey);
    if (status != UCS_OK) {
      ucp_rkey_destroy(rkey);
      ARROW_LOG(WARNING) << FromUcsStatus("ucp_ep_rkey_unpack", status).ToString();
      return;
    }

    auto* buffers =
        reinterpret_cast<const std::pair<uint64_t, uint64_t>*>(in_buf->data_as<void>());
    uint64_t total_size = buffers[0].first;
    uint64_t num_bufs = buffers[0].second;
    buffers++;

    if (arrow::cuda::IsCudaMemoryManager(*rndv_mem_mgr_)) {
      auto ctx = *(*arrow::cuda::AsCudaMemoryManager(rndv_mem_mgr_))
                      ->cuda_device()
                      ->GetContext();
      cuCtxPushCurrent(reinterpret_cast<CUcontext>(ctx->handle()));
    }

    auto outbuf = *rndv_mem_mgr_->AllocateBuffer(total_size);
    void* outptr = reinterpret_cast<void*>(outbuf->mutable_address());
    if (outbuf->is_cpu()) {
      std::memset(outptr, 0, outbuf->size());
    } else {
      cuMemsetD8(outbuf->mutable_address(), 0, outbuf->size());
    }

    uint64_t offset = 0;

    ucp_request_param_t param;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE | UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send = logging_callback;
    param.memory_type = (outbuf->is_cpu()) ? UCS_MEMORY_TYPE_HOST : UCS_MEMORY_TYPE_CUDA;

    std::vector<uint64_t> addresses;
    for (auto* p = buffers; p != buffers + num_bufs; p++) {
      if (p->first == 0) {
        offset += p->second;
        continue;
      }

      addresses.push_back(p->first);
      param.user_data = reinterpret_cast<void*>(std::distance(buffers, p));
      auto status = ucp_get_nbx(cnxn_->endpoint(), UCS_PTR_BYTE_OFFSET(outptr, offset),
                                p->second, p->first, rkey, &param);
      if (UCS_PTR_IS_ERR(status)) {
        ucp_rkey_destroy(rkey);
        ARROW_LOG(WARNING)
            << FromUcsStatus("ucp_get_nbx", UCS_PTR_STATUS(status)).ToString();
        return;
      }
      offset += p->second;
    }

    auto flush_status = cnxn_->Flush();
    if (!flush_status.ok()) {
      ucp_rkey_destroy(rkey);
      ARROW_LOG(WARNING) << flush_status.ToString();
      return;
    }

    ucp_rkey_destroy(rkey);

    auto free_status = cnxn_->SendTagSync(kFreeDataTag, reinterpret_cast<void*>(addresses.data()),
                       addresses.size() * sizeof(uint64_t));
    if (!free_status.ok()) {
      ARROW_LOG(WARNING) << free_status.ToString();
    }

    auto msg = *ipc::Message::Open(metadata, std::move(outbuf));
    p.set_value(std::move(msg));
    --outstanding_rndv_;
  }

  static ucs_status_t RecvMsg(void* arg, const void* header, size_t header_length,
                              void* data, size_t length,
                              const ucp_am_recv_param_t* param) {
    UcxStreamReader* rdr = reinterpret_cast<UcxStreamReader*>(arg);
    DCHECK(length);

    std::promise<std::unique_ptr<Buffer>> p;
    {
      std::lock_guard<std::mutex> lock(rdr->queue_mutex_);
      rdr->msg_queue_.push(p.get_future());
    }

    return rdr->cnxn_->RecvAM(std::move(p), header, header_length, data, length, param);
  }

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

  static void logging_callback(void* request, ucs_status_t status, void* user_data) {
    ucp_request_free(request);
    long val = reinterpret_cast<long>(user_data);
    if (status == UCS_OK) {
      ARROW_LOG(DEBUG) << "logging_callback success: " << val;
    } else {
      ARROW_LOG(DEBUG) << "logging_callback " << val << " "
                       << FromUcsStatus("ucp_get_nbx", status).ToString();
    }
  }

  void process_mapped_buffers(std::promise<processed_rb> promise,
                              std::shared_ptr<Buffer> metadata,
                              std::shared_ptr<Buffer> buffer_info) {
    if (arrow::cuda::IsCudaMemoryManager(*rndv_mem_mgr_)) {
      auto ctx = *(*arrow::cuda::AsCudaMemoryManager(rndv_mem_mgr_))
                      ->cuda_device()
                      ->GetContext();
      cuCtxPushCurrent(reinterpret_cast<CUcontext>(ctx->handle()));
    }
    ucp_rkey_h rkey;
    auto status = ucp_ep_rkey_unpack(cnxn_->endpoint(), rkey_buf_.data(), &rkey);
    if (status != UCS_OK) {
      ucp_rkey_destroy(rkey);
      promise.set_value(FromUcsStatus("ucp_ep_rkey_unpack", status));
      return;
    }

    auto* buffers = reinterpret_cast<const std::pair<uint64_t, uint64_t>*>(
        buffer_info->data_as<void>());
    uint64_t total_size = buffers[0].first;
    uint64_t num_bufs = buffers[0].second;
    buffers++;

    auto outbuf = rndv_mem_mgr_->AllocateBuffer(total_size).ValueOrDie();
    void* outptr = reinterpret_cast<void*>(outbuf->mutable_address());
    if (outbuf->is_cpu()) {
      std::memset(outptr, 0, outbuf->size());
    } else {
      cuMemsetD8(outbuf->mutable_address(), 0, outbuf->size());
    }
    uint64_t offset = 0;

    ucp_request_param_t param;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE | UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send = logging_callback;
    param.memory_type = (outbuf->is_cpu()) ? UCS_MEMORY_TYPE_HOST : UCS_MEMORY_TYPE_CUDA;

    for (auto* p = buffers; p != buffers + num_bufs; p++) {
      if (p->first == 0) {
        offset += p->second;
        continue;
      }

      param.user_data = reinterpret_cast<void*>(std::distance(buffers, p));

      auto status = ucp_get_nbx(cnxn_->endpoint(), UCS_PTR_BYTE_OFFSET(outptr, offset),
                                p->second, p->first, rkey, &param);
      if (UCS_PTR_IS_ERR(status)) {
        ucp_rkey_destroy(rkey);
        promise.set_value(FromUcsStatus("ucp_get_nbx", UCS_PTR_STATUS(status)));
        return;
      }

      offset += p->second;
    }

    auto flush_status = cnxn_->Flush();
    if (!flush_status.ok()) {
      ucp_rkey_destroy(rkey);
      promise.set_value(flush_status);
      return;
    }

    ucp_rkey_destroy(rkey);

    auto msg = *ipc::Message::Open(metadata, std::move(outbuf));
    auto rec = ipc::ReadRecordBatch(*msg, schema_, &dictionary_memo_, ipc_options_);
    promise.set_value(std::move(rec));
    --outstanding_rndv_;
  }

  static ucs_status_t RecvMappedRecordBatchMsg(void* arg, const void* header,
                                               size_t header_length, void* data,
                                               size_t length,
                                               const ucp_am_recv_param_t* param) {
    UcxStreamReader* rdr = reinterpret_cast<UcxStreamReader*>(arg);
    std::lock_guard<std::mutex> lock(rdr->stream_lock_);

    std::promise<processed_rb> promise;
    rdr->queue_.push(promise.get_future());
    auto metadata =
        std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(header), header_length);
    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
      auto buffer_info =
          std::make_shared<UcxDataBuffer>(rdr->cnxn_->worker(), data, length);
      std::thread(&UcxStreamReader::process_mapped_buffers, rdr, std::move(promise),
                  std::move(metadata), std::move(buffer_info))
          .detach();
      return UCS_INPROGRESS;
    }

    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
      auto maybe_buffer = rdr->rndv_mem_mgr_->AllocateBuffer(length);
      if (!maybe_buffer.ok()) {
        ARROW_LOG(WARNING) << "failed mem mgr AllocateBuffer(" << length
                           << "):" << maybe_buffer.status().ToString();
        return UCS_ERR_NO_MEMORY;
      }

      PendingDataRecv* pending_recv = new PendingDataRecv{
          std::move(promise), std::move(metadata), maybe_buffer.MoveValueUnsafe()};

      void* dest = pending_recv->buffer->mutable_data_as<void>();
      ucp_request_param_t recv_param;
      recv_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
      recv_param.user_data = reinterpret_cast<void*>(pending_recv);
      recv_param.cb.recv_am = RecvDataMappedCB;

      void* request = ucp_am_recv_data_nbx(rdr->cnxn_->worker()->get(), data, dest,
                                           length, &recv_param);
      if (UCS_PTR_IS_ERR(request)) {
        ucs_status_t st = UCS_PTR_STATUS(request);
        pending_recv->promise.set_value(FromUcsStatus("ucp_am_recv_data_nbx", st));
        delete pending_recv;
        return st;
      } else if (!request) {
        RecvDataMappedCB(request, UCS_OK, length, recv_param.user_data);
        return UCS_OK;
      }

      ++rdr->outstanding_rndv_;
      return UCS_OK;
    }

    return UCS_ERR_IO_ERROR;
  }

  static ucs_status_t RecvRecordBatchMsg(void* arg, const void* header,
                                         size_t header_length, void* data, size_t length,
                                         const ucp_am_recv_param_t* param) {
    UcxStreamReader* rdr = reinterpret_cast<UcxStreamReader*>(arg);
    std::lock_guard<std::mutex> lock(rdr->stream_lock_);
    // TODO: check memory type and build buffers based on the memory type
    // so we can try passing data direct from GPU to GPU and avoid copies

    std::promise<processed_rb> promise;
    auto metadata =
        std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(header), header_length);
    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
      // UcxDataBuffer will free the data later, returning UCS_INPROGRESS
      // maintains the lifetime after the callback returns.
      auto body = std::make_shared<UcxDataBuffer>(rdr->cnxn_->worker(), data, length);
      // scatter gather IOV returns the buffers here as a single contiguous chunk
      // of data, which is perfect for processing IPC
      // could we get better performance by memh? is there a better way we should be doing
      // this?
      auto msg = *ipc::Message::Open(metadata, body);
      auto rec = ipc::ReadRecordBatch(*msg, rdr->schema_, &rdr->dictionary_memo_,
                                      rdr->ipc_options_);
      rdr->queue_.push(promise.get_future());
      promise.set_value(std::move(rec));
      return UCS_INPROGRESS;
    }

    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
      rdr->queue_.push(promise.get_future());

      // handle rndv protocol
      auto maybe_buffer = rdr->rndv_mem_mgr_->AllocateBuffer(length);
      if (!maybe_buffer.ok()) {
        ARROW_LOG(WARNING) << "failed mem mgr AllocateBuffer(" << length
                           << "):" << maybe_buffer.status().ToString();
        return UCS_ERR_NO_MEMORY;
      }

      PendingDataRecv* pending_recv = new PendingDataRecv{
          std::move(promise), std::move(metadata), maybe_buffer.MoveValueUnsafe(), rdr};
      void* dest = reinterpret_cast<void*>(pending_recv->buffer->address());

      ucp_request_param_t recv_param;
      recv_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                                UCP_OP_ATTR_FIELD_USER_DATA;
      recv_param.memory_type =
          (pending_recv->buffer->is_cpu()) ? UCS_MEMORY_TYPE_HOST : UCS_MEMORY_TYPE_CUDA;
      recv_param.cb.recv_am = RecvAMDataCB;
      recv_param.user_data = reinterpret_cast<void*>(pending_recv);

      void* request = ucp_am_recv_data_nbx(rdr->cnxn_->worker()->get(), data, dest,
                                           length, &recv_param);
      if (UCS_PTR_IS_ERR(request)) {
        ucs_status_t st = UCS_PTR_STATUS(request);
        pending_recv->promise.set_value(FromUcsStatus("ucp_am_recv_data_nbx", st));
        delete pending_recv;
        return st;
      } else if (!request) {
        auto msg =
            *ipc::Message::Open(pending_recv->metadata, std::move(pending_recv->buffer));
        auto rec = ipc::ReadRecordBatch(*msg, rdr->schema_, &rdr->dictionary_memo_,
                                        rdr->ipc_options_);
        pending_recv->promise.set_value(std::move(rec));
        delete pending_recv;
      }
      ++rdr->outstanding_rndv_;

      return UCS_OK;
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

  static void RecvDataMappedCB(void* request, ucs_status_t status, size_t length,
                               void* user_data) {
    PendingDataRecv* pending_recv = reinterpret_cast<PendingDataRecv*>(user_data);
    if (request) {
      ucp_request_free(request);
    }
    if (status != UCS_OK) {
      pending_recv->promise.set_value(
          FromUcsStatus("ucp_am_recv_data_nbx (callback)", status));
    } else {
      std::thread(&UcxStreamReader::process_mapped_buffers, pending_recv->rdr,
                  std::move(pending_recv->promise), std::move(pending_recv->metadata),
                  std::move(pending_recv->buffer))
          .detach();
    }

    delete pending_recv;
  }

  static void RecvAMDataCB(void* request, ucs_status_t status, size_t length,
                           void* user_data) {
    PendingDataRecv* pending_recv = reinterpret_cast<PendingDataRecv*>(user_data);
    ucp_request_free(request);
    if (status != UCS_OK) {
      pending_recv->promise.set_value(
          FromUcsStatus("ucp_am_recv_data_nbx (callback)", status));
      delete pending_recv;
    }

    auto msg =
        *ipc::Message::Open(pending_recv->metadata, std::move(pending_recv->buffer));
    auto rec = ipc::ReadRecordBatch(*msg, pending_recv->rdr->schema_,
                                    &pending_recv->rdr->dictionary_memo_,
                                    pending_recv->rdr->ipc_options_);
    pending_recv->promise.set_value(std::move(rec));
    delete pending_recv;
    --pending_recv->rdr->outstanding_rndv_;
  }

 private:
  Connection* cnxn_;
  std::string rkey_buf_;
  std::atomic<bool> finished_{false};
  std::condition_variable cv_progress_;
  std::mutex stream_lock_;
  std::atomic<int64_t> outstanding_rndv_{0};
  std::queue<std::future<processed_rb>> queue_;

  ipc::IpcReadOptions ipc_options_;

  std::shared_ptr<Schema> schema_;
  ipc::DictionaryMemo dictionary_memo_;
  bool swap_endian_;

  std::shared_ptr<MemoryManager> rndv_mem_mgr_{
      CPUDevice::Instance()->default_memory_manager()};

  std::mutex tag_polling_mutex_;
  std::unordered_map<ucp_tag_t, PendingMsg> tags_to_poll_;
  std::mutex queue_mutex_;
  std::queue<std::future<std::unique_ptr<Buffer>>> msg_queue_;
  std::mutex msg_mutex_;
  std::unordered_map<uint32_t, std::future<std::unique_ptr<ipc::Message>>> msg_map_;
  uint32_t counter_{0};
  uint32_t next_counter_{0};
};

class UcxStreamWriter {
 public:
  explicit UcxStreamWriter(Connection* cnxn)
      : cnxn_(cnxn), ipc_options_(ipc::IpcWriteOptions::Defaults()) {
    auto mgr = arrow::cuda::CudaDeviceManager::Instance().ValueOrDie();
    auto context = mgr->GetContext(0).ValueOrDie();

    cuda_padding_bytes_ = context->Allocate(8).ValueOrDie();
    cuMemsetD8(cuda_padding_bytes_->address(), 0, 8);
  }

  Status Begin(const Schema& schema) {
    if (started_) {
      return Status::Invalid("writer has already been started");
    }
    started_ = true;

    RETURN_NOT_OK(mapper_.AddSchemaFields(schema));
    ipc::IpcPayload payload;
    RETURN_NOT_OK(ipc::GetSchemaPayload(schema, ipc_options_, mapper_, &payload));

    return WriteIPC(payload);
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
    RETURN_NOT_OK(WriteIPC(payload));
    ++stats_.num_record_batches;
    return Status::OK();
  }

  Status WriteMappedRecordBatch(const RecordBatch& batch, BufferVector* buf_keep_alive) {
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

    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;

    ++stats_.num_messages;
    auto pending = std::make_unique<PendingIOV>();
    pending->iovs.resize(2);
    pending->iovs[0].buffer = const_cast<void*>(payload.metadata->data_as<void>());
    pending->iovs[0].length = payload.metadata->size();
    pending->body_buffers.emplace_back(payload.metadata);

    pending->iovs[1].buffer = malloc(4);
    pending->iovs[1].length = 4;
    Uint32ToBytesLE(sequence_num, reinterpret_cast<uint8_t*>(pending->iovs[1].buffer));

    auto* pending_iov = pending.get();
    void* user_data = pending.release();

    ARROW_RETURN_NOT_OK(cnxn_->SendAMIov(
        0, nullptr, 0, pending_iov->iovs.data(), pending_iov->iovs.size(), pending_iov,
        [](void* request, ucs_status_t status, void* user_data) {
          auto pending_iov = reinterpret_cast<PendingIOV*>(user_data);
          if (status != UCS_OK) {
            ARROW_LOG(WARNING) << FromUcsStatus("ucp_am_send_nbx_cb", status).ToString();
          }
          if (request) ucp_request_free(request);
          free(pending_iov->iovs[1].buffer);
          delete pending_iov;
        },
        mem_type));

    if (!payload.body_buffers.size()) {
      ++sequence_num;
      return Status::OK();
    }

    // unsigned int id = RMA_RECORD_BATCH_ID;
    // const void* header = payload.metadata->data_as<void>();
    // const size_t header_length = payload.metadata->size();

    std::vector<std::pair<uint64_t, uint64_t>> buffers;
    buffers.emplace_back(payload.body_length, 0);

    for (const auto& buffer : payload.body_buffers) {
      if (!buffer || buffer->size() == 0) continue;
      buffers.emplace_back(buffer->address(), buffer->size());
      if (buf_keep_alive) buf_keep_alive->emplace_back(buffer);
      const auto remainder = static_cast<int>(
          bit_util::RoundUpToMultipleOf8(buffer->size()) - buffer->size());
      if (remainder) {
        buffers.emplace_back(0, remainder);
      }
    }
    buffers[0].second = buffers.size() - 1;

    ucp_tag_t tag = (static_cast<uint64_t>(payload.type) << 56) | (uint64_t(1) << 55) |
                    bit_util::ToLittleEndian(sequence_num);
    ++sequence_num;
    auto* pending_buffers =
        new std::vector<std::pair<uint64_t, uint64_t>>(std::move(buffers));
    return cnxn_->SendTagData(
        tag, pending_buffers->data(), pending_buffers->size() * (2 * sizeof(uint64_t)),
        pending_buffers,
        [](void* request, ucs_status_t status, void* user_data) {
          auto pending =
              reinterpret_cast<std::vector<std::pair<uint64_t, uint64_t>>*>(user_data);
          delete pending;
          if (status != UCS_OK) {
            ARROW_LOG(WARNING) << FromUcsStatus("ucp_tag_send_nbx_cb", status).ToString();
          }
          if (request) {
            ucp_request_free(request);
          }
        },
        mem_type);
  }

  ipc::WriteStats stats() const { return stats_; }

  Status Close() {
    if (!started_) {
      return Status::Invalid("close called on writer that was not started");
    }

    RETURN_NOT_OK(cnxn_->Flush());
    std::array<uint8_t, 8> eos_bytes{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0};
    uint32_t num = bit_util::ToLittleEndian(sequence_num);
    std::memcpy(eos_bytes.data() + 4, reinterpret_cast<void*>(&num), 4);

    return cnxn_->SendAM(0, eos_bytes.data(), eos_bytes.size());
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

  Status WriteIPC(const ipc::IpcPayload& payload) {
    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;

    ++stats_.num_messages;

    auto pending = std::make_unique<PendingIOV>();
    pending->iovs.resize(2);
    pending->iovs[0].buffer = const_cast<void*>(payload.metadata->data_as<void>());
    pending->iovs[0].length = payload.metadata->size();
    pending->body_buffers.emplace_back(payload.metadata);

    pending->iovs[1].buffer = malloc(4);
    pending->iovs[1].length = 4;
    Uint32ToBytesLE(sequence_num, reinterpret_cast<uint8_t*>(pending->iovs[1].buffer));

    auto* pending_iov = pending.get();
    void* user_data = pending.release();

    ARROW_RETURN_NOT_OK(cnxn_->SendAMIov(
        0, nullptr, 0, pending_iov->iovs.data(), pending_iov->iovs.size(), pending_iov,
        [](void* request, ucs_status_t status, void* user_data) {
          auto pending_iov = reinterpret_cast<PendingIOV*>(user_data);
          if (status != UCS_OK) {
            ARROW_LOG(WARNING) << FromUcsStatus("ucp_am_send_nbx_cb", status).ToString();
          }
          if (request) ucp_request_free(request);
          free(pending_iov->iovs[1].buffer);
          delete pending_iov;
        },
        mem_type));

    if (!payload.body_buffers.size()) {
      ++sequence_num;
      return Status::OK();
    }

    pending = std::make_unique<PendingIOV>();
    int32_t total_buffers = 0;
    for (const auto& buffer : payload.body_buffers) {
      if (!buffer || buffer->size() == 0) continue;

      if (!buffer->is_cpu()) {
        mem_type = UCS_MEMORY_TYPE_CUDA;
      }

      total_buffers++;
      // arrow ipc requires aligning buffers to 8 byte boundary
      const auto remainder = static_cast<int>(
          bit_util::RoundUpToMultipleOf8(buffer->size()) - buffer->size());
      if (remainder) total_buffers++;
    }

    pending->iovs.resize(total_buffers);
    ucp_dt_iov_t* iov = pending->iovs.data();
    pending->body_buffers = payload.body_buffers;

    void* padding_bytes =
        const_cast<void*>(reinterpret_cast<const void*>(padding_bytes_.data()));
    if (mem_type == UCS_MEMORY_TYPE_CUDA) {
      padding_bytes = const_cast<void*>(
          reinterpret_cast<const void*>(cuda_padding_bytes_->address()));
    }

    for (const auto& buffer : payload.body_buffers) {
      if (!buffer || buffer->size() == 0) continue;

      iov->buffer = const_cast<void*>(reinterpret_cast<const void*>(buffer->address()));
      iov->length = buffer->size();
      ++iov;

      const auto remainder = static_cast<int>(
          bit_util::RoundUpToMultipleOf8(buffer->size()) - buffer->size());
      if (remainder) {
        iov->buffer = padding_bytes;
        iov->length = remainder;
        ++iov;
      }
    }

    pending_iov = pending.release();

    ucp_tag_t tag = (static_cast<uint64_t>(payload.type) << 56) | (uint64_t(0) << 55) |
                    bit_util::ToLittleEndian(sequence_num);
    ++sequence_num;
    return cnxn_->SendTagMsgIov(
        tag, pending_iov->iovs.data(), pending_iov->iovs.size(), pending_iov,
        [](void* request, ucs_status_t status, void* user_data) {
          auto pending_iov = reinterpret_cast<PendingIOV*>(user_data);
          if (status != UCS_OK) {
            ARROW_LOG(WARNING) << FromUcsStatus("ucp_tag_send_nbx_cb", status).ToString();
          }
          if (request) {
            ucp_request_free(request);
          }
          delete pending_iov;
        },
        mem_type);
  }

  struct PendingIOV {
    std::vector<ucp_dt_iov_t> iovs;
    BufferVector body_buffers;
  };

  Status WritePayload(const ipc::IpcPayload& payload) {
    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;

    ++stats_.num_messages;
    unsigned int id = static_cast<unsigned int>(payload.type);
    const void* header = payload.metadata->data_as<void>();
    const size_t header_length = payload.metadata->size();

    auto pending = std::make_unique<PendingIOV>();

    int32_t total_buffers = 0;
    for (const auto& buffer : payload.body_buffers) {
      if (!buffer || buffer->size() == 0) continue;

      if (!buffer->is_cpu()) {
        mem_type = UCS_MEMORY_TYPE_CUDA;
      }

      total_buffers++;
      // arrow ipc requires aligning buffers to 8 byte boundary
      const auto remainder = static_cast<int>(
          bit_util::RoundUpToMultipleOf8(buffer->size()) - buffer->size());
      if (remainder) total_buffers++;
    }

    pending->iovs.resize(total_buffers);
    ucp_dt_iov_t* iov = pending->iovs.data();
    pending->body_buffers = payload.body_buffers;

    void* padding_bytes =
        const_cast<void*>(reinterpret_cast<const void*>(padding_bytes_.data()));
    if (mem_type == UCS_MEMORY_TYPE_CUDA) {
      padding_bytes = const_cast<void*>(
          reinterpret_cast<const void*>(cuda_padding_bytes_->address()));
    }

    for (const auto& buffer : payload.body_buffers) {
      if (!buffer || buffer->size() == 0) continue;

      iov->buffer = const_cast<void*>(reinterpret_cast<const void*>(buffer->address()));
      iov->length = buffer->size();
      ++iov;

      const auto remainder = static_cast<int>(
          bit_util::RoundUpToMultipleOf8(buffer->size()) - buffer->size());
      if (remainder) {
        iov->buffer = padding_bytes;
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
        pending_iov,
        [](void* request, ucs_status_t status, void* user_data) {
          auto pending_iov = reinterpret_cast<PendingIOV*>(user_data);
          if (status != UCS_OK) {
            ARROW_LOG(WARNING)
                << FromUcsStatus("ucp_am_send_nbx callback", status).ToString();
          }
          delete pending_iov;
          ucp_request_free(request);
        },
        mem_type);
  }

  Connection* cnxn_;

  ipc::IpcWriteOptions ipc_options_;
  ipc::DictionaryFieldMapper mapper_;
  ipc::WriteStats stats_;
  bool started_{false};
  bool dicts_written_{false};

  const std::array<uint8_t, 8> padding_bytes_{0, 0, 0, 0, 0, 0, 0, 0};
  std::unique_ptr<cuda::CudaBuffer> cuda_padding_bytes_;

  uint32_t sequence_num{0};
};

class Requester {
 public:
  Requester(Connection* conn, std::shared_ptr<Device> device)
      : cnxn_{conn}, device_{std::move(device)} {}

  arrow::Result<std::unique_ptr<UcxStreamReader>> GetData(flight::Ticket& tkt,
                                                          std::string rkey) {
    auto reader = std::make_unique<UcxStreamReader>(cnxn_, rkey);
    // cnxn_->set_memory_manager(device_->default_memory_manager());
    reader->set_memory_mgr(device_->default_memory_manager());

    RETURN_NOT_OK(reader->SetHandlers());
    return reader;
  }

 private:
  Connection* cnxn_;
  std::shared_ptr<Device> device_{CPUDevice::Instance()};
};

class DataServer : public UcxServer {
 public:
  DataServer() : UcxServer() {}
  ~DataServer() {
    if (rkey_buffer != nullptr) {
      ucp_rkey_buffer_release(rkey_buffer);
      rkey_buffer = nullptr;
    }
    if (cuda_rkey_buffer_ != nullptr) {
      ucp_rkey_buffer_release(cuda_rkey_buffer_);
      cuda_rkey_buffer_ = nullptr;
    }
  }

  Status Init(const internal::Uri& uri) override {
    RETURN_NOT_OK(UcxServer::Init(uri));

    ARROW_ASSIGN_OR_RAISE(auto mapped_pool, arrow::ucx::UcxMappedPool::Make(
                                                ucp_context_->get(), 1024 * 1024 * 1024));

    RETURN_NOT_OK(MakeIntRecordBatch(&sample_record_batch_, mapped_pool.get()));
    mapped_pool_ = std::move(mapped_pool);

    ucs_status_t status = ucp_rkey_pack(
        ucp_context_->get(), mapped_pool_->get_mem_handle(), &rkey_buffer, &rkey_size);
    if (status != UCS_OK) {
      return FromUcsStatus("ucp_rkey_pack", status);
    }

    return copy_to_cuda();
  }

  Status Shutdown() override {
    sample_record_batch_.reset();
    ucs_status_t st = ucp_mem_unmap(ucp_context_->get(), mapped_pool_->get_mem_handle());
    if (st != UCS_OK) {
      return FromUcsStatus("ucp_mem_unmap", st);
    }
    mapped_pool_.reset();

    cuda_record_batch_.reset();
    st = ucp_mem_unmap(ucp_context_->get(), cuda_mem_handle_);
    if (st != UCS_OK) {
      return FromUcsStatus("ucp_mem_unmap cuda", st);
    }
    return UcxServer::Shutdown();
  }

  std::string_view get_rkey_buffer() const {
    return std::string_view{reinterpret_cast<char*>(rkey_buffer), rkey_size};
  }

  std::string_view get_cuda_rkey_buffer() const {
    return std::string_view{reinterpret_cast<char*>(cuda_rkey_buffer_), cuda_rkey_size_};
  }

 protected:
  Status copy_to_cuda() {
    ARROW_ASSIGN_OR_RAISE(auto mgr, arrow::cuda::CudaDeviceManager::Instance());
    ARROW_ASSIGN_OR_RAISE(auto context, mgr->GetContext(0));
    cuCtxPushCurrent(reinterpret_cast<CUcontext>(context->handle()));

    // for now the simplest thing to do is to just cuda serialize and read the
    // record batch batch onto the device. that'll give us a record batch
    // whose data is on the cuda device, and we can play from there.
    ARROW_ASSIGN_OR_RAISE(
        auto buf, cuda::SerializeRecordBatch(*sample_record_batch_, context.get()));

    ucp_mem_map_params_t params;
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE |
                        UCP_MEM_MAP_PARAM_FIELD_ADDRESS;
    params.length = buf->size();
    params.address = reinterpret_cast<void*>(buf->address());
    // params.flags = UCP_MEM_MAP_NONBLOCK;
    params.memory_type = UCS_MEMORY_TYPE_CUDA;

    auto status = ucp_mem_map(ucp_context_->get(), &params, &cuda_mem_handle_);
    if (status != UCS_OK) {
      return FromUcsStatus("ucp_mem_map cuda", status);
    }

    status = ucp_rkey_pack(ucp_context_->get(), cuda_mem_handle_, &cuda_rkey_buffer_,
                           &cuda_rkey_size_);
    if (status != UCS_OK) {
      return FromUcsStatus("ucp_memh_pack cuda", status);
    }

    ARROW_ASSIGN_OR_RAISE(
        cuda_record_batch_,
        cuda::ReadRecordBatch(sample_record_batch_->schema(), nullptr, buf));
    return context->Synchronize();
  }

  Status do_work(UcxServer::ClientWorker* worker) override {
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag;
    while (true) {
      msg_tag = ucp_tag_probe_nb(worker->worker_->get(), kWantDataTag, UINT64_MAX, 1,
                                 &info_tag);
      if (msg_tag != nullptr) {
        // we got one!
        break;
      } else if (ucp_worker_progress(worker->worker_->get())) {
        // some events polled, try again
        continue;
      }

      // ucp_worker_progress was 0, so we sleep
      // following blocked method used to polling internal file descriptor
      // to make CPU idle and not spin loop
      ARROW_RETURN_NOT_OK(
          FromUcsStatus("ucp_worker_wait", ucp_worker_wait(worker->worker_->get())));
    }

    auto mm = worker->conn_->memory_manager();
    ARROW_ASSIGN_OR_RAISE(auto msg_buf, mm->AllocateBuffer(info_tag.length));
    ARROW_RETURN_NOT_OK(worker->conn_->RecvTagData(
        msg_tag, reinterpret_cast<void*>(msg_buf->address()), info_tag.length, nullptr,
        nullptr, UCS_MEMORY_TYPE_HOST));

    std::cout << msg_buf->ToString() << std::endl;

    // std::future<std::unique_ptr<Buffer>> fut;
    // {
    //   std::lock_guard<std::mutex> lock(worker->queue_mutex_);
    //   if (worker->buffer_queue_.empty()) {
    //     return Status::OK();
    //   }
    //   fut = std::move(worker->buffer_queue_.front());
    //   worker->buffer_queue_.pop();
    // }

    // while (fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    //   ucp_worker_progress(worker->worker_->get());
    // }

    // auto buf = fut.get();
    // std::cout << buf->ToString() << std::endl;
    BufferVector buf_keep_alive;
    UcxStreamWriter wr(worker->conn_.get());
    if (msg_buf->ToString() == kCPUData) {
      RETURN_NOT_OK(wr.Begin(*sample_record_batch_->schema()));
      RETURN_NOT_OK(wr.WriteMappedRecordBatch(*sample_record_batch_, &buf_keep_alive));
      RETURN_NOT_OK(wr.WriteRecordBatch(*sample_record_batch_));
    } else if (msg_buf->ToString() == kGPUData) {
      RETURN_NOT_OK(wr.Begin(*cuda_record_batch_->schema()));
      RETURN_NOT_OK(wr.WriteMappedRecordBatch(*cuda_record_batch_, &buf_keep_alive));
      RETURN_NOT_OK(wr.WriteRecordBatch(*cuda_record_batch_));
    } else {
      return Status::Invalid("invalid argument");
    }

    RETURN_NOT_OK(wr.Close());
    std::cout << "num messages sent: " << wr.stats().num_messages << std::endl;

    std::vector<uint64_t> addresses;
    addresses.resize(buf_keep_alive.size());
    while (buf_keep_alive.size()) {
      ucp_tag_recv_info_t info_tag;
      ucp_tag_message_h msg_tag;
      while (true) {
        msg_tag = ucp_tag_probe_nb(worker->worker_->get(), kFreeDataTag, UINT64_MAX, 1,
                                   &info_tag);
        if (msg_tag != nullptr)
          break;
        else if (ucp_worker_progress(worker->worker_->get()))
          continue;

        ARROW_RETURN_NOT_OK(
            FromUcsStatus("ucp_worker_wait", ucp_worker_wait(worker->worker_->get())));
      }

      ARROW_RETURN_NOT_OK(worker->conn_->RecvTagData(
          msg_tag, reinterpret_cast<void*>(addresses.data()), info_tag.length, nullptr,
          nullptr, UCS_MEMORY_TYPE_HOST));
      size_t count = info_tag.length / sizeof(uint64_t);
      for (int i = 0; i < count; ++i) {        
        const uint64_t add = addresses[i];
        auto it = std::find_if(buf_keep_alive.begin(), buf_keep_alive.end(), [add](const std::shared_ptr<Buffer>& b) -> bool {
          return static_cast<const uint64_t>(b->address()) == add;
        });
        if (it != buf_keep_alive.end()) {
          buf_keep_alive.erase(it);
        }
      }
    }

    return Status::OK();
  }

  Status setup_handlers(UcxServer::ClientWorker* worker) override {
    // ucp_am_handler_param_t handler_params;
    // std::memset(&handler_params, 0, sizeof(handler_params));
    // handler_params.field_mask =
    //     UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_CB |
    //     UCP_AM_HANDLER_PARAM_FIELD_FLAGS | UCP_AM_HANDLER_PARAM_FIELD_ARG;
    // handler_params.id = kTicketAmHandlerId;
    // handler_params.flags = UCP_AM_FLAG_PERSISTENT_DATA;
    // handler_params.cb = HandleIncomingTicketMessage;
    // handler_params.arg = worker;

    // auto status = ucp_worker_set_am_recv_handler(worker->worker_->get(),
    // &handler_params); RETURN_NOT_OK(FromUcsStatus("ucp_worker_set_am_recv_handler",
    // status));
    return Status::OK();
  }

 private:
  std::shared_ptr<RecordBatch> sample_record_batch_;
  std::shared_ptr<RecordBatch> cuda_record_batch_;
  std::unique_ptr<UcxMappedPool> mapped_pool_;
  ucp_mem_h cuda_mem_handle_;
  void* rkey_buffer{nullptr};
  size_t rkey_size{0};

  void* cuda_rkey_buffer_{nullptr};
  size_t cuda_rkey_size_{0};

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
    std::string ticket, meta;
    if (request.cmd == "cpu") {
      ticket = ucx::kCPUData;
      meta = srv.get_rkey_buffer();
    } else if (request.cmd == "gpu") {
      ticket = ucx::kGPUData;
      meta = srv.get_cuda_rkey_buffer();
    }

    FlightEndpoint endpoint{{ticket}, {srv.location()}, std::nullopt, {}};

    ARROW_ASSIGN_OR_RAISE(
        auto flightinfo,
        FlightInfo::Make(*test_schema, request, {endpoint}, 10, -1, false, meta));
    *info = std::make_unique<FlightInfo>(std::move(flightinfo));
    return Status::OK();
  }

  ucx::DataServer srv;
};

}  // namespace flight

namespace cuda {
Result<std::shared_ptr<ArrayData>> CopyToHost(const ArrayData& data, MemoryManager& mm) {
  auto output = ArrayData::Make(data.type, data.length, data.null_count, data.offset);

  output->buffers.reserve(data.buffers.size());
  for (const auto& buf : data.buffers) {
    if (!buf) {
      output->buffers.push_back(nullptr);
      continue;
    }
    if (buf->size() == 0) {
      output->buffers.push_back(std::make_shared<Buffer>(nullptr, 0));
      continue;
    }
    ARROW_ASSIGN_OR_RAISE(auto temp_buf, mm.AllocateBuffer(buf->size()));
    auto cuda_buf = std::dynamic_pointer_cast<CudaBuffer>(buf);
    RETURN_NOT_OK(cuda_buf->CopyToHost(0, cuda_buf->size(), temp_buf->mutable_data()));
    output->buffers.push_back(std::move(temp_buf));
  }

  output->child_data.reserve(data.child_data.size());
  for (const auto& child : data.child_data) {
    ARROW_ASSIGN_OR_RAISE(auto copied, CopyToHost(*child, mm));
    output->child_data.push_back(std::move(copied));
  }

  if (data.dictionary) {
    ARROW_ASSIGN_OR_RAISE(output->dictionary, CopyToHost(*data.dictionary, mm));
  }

  return output;
}

Result<std::shared_ptr<Array>> CopyToHost(const Array& array, MemoryManager& mm) {
  ARROW_ASSIGN_OR_RAISE(auto copied_data, CopyToHost(*array.data(), mm));
  return MakeArray(copied_data);
}

Result<std::shared_ptr<RecordBatch>> CopyToHost(const RecordBatch& rb) {
  auto default_mem_manager = default_cpu_memory_manager();
  ArrayVector columns;
  columns.reserve(rb.num_columns());
  for (const auto& col : rb.columns()) {
    ARROW_ASSIGN_OR_RAISE(auto c, CopyToHost(*col, *default_mem_manager));
    columns.push_back(std::move(c));
  }

  return RecordBatch::Make(rb.schema(), rb.num_rows(), columns);
}
}  // namespace cuda

}  // namespace arrow

DEFINE_int32(port, 31337, "port to listen or connect");
DEFINE_string(address, "127.0.0.1", "address to connect to");
DEFINE_bool(gpu, false, "use gpu memory");
DEFINE_bool(client, false, "run the client");

arrow::Status run_server(const std::string& addr, const int port, const bool use_gpu) {
  const std::string command = use_gpu ? "gpu" : "cpu";
  auto flight_server =
      std::make_shared<arrow::flight::FlightServerWithUCXData>("ucx://127.0.0.1:0");
  auto location = *arrow::flight::Location::ForGrpcTcp(addr, port);
  arrow::flight::FlightServerOptions options(location);

  RETURN_NOT_OK(flight_server->Init(options));
  RETURN_NOT_OK(flight_server->SetShutdownOnSignals({SIGTERM}));

  std::cout << "Flight Server Listening on " << flight_server->location().ToString()
            << std::endl;

  return flight_server->Serve();
}

arrow::Status run_client(const std::string& addr, const int port, const bool use_gpu) {
  const std::string command = use_gpu ? "gpu" : "cpu";
  auto location = *arrow::flight::Location::ForGrpcTcp(addr, port);
  ARROW_ASSIGN_OR_RAISE(auto client, arrow::flight::FlightClient::Connect(location));

  ARROW_ASSIGN_OR_RAISE(
      auto info,
      client->GetFlightInfo(arrow::flight::FlightDescriptor::Command(command)));
  std::cout << info->endpoints()[0].locations[0].ToString() << std::endl;

  std::shared_ptr<arrow::Device> device = arrow::CPUDevice::Instance();
  if (use_gpu) {
    ARROW_ASSIGN_OR_RAISE(auto cuda_mgr, arrow::cuda::CudaDeviceManager::Instance());
    ARROW_ASSIGN_OR_RAISE(device, cuda_mgr->GetDevice(0));

    ARROW_ASSIGN_OR_RAISE(auto cuda_device, arrow::cuda::AsCudaDevice(device));
    ARROW_ASSIGN_OR_RAISE(auto ctx, cuda_device->GetContext());
    cuCtxPushCurrent(reinterpret_cast<CUcontext>(ctx->handle()));
  }

  std::cout << device->ToString() << std::endl;

  arrow::ucx::UcxClient ucx_client;
  RETURN_NOT_OK(ucx_client.Init(info->endpoints()[0].locations[0]));
  ARROW_ASSIGN_OR_RAISE(auto conn, ucx_client.Get());
  const auto rkey_buf = info->app_metadata();

  auto tkt = info->endpoints()[0].ticket;
  arrow::ucx::Requester req{&conn, device};

  ARROW_ASSIGN_OR_RAISE(auto rdr, req.GetData(tkt, rkey_buf));
  RETURN_NOT_OK(rdr->Start(tkt));
  ARROW_ASSIGN_OR_RAISE(auto sc, rdr->GetSchema());
  std::cout << sc->ToString() << std::endl;

  ARROW_ASSIGN_OR_RAISE(auto rec, rdr->ReadNext());
  if (command == "gpu") {
    ARROW_ASSIGN_OR_RAISE(rec, arrow::cuda::CopyToHost(*rec));
  }
  std::cout << rec->ToString() << std::endl;

  ARROW_ASSIGN_OR_RAISE(rec, rdr->ReadNext());
  if (command == "gpu") {
    ARROW_ASSIGN_OR_RAISE(rec, arrow::cuda::CopyToHost(*rec));
  }
  std::cout << rec->ToString() << std::endl;

  RETURN_NOT_OK(ucx_client.Put(std::move(conn)));
  RETURN_NOT_OK(ucx_client.Close());
  return arrow::Status::OK();
}

int main(int argc, char** argv) {
  arrow::util::ArrowLog::StartArrowLog("ucxpoc", arrow::util::ArrowLogLevel::ARROW_DEBUG);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::thread t1(run_server, FLAGS_address, FLAGS_port, FLAGS_gpu);

  using namespace std::chrono_literals;
  std::this_thread::sleep_for(2000ms);
  std::thread t2(run_client, FLAGS_address, FLAGS_port, FLAGS_gpu);
  t1.join();
  t2.join();

  // if (FLAGS_client) {
  //   ARROW_CHECK_OK(run_client(FLAGS_address, FLAGS_port, FLAGS_gpu));
  // } else {
  //   ARROW_CHECK_OK(run_server(FLAGS_address, FLAGS_port, FLAGS_gpu));
  // }
}
