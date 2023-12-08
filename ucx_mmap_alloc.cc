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

#include "ucx_mmap_alloc.h"
#include "flight_ucx_utils.h"

namespace {
// simple allocator based on github.com/CCareaga/heap_allocator
static constexpr int MIN_WILDERNESS = 0x2000;
static constexpr int MAX_WILDERNESS = 0x1000000;

static constexpr int MIN_ALLOC_SZ = 4;
static constexpr int BIN_COUNT = 9;
static constexpr int BIN_MAX_IDX = BIN_COUNT - 1;

struct node_t {
  unsigned int hole{0};
  unsigned int size{0};
  node_t* next{nullptr};
  node_t* prev{nullptr};
};

struct footer_t {
  node_t* header{nullptr};
};

footer_t* get_foot(node_t* head) {
  return reinterpret_cast<footer_t*>(reinterpret_cast<uint8_t*>(head) + sizeof(node_t) +
                                     head->size);
}

unsigned int get_bin_index(size_t sz) {
  unsigned int index = 0;
  sz = sz < 4 ? 4 : sz;
  while (sz >>= 1) index++;
  index -= 2;
  if (index > BIN_MAX_IDX) index = BIN_MAX_IDX;
  return index;
}

void create_foot(node_t* head) {
  footer_t* foot = get_foot(head);
  foot->header = head;
}

struct bin_t {
  node_t* head_{nullptr};

  void add_node(node_t* node) {
    node->next = node->prev = nullptr;

    if (head_ == nullptr) {
      head_ = node;
      return;
    }

    // save next and prev while we iterate
    node_t* current = head_;
    node_t* previous = nullptr;
    while (current != nullptr && current->size <= node->size) {
      previous = current;
      current = current->next;
    }

    if (current == nullptr) {  // reached end of the list
      previous->next = node;
      node->prev = previous;
    } else {
      if (previous != nullptr) {  // middle of list, connect all links
        node->next = current;
        previous->next = node;

        node->prev = previous;
        current->prev = node;
      } else {
        // head is the only element
        node->next = head_;
        head_->prev = node;
        head_ = node;
      }
    }
  }

  void remove_node(node_t* node) {
    if (head_ == nullptr) return;
    if (head_ == node) {
      head_ = head_->next;
      return;
    }

    node_t* temp = head_->next;
    while (temp != nullptr) {
      if (temp == node) {             // found the node
        if (temp->next == nullptr) {  // last item
          temp->prev->next = nullptr;
        } else {  // middle item
          temp->prev->next = temp->next;
          temp->next->prev = temp->prev;
        }
        // don't worry about deleting the head here because we already checked that
        return;
      }
      temp = temp->next;
    }
  }

  node_t* get_best_fit(size_t size) {
    if (head_ == nullptr) return nullptr;  // empty list
    node_t* temp = head_;
    while (temp != nullptr) {
      if (temp->size >= size) {
        return temp;  // found a fit
      }
      temp = temp->next;
    }
    return nullptr;  // no fit
  }

  node_t* get_last_node() {
    node_t* temp = head_;
    while (temp->next != nullptr) {
      temp = temp->next;
    }
    return temp;
  }
};

static constexpr unsigned int overhead = sizeof(footer_t) + sizeof(node_t);

struct pool_t {
  uintptr_t start_;
  uintptr_t end_;
  bin_t bins_[BIN_COUNT];

  pool_t(size_t init_size, uintptr_t start) : start_{start}, end_{start + init_size} {
    // pool starts as a single big chunk of mem
    node_t* init_region = reinterpret_cast<node_t*>(start_);
    init_region->hole = 1;
    init_region->size = init_size - sizeof(node_t) - sizeof(footer_t);

    create_foot(init_region);
    // add the region to the correct bin and setup our pool
    bins_[get_bin_index(init_region->size)].add_node(init_region);
  }

  void* alloc(size_t sz) {
    // get the bin index that this chunk size should be in
    auto idx = get_bin_index(sz);
    // use this bin to find a good fitting chunk
    auto temp = &bins_[idx];
    auto found = temp->get_best_fit(sz);

    // while there are no chunks found, advance through the bins
    // until we find a chunk or get to the wilderness
    while (found == nullptr) {
      if (idx + 1 >= BIN_COUNT) {
        return nullptr;
      }
      temp = &bins_[++idx];
      found = temp->get_best_fit(sz);
    }

    // if the difference between the found chunk and the requested chunk
    // is bigger than the overhead (metadata size) + the min alloc size
    // then we should split this chunk, otherwise just return the chunk
    if ((found->size - sz) > (overhead + MIN_ALLOC_SZ)) {
      // do the math to get where to split at, then set metadata
      node_t* split =
          reinterpret_cast<node_t*>(reinterpret_cast<uint8_t*>(found) + overhead + sz);
      split->size = found->size - sz - overhead;
      split->hole = 1;

      create_foot(split);  // create a footer for the split
      // get new index for this split chunk and place it in the correct bin
      unsigned int new_idx = get_bin_index(split->size);
      bins_[new_idx].add_node(split);

      // set the found chunks size and remake the foot
      found->size = sz;
      create_foot(found);
    }
    // not a hole anymore, remove from its bin
    found->hole = 0;
    bins_[idx].remove_node(found);

    // determine if we need to expand or contract the pool
    node_t* wild = get_wilderness();
    if (wild->size < MIN_WILDERNESS) {
      unsigned int success = expand(0x1000);
      if (!success) {
        return nullptr;
      }
    } else if (wild->size > MAX_WILDERNESS) {
      contract(0x1000);
    }

    // when the chunk is in use, we don't need prev and next fields
    // the address of the next field is the address we'll use to return
    found->prev = nullptr;
    found->next = nullptr;
    return &found->next;
  }

  static constexpr int node_offset = sizeof(unsigned int) * 2;

  void free(void* p) {
    bin_t* list;
    footer_t* new_foot;
    footer_t* old_foot;

    // the actual head of the node is not p, it is p minus the size of the
    // fields which preced "next" in the node structure.
    // if the node being freed is the start of the pool, then we don't need
    // to coalesce, so just put it in the right list
    node_t* head = reinterpret_cast<node_t*>(reinterpret_cast<uint8_t*>(p) - node_offset);
    if (reinterpret_cast<uintptr_t>(head) == start_) {
      head->hole = 1;
      bins_[get_bin_index(head->size)].add_node(head);
      return;
    }

    // these are the next and previous nodes in the pool, not the prev and
    // next in a bin. to find prev we just substract from the start of the
    // head node, to get the footer of the previous node (which gives us the
    // header pointer). to get the next node we simply get the footer and
    // add the sizeof(footer_t).
    node_t* next = reinterpret_cast<node_t*>(reinterpret_cast<uint8_t*>(get_foot(head)) +
                                             sizeof(footer_t));
    node_t* prev =
        reinterpret_cast<node_t*>(*(reinterpret_cast<uint8_t*>(head) - sizeof(footer_t)));

    // if the previous node is a hole we can coalesce
    if (prev->hole) {
      // remove the previous node from its bin
      list = &bins_[get_bin_index(prev->size)];
      list->remove_node(prev);

      // re-calculate the size of the node and recreate the footer
      prev->size += overhead + head->size;
      create_foot(prev);

      // previous is now the node we are working with, we head to prev
      // because the next if statement will coalesce with the next node
      // and we want that statement to work even when we coalesce with prev
      head = prev;
    }

    // if next node is free, coalesce
    if (next->hole) {
      // remove from its bin
      list = &bins_[get_bin_index(next->size)];
      list->remove_node(next);

      // recalculate new size of head
      head->size += overhead + next->size;

      // clear out old metadata from next
      old_foot = get_foot(next);
      old_foot->header = 0;
      next->size = 0;
      next->hole = 0;
      create_foot(head);
    }

    // this chunk is now a hole, so put it in the right bin
    head->hole = 1;
    bins_[get_bin_index(head->size)].add_node(head);
  }

  unsigned int expand(size_t sz) { return 0; }
  void contract(size_t sz) {}
  // relies on the end field being correct, simply uses the footer at
  // the end of the pool because that is always the wilderness
  node_t* get_wilderness() {
    footer_t* wild_foot =
        reinterpret_cast<footer_t*>(reinterpret_cast<uint8_t*>(end_) - sizeof(footer_t));
    return wild_foot->header;
  }
};

}  // namespace

namespace arrow {
namespace ucx {
struct UcxMappedPool::Impl {
  Impl(ucp_context_h ctx, ucp_mem_h handle, size_t initial, void* address,
       void* exported_memh)
      : ctx_{ctx},
        handle_(handle),
        exported_memh_buf_{exported_memh},
        pool_{initial, reinterpret_cast<uintptr_t>(address)} {}

  ~Impl() {    
  }

  ucp_context_h ctx_;
  ucp_mem_h handle_;
  void* exported_memh_buf_;
  pool_t pool_;
};

UcxMappedPool::UcxMappedPool(ucp_context_h ctx, ucp_mem_h handle, size_t length,
                             void* address, void* exported_memh)
    : initial_{length},
      impl_{std::make_unique<Impl>(ctx, handle, length, address, exported_memh)} {}

UcxMappedPool::~UcxMappedPool() {}

ucp_mem_h UcxMappedPool::get_mem_handle() const { return impl_->handle_; }
void* UcxMappedPool::get_exported_memh() const { return impl_->exported_memh_buf_; }

Result<std::unique_ptr<UcxMappedPool>> UcxMappedPool::Make(ucp_context_h ctx,
                                                           size_t initial) {
  ucp_mem_map_params_t params;
  ucp_mem_h mem_handle;

  params.field_mask = UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                      UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE |
                      UCP_MEM_MAP_PARAM_FIELD_ADDRESS;
  params.length = initial;
  params.address = nullptr;
  params.flags = UCP_MEM_MAP_ALLOCATE;
  params.memory_type = UCS_MEMORY_TYPE_HOST;

  auto status = ucp_mem_map(ctx, &params, &mem_handle);
  if (status != UCS_OK) {
    return FromUcsStatus("ucp_mem_map", status);
  }

  ucp_mem_attr_t attrs;
  attrs.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
  status = ucp_mem_query(mem_handle, &attrs);
  if (status != UCS_OK) {
    return FromUcsStatus("ucp_mem_query", status);
  }

  return std::make_unique<UcxMappedPool>(ctx, mem_handle, initial, attrs.address,
                                         const_cast<void*>(params.exported_memh_buffer));
}

Status UcxMappedPool::Allocate(int64_t size, int64_t alignment, uint8_t** out) {
  // ignore alignment for now
  auto ptr = impl_->pool_.alloc(size);
  if (!ptr) {
    return Status::Invalid("UcxMappedPool bad allocation");
  }

  *out = reinterpret_cast<uint8_t*>(ptr);
  allocated_ += size;
  ++num_allocs_;
  return Status::OK();
}

void UcxMappedPool::Free(uint8_t* buffer, int64_t size, int64_t alignment) {
  // ignore alignment for now
  auto ptr = reinterpret_cast<void*>(buffer);
  impl_->pool_.free(ptr);
  allocated_ -= size;
  --num_allocs_;
}

Status UcxMappedPool::Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                                 uint8_t** ptr) {
  // let's be simplistic for now and just allocate a new block
  // copy to it and free the old one
  // we're still ignoring alignment for the time being
  if (*ptr == nullptr) {
    return Allocate(new_size, alignment, ptr);
  }

  if (new_size == 0) {
    Free(*ptr, old_size, alignment);
    *ptr == nullptr;
    return Status::OK();
  }

  uint8_t* out;
  RETURN_NOT_OK(Allocate(new_size, alignment, &out));
  copy_data(out, *ptr, old_size);
  Free(*ptr, old_size, alignment);

  *ptr = out;
  return Status::OK();
}

void UcxMappedPool::copy_data(uint8_t* dest, uint8_t* src, size_t n) {
  std::memcpy(dest, src, n);
}

}  // namespace ucx
}  // namespace arrow