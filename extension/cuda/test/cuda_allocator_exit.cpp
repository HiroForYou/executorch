/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/cuda/cuda_allocator.h>

#include <cstdio>
#include <cstdlib>

using namespace executorch::extension::cuda;

namespace {
alignas(4096) char storage[8192];
void* rounded = nullptr;
executorch::runtime::DeviceAllocator* allocator = nullptr;
int frees = 0;

void free_at_exit() {
  frees = 0;
  allocator->deallocate(rounded, -1);
  if (frees != 1) {
    std::fprintf(stderr, "late deallocation did not free the original block\n");
    std::_Exit(1);
  }
}
} // namespace

// Stand-ins keep the test independent of GPU packing and driver teardown order.
extern "C" cudaError_t cudaMalloc(void** ptr, size_t) {
  *ptr = storage + 128;
  return cudaSuccess;
}

extern "C" cudaError_t cudaFree(void* ptr) {
  if (ptr != storage + 128) {
    std::fprintf(stderr, "late deallocation lost the original block\n");
    std::_Exit(1);
  }
  ++frees;
  return cudaSuccess;
}

int main() {
  // Registered first, so this runs after any destructible allocator state.
  if (std::atexit(free_at_exit) != 0) {
    return 1;
  }
  allocator = &CudaAllocator::instance();
  auto result = allocator->allocate(1024, -1, 4096);
  if (!result.ok() || result.get() != storage + 4096) {
    std::_Exit(1);
  }
  rounded = result.get();
  return 0;
}
