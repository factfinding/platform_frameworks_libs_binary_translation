/*
 * Copyright (C) 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <thread>
#include <vector>

#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/runtime/berberis.h"
#include "berberis/runtime_primitives/native_art_hooks.h"
#include "gtest/gtest.h"

namespace berberis {
namespace {
using ReadFn = ssize_t (*)(int, void*, size_t);
using MmapFn = void* (*)(void*, size_t, int, int, int, off_t);

ssize_t OriginalRead(int fd, void* buffer, size_t size) {
  if (fd == -1) {
    errno = EBADF;
    return -1;
  }
  if (size) *static_cast<uint8_t*>(buffer) = fd;
  return static_cast<ssize_t>(size);
}

void* OriginalMmap(void* address, size_t size, int prot, int flags, int fd, off_t offset) {
  EXPECT_EQ(address, reinterpret_cast<void*>(0x123456789000ULL));
  EXPECT_EQ(size, 0x200004000ULL);
  EXPECT_EQ(prot, PROT_READ);
  EXPECT_EQ(flags, MAP_PRIVATE);
  EXPECT_EQ(fd, -1);
  EXPECT_EQ(offset, 0x12345678000LL);
  return reinterpret_cast<void*>(0x76543210000ULL);
}

class NativeArtHooksTest : public ::testing::Test {
 protected:
  void SetUp() override {
    InitBerberis();
    size_ = getpagesize();
    code_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(code_, MAP_FAILED);
  }
  void TearDown() override {
    if (code_ != MAP_FAILED) {
      GuestMapShadow::GetInstance()->ClearExecutable(reinterpret_cast<uintptr_t>(code_), size_);
      munmap(code_, size_);
    }
  }
  uintptr_t Stub(NativeArtHookKind kind, uintptr_t original, bool increment_read_size = false) {
    // An ARM64 hook that changes an argument and tail-calls its saved native
    // original. Both directions must bridge; calling host code as ARM64 fails.
    const uint32_t code[] = {
        increment_read_size ? 0x91000442u : 0xd503201fu,  // add x2,x2,#1 / nop
        0x58000070,                                       // ldr x16, literal at +16
        0xd61f0200,                                       // br x16
        0xd503201f,
    };
    memcpy(code_, code, sizeof(code));
    memcpy(static_cast<char*>(code_) + 16, &original, sizeof(original));
    RegisterNativeArtHookOriginal(kind, original);
    GuestMapShadow::GetInstance()->SetExecutable(reinterpret_cast<uintptr_t>(code_), size_);
    return WrapNativeArtHook(kind, reinterpret_cast<uintptr_t>(code_));
  }
  void* code_ = MAP_FAILED;
  size_t size_ = 0;
};

TEST_F(NativeArtHooksTest, ReadHookAndOriginalRoundTrip) {
  const auto wrapper =
      Stub(NativeArtHookKind::kRead, reinterpret_cast<uintptr_t>(&OriginalRead), true);
  uint8_t byte = 0;
  EXPECT_EQ(reinterpret_cast<ReadFn>(wrapper)(37, &byte, 0x100000000ULL), 0x100000001LL);
  EXPECT_EQ(byte, 37);
}

TEST_F(NativeArtHooksTest, OriginalErrorAndErrno) {
  const auto wrapper = Stub(NativeArtHookKind::kRead, reinterpret_cast<uintptr_t>(&OriginalRead));
  errno = 0;
  EXPECT_EQ(reinterpret_cast<ReadFn>(wrapper)(-1, nullptr, 0), -1);
  EXPECT_EQ(errno, EBADF);
}

TEST_F(NativeArtHooksTest, MmapPreservesSixArgumentsAnd64BitResult) {
  const auto wrapper = Stub(NativeArtHookKind::kMmap, reinterpret_cast<uintptr_t>(&OriginalMmap));
  EXPECT_EQ(reinterpret_cast<MmapFn>(wrapper)(reinterpret_cast<void*>(0x123456789000ULL),
                                              0x200004000ULL,
                                              PROT_READ,
                                              MAP_PRIVATE,
                                              -1,
                                              0x12345678000LL),
            reinterpret_cast<void*>(0x76543210000ULL));
}

TEST_F(NativeArtHooksTest, NativeAndNullPointersAreUnchanged) {
  auto target = reinterpret_cast<uintptr_t>(&OriginalRead);
  EXPECT_EQ(WrapNativeArtHook(NativeArtHookKind::kRead, target), target);
  EXPECT_EQ(WrapNativeArtHook(NativeArtHookKind::kRead, 0), 0u);
  EXPECT_EQ(WrapNativeArtHook(NativeArtHookKind::kRead, reinterpret_cast<uintptr_t>(code_)),
            reinterpret_cast<uintptr_t>(code_));
}

TEST_F(NativeArtHooksTest, RepeatedSyncDoesNotWrapAWrapper) {
  const auto wrapper = Stub(NativeArtHookKind::kRead, reinterpret_cast<uintptr_t>(&OriginalRead));
  EXPECT_NE(wrapper, reinterpret_cast<uintptr_t>(code_));
  EXPECT_EQ(WrapNativeArtHook(NativeArtHookKind::kRead, wrapper), wrapper);
}

TEST_F(NativeArtHooksTest, CallbackWorksOnNativeThreads) {
  const auto wrapper = Stub(NativeArtHookKind::kRead, reinterpret_cast<uintptr_t>(&OriginalRead));
  std::vector<std::thread> threads;
  for (int i = 1; i <= 4; ++i) {
    threads.emplace_back([wrapper, i] {
      uint8_t byte = 0;
      EXPECT_EQ(reinterpret_cast<ReadFn>(wrapper)(i, &byte, 7), 7);
      EXPECT_EQ(byte, i);
    });
  }
  for (auto& thread : threads) thread.join();
}
}  // namespace
}  // namespace berberis
