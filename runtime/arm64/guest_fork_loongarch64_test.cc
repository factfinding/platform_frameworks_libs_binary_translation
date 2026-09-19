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

#include <sched.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "gtest/gtest.h"

#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_state/guest_state_opaque.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {
namespace {

class GuestForkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    InitHostEntries();
    thread_ = GuestThread::CreateForTest(CreateThreadState());
    ASSERT_NE(thread_, nullptr);
    cache_ = TranslationCache::GetInstance();
    cache_->InvalidateGuestRange(kPc, kPc + 0x1000);
  }

  void TearDown() override {
    cache_->InvalidateGuestRange(kPc, kPc + 0x1000);
    GuestThread::Destroy(thread_);
  }

  pid_t Clone() { return CloneGuestThread(thread_, SIGCHLD, 0, 0, 0, 0); }

  void ExpectChildSuccess(pid_t pid) {
    ASSERT_GT(pid, 0);
    // Bound even deadlocks before CloneGuestThread returns in the child.
    int status = 0;
    for (int i = 0; i < 300; ++i) {
      pid_t result = waitpid(pid, &status, WNOHANG);
      if (result == pid) {
        ASSERT_TRUE(WIFEXITED(status)) << status;
        EXPECT_EQ(WEXITSTATUS(status), 0);
        return;
      }
      ASSERT_EQ(result, 0);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    FAIL() << "fork child did not finish within 3 seconds";
  }

  static constexpr GuestAddr kPc = 0x3456'0000;
  GuestThread* thread_ = nullptr;
  TranslationCache* cache_ = nullptr;
};

TEST_F(GuestForkTest, ChildCanTranslateWhileAnotherThreadHeldCacheLock) {
  std::atomic<bool> locked{false};
  std::thread worker([&] {
    cache_->PrepareForFork();
    locked.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    cache_->FinishFork(false);
  });
  while (!locked.load()) {
    std::this_thread::yield();
  }
  pid_t pid = Clone();
  if (pid == 0) {
    auto* entry = cache_->AddAndLockForTranslation(kPc, 0);
    if (!entry) _exit(1);
    cache_->SetTranslatedAndUnlock(
        kPc, entry, 4, GuestCodeEntry::Kind::kInterpreted, {kEntryInterpret, 0});
    _exit(0);
  }
  worker.join();
  ExpectChildSuccess(pid);
  EXPECT_EQ(cache_->GetHostCodePtr(kPc)->load(), kEntryNotTranslated);
}

TEST_F(GuestForkTest, ChildDiscardsUnfinishedWorkAndPreservesFinishedEntries) {
  auto* invalidating = cache_->AddAndLockForTranslation(kPc, 0);
  ASSERT_NE(invalidating, nullptr);
  cache_->InvalidateGuestRange(kPc + 0x800, kPc + 0x804);
  auto* translating = cache_->AddAndLockForTranslation(kPc + 4, 0);
  auto* wrapping = cache_->AddAndLockForWrapping(kPc + 8);
  auto* finished = cache_->AddAndLockForTranslation(kPc + 12, 0);
  auto* wrapped = cache_->AddAndLockForWrapping(kPc + 16);
  ASSERT_NE(translating, nullptr);
  ASSERT_NE(wrapping, nullptr);
  ASSERT_NE(finished, nullptr);
  ASSERT_NE(wrapped, nullptr);
  cache_->SetTranslatedAndUnlock(
      kPc + 12, finished, 4, GuestCodeEntry::Kind::kInterpreted, {kEntryInterpret, 0});
  cache_->SetWrappedAndUnlock(kPc + 16, wrapped, true, {kEntryNoExec, 0});
  ASSERT_EQ(cache_->GetHostCodePtr(kPc)->load(), kEntryInvalidating);

  pid_t pid = Clone();
  if (pid == 0) {
    for (GuestAddr pc : {kPc, kPc + 4, kPc + 8}) {
      if (cache_->GetHostCodePtr(pc)->load() != kEntryNotTranslated) _exit(1);
      auto* entry = cache_->AddAndLockForTranslation(pc, 0);
      if (!entry) _exit(2);
      cache_->SetTranslatedAndUnlock(
          pc, entry, 4, GuestCodeEntry::Kind::kInterpreted, {kEntryInterpret, 0});
    }
    if (cache_->GetHostCodePtr(kPc + 12)->load() != kEntryInterpret) _exit(3);
    if (!cache_->IsHostFunctionWrapped(kPc + 16)) _exit(4);
    _exit(0);
  }
  ExpectChildSuccess(pid);
  // The parent must still be able to finish its original transactions.
  EXPECT_EQ(cache_->GetHostCodePtr(kPc)->load(), kEntryInvalidating);
  EXPECT_EQ(cache_->GetHostCodePtr(kPc + 4)->load(), kEntryTranslating);
  EXPECT_EQ(cache_->GetHostCodePtr(kPc + 8)->load(), kEntryWrapping);
  cache_->SetTranslatedAndUnlock(
      kPc, invalidating, 4, GuestCodeEntry::Kind::kInterpreted, {kEntryInterpret, 0});
  cache_->SetTranslatedAndUnlock(
      kPc + 4, translating, 4, GuestCodeEntry::Kind::kInterpreted, {kEntryInterpret, 0});
  cache_->SetWrappedAndUnlock(kPc + 8, wrapping, true, {kEntryNoExec, 0});
}

TEST_F(GuestForkTest, ChildHasFreshHostIdentityAndGuestTidWrites) {
  int parent_tid = -1;
  int child_tid = -1;
  pid_t parent_pid = getpid();
  pid_t parent_host_tid = gettid();
  pid_t pid = CloneGuestThread(thread_,
                               SIGCHLD | CLONE_PARENT_SETTID | CLONE_CHILD_SETTID,
                               0,
                               ToGuestAddr(&parent_tid),
                               0,
                               ToGuestAddr(&child_tid));
  if (pid == 0) {
    pid_t actual_tid = syscall(__NR_gettid);
    if (gettid() != actual_tid) _exit(1);
    if (getpid() != syscall(__NR_getpid)) _exit(2);
    if (child_tid != actual_tid) _exit(3);
    _exit(0);
  }
  ExpectChildSuccess(pid);
  EXPECT_EQ(parent_tid, pid);
  EXPECT_EQ(child_tid, -1);
  EXPECT_EQ(getpid(), parent_pid);
  EXPECT_EQ(gettid(), parent_host_tid);
}

TEST_F(GuestForkTest, FailedCloneReleasesCacheLockAndPreservesErrno) {
  // CLONE_THREAD without CLONE_SIGHAND/CLONE_VM is rejected by the kernel.
  errno = 0;
  pid_t pid = CloneGuestThread(thread_, CLONE_THREAD | SIGCHLD, 0, 0, 0, 0);
  int saved_errno = errno;
  ASSERT_EQ(pid, -1);
  EXPECT_EQ(saved_errno, EINVAL);
  auto* entry = cache_->AddAndLockForTranslation(kPc, 0);
  ASSERT_NE(entry, nullptr);
  cache_->SetTranslatedAndUnlock(
      kPc, entry, 4, GuestCodeEntry::Kind::kInterpreted, {kEntryInterpret, 0});
}

}  // namespace
}  // namespace berberis
