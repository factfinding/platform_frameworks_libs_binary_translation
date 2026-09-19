/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include <sys/mman.h>
#include <unistd.h>

#include <cstring>

#include "gtest/gtest.h"

#include "berberis/runtime_primitives/runtime_library.h"

#include "../../guest_loader/executable_segments.h"

namespace berberis {
namespace {

class LinkerExecutableSegmentsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    InitHostEntries();
    page_ = static_cast<size_t>(getpagesize());
    memory_ = mmap(nullptr, page_ * 4, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(memory_, MAP_FAILED);
    first_.l_addr = reinterpret_cast<uintptr_t>(memory_);
    second_.l_addr = first_.l_addr + page_ * 2;
    InitElf(first_);
    InitElf(second_);
  }

  void TearDown() override {
    tracker_.Sync(nullptr, &shadow_);
    if (memory_ != MAP_FAILED) {
      munmap(memory_, page_ * 4);
    }
  }

  void InitElf(const link_map& link) {
    auto* header = reinterpret_cast<Elf64_Ehdr*>(link.l_addr);
    memset(header, 0, page_);
    memcpy(header->e_ident, ELFMAG, SELFMAG);
    header->e_machine = EM_AARCH64;
    header->e_phoff = sizeof(Elf64_Ehdr);
    header->e_phentsize = sizeof(Elf64_Phdr);
    header->e_phnum = 1;
    auto* segment = reinterpret_cast<Elf64_Phdr*>(link.l_addr + header->e_phoff);
    segment->p_type = PT_LOAD;
    segment->p_flags = PF_R | PF_X;
    segment->p_vaddr = page_;
    segment->p_memsz = page_;
  }

  GuestAddr Code(const link_map& link) { return link.l_addr + page_; }
  bool Executable(const link_map& link) { return shadow_.IsExecutable(Code(link), page_); }

  size_t page_ = 0;
  void* memory_ = MAP_FAILED;
  link_map first_{};
  link_map second_{};
  GuestMapShadow shadow_;
  LinkerExecutableSegments tracker_;
};

TEST_F(LinkerExecutableSegmentsTest, ErasedHeaderSurvivesUnrelatedDlopen) {
  tracker_.Sync(&first_, &shadow_);
  ASSERT_TRUE(Executable(first_));
  memset(reinterpret_cast<void*>(first_.l_addr), 0, page_);
  first_.l_next = &second_;
  tracker_.Sync(&first_, &shadow_);
  EXPECT_TRUE(Executable(first_));
  EXPECT_TRUE(Executable(second_));
}

TEST_F(LinkerExecutableSegmentsTest, ChangedProgramHeadersDoNotReplaceLivePermissions) {
  tracker_.Sync(&first_, &shadow_);
  auto* header = reinterpret_cast<Elf64_Ehdr*>(first_.l_addr);
  auto* segment = reinterpret_cast<Elf64_Phdr*>(first_.l_addr + header->e_phoff);
  segment->p_flags = PF_R;
  first_.l_next = &second_;
  tracker_.Sync(&first_, &shadow_);
  EXPECT_TRUE(Executable(first_));
  EXPECT_TRUE(Executable(second_));
}

TEST_F(LinkerExecutableSegmentsTest, ExplicitPermissionRevocationIsPreserved) {
  tracker_.Sync(&first_, &shadow_);
  shadow_.ClearExecutable(Code(first_), page_);
  first_.l_next = &second_;
  tracker_.Sync(&first_, &shadow_);
  EXPECT_FALSE(Executable(first_));
  EXPECT_TRUE(Executable(second_));
}

TEST_F(LinkerExecutableSegmentsTest, ActualUnloadClearsCachedRangesAfterHeaderErasure) {
  first_.l_next = &second_;
  tracker_.Sync(&first_, &shadow_);
  memset(reinterpret_cast<void*>(first_.l_addr), 0, page_);
  tracker_.Sync(&second_, &shadow_);
  EXPECT_FALSE(Executable(first_));
  EXPECT_TRUE(Executable(second_));
}

TEST_F(LinkerExecutableSegmentsTest, ReloadMayReuseBothLinkNodeAndAddress) {
  tracker_.Sync(&first_, &shadow_);
  tracker_.Sync(nullptr, &shadow_);
  EXPECT_FALSE(Executable(first_));
  tracker_.Sync(&first_, &shadow_);
  EXPECT_TRUE(Executable(first_));
}

TEST_F(LinkerExecutableSegmentsTest, ReplacementAtSameAddressRegistersAfterRemoval) {
  tracker_.Sync(&first_, &shadow_);
  link_map replacement = first_;
  tracker_.Sync(&replacement, &shadow_);
  EXPECT_TRUE(Executable(replacement));
}

}  // namespace
}  // namespace berberis
