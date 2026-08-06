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

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "gtest/gtest.h"

#include "../../lite_translator/include/berberis/lite_translator/lite_translate_region.h"
#include "berberis/assembler/loongarch64.h"
#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/test_utils/scoped_exec_region.h"

namespace berberis {
namespace {

TEST(LoongArch64RuntimeLibraryTest, RunsGeneratedCodeAndSynchronizesGuestPc) {
  InitHostEntries();

  ThreadState state{};
  constexpr GuestAddr kInitialPc = 0x1234'5000;
  SetInsnAddr(state.cpu, kInitialPc);
  state.cpu.x[0] = kEntryExitGeneratedCode;
  SetResidence(state, kOutsideGeneratedCode);

  constexpr int32_t kExitEntryOffset = offsetof(ThreadState, cpu) + offsetof(CPUState, x[0]);

  MachineCode code;
  loongarch64::Assembler as(&code);
  as.LdD(loongarch64::Assembler::t0, loongarch64::Assembler::s8, kExitEntryOffset);
  as.AddiD(loongarch64::Assembler::s7, loongarch64::Assembler::s7, 4);
  as.Jirl(loongarch64::Assembler::zero, loongarch64::Assembler::t0, 0);
  as.Finalize();

  ScopedExecRegion exec(&code);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));

  EXPECT_EQ(GetInsnAddr(state.cpu), kInitialPc + 4);
  EXPECT_EQ(GetResidence(state), kOutsideGeneratedCode);
}

template <size_t kSize>
void TranslateAndRun(const std::array<uint32_t, kSize>& guest_code, ThreadState* state) {
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(guest_code.data());
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(guest_code);
  params.allow_dispatch = false;
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  EXPECT_TRUE(success);
  EXPECT_GE(stop_pc, start_pc + sizeof(uint32_t));

  ScopedExecRegion exec(&code);
  SetInsnAddr(state->cpu, start_pc);
  SetResidence(*state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(state, AsHostCode(exec.GetHostCodeAddr()));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesMoveWideAndAddSubImmediate) {
  // movz x0, #0x1234; add x1, x0, #5; movk x1, #0xabcd, lsl #16;
  // sub w2, w1, #1
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0xd282'4680, 0x9100'1401, 0xf2b5'79a1, 0x5100'0422};

  ThreadState state{};
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(state.cpu.x[0], 0x1234u);
  EXPECT_EQ(state.cpu.x[1], 0xabcd'1239u);
  EXPECT_EQ(state.cpu.x[2], 0xabcd'1238u);
  EXPECT_EQ(GetInsnAddr(state.cpu), ToGuestAddr(kGuestCode.data() + kGuestCode.size()));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesCompareAndBranch) {
  // cbz x0, +8.  The following words only provide valid branch destinations;
  // the translated region ends at the conditional branch.
  constexpr std::array<uint32_t, 3> kGuestCode = {0xb400'0040, 0xd280'0021, 0xd280'0041};

  ThreadState zero_state{};
  TranslateAndRun(kGuestCode, &zero_state);
  EXPECT_EQ(GetInsnAddr(zero_state.cpu), ToGuestAddr(kGuestCode.data() + 2));

  ThreadState nonzero_state{};
  nonzero_state.cpu.x[0] = 1;
  TranslateAndRun(kGuestCode, &nonzero_state);
  EXPECT_EQ(GetInsnAddr(nonzero_state.cpu), ToGuestAddr(kGuestCode.data() + 1));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesBranchWithLink) {
  // bl +8; the link register receives the address after BL.
  constexpr std::array<uint32_t, 3> kGuestCode = {0x9400'0002, 0xd280'0020, 0xd280'0040};

  ThreadState state{};
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(state.cpu.x[30], ToGuestAddr(kGuestCode.data() + 1));
  EXPECT_EQ(GetInsnAddr(state.cpu), ToGuestAddr(kGuestCode.data() + 2));
}

}  // namespace
}  // namespace berberis
