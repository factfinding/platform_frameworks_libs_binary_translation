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
#include <cstring>
#include <tuple>

#include "gtest/gtest.h"

#include "../../lite_translator/include/berberis/lite_translator/lite_translate_region.h"
#include "berberis/assembler/loongarch64.h"
#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/code_pool.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"
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

TEST(LoongArch64RuntimeLibraryTest, LiteDirectDispatchLinksCachedRegion) {
  // b +8; nop; <cached target>.  The source region must load the target's
  // TranslationCache slot and enter the generated target without unwinding
  // the RunGeneratedCode frame.
  constexpr std::array<uint32_t, 3> kGuestCode = {0x1400'0002, 0xd503'201f, 0xd503'201f};
  GuestAddr source_pc = ToGuestAddr(kGuestCode.data());
  GuestAddr target_pc = ToGuestAddr(kGuestCode.data() + 2);
  GuestAddr exit_pc = ToGuestAddr(kGuestCode.data() + 3);

  InitHostEntries();
  MachineCode target_code;
  loongarch64::Assembler target_as(&target_code);
  target_as.Li(loongarch64::Assembler::t0, 0x1234'5678'9abc'def0);
  target_as.StD(loongarch64::Assembler::t0,
                loongarch64::Assembler::s8,
                offsetof(ThreadState, cpu) + offsetof(CPUState, x[0]));
  target_as.Li(loongarch64::Assembler::s7, exit_pc);
  target_as.Li(loongarch64::Assembler::t0, kEntryExitGeneratedCode);
  target_as.Jirl(loongarch64::Assembler::zero, loongarch64::Assembler::t0, 0);
  target_as.Finalize();

  TranslationCache* cache = TranslationCache::GetInstance();
  GuestCodeEntry* target_entry = cache->AddAndLockForTranslation(target_pc, 0);
  ASSERT_NE(target_entry, nullptr);
  HostCodeAddr target_host_code = GetDefaultCodePoolInstance()->Add(&target_code);
  cache->SetTranslatedAndUnlock(target_pc,
                                target_entry,
                                sizeof(uint32_t),
                                GuestCodeEntry::Kind::kLiteTranslated,
                                {target_host_code, target_code.install_size()});

  MachineCode source_code;
  LiteTranslateParams params;
  params.end_pc = source_pc + sizeof(uint32_t);
  params.allow_dispatch = true;
  auto [success, stop_pc] = TryLiteTranslateRegion(source_pc, &source_code, params);
  ASSERT_TRUE(success);
  EXPECT_EQ(stop_pc, params.end_pc);
  ScopedExecRegion source_exec(&source_code);

  ThreadState state{};
  SetInsnAddr(state.cpu, source_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(source_exec.GetHostCodeAddr()));
  EXPECT_EQ(state.cpu.x[0], 0x1234'5678'9abc'def0u);
  EXPECT_EQ(GetInsnAddr(state.cpu), exit_pc);

  // A pending signal must force a normal generated-code exit at the branch
  // target instead of entering the cached region.
  ThreadState pending_state{};
  pending_state.pending_signals_status.store(kPendingSignalsPresent);
  SetInsnAddr(pending_state.cpu, source_pc);
  SetResidence(pending_state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&pending_state, AsHostCode(source_exec.GetHostCodeAddr()));
  EXPECT_EQ(pending_state.cpu.x[0], 0u);
  EXPECT_EQ(GetInsnAddr(pending_state.cpu), target_pc);

  cache->InvalidateGuestRange(target_pc, target_pc + sizeof(uint32_t));
}

TEST(LoongArch64RuntimeLibraryTest, LiteIndirectDispatchLinksCachedRegion) {
  // br x5; the target address is resolved through TranslationCache's two-level
  // table and must enter the cached generated region without a dispatcher
  // round trip.
  constexpr std::array<uint32_t, 2> kGuestCode = {0xd61f'00a0, 0xd503'201f};
  GuestAddr source_pc = ToGuestAddr(kGuestCode.data());
  GuestAddr target_pc = ToGuestAddr(kGuestCode.data() + 1);
  GuestAddr exit_pc = target_pc + sizeof(uint32_t);

  InitHostEntries();
  MachineCode target_code;
  loongarch64::Assembler target_as(&target_code);
  target_as.Li(loongarch64::Assembler::t0, 0xfedc'ba98'7654'3210);
  target_as.StD(loongarch64::Assembler::t0,
                loongarch64::Assembler::s8,
                offsetof(ThreadState, cpu) + offsetof(CPUState, x[0]));
  target_as.Li(loongarch64::Assembler::s7, exit_pc);
  target_as.Li(loongarch64::Assembler::t0, kEntryExitGeneratedCode);
  target_as.Jirl(loongarch64::Assembler::zero, loongarch64::Assembler::t0, 0);
  target_as.Finalize();

  TranslationCache* cache = TranslationCache::GetInstance();
  GuestCodeEntry* target_entry = cache->AddAndLockForTranslation(target_pc, 0);
  ASSERT_NE(target_entry, nullptr);
  HostCodeAddr target_host_code = GetDefaultCodePoolInstance()->Add(&target_code);
  cache->SetTranslatedAndUnlock(target_pc,
                                target_entry,
                                sizeof(uint32_t),
                                GuestCodeEntry::Kind::kLiteTranslated,
                                {target_host_code, target_code.install_size()});

  MachineCode source_code;
  LiteTranslateParams params;
  params.end_pc = source_pc + sizeof(uint32_t);
  params.allow_dispatch = true;
  auto [success, stop_pc] = TryLiteTranslateRegion(source_pc, &source_code, params);
  ASSERT_TRUE(success);
  EXPECT_EQ(stop_pc, params.end_pc);
  ScopedExecRegion source_exec(&source_code);

  ThreadState state{};
  state.cpu.x[5] = target_pc;
  SetInsnAddr(state.cpu, source_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(source_exec.GetHostCodeAddr()));
  EXPECT_EQ(state.cpu.x[0], 0xfedc'ba98'7654'3210u);
  EXPECT_EQ(GetInsnAddr(state.cpu), exit_pc);

  cache->InvalidateGuestRange(target_pc, target_pc + sizeof(uint32_t));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesShiftedIntegerOperations) {
  // add x2, x0, x1, lsl #3; sub w3, w2, w1, lsr #1;
  // orr x4, x2, x3; eor w5, w4, w3, lsr #4
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x8b01'0c02, 0x4b41'0443, 0xaa03'0044, 0x4a43'1085};

  ThreadState state{};
  state.cpu.x[0] = 5;
  state.cpu.x[1] = 7;
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(state.cpu.x[2], 61u);
  EXPECT_EQ(state.cpu.x[3], 58u);
  EXPECT_EQ(state.cpu.x[4], 63u);
  EXPECT_EQ(state.cpu.x[5], 60u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesPcRelativeAddressesAndNop) {
  // adr x6, +8; adrp x7, current page; nop
  constexpr std::array<uint32_t, 3> kGuestCode = {0x1000'0046, 0x9000'0007, 0xd503'201f};

  ThreadState state{};
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(state.cpu.x[6], ToGuestAddr(kGuestCode.data() + 2));
  EXPECT_EQ(state.cpu.x[7], ToGuestAddr(kGuestCode.data() + 1) & ~GuestAddr{0xfff});
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesRegisterBranches) {
  constexpr std::array<uint32_t, 3> kBrCode = {0xd61f'00a0, 0xd503'201f, 0xd503'201f};
  ThreadState br_state{};
  br_state.cpu.x[5] = ToGuestAddr(kBrCode.data() + 2);
  TranslateAndRun(kBrCode, &br_state);
  EXPECT_EQ(GetInsnAddr(br_state.cpu), ToGuestAddr(kBrCode.data() + 2));

  constexpr std::array<uint32_t, 3> kBlrCode = {0xd63f'00a0, 0xd503'201f, 0xd503'201f};
  ThreadState blr_state{};
  blr_state.cpu.x[5] = ToGuestAddr(kBlrCode.data() + 2);
  TranslateAndRun(kBlrCode, &blr_state);
  EXPECT_EQ(blr_state.cpu.x[30], ToGuestAddr(kBlrCode.data() + 1));
  EXPECT_EQ(GetInsnAddr(blr_state.cpu), ToGuestAddr(kBlrCode.data() + 2));

  constexpr std::array<uint32_t, 3> kRetCode = {0xd65f'03c0, 0xd503'201f, 0xd503'201f};
  ThreadState ret_state{};
  ret_state.cpu.x[30] = ToGuestAddr(kRetCode.data() + 2);
  TranslateAndRun(kRetCode, &ret_state);
  EXPECT_EQ(GetInsnAddr(ret_state.cpu), ToGuestAddr(kRetCode.data() + 2));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesUnsignedImmediateLoadsAndStores) {
  // ldr x2, [x0, #8]; str x2, [x0, #16];
  // ldr w3, [x0, #4]; str w3, [x0]
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0xf940'0402, 0xf900'0802, 0xb940'0403, 0xb900'0003};
  std::array<uint64_t, 3> memory = {0x1122'3344'5566'7788, 0xaabb'ccdd'eeff'0011, 0};

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data());
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(state.cpu.x[2], 0xaabb'ccdd'eeff'0011u);
  EXPECT_EQ(state.cpu.x[3], 0x1122'3344u);
  EXPECT_EQ(memory[0], 0x1122'3344'1122'3344u);
  EXPECT_EQ(memory[2], 0xaabb'ccdd'eeff'0011u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesByteAndHalfwordMemory) {
  // ldrb w2, [x0, #1]; strb w2, [x0, #2];
  // ldrh w3, [x0, #2]; strh w3, [x0, #4]
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x3940'0402, 0x3900'0802, 0x7940'0403, 0x7900'0803};
  std::array<uint8_t, 8> memory = {0x11, 0xab, 0x34, 0x12, 0, 0, 0, 0};

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data());
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[2], 0xabu);
  EXPECT_EQ(memory[2], 0xabu);
  EXPECT_EQ(state.cpu.x[3], 0x12abu);
  EXPECT_EQ(memory[4], 0xabu);
  EXPECT_EQ(memory[5], 0x12u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteMemoryIgnoresPointerTopByte) {
  // ldr x2, [x0, #8]; str x2, [x0, #16]
  constexpr std::array<uint32_t, 2> kGuestCode = {0xf940'0402, 0xf900'0802};
  std::array<uint64_t, 3> memory = {0, 0x1122'3344'5566'7788, 0};

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data()) | 0xab00'0000'0000'0000ULL;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[2], memory[1]);
  EXPECT_EQ(memory[2], memory[1]);
  // TBI affects the memory access, not the architectural register value.
  EXPECT_EQ(state.cpu.x[0], ToGuestAddr(memory.data()) | 0xab00'0000'0000'0000ULL);
}

TEST(LoongArch64RuntimeLibraryTest, LiteMemorySupportsUnalignedAddresses) {
  // ldr x1, [x0]; str x1, [x0, #8]
  constexpr std::array<uint32_t, 2> kGuestCode = {0xf940'0001, 0xf900'0401};
  std::array<uint8_t, 17> memory{};
  constexpr uint64_t kValue = 0x0123'4567'89ab'cdef;
  memcpy(memory.data() + 1, &kValue, sizeof(kValue));

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data() + 1);
  TranslateAndRun(kGuestCode, &state);

  uint64_t stored = 0;
  memcpy(&stored, memory.data() + 9, sizeof(stored));
  EXPECT_EQ(state.cpu.x[1], kValue);
  EXPECT_EQ(stored, kValue);
}

TEST(LoongArch64RuntimeLibraryTest, LiteMemoryInstructionsHaveRecoveryPoints) {
  // ldr x2, [x0]; str x2, [x0, #8]; ldp x3, x4, [x0]; stp x3, x4, [x0]
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0xf940'0002, 0xf900'0402, 0xa940'1003, 0xa900'1003};
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;

  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  EXPECT_EQ(stop_pc, params.end_pc);

  ScopedExecRegion exec(&code);
  // One recovery entry for each scalar access and two for each pair.
  EXPECT_EQ(exec.recovery_map().size(), 6u);
  for (const auto& [fault_pc, recovery_pc] : exec.recovery_map()) {
    EXPECT_NE(fault_pc, 0u);
    EXPECT_NE(recovery_pc, 0u);
    EXPECT_NE(fault_pc, recovery_pc);
  }
}

TEST(LoongArch64RuntimeLibraryTest, GuestMemoryCanBeKeptInInterpreter) {
  // ldr x1, [x0]
  constexpr std::array<uint32_t, 1> kGuestCode = {0xf940'0001};
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;
  params.enable_guest_memory = false;

  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);

  EXPECT_FALSE(success);
  EXPECT_EQ(stop_pc, start_pc);
}

TEST(LoongArch64RuntimeLibraryTest, LiteRejectsSimdLoadsAndStores) {
  // ldr d0, [x1]; str d2, [x3]; ldr d4, [x5], #8; str d6, [x7, #-8]!
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0xfd40'0020, 0xfd00'0062, 0xfc40'84a4, 0xfc1f'8ce6};

  for (const uint32_t& insn : kGuestCode) {
    GuestAddr start_pc = ToGuestAddr(&insn);
    MachineCode code;
    LiteTranslateParams params;
    params.end_pc = start_pc + sizeof(insn);
    params.allow_dispatch = false;

    auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);

    EXPECT_FALSE(success) << std::hex << insn;
    EXPECT_EQ(stop_pc, start_pc) << std::hex << insn;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteRejectsConstrainedUnpredictableLoadPair) {
  // ldp x1, x1, [x0] has overlapping destination registers and is
  // CONSTRAINED UNPREDICTABLE.  Do not assign an arbitrary JIT meaning to it.
  constexpr uint32_t kInsn = 0xa940'0401;
  GuestAddr start_pc = ToGuestAddr(&kInsn);
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kInsn);
  params.allow_dispatch = false;

  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);

  EXPECT_FALSE(success);
  EXPECT_EQ(stop_pc, start_pc);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesPreAndPostIndexedMemory) {
  // str x1, [x0, #-8]!; ldr x2, [x0], #8
  constexpr std::array<uint32_t, 2> kGuestCode = {0xf81f'8c01, 0xf840'8402};
  std::array<uint64_t, 2> memory{};

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data() + 1);
  state.cpu.x[1] = 0x1234'5678'9abc'def0;
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(memory[0], state.cpu.x[1]);
  EXPECT_EQ(state.cpu.x[2], state.cpu.x[1]);
  EXPECT_EQ(state.cpu.x[0], ToGuestAddr(memory.data() + 1));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesIndexedByteMemory) {
  // strb w1, [x0, #-1]!; ldrb w2, [x0], #1
  constexpr std::array<uint32_t, 2> kGuestCode = {0x381f'fc01, 0x3840'1402};
  std::array<uint8_t, 2> memory{};

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data() + 1);
  state.cpu.x[1] = 0x1234'5678'9abc'def0;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(memory[0], 0xf0u);
  EXPECT_EQ(state.cpu.x[2], 0xf0u);
  EXPECT_EQ(state.cpu.x[0], ToGuestAddr(memory.data() + 1));
}

TEST(LoongArch64RuntimeLibraryTest, LiteIndexedMemoryPreservesPointerTagOnWriteback) {
  // str x1, [x0, #-8]!; ldr x2, [x0], #8
  constexpr std::array<uint32_t, 2> kGuestCode = {0xf81f'8c01, 0xf840'8402};
  std::array<uint64_t, 2> memory{};
  constexpr uint64_t kTag = 0xcd00'0000'0000'0000ULL;

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data() + 1) | kTag;
  state.cpu.x[1] = 0x1234'5678'9abc'def0;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(memory[0], state.cpu.x[1]);
  EXPECT_EQ(state.cpu.x[2], state.cpu.x[1]);
  EXPECT_EQ(state.cpu.x[0], ToGuestAddr(memory.data() + 1) | kTag);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesStackRegisterPairs) {
  // stp x29, x30, [sp, #-16]!; ldp x29, x30, [sp], #16
  constexpr std::array<uint32_t, 2> kGuestCode = {0xa9bf'7bfd, 0xa8c1'7bfd};
  std::array<uint64_t, 4> stack{};

  ThreadState state{};
  state.cpu.sp = ToGuestAddr(stack.data() + 2);
  state.cpu.x[29] = 0x1122'3344'5566'7788;
  state.cpu.x[30] = 0x8877'6655'4433'2211;
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(stack[0], state.cpu.x[29]);
  EXPECT_EQ(stack[1], state.cpu.x[30]);
  EXPECT_EQ(state.cpu.sp, ToGuestAddr(stack.data() + 2));
}

TEST(LoongArch64RuntimeLibraryTest, LitePairMemoryIgnoresPointerTopByte) {
  // stp x1, x2, [x0]; ldp x3, x4, [x0]
  constexpr std::array<uint32_t, 2> kGuestCode = {0xa900'0801, 0xa940'1003};
  std::array<uint64_t, 2> memory{};

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data()) | 0xef00'0000'0000'0000ULL;
  state.cpu.x[1] = 0x0123'4567'89ab'cdef;
  state.cpu.x[2] = 0xfedc'ba98'7654'3210;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(memory[0], state.cpu.x[1]);
  EXPECT_EQ(memory[1], state.cpu.x[2]);
  EXPECT_EQ(state.cpu.x[3], state.cpu.x[1]);
  EXPECT_EQ(state.cpu.x[4], state.cpu.x[2]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesCmpAndConditionalBranch) {
  // cmp x0, #5; b.eq +8
  constexpr std::array<uint32_t, 4> kEqCode = {0xf100'141f, 0x5400'0040, 0xd280'0021, 0xd280'0041};
  ThreadState equal_state{};
  equal_state.cpu.x[0] = 5;
  TranslateAndRun(kEqCode, &equal_state);
  EXPECT_EQ(equal_state.cpu.flags, 6u);  // Z | C
  EXPECT_EQ(GetInsnAddr(equal_state.cpu), ToGuestAddr(kEqCode.data() + 3));

  ThreadState unequal_state{};
  unequal_state.cpu.x[0] = 3;
  TranslateAndRun(kEqCode, &unequal_state);
  EXPECT_EQ(unequal_state.cpu.flags, 8u);  // N
  EXPECT_EQ(GetInsnAddr(unequal_state.cpu), ToGuestAddr(kEqCode.data() + 2));

  // cmp x0, #1; b.lt +8.  INT64_MIN - 1 overflows to INT64_MAX,
  // so signed LT is true because N != V.
  constexpr std::array<uint32_t, 4> kLtCode = {0xf100'041f, 0x5400'004b, 0xd280'0021, 0xd280'0041};
  ThreadState overflow_state{};
  overflow_state.cpu.x[0] = uint64_t{1} << 63;
  TranslateAndRun(kLtCode, &overflow_state);
  EXPECT_EQ(overflow_state.cpu.flags, 3u);  // C | V
  EXPECT_EQ(GetInsnAddr(overflow_state.cpu), ToGuestAddr(kLtCode.data() + 3));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesFlagSettingRegisterOperations) {
  // adds w2, w0, w1
  constexpr std::array<uint32_t, 1> kAddsCode = {0x2b01'0002};
  ThreadState carry_state{};
  carry_state.cpu.x[0] = UINT32_MAX;
  carry_state.cpu.x[1] = 1;
  TranslateAndRun(kAddsCode, &carry_state);
  EXPECT_EQ(carry_state.cpu.x[2], 0u);
  EXPECT_EQ(carry_state.cpu.flags, 6u);  // Z | C

  ThreadState overflow_state{};
  overflow_state.cpu.x[0] = 0x7fff'ffff;
  overflow_state.cpu.x[1] = 1;
  TranslateAndRun(kAddsCode, &overflow_state);
  EXPECT_EQ(overflow_state.cpu.x[2], 0x8000'0000u);
  EXPECT_EQ(overflow_state.cpu.flags, 9u);  // N | V

  // cmp x0, x1
  constexpr std::array<uint32_t, 1> kCmpCode = {0xeb01'001f};
  ThreadState cmp_state{};
  cmp_state.cpu.x[0] = 3;
  cmp_state.cpu.x[1] = 5;
  TranslateAndRun(kCmpCode, &cmp_state);
  EXPECT_EQ(cmp_state.cpu.flags, 8u);  // N

  // tst x0, x1
  constexpr std::array<uint32_t, 1> kTstCode = {0xea01'001f};
  ThreadState tst_state{};
  tst_state.cpu.x[0] = 0xf0;
  tst_state.cpu.x[1] = 0x0f;
  TranslateAndRun(kTstCode, &tst_state);
  EXPECT_EQ(tst_state.cpu.flags, 4u);  // Z
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesConditionalSelectFamily) {
  // csel x2, x0, x1, eq; csinc x3, x0, x1, ne;
  // csinv x4, x0, x1, ne; csneg x5, x0, x1, ne
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x9a81'0002, 0x9a81'1403, 0xda81'1004, 0xda81'1405};
  ThreadState state{};
  state.cpu.flags = 4;  // Z
  state.cpu.x[0] = 10;
  state.cpu.x[1] = 3;
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(state.cpu.x[2], 10u);
  EXPECT_EQ(state.cpu.x[3], 4u);
  EXPECT_EQ(state.cpu.x[4], ~uint64_t{3});
  EXPECT_EQ(state.cpu.x[5], uint64_t{0} - 3);
}

}  // namespace
}  // namespace berberis
