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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "../../lite_translator/include/berberis/lite_translator/lite_translate_region.h"
#include "berberis/assembler/loongarch64.h"
#include "berberis/assembler/machine_code.h"
#include "berberis/code_gen_lib/gen_wrapper.h"
#include "berberis/guest_abi/guest_arguments_arch.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/arm64/interpreter.h"
#include "berberis/runtime_primitives/code_pool.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"
#include "berberis/test_utils/scoped_exec_region.h"

namespace berberis {
namespace {

void RunIntegerClosure(GuestAddr pc, GuestArgumentBuffer* buffer) {
  buffer->argv[0] = buffer->argv[0] + buffer->argv[1] + buffer->argv[2] + pc;
}

void RunFloatingClosure(GuestAddr pc, GuestArgumentBuffer* buffer) {
  float first;
  float second;
  double third;
  memcpy(&first, &buffer->simd_argv[0], sizeof(first));
  memcpy(&second, &buffer->simd_argv[1], sizeof(second));
  memcpy(&third, &buffer->simd_argv[2], sizeof(third));
  const double result = first + second + third + static_cast<double>(pc);
  memcpy(&buffer->simd_argv[0], &result, sizeof(result));
}

void RunStackClosure(GuestAddr pc, GuestArgumentBuffer* buffer) {
  uint64_t result = pc;
  for (size_t i = 0; i < 8; ++i) {
    result += buffer->argv[i];
  }
  result += buffer->stack_argv[0];
  result += buffer->stack_argv[1];
  buffer->argv[0] = result;
}

void RunConcurrentClosure(GuestAddr pc, GuestArgumentBuffer* buffer) {
  buffer->argv[0] += pc;
}

TEST(LoongArch64RuntimeLibraryTest, StaticClosureTrampolinePreservesArguments) {
  using IntegerCallback = int64_t (*)(int32_t, int32_t, int64_t);
  IntegerCallback integer_callback = AsFuncPtr<IntegerCallback>(CreateGuestFunctionWrapper(
      7, "liil", AsHostCode(&RunIntegerClosure), "integer_closure_test"));
  EXPECT_EQ(integer_callback(11, -3, 100), 115);

  using FloatingCallback = double (*)(float, float, double);
  FloatingCallback floating_callback = AsFuncPtr<FloatingCallback>(CreateGuestFunctionWrapper(
      5, "dffd", AsHostCode(&RunFloatingClosure), "floating_closure_test"));
  EXPECT_DOUBLE_EQ(floating_callback(1.25f, 2.5f, 3.75), 12.5);

  using StackCallback = uint64_t (*)(uint64_t,
                                     uint64_t,
                                     uint64_t,
                                     uint64_t,
                                     uint64_t,
                                     uint64_t,
                                     uint64_t,
                                     uint64_t,
                                     uint64_t,
                                     uint64_t);
  StackCallback stack_callback = AsFuncPtr<StackCallback>(CreateGuestFunctionWrapper(
      10, "lllllllllll", AsHostCode(&RunStackClosure), "stack_closure_test"));
  EXPECT_EQ(stack_callback(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 65u);
}

TEST(LoongArch64RuntimeLibraryTest, StaticClosureTrampolineAllocationIsConcurrent) {
  constexpr size_t kThreadCount = 64;
  std::array<uintptr_t, kThreadCount> wrappers{};
  std::array<int64_t, kThreadCount> results{};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([i, &wrappers, &results]() {
      using Callback = int64_t (*)(int64_t);
      HostCode wrapper = CreateGuestFunctionWrapper(
          i + 1, "ll", AsHostCode(&RunConcurrentClosure), "concurrent_closure_test");
      wrappers[i] = reinterpret_cast<uintptr_t>(wrapper);
      Callback callback = AsFuncPtr<Callback>(wrapper);
      results[i] = callback(1000);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  for (size_t i = 0; i < kThreadCount; ++i) {
    EXPECT_EQ(results[i], static_cast<int64_t>(1001 + i));
  }
  std::sort(wrappers.begin(), wrappers.end());
  EXPECT_EQ(std::adjacent_find(wrappers.begin(), wrappers.end()), wrappers.end());
  for (size_t i = 1; i < wrappers.size(); ++i) {
    EXPECT_EQ(wrappers[i] - wrappers[i - 1], 16u);
  }
}

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

constexpr __uint128_t MakeUint128(uint64_t low, uint64_t high) {
  return static_cast<__uint128_t>(low) | (static_cast<__uint128_t>(high) << 64);
}

constexpr __uint128_t MakeUint32x4(uint32_t lane0, uint32_t lane1, uint32_t lane2, uint32_t lane3) {
  return static_cast<__uint128_t>(lane0) | (static_cast<__uint128_t>(lane1) << 32) |
         (static_cast<__uint128_t>(lane2) << 64) | (static_cast<__uint128_t>(lane3) << 96);
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

TEST(LoongArch64RuntimeLibraryTest, LiteWriteThroughRegisterMappingPreservesIntegerRegions) {
  // Keep x0-x2 hot enough to map.  Mapped writes must remain visible in
  // ThreadState rather than depending on a deferred region-exit flush.
  constexpr std::array<uint32_t, 13> kGuestCode = {
      0xd280'0020,
      0x9100'0401,
      0x9100'0422,
      0x9100'0440,
      0x9100'0401,
      0x9100'0422,
      0x9100'0440,
      0x9100'0401,
      0x9100'0422,
      0x9100'0440,
      0x9100'0401,
      0x9100'0422,
      0x9100'0440,
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());

  MachineCode unmapped_code;
  LiteTranslateParams unmapped_params;
  unmapped_params.end_pc = start_pc + sizeof(kGuestCode);
  unmapped_params.allow_dispatch = false;
  unmapped_params.enable_reg_mapping = false;
  auto [unmapped_success, unmapped_stop_pc] =
      TryLiteTranslateRegion(start_pc, &unmapped_code, unmapped_params);
  ASSERT_TRUE(unmapped_success);
  ASSERT_EQ(unmapped_stop_pc, start_pc + sizeof(kGuestCode));

  MachineCode mapped_code;
  LiteTranslateParams mapped_params = unmapped_params;
  mapped_params.enable_reg_mapping = true;
  auto [mapped_success, mapped_stop_pc] =
      TryLiteTranslateRegion(start_pc, &mapped_code, mapped_params);
  ASSERT_TRUE(mapped_success);
  ASSERT_EQ(mapped_stop_pc, start_pc + sizeof(kGuestCode));
  EXPECT_NE(mapped_code.install_size(), unmapped_code.install_size());

  ThreadState state{};
  ScopedExecRegion exec(&mapped_code);
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  EXPECT_EQ(state.cpu.x[0], 13u);
  EXPECT_EQ(state.cpu.x[1], 11u);
  EXPECT_EQ(state.cpu.x[2], 12u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesCompareAndBranch) {
  // cbz x0, +8.  Taken dispatches to the target while the fallthrough remains
  // in the current region and executes both following instructions.
  constexpr std::array<uint32_t, 3> kGuestCode = {0xb400'0040, 0xd280'0021, 0xd280'0041};

  ThreadState zero_state{};
  TranslateAndRun(kGuestCode, &zero_state);
  EXPECT_EQ(GetInsnAddr(zero_state.cpu), ToGuestAddr(kGuestCode.data() + 2));

  ThreadState nonzero_state{};
  nonzero_state.cpu.x[0] = 1;
  TranslateAndRun(kGuestCode, &nonzero_state);
  EXPECT_EQ(nonzero_state.cpu.x[1], 2u);
  EXPECT_EQ(GetInsnAddr(nonzero_state.cpu), ToGuestAddr(kGuestCode.data() + 3));
}

TEST(LoongArch64RuntimeLibraryTest, LiteConditionalFallthroughExtendsRegion) {
  // cbnz x0, +8; add x1, x1, #1; add x2, x2, #1.  A conditional branch must
  // leave its not-taken path open so the linear fallthrough is translated as
  // part of the same region.
  constexpr std::array<uint32_t, 3> kGuestCode = {0xb500'0040, 0x9100'0421, 0x9100'0442};
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  EXPECT_EQ(stop_pc, params.end_pc);

  ThreadState state{};
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(state.cpu.x[1], 1u);
  EXPECT_EQ(state.cpu.x[2], 1u);
  EXPECT_EQ(GetInsnAddr(state.cpu), params.end_pc);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesTestAndBranch) {
  // tbz x0, #5, +8
  constexpr std::array<uint32_t, 3> kTbzCode = {0x3628'0040, 0xd503'201f, 0xd503'201f};
  ThreadState clear_state{};
  TranslateAndRun(kTbzCode, &clear_state);
  EXPECT_EQ(GetInsnAddr(clear_state.cpu), ToGuestAddr(kTbzCode.data() + 2));

  ThreadState set_state{};
  set_state.cpu.x[0] = (uint64_t{1} << 63) | (uint64_t{1} << 5);
  TranslateAndRun(kTbzCode, &set_state);
  EXPECT_EQ(GetInsnAddr(set_state.cpu), ToGuestAddr(kTbzCode.data() + 3));

  ThreadState high_bits_only_state{};
  high_bits_only_state.cpu.x[0] = uint64_t{1} << 63;
  TranslateAndRun(kTbzCode, &high_bits_only_state);
  EXPECT_EQ(GetInsnAddr(high_bits_only_state.cpu), ToGuestAddr(kTbzCode.data() + 2));

  // tbnz w0, #5, +8
  constexpr std::array<uint32_t, 3> kTbnzCode = {0x3728'0040, 0xd503'201f, 0xd503'201f};
  TranslateAndRun(kTbnzCode, &set_state);
  EXPECT_EQ(GetInsnAddr(set_state.cpu), ToGuestAddr(kTbnzCode.data() + 2));

  // Exercise the top testable bit in both W and X forms.
  constexpr std::array<uint32_t, 3> kTbzW31Code = {0x36f8'0040, 0xd503'201f, 0xd503'201f};
  ThreadState w31_state{};
  w31_state.cpu.x[0] = uint64_t{1} << 31;
  TranslateAndRun(kTbzW31Code, &w31_state);
  EXPECT_EQ(GetInsnAddr(w31_state.cpu), ToGuestAddr(kTbzW31Code.data() + 3));

  constexpr std::array<uint32_t, 3> kTbnzX63Code = {0xb7f8'0040, 0xd503'201f, 0xd503'201f};
  ThreadState x63_state{};
  x63_state.cpu.x[0] = uint64_t{1} << 63;
  TranslateAndRun(kTbnzX63Code, &x63_state);
  EXPECT_EQ(GetInsnAddr(x63_state.cpu), ToGuestAddr(kTbnzX63Code.data() + 2));
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

TEST(LoongArch64RuntimeLibraryTest, LiteConditionalSideExitAndFallthroughDispatch) {
  // cbz x0, +8; mov x1, #1; <cached target>.  Taken must side-exit directly
  // to the cached target, while not-taken executes the fallthrough in this
  // region before its final dispatch reaches that same target.
  constexpr std::array<uint32_t, 3> kGuestCode = {0xb400'0040, 0xd280'0021, 0xd503'201f};
  GuestAddr source_pc = ToGuestAddr(kGuestCode.data());
  GuestAddr target_pc = ToGuestAddr(kGuestCode.data() + 2);
  GuestAddr exit_pc = ToGuestAddr(kGuestCode.data() + 3);

  InitHostEntries();
  MachineCode target_code;
  loongarch64::Assembler target_as(&target_code);
  target_as.Li(loongarch64::Assembler::t0, 0x55aa);
  target_as.StD(loongarch64::Assembler::t0,
                loongarch64::Assembler::s8,
                offsetof(ThreadState, cpu) + offsetof(CPUState, x[2]));
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
  params.end_pc = target_pc;
  params.allow_dispatch = true;
  auto [success, stop_pc] = TryLiteTranslateRegion(source_pc, &source_code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, target_pc);
  ScopedExecRegion source_exec(&source_code);

  ThreadState taken_state{};
  SetInsnAddr(taken_state.cpu, source_pc);
  SetResidence(taken_state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&taken_state, AsHostCode(source_exec.GetHostCodeAddr()));
  EXPECT_EQ(taken_state.cpu.x[1], 0u);
  EXPECT_EQ(taken_state.cpu.x[2], 0x55aau);
  EXPECT_EQ(GetInsnAddr(taken_state.cpu), exit_pc);

  ThreadState fallthrough_state{};
  fallthrough_state.cpu.x[0] = 1;
  SetInsnAddr(fallthrough_state.cpu, source_pc);
  SetResidence(fallthrough_state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&fallthrough_state, AsHostCode(source_exec.GetHostCodeAddr()));
  EXPECT_EQ(fallthrough_state.cpu.x[1], 1u);
  EXPECT_EQ(fallthrough_state.cpu.x[2], 0x55aau);
  EXPECT_EQ(GetInsnAddr(fallthrough_state.cpu), exit_pc);

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

TEST(LoongArch64RuntimeLibraryTest, LiteInvertedLogicalShiftedMatchesInterpreter) {
  // bic x3, x1, x2; orn w4, wzr, w2 (mvn w4, w2); eon x5, x1, x2;
  // bics w6, w1, w2
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x8a22'0023, 0x2a22'03e4, 0xca22'0025, 0x6a22'0026};
  ThreadState interpreted{};
  interpreted.cpu.x[1] = 0x0123'4567'89ab'cdef;
  interpreted.cpu.x[2] = 0xfedc'ba98'7654'3210;
  SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));

  ThreadState translated{};
  translated.cpu.x[1] = interpreted.cpu.x[1];
  translated.cpu.x[2] = interpreted.cpu.x[2];
  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    InterpretInsn(&interpreted);
  }
  TranslateAndRun(kGuestCode, &translated);

  for (uint32_t rd : {3u, 4u, 5u, 6u}) {
    EXPECT_EQ(translated.cpu.x[rd], interpreted.cpu.x[rd]) << rd;
  }
  EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesBitfieldExtracts) {
  // sxth w7, w7; ubfx x2, x1, #8, #8; sbfx x3, x1, #8, #8
  constexpr std::array<uint32_t, 3> kGuestCode = {0x1300'3ce7, 0xd348'3c22, 0x9348'3c23};

  ThreadState state{};
  state.cpu.x[1] = 0x0000'0000'0000'80ff;
  state.cpu.x[7] = 0x8001;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[7], 0xffff'8001u);
  EXPECT_EQ(state.cpu.x[2], 0x80u);
  EXPECT_EQ(state.cpu.x[3], UINT64_C(0xffff'ffff'ffff'ff80));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesBitfieldInserts) {
  // bfi w2, w1, #8, #8; bfxil w3, w1, #4, #8;
  // bfi x4, x1, #40, #8; bfxil x5, x1, #8, #16
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x3318'1c22, 0x3304'2c23, 0xb358'1c24, 0xb348'5c25};

  ThreadState state{};
  state.cpu.x[1] = 0x1122'3344'5566'77ab;
  state.cpu.x[2] = 0xffff'ffff'aabb'ccdd;
  state.cpu.x[3] = 0xffff'ffff'aabb'ccdd;
  state.cpu.x[4] = 0xff00'ff00'ff00'ff00;
  state.cpu.x[5] = 0x0123'4567'89ab'cdef;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[2], 0xaabb'abddu);
  EXPECT_EQ(state.cpu.x[3], 0xaabb'cc7au);
  EXPECT_EQ(state.cpu.x[4], 0xff00'ab00'ff00'ff00u);
  EXPECT_EQ(state.cpu.x[5], 0x0123'4567'89ab'6677u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteWrappingBitfieldsMatchInterpreter) {
  // Exercise every wrapping SBFM/UBFM immediate pair for W and X forms,
  // including the hot `sbfiz x3, x2, #3, #32` encoding from jkchess.
  constexpr std::array<uint64_t, 5> kInputs = {
      0, 1, UINT64_MAX, 0x8000'0000'0000'0001, 0xa55a'f00f'963c'c369};
  for (uint32_t is_64_bit = 0; is_64_bit < 2; ++is_64_bit) {
    uint32_t data_size = is_64_bit ? 64 : 32;
    for (uint32_t opc : {0u, 2u}) {
      for (uint32_t immr = 1; immr < data_size; ++immr) {
        for (uint32_t imms = 0; imms < immr; ++imms) {
          const std::array<uint32_t, 1> guest_code = {0x1300'0001u | (is_64_bit << 31) |
                                                      (opc << 29) | (is_64_bit << 22) |
                                                      (immr << 16) | (imms << 10)};
          for (uint64_t input : kInputs) {
            ThreadState interpreted{};
            interpreted.cpu.x[0] = input;
            SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
            InterpretInsn(&interpreted);

            ThreadState translated{};
            translated.cpu.x[0] = input;
            TranslateAndRun(guest_code, &translated);

            SCOPED_TRACE(testing::Message() << "sf=" << is_64_bit << " opc=" << opc << " immr="
                                            << immr << " imms=" << imms << " input=" << input);
            EXPECT_EQ(translated.cpu.x[1], interpreted.cpu.x[1]);
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteExtractAndRorMatchInterpreter) {
  constexpr std::array<uint64_t, 3> kInputs = {
      0x0123'4567'89ab'cdef, 0xfedc'ba98'7654'3210, UINT64_MAX};
  for (uint32_t is_64_bit = 0; is_64_bit < 2; ++is_64_bit) {
    uint32_t width = is_64_bit ? 64 : 32;
    for (uint32_t same_source = 0; same_source < 2; ++same_source) {
      uint32_t rm = same_source ? 1 : 2;
      for (uint32_t lsb = 0; lsb < width; ++lsb) {
        const std::array<uint32_t, 1> guest_code = {0x1380'0023u | (is_64_bit << 31) |
                                                    (is_64_bit << 22) | (rm << 16) | (lsb << 10)};
        for (uint64_t input : kInputs) {
          ThreadState interpreted{};
          interpreted.cpu.x[1] = input;
          interpreted.cpu.x[2] = 0xa55a'f00f'963c'c369;
          SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));

          ThreadState translated{};
          translated.cpu.x[1] = interpreted.cpu.x[1];
          translated.cpu.x[2] = interpreted.cpu.x[2];
          InterpretInsn(&interpreted);
          TranslateAndRun(guest_code, &translated);

          SCOPED_TRACE(testing::Message() << "sf=" << is_64_bit << " same_source=" << same_source
                                          << " lsb=" << lsb << " input=" << input);
          EXPECT_EQ(translated.cpu.x[3], interpreted.cpu.x[3]);
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesLslImmediateAndExtendedAddSub) {
  // lsl x2, x1, #5; lsl w3, w4, #7;
  // add x5, x6, w7, uxtb #3; sub x8, x9, w10, sxtw #2;
  // add w11, wsp, w12, uxth #1; subs w13, w14, w15, sxtb
  constexpr std::array<uint32_t, 6> kGuestCode = {
      0xd37b'e822, 0x5319'6083, 0x8b27'0cc5, 0xcb2a'c928, 0x0b2c'27eb, 0x6b2f'81cd};

  ThreadState state{};
  state.cpu.x[1] = 0x123;
  state.cpu.x[4] = 0x1234'5678;
  state.cpu.x[6] = 1000;
  state.cpu.x[7] = 0xffff'ffff'ffff'fffe;
  state.cpu.x[9] = 1000;
  state.cpu.x[10] = 0xffff'fffe;
  state.cpu.x[12] = 0x12345;
  state.cpu.x[14] = 1;
  state.cpu.x[15] = 0xff;
  state.cpu.sp = 0x1000;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[2], 0x2460u);
  EXPECT_EQ(state.cpu.x[3], 0x1a2b'3c00u);
  EXPECT_EQ(state.cpu.x[5], 3032u);
  EXPECT_EQ(state.cpu.x[8], 1008u);
  EXPECT_EQ(state.cpu.x[11], 0x568au);
  EXPECT_EQ(state.cpu.x[13], 2u);
  EXPECT_EQ(state.cpu.flags, 0u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesVariableShiftsAndDivision) {
  // lslv/lsrv/asrv/rorv x[2-5], x0, x1; udiv x6, x0, x1;
  // sdiv x7, x8, x9; then the equivalent W-register operations.
  constexpr std::array<uint32_t, 12> kGuestCode = {0x9ac1'2002,
                                                   0x9ac1'2403,
                                                   0x9ac1'2804,
                                                   0x9ac1'2c05,
                                                   0x9ac1'0806,
                                                   0x9ac9'0d07,
                                                   0x1acc'216a,
                                                   0x1acc'256d,
                                                   0x1acc'296e,
                                                   0x1acc'2d6f,
                                                   0x1acc'0970,
                                                   0x1ad3'0e51};

  ThreadState state{};
  state.cpu.x[0] = 0x8000'0000'0000'0001;
  state.cpu.x[1] = 65;
  state.cpu.x[8] = static_cast<uint64_t>(-100);
  state.cpu.x[9] = 7;
  state.cpu.x[11] = 0x8000'0001;
  state.cpu.x[12] = 33;
  state.cpu.x[18] = static_cast<uint32_t>(-100);
  state.cpu.x[19] = 7;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[2], 2u);
  EXPECT_EQ(state.cpu.x[3], 0x4000'0000'0000'0000u);
  EXPECT_EQ(state.cpu.x[4], 0xc000'0000'0000'0000u);
  EXPECT_EQ(state.cpu.x[5], 0xc000'0000'0000'0000u);
  EXPECT_EQ(state.cpu.x[6], UINT64_C(0x8000'0000'0000'0001) / 65);
  EXPECT_EQ(state.cpu.x[7], static_cast<uint64_t>(-14));
  EXPECT_EQ(state.cpu.x[10], 2u);
  EXPECT_EQ(state.cpu.x[13], 0x4000'0000u);
  EXPECT_EQ(state.cpu.x[14], 0xc000'0000u);
  EXPECT_EQ(state.cpu.x[15], 0xc000'0000u);
  EXPECT_EQ(state.cpu.x[16], UINT32_C(0x8000'0001) / 33);
  EXPECT_EQ(state.cpu.x[17], static_cast<uint32_t>(-14));

  ThreadState edge_state{};
  edge_state.cpu.x[0] = UINT64_C(0x8000'0000'0000'0000);
  edge_state.cpu.x[1] = 0;
  edge_state.cpu.x[8] = UINT64_C(0x8000'0000'0000'0000);
  edge_state.cpu.x[9] = UINT64_MAX;
  edge_state.cpu.x[11] = UINT32_C(0x8000'0000);
  edge_state.cpu.x[12] = 0;
  edge_state.cpu.x[18] = UINT32_C(0x8000'0000);
  edge_state.cpu.x[19] = UINT32_MAX;
  TranslateAndRun(kGuestCode, &edge_state);

  EXPECT_EQ(edge_state.cpu.x[6], 0u);
  EXPECT_EQ(edge_state.cpu.x[7], UINT64_C(0x8000'0000'0000'0000));
  EXPECT_EQ(edge_state.cpu.x[16], 0u);
  EXPECT_EQ(edge_state.cpu.x[17], UINT32_C(0x8000'0000));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesMultiplyAddSub) {
  // mul/madd/msub x[4-6], x0, x1, x2; then the equivalent W operations.
  constexpr std::array<uint32_t, 6> kGuestCode = {
      0x9b01'7c04, 0x9b01'0805, 0x9b01'8806, 0x1b01'7c07, 0x1b01'0808, 0x1b01'8809};

  ThreadState state{};
  state.cpu.x[0] = UINT64_MAX;
  state.cpu.x[1] = 3;
  state.cpu.x[2] = 10;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[4], UINT64_MAX - 2);
  EXPECT_EQ(state.cpu.x[5], 7u);
  EXPECT_EQ(state.cpu.x[6], 13u);
  EXPECT_EQ(state.cpu.x[7], UINT32_MAX - 2);
  EXPECT_EQ(state.cpu.x[8], 7u);
  EXPECT_EQ(state.cpu.x[9], 13u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteMultiplyAddSubLongMatchesInterpreter) {
  // smaddl x4, w0, w1, x2; smsubl x5, w0, w1, x2;
  // umaddl x6, w0, w1, x2; umsubl x7, w0, w1, x2.
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x9b21'0804, 0x9b21'8805, 0x9ba1'0806, 0x9ba1'8807};
  ThreadState interpreted{};
  ThreadState translated{};
  auto initialize = [](ThreadState* state) {
    state->cpu.x[0] = 0xffff'ffff'8000'0001;
    state->cpu.x[1] = 0xaaaa'aaaa'ffff'fffd;
    state->cpu.x[2] = 0x1234'5678'9abc'def0;
  };
  initialize(&interpreted);
  initialize(&translated);
  SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));

  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    InterpretInsn(&interpreted);
  }
  TranslateAndRun(kGuestCode, &translated);

  for (uint32_t reg = 4; reg <= 7; ++reg) {
    EXPECT_EQ(translated.cpu.x[reg], interpreted.cpu.x[reg]) << "x" << reg;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesUmulhAndRev64) {
  // umulh x14, x13, x23; rev x6, x6
  constexpr std::array<uint32_t, 2> kGuestCode = {0x9bd7'7dae, 0xdac0'0cc6};
  ThreadState interpreted{};
  interpreted.cpu.x[6] = 0x0123'4567'89ab'cdef;
  interpreted.cpu.x[13] = 0xfedc'ba98'7654'3210;
  interpreted.cpu.x[23] = 0x89ab'cdef'0123'4567;
  SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));

  ThreadState translated{};
  translated.cpu.x[6] = interpreted.cpu.x[6];
  translated.cpu.x[13] = interpreted.cpu.x[13];
  translated.cpu.x[23] = interpreted.cpu.x[23];
  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    InterpretInsn(&interpreted);
  }
  TranslateAndRun(kGuestCode, &translated);

  EXPECT_EQ(translated.cpu.x[14], interpreted.cpu.x[14]);
  EXPECT_EQ(translated.cpu.x[6], interpreted.cpu.x[6]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesUmullClzAndRev32) {
  // umull x8, w25, w8; clz w8, w1; clz x9, x6; rev w15, w14
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x9ba8'7f28, 0x5ac0'1028, 0xdac0'10c9, 0x5ac0'09cf};
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.x[1] = 0x0000'00f0;
      state->cpu.x[6] = 0x0000'0000'0000'0100;
      state->cpu.x[8] = 0xffff'ffff;
      state->cpu.x[14] = 0xffff'ffff'0123'4567;
      state->cpu.x[25] = 0x8000'0001;
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.x[rd], interpreted.cpu.x[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesLogicalImmediateAnd) {
  // and w9, w8, #0xff
  constexpr std::array<uint32_t, 1> kGuestCode = {0x1200'1d09};

  ThreadState state{};
  state.cpu.x[8] = 0xffff'ffff'1234'56ab;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[9], 0xabu);
}

TEST(LoongArch64RuntimeLibraryTest, LiteLogicalImmediateAndWritesWsp) {
  // and wsp, w8, #0xff
  constexpr std::array<uint32_t, 1> kGuestCode = {0x1200'1d1f};

  ThreadState state{};
  state.cpu.x[8] = 0xffff'ffff'1234'56ab;
  state.cpu.sp = UINT64_MAX;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.sp, 0xabu);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesLogicalImmediateAndX) {
  // and x3, x1, #0xfffffffffffffffc
  constexpr std::array<uint32_t, 1> kGuestCode = {0x927e'f423};

  ThreadState state{};
  state.cpu.x[1] = 0x1234'5678'9abc'def3;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[3], 0x1234'5678'9abc'def0u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteLogicalImmediateAndXMatchesInterpreterExhaustively) {
  constexpr uint64_t kInput = 0xa55a'f00f'963c'c369;
  for (uint32_t n = 0; n < 2; ++n) {
    for (uint32_t immr = 0; immr < 64; ++immr) {
      for (uint32_t imms = 0; imms < 64; ++imms) {
        const uint32_t len_source = (n << 6) | (~imms & 0x3f);
        int32_t len = -1;
        for (int32_t bit = 6; bit >= 0; --bit) {
          if ((len_source & (uint32_t{1} << bit)) != 0) {
            len = bit;
            break;
          }
        }
        if (len < 1) {
          continue;
        }
        const uint32_t levels = (uint32_t{1} << len) - 1;
        if ((imms & levels) == levels) {
          continue;
        }

        const std::array<uint32_t, 1> guest_code = {0x9200'0001u | (n << 22) | (immr << 16) |
                                                    (imms << 10)};
        ThreadState interpreted{};
        interpreted.cpu.x[0] = kInput;
        SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
        InterpretInsn(&interpreted);

        ThreadState translated{};
        translated.cpu.x[0] = kInput;
        TranslateAndRun(guest_code, &translated);

        SCOPED_TRACE(testing::Message() << "n=" << n << " immr=" << immr << " imms=" << imms);
        EXPECT_EQ(translated.cpu.x[1], interpreted.cpu.x[1]);
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteLogicalImmediateAndXMatchesInterpreterForAllRegisters) {
  // Immediate fields from an AND instruction observed in jkchess.
  constexpr uint32_t kInsnTemplate = 0x9240'7c00u;
  for (uint32_t rn = 0; rn < 32; ++rn) {
    for (uint32_t rd = 0; rd < 32; ++rd) {
      const std::array<uint32_t, 1> guest_code = {kInsnTemplate | (rn << 5) | rd};
      ThreadState interpreted{};
      ThreadState translated{};
      for (uint32_t reg = 0; reg < 31; ++reg) {
        const uint64_t value = 0xa55a'f00f'963c'c369u ^ (uint64_t{reg} * 0x0101'0101'0101'0101u);
        interpreted.cpu.x[reg] = value;
        translated.cpu.x[reg] = value;
      }
      interpreted.cpu.sp = 0x1234'5678'9abc'def0u;
      translated.cpu.sp = interpreted.cpu.sp;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
      InterpretInsn(&interpreted);
      TranslateAndRun(guest_code, &translated);

      SCOPED_TRACE(testing::Message() << "rn=" << rn << " rd=" << rd);
      for (uint32_t reg = 0; reg < 31; ++reg) {
        EXPECT_EQ(translated.cpu.x[reg], interpreted.cpu.x[reg]);
      }
      EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesLogicalImmediateOrr) {
  // orr w25, wzr, #3; orr x8, x8, #8
  constexpr std::array<uint32_t, 2> kGuestCode = {0x3200'07f9, 0xb27d'0108};

  ThreadState state{};
  state.cpu.x[8] = 0x1234'5678'9abc'def0u;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[25], 3u);
  EXPECT_EQ(state.cpu.x[8], 0x1234'5678'9abc'def8u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteLogicalImmediateOrrXMatchesInterpreterExhaustively) {
  constexpr uint64_t kInput = 0xa55a'f00f'963c'c369;
  for (uint32_t n = 0; n < 2; ++n) {
    for (uint32_t immr = 0; immr < 64; ++immr) {
      for (uint32_t imms = 0; imms < 64; ++imms) {
        const uint32_t len_source = (n << 6) | (~imms & 0x3f);
        int32_t len = -1;
        for (int32_t bit = 6; bit >= 0; --bit) {
          if ((len_source & (uint32_t{1} << bit)) != 0) {
            len = bit;
            break;
          }
        }
        if (len < 1) {
          continue;
        }
        const uint32_t levels = (uint32_t{1} << len) - 1;
        if ((imms & levels) == levels) {
          continue;
        }

        const std::array<uint32_t, 1> guest_code = {0xb200'0001u | (n << 22) | (immr << 16) |
                                                    (imms << 10)};
        ThreadState interpreted{};
        interpreted.cpu.x[0] = kInput;
        SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
        InterpretInsn(&interpreted);

        ThreadState translated{};
        translated.cpu.x[0] = kInput;
        TranslateAndRun(guest_code, &translated);

        SCOPED_TRACE(testing::Message() << "n=" << n << " immr=" << immr << " imms=" << imms);
        EXPECT_EQ(translated.cpu.x[1], interpreted.cpu.x[1]);
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteLogicalImmediateOrrMatchesInterpreterForAllRegisters) {
  // Immediate fields from "orr w25, wzr, #3", the hottest jkchess ORR fallback.
  constexpr uint32_t kInsnTemplate = 0x3200'0400u;
  for (uint32_t rn = 0; rn < 32; ++rn) {
    for (uint32_t rd = 0; rd < 32; ++rd) {
      const std::array<uint32_t, 1> guest_code = {kInsnTemplate | (rn << 5) | rd};
      ThreadState interpreted{};
      ThreadState translated{};
      for (uint32_t reg = 0; reg < 31; ++reg) {
        const uint64_t value = 0xa55a'f00f'963c'c369u ^ (uint64_t{reg} * 0x0101'0101'0101'0101u);
        interpreted.cpu.x[reg] = value;
        translated.cpu.x[reg] = value;
      }
      interpreted.cpu.sp = 0x1234'5678'9abc'def0u;
      translated.cpu.sp = interpreted.cpu.sp;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
      InterpretInsn(&interpreted);
      TranslateAndRun(guest_code, &translated);

      SCOPED_TRACE(testing::Message() << "rn=" << rn << " rd=" << rd);
      for (uint32_t reg = 0; reg < 31; ++reg) {
        EXPECT_EQ(translated.cpu.x[reg], interpreted.cpu.x[reg]);
      }
      EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesLogicalImmediateEor) {
  // eor w16, w9, #1; eor x11, x8, #1
  constexpr std::array<uint32_t, 2> kGuestCode = {0x5200'0130, 0xd240'010b};

  ThreadState state{};
  state.cpu.x[8] = 0x1234'5678'9abc'def0u;
  state.cpu.x[9] = 0xffff'ffff'1234'5679u;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[16], 0x1234'5678u);
  EXPECT_EQ(state.cpu.x[11], 0x1234'5678'9abc'def1u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteLogicalImmediateEorMatchesInterpreterExhaustively) {
  constexpr uint64_t kInput = 0xa55a'f00f'963c'c369;
  for (uint32_t is_64_bit = 0; is_64_bit < 2; ++is_64_bit) {
    for (uint32_t n = 0; n < 2; ++n) {
      for (uint32_t immr = 0; immr < 64; ++immr) {
        for (uint32_t imms = 0; imms < 64; ++imms) {
          const uint32_t len_source = (n << 6) | (~imms & 0x3f);
          int32_t len = -1;
          for (int32_t bit = 6; bit >= 0; --bit) {
            if ((len_source & (uint32_t{1} << bit)) != 0) {
              len = bit;
              break;
            }
          }
          if (len < 1 || (!is_64_bit && n != 0) || (uint32_t{1} << len) > (32u << is_64_bit)) {
            continue;
          }
          const uint32_t levels = (uint32_t{1} << len) - 1;
          if ((imms & levels) == levels) {
            continue;
          }

          const uint32_t sf = is_64_bit << 31;
          const std::array<uint32_t, 1> guest_code = {0x5200'0001u | sf | (n << 22) | (immr << 16) |
                                                      (imms << 10)};
          ThreadState interpreted{};
          interpreted.cpu.x[0] = kInput;
          SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
          InterpretInsn(&interpreted);

          ThreadState translated{};
          translated.cpu.x[0] = kInput;
          TranslateAndRun(guest_code, &translated);

          SCOPED_TRACE(testing::Message()
                       << "sf=" << is_64_bit << " n=" << n << " immr=" << immr << " imms=" << imms);
          EXPECT_EQ(translated.cpu.x[1], interpreted.cpu.x[1]);
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteLogicalImmediateEorMatchesInterpreterForAllRegisters) {
  // Immediate fields from "eor w16, w9, #1", Mingchao's hottest EOR fallback.
  constexpr uint32_t kInsnTemplate = 0x5200'0000u;
  for (uint32_t rn = 0; rn < 32; ++rn) {
    for (uint32_t rd = 0; rd < 32; ++rd) {
      const std::array<uint32_t, 1> guest_code = {kInsnTemplate | (rn << 5) | rd};
      ThreadState interpreted{};
      ThreadState translated{};
      for (uint32_t reg = 0; reg < 31; ++reg) {
        const uint64_t value = 0xa55a'f00f'963c'c369u ^ (uint64_t{reg} * 0x0101'0101'0101'0101u);
        interpreted.cpu.x[reg] = value;
        translated.cpu.x[reg] = value;
      }
      interpreted.cpu.sp = 0x1234'5678'9abc'def0u;
      translated.cpu.sp = interpreted.cpu.sp;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
      InterpretInsn(&interpreted);
      TranslateAndRun(guest_code, &translated);

      SCOPED_TRACE(testing::Message() << "rn=" << rn << " rd=" << rd);
      for (uint32_t reg = 0; reg < 31; ++reg) {
        EXPECT_EQ(translated.cpu.x[reg], interpreted.cpu.x[reg]);
      }
      EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesLogicalImmediateAndsAndTst) {
  // ands x3, x1, #0xfffffffffffffffc
  constexpr std::array<uint32_t, 1> kAndsCode = {0xf27e'f423};
  ThreadState ands_state{};
  ands_state.cpu.x[1] = 0x8000'0000'0000'0003u;
  ands_state.cpu.flags = 0xf;
  TranslateAndRun(kAndsCode, &ands_state);
  EXPECT_EQ(ands_state.cpu.x[3], 0x8000'0000'0000'0000u);
  EXPECT_EQ(ands_state.cpu.flags, 8u);  // N

  // tst w8, #0xc000
  constexpr std::array<uint32_t, 1> kTstCode = {0x7212'051f};
  ThreadState tst_state{};
  tst_state.cpu.x[8] = 0x1234u;
  tst_state.cpu.sp = 0x1234'5678'9abc'def0u;
  tst_state.cpu.flags = 0xf;
  TranslateAndRun(kTstCode, &tst_state);
  EXPECT_EQ(tst_state.cpu.sp, 0x1234'5678'9abc'def0u);
  EXPECT_EQ(tst_state.cpu.flags, 4u);  // Z
}

TEST(LoongArch64RuntimeLibraryTest, LiteLogicalImmediateAndsXMatchesInterpreterExhaustively) {
  constexpr uint64_t kInput = 0xa55a'f00f'963c'c369;
  for (uint32_t n = 0; n < 2; ++n) {
    for (uint32_t immr = 0; immr < 64; ++immr) {
      for (uint32_t imms = 0; imms < 64; ++imms) {
        const uint32_t len_source = (n << 6) | (~imms & 0x3f);
        int32_t len = -1;
        for (int32_t bit = 6; bit >= 0; --bit) {
          if ((len_source & (uint32_t{1} << bit)) != 0) {
            len = bit;
            break;
          }
        }
        if (len < 1) {
          continue;
        }
        const uint32_t levels = (uint32_t{1} << len) - 1;
        if ((imms & levels) == levels) {
          continue;
        }

        const std::array<uint32_t, 1> guest_code = {0xf200'0001u | (n << 22) | (immr << 16) |
                                                    (imms << 10)};
        ThreadState interpreted{};
        interpreted.cpu.x[0] = kInput;
        interpreted.cpu.flags = 0xf;
        SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
        InterpretInsn(&interpreted);

        ThreadState translated{};
        translated.cpu.x[0] = kInput;
        translated.cpu.flags = 0xf;
        TranslateAndRun(guest_code, &translated);

        SCOPED_TRACE(testing::Message() << "n=" << n << " immr=" << immr << " imms=" << imms);
        EXPECT_EQ(translated.cpu.x[1], interpreted.cpu.x[1]);
        EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesPcRelativeAddressesAndNop) {
  // adr x6, +8; adrp x7, current page; nop
  constexpr std::array<uint32_t, 3> kGuestCode = {0x1000'0046, 0x9000'0007, 0xd503'201f};

  ThreadState state{};
  TranslateAndRun(kGuestCode, &state);
  EXPECT_EQ(state.cpu.x[6], ToGuestAddr(kGuestCode.data() + 2));
  EXPECT_EQ(state.cpu.x[7], ToGuestAddr(kGuestCode.data() + 1) & ~GuestAddr{0xfff});
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesArchitecturalHints) {
  // nop; yield; wfe; bti c; paciasp; autiasp
  constexpr std::array<uint32_t, 6> kGuestCode = {
      0xd503'201f, 0xd503'203f, 0xd503'205f, 0xd503'245f, 0xd503'233f, 0xd503'23bf};

  ThreadState state{};
  state.cpu.x[30] = 0x1234'5678'9abc'def0;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[30], 0x1234'5678'9abc'def0u);
  EXPECT_EQ(GetInsnAddr(state.cpu), ToGuestAddr(kGuestCode.data() + kGuestCode.size()));
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

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesReadTpidrEl0) {
  // mrs x19, tpidr_el0
  constexpr std::array<uint32_t, 1> kGuestCode = {0xd53b'd053};
  ThreadState state{};
  state.tls = 0x0123'4567'89ab'cdef;

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[19], state.tls);
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

TEST(LoongArch64RuntimeLibraryTest, LiteTreatsPrfumAsPrefetchHint) {
  // prfum pldl1strm, [x20, #-112].  It must neither access memory nor treat
  // the prefetch operation encoded in Rt as a destination register.
  constexpr std::array<uint32_t, 1> kGuestCode = {0xf899'0281};
  ThreadState state{};
  state.cpu.x[1] = 0x0123'4567'89ab'cdef;
  state.cpu.x[20] = 1;

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[1], 0x0123'4567'89ab'cdefu);
  EXPECT_EQ(state.cpu.x[20], 1u);
  EXPECT_EQ(GetInsnAddr(state.cpu), ToGuestAddr(kGuestCode.data() + 1));
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

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesMteTagLoadsAndWriteback) {
  // ldg x0, [x1, #16]; stg x2, [x3], #16; st2g x4, [x5, #-32]!
  constexpr std::array<uint32_t, 3> kGuestCode = {0xd960'1020, 0xd920'1462, 0xd9bf'eca4};
  alignas(32) std::array<uint8_t, 64> memory{};
  constexpr uint64_t kTag = UINT64_C(0xab00'0000'0000'0000);

  ThreadState state{};
  state.cpu.x[0] = UINT64_C(0xffff'ffff'ffff'ffff);
  state.cpu.x[1] = ToGuestAddr(memory.data()) | kTag;
  state.cpu.x[2] = UINT64_C(0x1200'0000'0000'0000);
  state.cpu.x[3] = ToGuestAddr(memory.data()) | kTag;
  state.cpu.x[4] = UINT64_C(0x3400'0000'0000'0000);
  state.cpu.x[5] = ToGuestAddr(memory.data() + 32) | kTag;

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[0], UINT64_C(0xf0ff'ffff'ffff'ffff));
  EXPECT_EQ(state.cpu.x[1], ToGuestAddr(memory.data()) | kTag);
  EXPECT_EQ(state.cpu.x[3], ToGuestAddr(memory.data() + 16) | kTag);
  EXPECT_EQ(state.cpu.x[5], ToGuestAddr(memory.data()) | kTag);
  EXPECT_EQ(GetInsnAddr(state.cpu), ToGuestAddr(kGuestCode.data() + kGuestCode.size()));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesMteTagAndZero) {
  // stzg x0, [x1, #16]; stz2g x0, [x2], #16
  constexpr std::array<uint32_t, 2> kGuestCode = {0xd960'1820, 0xd9e0'1440};
  alignas(32) std::array<uint8_t, 96> memory;
  memory.fill(0xa5);
  constexpr uint64_t kTag = UINT64_C(0xcd00'0000'0000'0000);

  ThreadState state{};
  state.cpu.x[1] = ToGuestAddr(memory.data()) | kTag;
  state.cpu.x[2] = ToGuestAddr(memory.data() + 32) | kTag;
  TranslateAndRun(kGuestCode, &state);

  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(memory[i], 0xa5u);
    EXPECT_EQ(memory[16 + i], 0u);
  }
  for (size_t i = 32; i < 64; ++i) {
    EXPECT_EQ(memory[i], 0u);
  }
  EXPECT_EQ(memory[64], 0xa5u);
  EXPECT_EQ(state.cpu.x[1], ToGuestAddr(memory.data()) | kTag);
  EXPECT_EQ(state.cpu.x[2], ToGuestAddr(memory.data() + 48) | kTag);
}

TEST(LoongArch64RuntimeLibraryTest, LiteMteZeroStoresHaveRecoveryPoints) {
  // stzg x0, [x1]; stz2g x0, [x2]
  constexpr std::array<uint32_t, 2> kGuestCode = {0xd960'0820, 0xd9e0'0840};
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;

  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  EXPECT_EQ(stop_pc, params.end_pc);

  ScopedExecRegion exec(&code);
  EXPECT_EQ(exec.recovery_map().size(), 6u);
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

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesSimd64LoadsAndStores) {
  // ldr d0, [x1]; str d2, [x3]; ldr d4, [x5], #8; str d6, [x7, #-8]!
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0xfd40'0020, 0xfd00'0062, 0xfc40'84a4, 0xfc1f'8ce6};
  std::array<uint64_t, 5> memory = {0x0123'4567'89ab'cdef, 0, 0x1122'3344'5566'7788, 0, 0};

  ThreadState state{};
  state.cpu.x[1] = ToGuestAddr(memory.data());
  state.cpu.x[3] = ToGuestAddr(memory.data() + 1);
  state.cpu.x[5] = ToGuestAddr(memory.data() + 2);
  state.cpu.x[7] = ToGuestAddr(memory.data() + 4);
  state.cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
  state.cpu.v[2] = MakeUint128(0xaabb'ccdd'eeff'0011, UINT64_MAX);
  state.cpu.v[4] = MakeUint128(UINT64_MAX, UINT64_MAX);
  state.cpu.v[6] = MakeUint128(0x99aa'bbcc'ddee'ff00, UINT64_MAX);
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[0], static_cast<__uint128_t>(0x0123'4567'89ab'cdef));
  EXPECT_EQ(memory[1], 0xaabb'ccdd'eeff'0011u);
  EXPECT_EQ(state.cpu.v[4], static_cast<__uint128_t>(0x1122'3344'5566'7788));
  EXPECT_EQ(state.cpu.x[5], ToGuestAddr(memory.data() + 3));
  EXPECT_EQ(memory[3], 0x99aa'bbcc'ddee'ff00u);
  EXPECT_EQ(state.cpu.x[7], ToGuestAddr(memory.data() + 3));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesSimd32LoadsAndStores) {
  // ldr s9, [x10, w11, sxtw #2]; str s9, [x10, #4]
  constexpr std::array<uint32_t, 2> kGuestCode = {0xbc6b'd949, 0xbd00'0549};
  std::array<uint32_t, 4> memory = {0x1111'1111, 0, 0x4049'0fdb, 0x4444'4444};

  ThreadState state{};
  state.cpu.x[10] = ToGuestAddr(memory.data());
  state.cpu.x[11] = 2;
  state.cpu.v[9] = MakeUint128(UINT64_MAX, UINT64_MAX);
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[9], static_cast<__uint128_t>(memory[2]));
  EXPECT_EQ(memory[1], memory[2]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteLsxVectorFloatMatchesInterpreter) {
  // fmla v0.4s, v1.4s, v2.4s
  // fmul v3.4s, v1.4s, v2.4s
  // fmul v4.4s, v1.4s, v2.s[0]
  // fmla v22.4s, v18.4s, v4.s[3]
  // fmul v17.4s, v5.4s, v0.s[2]
  // fadd v2.4s, v2.4s, v3.4s
  constexpr std::array<uint32_t, 6> kGuestCode = {
      0x4e22'cc20, 0x6e22'dc23, 0x4f82'9024, 0x4fa4'1a56, 0x4f80'98b1, 0x4e23'd442};
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint32x4(0x3f00'0000, 0xbf80'0000, 0x4000'0000, 0xc040'0000);
      state->cpu.v[1] = MakeUint32x4(0x3fc0'0000, 0xc000'0000, 0x4080'0000, 0x3e80'0000);
      state->cpu.v[2] = MakeUint32x4(0x4000'0000, 0x3f00'0000, 0xbf80'0000, 0x4100'0000);
      state->cpu.v[3] = MakeUint32x4(0x3e80'0000, 0x3f00'0000, 0xbf80'0000, 0x4100'0000);
      state->cpu.v[4] = MakeUint32x4(0x3f80'0000, 0xc000'0000, 0x4040'0000, 0xbf00'0000);
      state->cpu.v[5] = MakeUint32x4(0x3f80'0000, 0x4000'0000, 0x4040'0000, 0x4080'0000);
      state->cpu.v[18] = MakeUint32x4(0x3f00'0000, 0xbf00'0000, 0x40a0'0000, 0xc0c0'0000);
      state->cpu.v[22] = MakeUint32x4(0x3f80'0000, 0x4000'0000, 0x4040'0000, 0x4080'0000);
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarFmaddAndFmovDMatchInterpreter) {
  // fmadd s18, s19, s18, s20; fmadd d3, d4, d3, d0; fmov d0, d1
  constexpr std::array<uint32_t, 3> kGuestCode = {0x1f12'5272, 0x1f43'0083, 0x1e60'4020};
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint128(0x4008'0000'0000'0000, UINT64_MAX);  // 3.0
      state->cpu.v[1] = MakeUint128(0xc02a'0000'0000'0000, UINT64_MAX);  // -13.0
      state->cpu.v[3] = MakeUint128(0x4000'0000'0000'0000, UINT64_MAX);  // 2.0
      state->cpu.v[4] = MakeUint128(0x3ff8'0000'0000'0000, UINT64_MAX);  // 1.5
      state->cpu.v[18] = MakeUint32x4(0x4000'0000, UINT32_MAX, UINT32_MAX, UINT32_MAX);
      state->cpu.v[19] = MakeUint32x4(0x3fc0'0000, UINT32_MAX, UINT32_MAX, UINT32_MAX);
      state->cpu.v[20] = MakeUint32x4(0x4040'0000, UINT32_MAX, UINT32_MAX, UINT32_MAX);
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteVectorRev64MatchesInterpreter) {
  // rev64 v2.16b/v2.8h, v3; rev64 v0.4s, v0; rev64 v2.2s, v3.
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x4e20'0862, 0x4e60'0862, 0x4ea0'0800, 0x0ea0'0862};
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint128(0x0011'2233'4455'6677, 0x8899'aabb'ccdd'eeff);
      state->cpu.v[2] = MakeUint128(UINT64_MAX, UINT64_MAX);
      state->cpu.v[3] = MakeUint128(0x0123'4567'89ab'cdef, 0xfedc'ba98'7654'3210);
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteVectorLogicalMatchesInterpreter) {
  // Exercise all logical operations, both vector widths, and destination
  // aliasing with either source.
  constexpr std::array<uint32_t, 12> kGuestCode = {
      0x0e22'1c20,  // and v0.8b, v1.8b, v2.8b
      0x4e22'1c20,  // and v0.16b, v1.16b, v2.16b
      0x0e22'1c21,  // and v1.8b, v1.8b, v2.8b
      0x4e22'1c22,  // and v2.16b, v1.16b, v2.16b
      0x0ea2'1c20,  // orr v0.8b, v1.8b, v2.8b
      0x4ea2'1c20,  // orr v0.16b, v1.16b, v2.16b
      0x0ea2'1c21,  // orr v1.8b, v1.8b, v2.8b
      0x4ea2'1c22,  // orr v2.16b, v1.16b, v2.16b
      0x2e22'1c20,  // eor v0.8b, v1.8b, v2.8b
      0x6e22'1c20,  // eor v0.16b, v1.16b, v2.16b
      0x2e22'1c21,  // eor v1.8b, v1.8b, v2.8b
      0x6e22'1c22,  // eor v2.16b, v1.16b, v2.16b
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
    interpreted.cpu.v[1] = MakeUint128(0x0123'4567'89ab'cdef, 0xfedc'ba98'7654'3210);
    interpreted.cpu.v[2] = MakeUint128(0xf0f0'0f0f'55aa'aa55, 0x0ff0'f00f'a5a5'5a5a);
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteVectorLogicalUsesSourceCache) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x6e27'1c22,  // eor v2.16b, v1.16b, v7.16b
      0x6e27'1c23,  // eor v3.16b, v1.16b, v7.16b
      0x6e27'1c24,  // eor v4.16b, v1.16b, v7.16b
      0x6e27'1c25,  // eor v5.16b, v1.16b, v7.16b
      0x6e27'1c26,  // eor v6.16b, v1.16b, v7.16b
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());

  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;
  params.enable_reg_mapping = false;
  MachineCode uncached_code;
  auto [uncached_success, uncached_stop_pc] =
      TryLiteTranslateRegion(start_pc, &uncached_code, params);
  ASSERT_TRUE(uncached_success);
  ASSERT_EQ(uncached_stop_pc, params.end_pc);

  params.enable_reg_mapping = true;
  MachineCode cached_code;
  auto [cached_success, cached_stop_pc] = TryLiteTranslateRegion(start_pc, &cached_code, params);
  ASSERT_TRUE(cached_success);
  ASSERT_EQ(cached_stop_pc, params.end_pc);

  auto count_vector_loads = [](const MachineCode& code) {
    size_t count = 0;
    for (size_t offset = 0; offset < code.install_size(); offset += sizeof(uint32_t)) {
      uint32_t insn = *code.AddrAs<const uint32_t>(offset);
      if ((insn & 0xffc0'0000u) == 0x2c00'0000u) {
        ++count;
      }
    }
    return count;
  };
  EXPECT_EQ(count_vector_loads(uncached_code), 10u);
  EXPECT_EQ(count_vector_loads(cached_code), 2u);

  ThreadState state{};
  state.cpu.v[1] = MakeUint128(0x0123'4567'89ab'cdef, 0xfedc'ba98'7654'3210);
  state.cpu.v[7] = MakeUint128(0xf0f0'0f0f'55aa'aa55, 0x0ff0'f00f'a5a5'5a5a);
  const __uint128_t expected = state.cpu.v[1] ^ state.cpu.v[7];
  ScopedExecRegion exec(&cached_code);
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  for (uint32_t reg = 2; reg <= 6; ++reg) {
    EXPECT_EQ(state.cpu.v[reg], expected) << reg;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteCachesReadOnlyVectorSources) {
  // Reuse v1 and v7 as sources while writing distinct destinations.  A mapped
  // region should load each source once and copy it from vr4-vr8 thereafter.
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x6e27'dc22,  // fmul v2.4s, v1.4s, v7.4s
      0x6e27'dc23,  // fmul v3.4s, v1.4s, v7.4s
      0x6e27'dc24,  // fmul v4.4s, v1.4s, v7.4s
      0x6e27'dc25,  // fmul v5.4s, v1.4s, v7.4s
      0x6e27'dc26,  // fmul v6.4s, v1.4s, v7.4s
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());

  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;
  params.enable_reg_mapping = false;
  MachineCode uncached_code;
  auto [uncached_success, uncached_stop_pc] =
      TryLiteTranslateRegion(start_pc, &uncached_code, params);
  ASSERT_TRUE(uncached_success);
  ASSERT_EQ(uncached_stop_pc, params.end_pc);

  params.enable_reg_mapping = true;
  MachineCode cached_code;
  auto [cached_success, cached_stop_pc] = TryLiteTranslateRegion(start_pc, &cached_code, params);
  ASSERT_TRUE(cached_success);
  ASSERT_EQ(cached_stop_pc, params.end_pc);

  auto count_vector_loads = [](const MachineCode& code) {
    size_t count = 0;
    for (size_t offset = 0; offset < code.install_size(); offset += sizeof(uint32_t)) {
      uint32_t insn = *code.AddrAs<const uint32_t>(offset);
      if ((insn & 0xffc0'0000u) == 0x2c00'0000u) {
        ++count;
      }
    }
    return count;
  };
  EXPECT_EQ(count_vector_loads(uncached_code), 10u);
  EXPECT_EQ(count_vector_loads(cached_code), 2u);

  ThreadState state{};
  state.cpu.v[1] = MakeUint32x4(0x3f80'0000, 0x4000'0000, 0x4040'0000, 0x4080'0000);
  state.cpu.v[7] = MakeUint32x4(0x4000'0000, 0x4040'0000, 0x4080'0000, 0x40a0'0000);
  const __uint128_t expected = MakeUint32x4(0x4000'0000, 0x40c0'0000, 0x4140'0000, 0x41a0'0000);
  ScopedExecRegion exec(&cached_code);
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  for (uint32_t reg = 2; reg <= 6; ++reg) {
    EXPECT_EQ(state.cpu.v[reg], expected) << reg;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteVectorCacheObservesLd1rWrites) {
  // LD1R writes v1 before the repeated FMUL sources read it.  Structure-load
  // destinations must therefore be excluded from the region's vector cache.
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x4d40'c801,  // ld1r {v1.4s}, [x0]
      0x6e27'dc22,  // fmul v2.4s, v1.4s, v7.4s
      0x6e27'dc23,  // fmul v3.4s, v1.4s, v7.4s
      0x6e27'dc24,  // fmul v4.4s, v1.4s, v7.4s
      0x6e27'dc25,  // fmul v5.4s, v1.4s, v7.4s
  };
  uint32_t source = 0x4000'0000;  // 2.0f
  ThreadState state{};
  state.cpu.x[0] = reinterpret_cast<uintptr_t>(&source);
  state.cpu.v[1] = MakeUint32x4(0x42c8'0000, 0x42c8'0000, 0x42c8'0000, 0x42c8'0000);
  state.cpu.v[7] = MakeUint32x4(0x3f80'0000, 0x4000'0000, 0x4040'0000, 0x4080'0000);

  TranslateAndRun(kGuestCode, &state);

  const __uint128_t expected =
      MakeUint32x4(0x4000'0000, 0x4080'0000, 0x40c0'0000, 0x4100'0000);
  for (uint32_t reg = 2; reg <= 5; ++reg) {
    EXPECT_EQ(state.cpu.v[reg], expected) << reg;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteVectorNeg2S4SMatchesInterpreter) {
  // Include the exact Mingchao hot form and an in-place destination.
  constexpr std::array<uint32_t, 3> kGuestCode = {
      0x2ea0'b820,  // neg v0.2s, v1.2s
      0x6ea0'b820,  // neg v0.4s, v1.4s
      0x2ea0'b821,  // neg v1.2s, v1.2s
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
    interpreted.cpu.v[1] = MakeUint32x4(0, 1, 0x8000'0000, 0xffff'ffff);
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteExtAllOffsetsMatchInterpreter) {
  for (bool is_128_bit : {false, true}) {
    const uint32_t limit = is_128_bit ? 16 : 8;
    for (uint32_t byte_offset = 0; byte_offset < limit; ++byte_offset) {
      // ext v1.{8b,16b}, v1.{8b,16b}, v2.{8b,16b}, #byte_offset.  Vd == Vn
      // exercises the aliasing case in addition to every legal immediate.
      const std::array<uint32_t, 1> guest_code = {0x2e00'0021u |
                                                  (static_cast<uint32_t>(is_128_bit) << 30) |
                                                  (2u << 16) | (byte_offset << 11)};
      ThreadState interpreted{};
      ThreadState translated{};
      interpreted.cpu.v[1] = MakeUint128(0x0706'0504'0302'0100, 0x0f0e'0d0c'0b0a'0908);
      interpreted.cpu.v[2] = MakeUint128(0x1716'1514'1312'1110, 0x1f1e'1d1c'1b1a'1918);
      translated.cpu.v[1] = interpreted.cpu.v[1];
      translated.cpu.v[2] = interpreted.cpu.v[2];
      SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));

      InterpretInsn(&interpreted);
      TranslateAndRun(guest_code, &translated);

      SCOPED_TRACE(testing::Message() << "q=" << is_128_bit << " byte_offset=" << byte_offset);
      EXPECT_EQ(translated.cpu.v[1], interpreted.cpu.v[1]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTbl16BMatchesInterpreterWithAliasedIndex) {
  // FFmpeg HEVC hot path: tbl v19.16b, {v8.16b-v11.16b}, v19.16b.
  // Vd == Vm is intentional and verifies that translation consumes every
  // index before committing the destination.
  constexpr std::array<uint32_t, 1> kGuestCode = {0x4e13'6113};
  constexpr std::array<uint8_t, 16> kIndices = {
      0, 15, 16, 31, 32, 47, 48, 63, 64, 255, 7, 23, 39, 55, 1, 62};

  ThreadState interpreted{};
  ThreadState translated{};
  for (uint32_t reg = 8; reg <= 11; ++reg) {
    uint8_t bytes[16];
    for (uint32_t lane = 0; lane < 16; ++lane) {
      bytes[lane] = static_cast<uint8_t>((reg - 8) * 16 + lane + 0x40);
    }
    memcpy(&interpreted.cpu.v[reg], bytes, sizeof(bytes));
    memcpy(&translated.cpu.v[reg], bytes, sizeof(bytes));
  }
  memcpy(&interpreted.cpu.v[19], kIndices.data(), kIndices.size());
  memcpy(&translated.cpu.v[19], kIndices.data(), kIndices.size());
  SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));

  InterpretInsn(&interpreted);
  TranslateAndRun(kGuestCode, &translated);

  EXPECT_EQ(translated.cpu.v[19], interpreted.cpu.v[19]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteIntegerMulMlaMls4SMatchInterpreter) {
  constexpr std::array<uint32_t, 3> kGuestCode = {
      0x4ea2'9c20,  // mul v0.4s,v1.4s,v2.4s
      0x4ea2'9420,  // mla v0.4s,v1.4s,v2.4s
      0x6ea2'9420,  // mls v0.4s,v1.4s,v2.4s
  };
  constexpr std::array<std::array<__uint128_t, 3>, 2> kInputs = {{
      {MakeUint32x4(1, 2, 3, 4),
       MakeUint32x4(5, 0xffff'ffff, 0x8000'0000, 0x7fff'ffff),
       MakeUint32x4(7, 3, 2, 0xffff'ffff)},
      {MakeUint32x4(0xffff'fff0, 0x8000'0000, 0x7fff'ffff, 0x1234'5678),
       MakeUint32x4(0xffff'ffff, 0x8000'0001, 0x4000'0000, 0xdead'beef),
       MakeUint32x4(0xffff'ffff, 3, 8, 0x1020'3040)},
  }};

  for (uint32_t insn : kGuestCode) {
    for (const auto& input : kInputs) {
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      interpreted.cpu.v[0] = input[0];
      interpreted.cpu.v[1] = input[1];
      interpreted.cpu.v[2] = input[2];
      translated.cpu.v[0] = input[0];
      translated.cpu.v[1] = input[1];
      translated.cpu.v[2] = input[2];
      SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

      InterpretInsn(&interpreted);
      TranslateAndRun(one_insn, &translated);

      SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
      EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteIntegerAddSub4SMatchInterpreter) {
  constexpr std::array<uint32_t, 6> kGuestCode = {
      0x4ea2'8420,  // add v0.4s, v1.4s, v2.4s
      0x6ea2'8420,  // sub v0.4s, v1.4s, v2.4s
      0x4ea2'8421,  // add v1.4s, v1.4s, v2.4s (rd == rn)
      0x6ea2'8421,  // sub v1.4s, v1.4s, v2.4s (rd == rn)
      0x4ea2'8422,  // add v2.4s, v1.4s, v2.4s (rd == rm)
      0x6ea2'8422,  // sub v2.4s, v1.4s, v2.4s (rd == rm)
  };
  constexpr std::array<std::array<__uint128_t, 2>, 3> kInputs = {{
      {MakeUint32x4(0, 1, 0xffff'ffff, 0x8000'0000), MakeUint32x4(0, 0xffff'ffff, 1, 0x8000'0000)},
      {MakeUint32x4(0x7fff'ffff, 0x8000'0000, 0xdead'beef, 0x1234'5678),
       MakeUint32x4(1, 1, 0x2152'4111, 0xedcb'a988)},
      {MakeUint32x4(0xffff'fff0, 0x0000'0010, 0xaaaa'5555, 0x5555'aaaa),
       MakeUint32x4(0x0000'0020, 0xffff'ffe0, 0x5555'aaaa, 0xaaaa'5555)},
  }};

  for (uint32_t insn : kGuestCode) {
    for (const auto& input : kInputs) {
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      interpreted.cpu.v[1] = input[0];
      interpreted.cpu.v[2] = input[1];
      translated.cpu.v[1] = input[0];
      translated.cpu.v[2] = input[1];
      SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

      InterpretInsn(&interpreted);
      TranslateAndRun(one_insn, &translated);

      const uint32_t rd = insn & 31;
      SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
      EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteSshll4SMatchesInterpreter) {
  constexpr std::array<uint32_t, 6> kGuestCode = {
      0x0f10'a420,  // sshll v0.4s, v1.4h, #0
      0x0f15'a420,  // sshll v0.4s, v1.4h, #5
      0x0f1f'a420,  // sshll v0.4s, v1.4h, #15
      0x4f10'a420,  // sshll2 v0.4s, v1.8h, #0
      0x4f15'a420,  // sshll2 v0.4s, v1.8h, #5
      0x4f1f'a420,  // sshll2 v0.4s, v1.8h, #15
  };
  constexpr std::array<__uint128_t, 2> kInputs = {
      MakeUint128(0x7fff'8000'ffff'0001, 0x1234'c000'4000'fffe),
      MakeUint128(0x8001'ffff'0000'7ffe, 0x8000'7fff'0001'ffff),
  };

  for (uint32_t insn : kGuestCode) {
    for (const __uint128_t input : kInputs) {
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      interpreted.cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
      translated.cpu.v[0] = interpreted.cpu.v[0];
      interpreted.cpu.v[1] = input;
      translated.cpu.v[1] = input;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

      InterpretInsn(&interpreted);
      TranslateAndRun(one_insn, &translated);

      SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
      EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteSqrshrn4HMatchesInterpreter) {
  constexpr std::array<uint32_t, 8> kGuestCode = {
      0x0f1f'9c20,  // sqrshrn v0.4h, v1.4s, #1
      0x0f1b'9c20,  // sqrshrn v0.4h, v1.4s, #5
      0x0f14'9c20,  // sqrshrn v0.4h, v1.4s, #12
      0x0f10'9c20,  // sqrshrn v0.4h, v1.4s, #16
      0x4f1f'9c20,  // sqrshrn2 v0.8h, v1.4s, #1
      0x4f1b'9c20,  // sqrshrn2 v0.8h, v1.4s, #5
      0x4f14'9c20,  // sqrshrn2 v0.8h, v1.4s, #12
      0x4f10'9c20,  // sqrshrn2 v0.8h, v1.4s, #16
  };
  constexpr std::array<__uint128_t, 3> kInputs = {
      MakeUint32x4(0, 1, 0xffff'ffff, 0x7fff'ffff),
      MakeUint32x4(0x8000'0000, 0x0000'7fff, 0xffff'8000, 0x4000'0800),
      MakeUint32x4(0xffff'f800, 0x07ff'ffff, 0xf800'0000, 0x0000'1800),
  };
  constexpr __uint128_t kOldDestination = MakeUint128(0x7654'fedc'8000'7fff, 0xaaaa'bbbb'cccc'dddd);

  for (uint32_t insn : kGuestCode) {
    for (const __uint128_t input : kInputs) {
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      interpreted.cpu.v[0] = kOldDestination;
      translated.cpu.v[0] = kOldDestination;
      interpreted.cpu.v[1] = input;
      translated.cpu.v[1] = input;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

      InterpretInsn(&interpreted);
      TranslateAndRun(one_insn, &translated);

      SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
      EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteFmls4SMatchesInterpreter) {
  constexpr std::array<uint32_t, 1> kGuestCode = {
      0x4ea2'cc20,  // fmls v0.4s,v1.4s,v2.4s
  };
  constexpr std::array<std::array<__uint128_t, 3>, 3> kInputs = {{
      {MakeUint32x4(0x3f80'0000, 0x4000'0000, 0x4040'0000, 0x4080'0000),
       MakeUint32x4(0x4000'0000, 0xc000'0000, 0x3f00'0000, 0xbf80'0000),
       MakeUint32x4(0x4040'0000, 0x3f00'0000, 0xc000'0000, 0x4100'0000)},
      {MakeUint32x4(0x0000'0000, 0x8000'0000, 0x7f80'0000, 0xff80'0000),
       MakeUint32x4(0x8000'0000, 0x0000'0000, 0x3f80'0000, 0xbf80'0000),
       MakeUint32x4(0x3f80'0000, 0xbf80'0000, 0x0000'0000, 0x8000'0000)},
      {MakeUint32x4(0x7fc0'1234, 0x0080'0000, 0x0000'0001, 0x3f80'0001),
       MakeUint32x4(0x3f80'0000, 0x7fc0'5678, 0x7f7f'ffff, 0x3f7f'ffff),
       MakeUint32x4(0x4000'0000, 0x4040'0000, 0x0080'0000, 0x3f80'0001)},
  }};

  for (const auto& input : kInputs) {
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[0] = input[0];
    interpreted.cpu.v[1] = input[1];
    interpreted.cpu.v[2] = input[2];
    translated.cpu.v[0] = input[0];
    translated.cpu.v[1] = input[1];
    translated.cpu.v[2] = input[2];
    SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(kGuestCode, &translated);

    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarAndVectorFpBinaryMatchInterpreter) {
  // Scalar S/D and AdvSIMD 2S/4S/2D forms of FMUL, FDIV, FADD and FSUB.
  // These dominate Unity startup math, and the narrow forms also verify that
  // architectural upper-lane zeroing is preserved by the LSX implementation.
  constexpr std::array<uint32_t, 20> kGuestCode = {
      0x1e22'0820,  // fmul s0, s1, s2
      0x1e22'1820,  // fdiv s0, s1, s2
      0x1e22'2820,  // fadd s0, s1, s2
      0x1e22'3820,  // fsub s0, s1, s2
      0x1e62'0820,  // fmul d0, d1, d2
      0x1e62'1820,  // fdiv d0, d1, d2
      0x1e62'2820,  // fadd d0, d1, d2
      0x1e62'3820,  // fsub d0, d1, d2
      0x2e22'dc20,  // fmul v0.2s, v1.2s, v2.2s
      0x2e22'fc20,  // fdiv v0.2s, v1.2s, v2.2s
      0x0e22'd420,  // fadd v0.2s, v1.2s, v2.2s
      0x0ea2'd420,  // fsub v0.2s, v1.2s, v2.2s
      0x6e22'dc20,  // fmul v0.4s, v1.4s, v2.4s
      0x6e22'fc20,  // fdiv v0.4s, v1.4s, v2.4s
      0x4e22'd420,  // fadd v0.4s, v1.4s, v2.4s
      0x4ea2'd420,  // fsub v0.4s, v1.4s, v2.4s
      0x6e62'dc20,  // fmul v0.2d, v1.2d, v2.2d
      0x6e62'fc20,  // fdiv v0.2d, v1.2d, v2.2d
      0x4e62'd420,  // fadd v0.2d, v1.2d, v2.2d
      0x4ee2'd420,  // fsub v0.2d, v1.2d, v2.2d
  };

  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
      state->cpu.v[1] = MakeUint32x4(0x40c0'0000, 0xc100'0000, 0x3f00'0000, 0xc020'0000);
      state->cpu.v[2] = MakeUint32x4(0x4000'0000, 0xc000'0000, 0x4080'0000, 0x3f80'0000);
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteVectorBitOperationsMatchInterpreter) {
  // fabs v3.4s, v6.4s; dup v0.4s, v3.s[0]; dup v1.2d, x8
  // ext v0.16b, v0.16b, v0.16b, #8
  // and v4.16b, v4.16b, v1.16b; orr v8.16b, v0.16b, v0.16b
  // eor v1.16b, v2.16b, v1.16b; movi d0, #0; movi v0.16b, #1
  constexpr std::array<uint32_t, 9> kGuestCode = {0x4ea0'f8c3,
                                                  0x4e04'0460,
                                                  0x4e08'0d01,
                                                  0x6e00'4000,
                                                  0x4e21'1c84,
                                                  0x4ea0'1c08,
                                                  0x6e21'1c41,
                                                  0x2f00'e400,
                                                  0x4f00'e420};
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.x[8] = 0x0123'4567'89ab'cdef;
      state->cpu.v[0] = MakeUint128(0x0011'2233'4455'6677, 0x8899'aabb'ccdd'eeff);
      state->cpu.v[1] = MakeUint128(0x0f0f'f0f0'55aa'aa55, 0xffff'0000'ffff'0000);
      state->cpu.v[2] = MakeUint128(0x3333'cccc'5a5a'a5a5, 0x0123'4567'89ab'cdef);
      state->cpu.v[3] = MakeUint32x4(0x8123'4567, 0x1111'1111, 0x2222'2222, 0x3333'3333);
      state->cpu.v[4] = MakeUint128(0xffff'0000'ffff'0000, 0xaaaa'5555'aaaa'5555);
      // Include -0, a negative quiet NaN, -inf, and a normal negative value.
      state->cpu.v[6] = MakeUint32x4(0x8000'0000, 0xffc1'2345, 0xff80'0000, 0xc049'0fdb);
      state->cpu.v[8] = MakeUint128(UINT64_MAX, UINT64_MAX);
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesFmovSAndDup4S) {
  // fmov s5, s6; dup v7.4s, w8
  constexpr std::array<uint32_t, 2> kGuestCode = {0x1e20'40c5, 0x4e04'0d07};
  ThreadState state{};
  state.cpu.v[5] = MakeUint128(UINT64_MAX, UINT64_MAX);
  state.cpu.v[6] = MakeUint32x4(0x4049'0fdb, 0x1111'1111, 0x2222'2222, 0x3333'3333);
  state.cpu.x[8] = 0x0123'4567'89ab'cdef;

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[5], static_cast<__uint128_t>(0x4049'0fdb));
  EXPECT_EQ(state.cpu.v[7], MakeUint32x4(0x89ab'cdef, 0x89ab'cdef, 0x89ab'cdef, 0x89ab'cdef));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesFmovFromGeneralRegisters) {
  // fmov s1, wzr; fmov d0, x15
  constexpr std::array<uint32_t, 2> kGuestCode = {0x1e27'03e1, 0x9e67'01e0};
  ThreadState state{};
  state.cpu.x[15] = 0x0123'4567'89ab'cdef;
  state.cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
  state.cpu.v[1] = MakeUint128(UINT64_MAX, UINT64_MAX);

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[1], static_cast<__uint128_t>(0));
  EXPECT_EQ(state.cpu.v[0], static_cast<__uint128_t>(0x0123'4567'89ab'cdef));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesFmovToGeneralRegisters) {
  // fmov w3, s14; fmov x0, d10; fmov xzr, d11
  constexpr std::array<uint32_t, 3> kGuestCode = {
      0x1e26'01c3,
      0x9e66'0140,
      0x9e66'017f,
  };
  ThreadState interpreted{};
  ThreadState translated{};
  auto initialize = [](ThreadState* state) {
    state->cpu.x[0] = UINT64_MAX;
    state->cpu.x[3] = UINT64_MAX;
    state->cpu.v[10] = MakeUint128(0x0123'4567'89ab'cdef, 0xfedc'ba98'7654'3210);
    state->cpu.v[11] = MakeUint128(0x1111'2222'3333'4444, 0x5555'6666'7777'8888);
    state->cpu.v[14] = MakeUint128(0xaaaa'bbbb'89ab'cdef, 0x9999'8888'7777'6666);
  };
  initialize(&interpreted);
  initialize(&translated);
  SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));

  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    InterpretInsn(&interpreted);
  }
  TranslateAndRun(kGuestCode, &translated);

  EXPECT_EQ(translated.cpu.x[0], interpreted.cpu.x[0]);
  EXPECT_EQ(translated.cpu.x[3], interpreted.cpu.x[3]);
  EXPECT_EQ(translated.cpu.v[10], interpreted.cpu.v[10]);
  EXPECT_EQ(translated.cpu.v[14], interpreted.cpu.v[14]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesLd1Two4SPostIndex) {
  // ld1 {v16.4s, v17.4s}, [x7], #32
  constexpr std::array<uint32_t, 1> kGuestCode = {0x4cdf'a8f0};
  constexpr uint64_t kTag = 0xab00'0000'0000'0000ULL;
  std::array<uint64_t, 4> memory = {
      0x0123'4567'89ab'cdef,
      0xfedc'ba98'7654'3210,
      0x1122'3344'5566'7788,
      0x99aa'bbcc'ddee'ff00,
  };
  ThreadState state{};
  state.cpu.x[7] = ToGuestAddr(memory.data()) | kTag;

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[16], MakeUint128(memory[0], memory[1]));
  EXPECT_EQ(state.cpu.v[17], MakeUint128(memory[2], memory[3]));
  EXPECT_EQ(state.cpu.x[7], (ToGuestAddr(memory.data()) | kTag) + 32);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesLd1Two4SWithoutWriteback) {
  // ld1 {v18.4s, v19.4s}, [x7]
  constexpr std::array<uint32_t, 1> kGuestCode = {0x4c40'a8f2};
  constexpr uint64_t kTag = 0xab00'0000'0000'0000ULL;
  std::array<uint64_t, 4> memory = {
      0x0123'4567'89ab'cdef,
      0xfedc'ba98'7654'3210,
      0x1122'3344'5566'7788,
      0x99aa'bbcc'ddee'ff00,
  };
  ThreadState state{};
  state.cpu.x[7] = ToGuestAddr(memory.data()) | kTag;

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[18], MakeUint128(memory[0], memory[1]));
  EXPECT_EQ(state.cpu.v[19], MakeUint128(memory[2], memory[3]));
  EXPECT_EQ(state.cpu.x[7], ToGuestAddr(memory.data()) | kTag);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesSimd32And64Pairs) {
  // ldp s0, s1, [x9]; stp s0, s0, [x19, #8]
  // ldp d8, d9, [sp, #0x58]; stp d8, d9, [sp, #0x78]
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x2d40'0520, 0x2d01'0260, 0x6d45'a7e8, 0x6d07'a7e8};
  std::array<uint64_t, 20> memory{};
  memory[0] = 0x1122'3344'5566'7788;
  memory[11] = 0x0123'4567'89ab'cdef;
  memory[12] = 0xfedc'ba98'7654'3210;
  ThreadState state{};
  state.cpu.x[9] = ToGuestAddr(memory.data());
  state.cpu.x[19] = ToGuestAddr(memory.data());
  state.cpu.sp = ToGuestAddr(memory.data());
  state.cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
  state.cpu.v[1] = MakeUint128(UINT64_MAX, UINT64_MAX);
  state.cpu.v[8] = MakeUint128(UINT64_MAX, UINT64_MAX);
  state.cpu.v[9] = MakeUint128(UINT64_MAX, UINT64_MAX);

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[0], static_cast<__uint128_t>(0x5566'7788));
  EXPECT_EQ(state.cpu.v[1], static_cast<__uint128_t>(0x1122'3344));
  EXPECT_EQ(memory[1], 0x5566'7788'5566'7788u);
  EXPECT_EQ(state.cpu.v[8], static_cast<__uint128_t>(memory[11]));
  EXPECT_EQ(state.cpu.v[9], static_cast<__uint128_t>(memory[12]));
  EXPECT_EQ(memory[15], memory[11]);
  EXPECT_EQ(memory[16], memory[12]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteLd1Two4SHasRecoveryPoints) {
  // ld1 {v16.4s, v17.4s}, [x7], #32
  constexpr std::array<uint32_t, 1> kGuestCode = {0x4cdf'a8f0};
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;

  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  EXPECT_EQ(stop_pc, params.end_pc);

  ScopedExecRegion exec(&code);
  EXPECT_EQ(exec.recovery_map().size(), 4u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteLdSt1Multiple4SMatchesInterpreter) {
  constexpr std::array<uint32_t, 6> kGuestCode = {
      0x4c40'2820,  // ld1 {v0.4s-v3.4s},[x1]
      0x4cc9'7820,  // ld1 {v0.4s},[x1],x9
      0x4c89'7820,  // st1 {v0.4s},[x1],x9
      0x4cdf'6820,  // ld1 {v0.4s-v2.4s},[x1],#48
      0x4c9f'2820,  // st1 {v0.4s-v3.4s},[x1],#64
      0x4c40'283e,  // ld1 {v30.4s,v31.4s,v0.4s,v1.4s},[x1]
  };
  constexpr uint64_t kTag = 0xab00'0000'0000'0000ULL;

  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    alignas(16) std::array<uint64_t, 12> interpreted_memory = {0x0001'0203'0405'0607,
                                                               0x1011'1213'1415'1617,
                                                               0x2021'2223'2425'2627,
                                                               0x3031'3233'3435'3637,
                                                               0x4041'4243'4445'4647,
                                                               0x5051'5253'5455'5657,
                                                               0x6061'6263'6465'6667,
                                                               0x7071'7273'7475'7677,
                                                               0,
                                                               0,
                                                               0,
                                                               0};
    alignas(16) std::array<uint64_t, 12> translated_memory = interpreted_memory;
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      for (uint32_t reg = 0; reg < 32; ++reg) {
        state->cpu.v[reg] =
            MakeUint128(0x1111'0000'0000'0000ULL + reg, 0xaaaa'0000'0000'0000ULL + reg);
      }
      state->cpu.x[9] = 16;
    };
    initialize(&interpreted);
    initialize(&translated);
    interpreted.cpu.x[1] = ToGuestAddr(interpreted_memory.data()) | kTag;
    translated.cpu.x[1] = ToGuestAddr(translated_memory.data()) | kTag;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated_memory, interpreted_memory);
    EXPECT_EQ(translated.cpu.x[1] - ToGuestAddr(translated_memory.data()),
              interpreted.cpu.x[1] - ToGuestAddr(interpreted_memory.data()));
    for (uint32_t reg = 0; reg < 32; ++reg) {
      EXPECT_EQ(translated.cpu.v[reg], interpreted.cpu.v[reg]) << reg;
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteLdSt1Multiple4SHasRecoveryPoints) {
  constexpr std::array<uint32_t, 2> kGuestCode = {
      0x4c40'2820,  // ld1 {v0.4s-v3.4s},[x1]
      0x4c9f'2820,  // st1 {v0.4s-v3.4s},[x1],#64
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    GuestAddr start_pc = ToGuestAddr(one_insn.data());
    MachineCode code;
    LiteTranslateParams params;
    params.end_pc = start_pc + sizeof(one_insn);
    params.allow_dispatch = false;
    auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
    ASSERT_TRUE(success);
    EXPECT_EQ(stop_pc, params.end_pc);
    ScopedExecRegion exec(&code);
    EXPECT_EQ(exec.recovery_map().size(), 8u);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesSimd128LoadsAndStores) {
  // ldr q1, [x0, #16]; str q1, [x0, #32]
  // ldr q2, [x0, x3]; str q2, [x0, x4, lsl #4]
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x3dc0'0401, 0x3d80'0801, 0x3ce3'6802, 0x3ca4'7802};
  std::array<uint64_t, 8> memory = {
      0x0123'4567'89ab'cdef,
      0xfedc'ba98'7654'3210,
      0x1122'3344'5566'7788,
      0x99aa'bbcc'ddee'ff00,
      0,
      0,
      0,
      0,
  };

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data());
  state.cpu.x[3] = 0;
  state.cpu.x[4] = 3;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[1], MakeUint128(memory[2], memory[3]));
  EXPECT_EQ(memory[4], memory[2]);
  EXPECT_EQ(memory[5], memory[3]);
  EXPECT_EQ(state.cpu.v[2], MakeUint128(memory[0], memory[1]));
  EXPECT_EQ(memory[6], memory[0]);
  EXPECT_EQ(memory[7], memory[1]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesSimd128Pairs) {
  // stp q0, q1, [x2]; ldp q3, q4, [x2]
  constexpr std::array<uint32_t, 2> kGuestCode = {0xad00'0440, 0xad40'1043};
  std::array<uint64_t, 4> memory{};

  ThreadState state{};
  state.cpu.x[2] = ToGuestAddr(memory.data());
  state.cpu.v[0] = MakeUint128(0x0123'4567'89ab'cdef, 0xfedc'ba98'7654'3210);
  state.cpu.v[1] = MakeUint128(0x1122'3344'5566'7788, 0x99aa'bbcc'ddee'ff00);
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[3], state.cpu.v[0]);
  EXPECT_EQ(state.cpu.v[4], state.cpu.v[1]);
  EXPECT_EQ(memory[0], 0x0123'4567'89ab'cdefu);
  EXPECT_EQ(memory[1], 0xfedc'ba98'7654'3210u);
  EXPECT_EQ(memory[2], 0x1122'3344'5566'7788u);
  EXPECT_EQ(memory[3], 0x99aa'bbcc'ddee'ff00u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesDup16B) {
  // dup v0.16b, w1
  constexpr std::array<uint32_t, 1> kGuestCode = {0x4e01'0c20};
  ThreadState state{};
  state.cpu.x[1] = 0x1234'5678'9abc'de5a;

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[0], MakeUint128(0x5a5a'5a5a'5a5a'5a5a, 0x5a5a'5a5a'5a5a'5a5a));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesMovi2DZero) {
  // movi v5.2d, #0
  constexpr std::array<uint32_t, 1> kGuestCode = {0x6f00'e405};
  ThreadState state{};
  state.cpu.v[5] = MakeUint128(0x0123'4567'89ab'cdef, 0xfedc'ba98'7654'3210);

  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[5], static_cast<__uint128_t>(0));
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

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesRegisterOffsetMemory) {
  // strb w1, [x0, x2]; ldrb w3, [x0, x2];
  // ldrh w4, [x0, w2, uxtw #1]
  constexpr std::array<uint32_t, 3> kGuestCode = {0x3822'6801, 0x3862'6803, 0x7862'5804};
  std::array<uint8_t, 8> memory = {0, 0, 0, 0, 0x34, 0x12, 0, 0};

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data());
  state.cpu.x[1] = 0xabu;
  state.cpu.x[2] = 2;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(memory[2], 0xabu);
  EXPECT_EQ(state.cpu.x[3], 0xabu);
  EXPECT_EQ(state.cpu.x[4], 0x1234u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesLoadSignedWord) {
  // ldrsw x2, [x0]; ldrsw x3, [x0, #-4]!;
  // ldrsw x4, [x0], #4; ldrsw x5, [x0, x6, lsl #2]
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0xb980'0002, 0xb89f'cc03, 0xb880'4404, 0xb8a6'7805};
  constexpr uint32_t kNegative = 0x8000'0001u;
  constexpr uint32_t kPositive = 0x7fff'fffeu;
  std::array<uint32_t, 3> memory = {kNegative, kPositive, kNegative};

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data() + 1);
  state.cpu.x[6] = 1;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[2], kPositive);
  EXPECT_EQ(state.cpu.x[3], 0xffff'ffff'8000'0001u);
  EXPECT_EQ(state.cpu.x[4], 0xffff'ffff'8000'0001u);
  EXPECT_EQ(state.cpu.x[5], 0xffff'ffff'8000'0001u);
  EXPECT_EQ(state.cpu.x[0], ToGuestAddr(memory.data() + 1));
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesSignedByteAndHalfwordLoads) {
  // ldrsb x1, [x0, #1]; ldrsb w2, [x0, #2]
  // ldrsh x3, [x0, x4]; ldrsh w5, [x0, x4]
  // ldrsb w22, [x21], #8
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x3980'0401, 0x39c0'0802, 0x78a4'6803, 0x78e4'6805, 0x38c0'86b6};
  std::array<uint8_t, 16> memory{};
  memory[1] = 0x80;
  memory[2] = 0xfe;
  memory[4] = 0x34;
  memory[5] = 0x80;

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(memory.data());
  state.cpu.x[4] = 4;
  state.cpu.x[21] = ToGuestAddr(memory.data() + 1);
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[1], UINT64_C(0xffff'ffff'ffff'ff80));
  EXPECT_EQ(state.cpu.x[2], UINT64_C(0xffff'fffe));
  EXPECT_EQ(state.cpu.x[3], UINT64_C(0xffff'ffff'ffff'8034));
  EXPECT_EQ(state.cpu.x[5], UINT64_C(0xffff'8034));
  EXPECT_EQ(state.cpu.x[22], UINT64_C(0xffff'ff80));
  EXPECT_EQ(state.cpu.x[21], ToGuestAddr(memory.data() + 9));
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

TEST(LoongArch64RuntimeLibraryTest, LiteLoadExclusiveFeedsInterpretedStoreExclusive) {
  // ldaxr w2, [x0]; the following stxr w3, w1, [x0] deliberately remains in
  // the interpreter and must consume the reservation created by Lite JIT.
  constexpr std::array<uint32_t, 2> kGuestCode = {0x885f'fc02, 0x8803'7c01};
  uint32_t memory = 0x1234'5678;

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(&memory);
  state.cpu.x[1] = 0x89ab'cdef;
  TranslateAndRun(std::array<uint32_t, 1>{kGuestCode[0]}, &state);
  EXPECT_EQ(state.cpu.x[2], 0x1234'5678u);
  EXPECT_EQ(state.cpu.reservation_address, ToGuestAddr(&memory));

  SetInsnAddr(state.cpu, ToGuestAddr(kGuestCode.data() + 1));
  InterpretInsn(&state);
  EXPECT_EQ(state.cpu.x[3], 0u);
  EXPECT_EQ(memory, 0x89ab'cdefu);
}

TEST(LoongArch64RuntimeLibraryTest, LiteExclusiveRoundTripAndFailure) {
  // ldaxr w2,[x0]; stlxr w3,w1,[x0]
  constexpr std::array<uint32_t, 2> kRoundTripCode = {0x885f'fc02, 0x8803'fc01};
  uint32_t memory = 0x1234'5678;
  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(&memory);
  state.cpu.x[1] = 0x89ab'cdef;
  TranslateAndRun(kRoundTripCode, &state);
  EXPECT_EQ(state.cpu.x[2], 0x1234'5678u);
  EXPECT_EQ(state.cpu.x[3], 0u);
  EXPECT_EQ(memory, 0x89ab'cdefu);
  EXPECT_EQ(state.cpu.reservation_address, 0u);

  // stxr w3,w1,[x0] without a matching reservation must fail and not write.
  constexpr std::array<uint32_t, 1> kStoreOnlyCode = {0x8803'7c01};
  memory = 0x7654'3210;
  ThreadState failure{};
  failure.cpu.x[0] = ToGuestAddr(&memory);
  failure.cpu.x[1] = 0x1111'2222;
  TranslateAndRun(kStoreOnlyCode, &failure);
  EXPECT_EQ(failure.cpu.x[3], 1u);
  EXPECT_EQ(memory, 0x7654'3210u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteSubwordExclusiveRoundTripAndFailure) {
  // ldxrb w2,[x0]; stxrb w3,w1,[x0]
  constexpr std::array<uint32_t, 2> kByteCode = {0x085f'7c02, 0x0803'7c01};
  uint8_t byte = 0x7a;
  ThreadState byte_state{};
  byte_state.cpu.x[0] = ToGuestAddr(&byte) | 0xab00'0000'0000'0000ULL;
  byte_state.cpu.x[1] = 0xc5;
  TranslateAndRun(kByteCode, &byte_state);
  EXPECT_EQ(byte_state.cpu.x[2], 0x7au);
  EXPECT_EQ(byte_state.cpu.x[3], 0u);
  EXPECT_EQ(byte, 0xc5u);
  EXPECT_EQ(byte_state.cpu.reservation_address, 0u);

  // ldaxrh w6,[x4]; stlxrh w7,w5,[x4]
  constexpr std::array<uint32_t, 2> kHalfCode = {0x485f'fc86, 0x4807'fc85};
  alignas(2) uint16_t half = 0x1234;
  ThreadState half_state{};
  half_state.cpu.x[4] = ToGuestAddr(&half);
  half_state.cpu.x[5] = 0xabcd;
  TranslateAndRun(std::array<uint32_t, 1>{kHalfCode[0]}, &half_state);
  EXPECT_EQ(half_state.cpu.x[6], 0x1234u);
  half = 0x5678;  // A change after LDXRH must make STLXRH fail.
  TranslateAndRun(std::array<uint32_t, 1>{kHalfCode[1]}, &half_state);
  EXPECT_EQ(half_state.cpu.x[7], 1u);
  EXPECT_EQ(half, 0x5678u);
  EXPECT_EQ(half_state.cpu.reservation_address, 0u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteSubwordExclusiveIsAtomicUnderContention) {
  // Compile one atomic increment region per host thread, then invoke it until
  // 1000 STLXRH operations succeed.  Keeping the retry loop outside the guest
  // region is intentional: this test configures Lite translation with direct
  // dispatch disabled, so a backward guest branch ends the region.
  constexpr std::array<uint32_t, 3> kGuestCode = {
      0x485f'fc01,  // ldaxrh w1,[x0]
      0x1100'0421,  // add w1,w1,#1
      0x4802'fc01,  // stlxrh w2,w1,[x0]
  };
  alignas(2) uint16_t counter = 0;
  auto increment = [&] {
    InitHostEntries();
    const GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
    MachineCode code;
    LiteTranslateParams params;
    params.end_pc = start_pc + sizeof(kGuestCode);
    params.allow_dispatch = false;
    auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
    ASSERT_TRUE(success);
    ASSERT_EQ(stop_pc, start_pc + sizeof(kGuestCode));
    ScopedExecRegion exec(&code);

    ThreadState state{};
    state.cpu.x[0] = ToGuestAddr(&counter);
    size_t completed = 0;
    while (completed != 1000) {
      SetInsnAddr(state.cpu, start_pc);
      SetResidence(state, kOutsideGeneratedCode);
      berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
      if (state.cpu.x[2] == 0) {
        ++completed;
      }
    }
  };
  std::thread first(increment);
  std::thread second(increment);
  first.join();
  second.join();
  EXPECT_EQ(counter, 2000u);
}

TEST(LoongArch64RuntimeLibraryTest, LitePairExclusiveRoundTripAndFailure) {
  // ldxp w1,w2,[x0]; stxp w3,w4,w5,[x0]
  constexpr uint32_t kLdxpW = 0x887f'0801;
  constexpr uint32_t kStxpW = 0x8823'1404;
  alignas(16) std::array<uint64_t, 2> memory = {0x1122'3344'5566'7788ULL, 0x99aa'bbcc'ddee'ff00ULL};
  ThreadState word_state{};
  word_state.cpu.x[0] = ToGuestAddr(memory.data()) | 0xab00'0000'0000'0000ULL;
  TranslateAndRun(std::array<uint32_t, 1>{kLdxpW}, &word_state);
  EXPECT_EQ(word_state.cpu.x[1], 0x5566'7788u);
  EXPECT_EQ(word_state.cpu.x[2], 0x1122'3344u);
  word_state.cpu.x[4] = 0x0123'4567;
  word_state.cpu.x[5] = 0x89ab'cdef;
  TranslateAndRun(std::array<uint32_t, 1>{kStxpW}, &word_state);
  EXPECT_EQ(word_state.cpu.x[3], 0u);
  EXPECT_EQ(memory[0], 0x89ab'cdef'0123'4567ULL);

  // ldaxp x1,x2,[x0]; stlxp w3,x4,x5,[x0]
  constexpr uint32_t kLdaxpX = 0xc87f'8801;
  constexpr uint32_t kStlxpX = 0xc823'9404;
  ThreadState double_state{};
  double_state.cpu.x[0] = ToGuestAddr(memory.data());
  TranslateAndRun(std::array<uint32_t, 1>{kLdaxpX}, &double_state);
  EXPECT_EQ(double_state.cpu.x[1], memory[0]);
  EXPECT_EQ(double_state.cpu.x[2], memory[1]);
  double_state.cpu.x[4] = 0xdead'beef'0123'4567ULL;
  double_state.cpu.x[5] = 0x7654'3210'cafe'babeULL;
  memory[1] ^= 1;  // A change in either half must make STLXP fail.
  const auto changed_memory = memory;
  TranslateAndRun(std::array<uint32_t, 1>{kStlxpX}, &double_state);
  EXPECT_EQ(double_state.cpu.x[3], 1u);
  EXPECT_EQ(memory, changed_memory);
  EXPECT_EQ(double_state.cpu.reservation_address, 0u);
}

TEST(LoongArch64RuntimeLibraryTest, LitePairExclusiveIsAtomicUnderContention) {
  // ldaxp x1,x2,[x0]; add x1,x1,#1; stlxp w3,x1,x2,[x0]
  constexpr std::array<uint32_t, 3> kGuestCode = {0xc87f'8801, 0x9100'0421, 0xc823'8801};
  alignas(16) std::array<uint64_t, 2> counter{};
  auto increment = [&] {
    InitHostEntries();
    const GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
    MachineCode code;
    LiteTranslateParams params;
    params.end_pc = start_pc + sizeof(kGuestCode);
    params.allow_dispatch = false;
    auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
    ASSERT_TRUE(success);
    ASSERT_EQ(stop_pc, start_pc + sizeof(kGuestCode));
    ScopedExecRegion exec(&code);

    ThreadState state{};
    state.cpu.x[0] = ToGuestAddr(counter.data());
    size_t completed = 0;
    while (completed != 1000) {
      SetInsnAddr(state.cpu, start_pc);
      SetResidence(state, kOutsideGeneratedCode);
      berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
      if (state.cpu.x[3] == 0) {
        ++completed;
      }
    }
  };
  std::thread first(increment);
  std::thread second(increment);
  first.join();
  second.join();
  EXPECT_EQ(counter[0], 2000u);
  EXPECT_EQ(counter[1], 0u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesStoreReleaseWidths) {
  // stlrb w1,[x0]; stlrh w1,[x2]; stlr w1,[x4]; stlr x1,[x6]
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x089f'fc01, 0x489f'fc41, 0x889f'fc81, 0xc89f'fcc1};
  alignas(8) std::array<uint64_t, 4> memory{};
  constexpr uint64_t kTag = 0xab00'0000'0000'0000ULL;

  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(&memory[0]) | kTag;
  state.cpu.x[2] = ToGuestAddr(&memory[1]) | kTag;
  state.cpu.x[4] = ToGuestAddr(&memory[2]) | kTag;
  state.cpu.x[6] = ToGuestAddr(&memory[3]) | kTag;
  state.cpu.x[1] = 0x0123'4567'89ab'cdef;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(memory[0], 0xefu);
  EXPECT_EQ(memory[1], 0xcdefu);
  EXPECT_EQ(memory[2], 0x89ab'cdefu);
  EXPECT_EQ(memory[3], 0x0123'4567'89ab'cdefu);
}

TEST(LoongArch64RuntimeLibraryTest, LiteDmbAndClrexPreserveExclusiveSemantics) {
  // ldaxr w2,[x0]; dmb ish; clrex; stxr w3,w1,[x0]
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x885f'fc02, 0xd503'3bbf, 0xd503'3f5f, 0x8803'7c01};
  uint32_t memory = 0x1234'5678;
  ThreadState state{};
  state.cpu.x[0] = ToGuestAddr(&memory);
  state.cpu.x[1] = 0x89ab'cdef;
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.x[2], 0x1234'5678u);
  EXPECT_EQ(state.cpu.x[3], 1u);
  EXPECT_EQ(memory, 0x1234'5678u);
  EXPECT_EQ(state.cpu.reservation_address, 0u);
}

TEST(LoongArch64RuntimeLibraryTest, LiteFcvtzsWSMatchesInterpreter) {
  // fcvtzs w3, s2
  constexpr std::array<uint32_t, 1> kGuestCode = {0x1e38'0043};
  constexpr std::array<uint32_t, 14> kInputs = {
      0x0000'0000,  // +0
      0x8000'0000,  // -0
      0x4079'999a,  // 3.9
      0xc039'999a,  // -2.9
      0x0000'0001,  // positive subnormal
      0x4eff'ffff,  // largest float below 2^31
      0xceff'ffff,  // negative finite in range
      0x4f00'0000,  // +2^31
      0xcf00'0000,  // -2^31
      0x7f80'0000,  // +infinity
      0xff80'0000,  // -infinity
      0x7fc0'0000,  // positive quiet NaN
      0xffc0'0000,  // negative quiet NaN
      0x7f80'0001,  // signaling NaN payload
  };

  for (uint32_t input : kInputs) {
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[2] = input;
    translated.cpu.v[2] = input;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(kGuestCode, &translated);

    SCOPED_TRACE(testing::Message() << "input=" << std::hex << input);
    EXPECT_EQ(translated.cpu.x[3], interpreted.cpu.x[3]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteFcmpSMatchesInterpreter) {
  constexpr std::array<uint32_t, 3> kGuestCode = {
      0x1e22'2020,  // fcmp s1,s2
      0x1e22'2030,  // fcmpe s1,s2
      0x1e20'2078,  // fcmpe s3,#0.0
  };
  constexpr std::array<std::pair<uint32_t, uint32_t>, 7> kInputs = {{
      {0x3f80'0000, 0x4000'0000},  // less
      {0x4000'0000, 0x3f80'0000},  // greater
      {0x3f80'0000, 0x3f80'0000},  // equal
      {0x0000'0000, 0x8000'0000},  // signed zeros compare equal
      {0x7fc0'0000, 0x3f80'0000},  // NaN in lhs
      {0x3f80'0000, 0x7fc0'0000},  // NaN in rhs
      {0xff80'0000, 0x7f80'0000},  // -inf < +inf
  }};

  for (uint32_t insn : kGuestCode) {
    for (auto [lhs, rhs] : kInputs) {
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      const uint32_t rn = (insn >> 5) & 31;
      const uint32_t rm = (insn >> 16) & 31;
      interpreted.cpu.v[rn] = lhs;
      translated.cpu.v[rn] = lhs;
      interpreted.cpu.v[rm] = rhs;
      translated.cpu.v[rm] = rhs;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

      InterpretInsn(&interpreted);
      TranslateAndRun(one_insn, &translated);
      SCOPED_TRACE(testing::Message()
                   << "insn=" << std::hex << insn << " lhs=" << lhs << " rhs=" << rhs);
      EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteFmovSImmediateMatchesInterpreterExhaustively) {
  for (uint32_t imm8 = 0; imm8 < 256; ++imm8) {
    const std::array<uint32_t, 1> one_insn = {0x1e20'1005 | (imm8 << 13)};
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[5] = MakeUint128(UINT64_MAX, UINT64_MAX);
    translated.cpu.v[5] = interpreted.cpu.v[5];
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    SCOPED_TRACE(testing::Message() << "imm8=" << std::hex << imm8);
    EXPECT_EQ(translated.cpu.v[5], interpreted.cpu.v[5]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteInsElementAndDup4SMatchInterpreter) {
  std::vector<uint32_t> instructions;
  for (uint32_t dst = 0; dst < 4; ++dst) {
    for (uint32_t src = 0; src < 4; ++src) {
      instructions.push_back(0x6e00'0400 | (((dst << 3) | 4) << 16) | (src << 13) | (1 << 5));
    }
  }
  for (uint32_t dst = 0; dst < 2; ++dst) {
    for (uint32_t src = 0; src < 2; ++src) {
      instructions.push_back(0x6e00'0400 | (((dst << 4) | 8) << 16) | (src << 14) | (1 << 5));
    }
  }
  for (uint32_t src = 0; src < 4; ++src) {
    instructions.push_back(0x4e00'0400 | (((src << 3) | 4) << 16) | (1 << 5));
  }

  for (uint32_t insn : instructions) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[0] = MakeUint128(0x3333'3333'2222'2222, 0x7777'7777'6666'6666);
    interpreted.cpu.v[1] = MakeUint128(0xbbbb'bbbbaaaa'aaaa, 0xdddd'ddddcccc'cccc);
    translated.cpu.v[0] = interpreted.cpu.v[0];
    translated.cpu.v[1] = interpreted.cpu.v[1];
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteUcvtf4SMatchesInterpreter) {
  constexpr std::array<uint32_t, 1> kGuestCode = {0x6e21'd820};  // ucvtf v0.4s,v1.4s
  constexpr std::array<std::array<uint32_t, 4>, 3> kInputs = {{
      {0, 1, 2, 3},
      {0x7fff'ffff, 0x8000'0000, 0xffff'ffff, 16'777'217},
      {123, 456789, 0xffff'0000, 0x0102'0304},
  }};
  for (const auto& lanes : kInputs) {
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[1] = MakeUint32x4(lanes[0], lanes[1], lanes[2], lanes[3]);
    translated.cpu.v[1] = interpreted.cpu.v[1];
    SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(kGuestCode, &translated);
    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, UnityInsGeneralToLiteVectorHandoffMatchesInterpreter) {
  // Real libunity sequence observed immediately after LD1R during the original
  // rendering-corruption diagnosis.
  // INS (general) is intentionally interpreter-only; verify that handing its
  // partially constructed vector to the Lite AND/UCVTF/FMUL chain preserves
  // all lanes exactly.
  constexpr std::array<uint32_t, 4> kInterpretedPrefix = {
      0x4e04'0ff2,  // dup v18.4s,wzr
      0x4e04'1ef2,  // mov v18.s[0],w23
      0x4e0c'1f12,  // mov v18.s[1],w24
      0x4e14'1ed2,  // mov v18.s[2],w22
  };
  constexpr std::array<uint32_t, 5> kLiteSuffix = {
      0x4e31'1e51,  // and v17.16b,v18.16b,v17.16b
      0x0b15'06b5,  // add w21,w21,w21,lsl #1
      0x6e21'da31,  // ucvtf v17.4s,v17.4s
      0x2a1f'03f3,  // mov w19,wzr
      0x6e31'de10,  // fmul v16.4s,v16.4s,v17.4s
  };
  constexpr std::array<uint32_t, 9> kFullSequence = {
      kInterpretedPrefix[0],
      kInterpretedPrefix[1],
      kInterpretedPrefix[2],
      kInterpretedPrefix[3],
      kLiteSuffix[0],
      kLiteSuffix[1],
      kLiteSuffix[2],
      kLiteSuffix[3],
      kLiteSuffix[4],
  };

  auto initialize = [](ThreadState* state) {
    state->cpu.x[21] = 0x34;
    state->cpu.x[22] = 0x1020'3040;
    state->cpu.x[23] = 0x5060'7080;
    state->cpu.x[24] = 0x90a0'b0c0;
    state->cpu.x[19] = UINT64_MAX;
    state->cpu.v[16] = MakeUint32x4(0x3f80'0000, 0x4000'0000, 0x4040'0000, 0x4080'0000);
    state->cpu.v[17] = MakeUint32x4(0x0000'00ff, 0x0000'ffff, 0x00ff'ffff, 0xffff'ffff);
    state->cpu.v[18] = MakeUint128(UINT64_MAX, UINT64_MAX);
  };

  ThreadState interpreted{};
  initialize(&interpreted);
  SetInsnAddr(interpreted.cpu, ToGuestAddr(kFullSequence.data()));
  for (size_t i = 0; i < kFullSequence.size(); ++i) {
    InterpretInsn(&interpreted);
  }

  ThreadState mixed{};
  initialize(&mixed);
  SetInsnAddr(mixed.cpu, ToGuestAddr(kInterpretedPrefix.data()));
  for (size_t i = 0; i < kInterpretedPrefix.size(); ++i) {
    InterpretInsn(&mixed);
  }
  TranslateAndRun(kLiteSuffix, &mixed);

  EXPECT_EQ(mixed.cpu.x[19], interpreted.cpu.x[19]);
  EXPECT_EQ(mixed.cpu.x[21], interpreted.cpu.x[21]);
  EXPECT_EQ(mixed.cpu.v[16], interpreted.cpu.v[16]);
  EXPECT_EQ(mixed.cpu.v[17], interpreted.cpu.v[17]);
  EXPECT_EQ(mixed.cpu.v[18], interpreted.cpu.v[18]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteHotStructMemoryMatchesInterpreter) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x4c9f'a820,  // st1 {v0.4s,v1.4s},[x1],#32
      0x0ddf'8464,  // ld1 {v4.d}[0],[x3],#8
      0x4ddf'8464,  // ld1 {v4.d}[1],[x3],#8
      0x4d40'c930,  // ld1r {v16.4s},[x9]
      0x4d00'8121,  // st1 {v1.s}[2],[x9]
  };

  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    const std::array<uint32_t, 1> one_insn = {kGuestCode[i]};
    alignas(16) std::array<uint64_t, 4> interpreted_memory = {
        0x0123'4567'89ab'cdef, 0x1122'3344'5566'7788, 0x99aa'bbcc'ddee'ff00, 0xfedc'ba98'7654'3210};
    alignas(16) std::array<uint64_t, 4> translated_memory = interpreted_memory;
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint128(0x1111'1111'2222'2222, 0x3333'3333'4444'4444);
      state->cpu.v[1] = MakeUint128(0x5555'5555'6666'6666, 0x7777'7777'8888'8888);
      state->cpu.v[4] = MakeUint128(0xaaaa'aaaa'bbbb'bbbb, 0xcccc'cccc'dddd'dddd);
      state->cpu.v[16] = MakeUint128(UINT64_MAX, UINT64_MAX);
    };
    initialize(&interpreted);
    initialize(&translated);
    const uint32_t rn = (one_insn[0] >> 5) & 31;
    interpreted.cpu.x[rn] = ToGuestAddr(interpreted_memory.data());
    translated.cpu.x[rn] = ToGuestAddr(translated_memory.data());
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << one_insn[0]);
    EXPECT_EQ(translated_memory, interpreted_memory);
    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
    EXPECT_EQ(translated.cpu.v[1], interpreted.cpu.v[1]);
    EXPECT_EQ(translated.cpu.v[4], interpreted.cpu.v[4]);
    EXPECT_EQ(translated.cpu.v[16], interpreted.cpu.v[16]);
    EXPECT_EQ(translated.cpu.x[rn] - ToGuestAddr(translated_memory.data()),
              interpreted.cpu.x[rn] - ToGuestAddr(interpreted_memory.data()));
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteHotLseAtomicsMatchInterpreter) {
  // Exact encodings observed in the Unity workload, plus LDADD with Rs==Rt.
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x48e0'7c41,  // casah w0,w1,[x2]
      0x7860'8020,  // swplh w0,w0,[x1]
      0xf8e0'8020,  // swpal x0,x0,[x1]
      0xf820'0040,  // ldadd x0,x0,[x2]
  };

  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    const std::array<uint32_t, 1> one_insn = {kGuestCode[i]};
    alignas(8) uint64_t interpreted_memory = i < 2 ? 0x1122'3344'5566'7788ULL : 100;
    alignas(8) uint64_t translated_memory = interpreted_memory;
    ThreadState interpreted{};
    ThreadState translated{};
    const uint32_t rn = (one_insn[0] >> 5) & 31;
    interpreted.cpu.x[rn] = ToGuestAddr(&interpreted_memory);
    translated.cpu.x[rn] = ToGuestAddr(&translated_memory);
    interpreted.cpu.x[0] = i == 0 ? 0x7788 : (i == 3 ? 25 : 0xaabb'ccdd'eeff'0011ULL);
    translated.cpu.x[0] = interpreted.cpu.x[0];
    if (i == 0) {
      interpreted.cpu.x[1] = 0xface;
      translated.cpu.x[1] = interpreted.cpu.x[1];
    }
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << one_insn[0]);
    EXPECT_EQ(translated_memory, interpreted_memory);
    EXPECT_EQ(translated.cpu.x[0], interpreted.cpu.x[0]);
    if (i == 0) {
      EXPECT_EQ(translated.cpu.x[1], interpreted.cpu.x[1]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteLdaddIsAtomicUnderContention) {
  // ldadd x0,x0,[x2], deliberately using Rs==Rt as libc++ refcounts do.
  constexpr std::array<uint32_t, 1> kGuestCode = {0xf820'0040};
  constexpr size_t kThreadCount = 4;
  constexpr size_t kIterations = 10000;
  alignas(8) uint64_t counter = 0;

  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, start_pc + sizeof(kGuestCode));
  ScopedExecRegion exec(&code);
  HostCode host_code = AsHostCode(exec.GetHostCodeAddr());

  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t thread = 0; thread < kThreadCount; ++thread) {
    threads.emplace_back([&] {
      ThreadState state{};
      state.cpu.x[2] = ToGuestAddr(&counter);
      for (size_t i = 0; i < kIterations; ++i) {
        state.cpu.x[0] = 1;
        SetInsnAddr(state.cpu, start_pc);
        SetResidence(state, kOutsideGeneratedCode);
        berberis_RunGeneratedCode(&state, host_code);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(counter, kThreadCount * kIterations);
}

TEST(LoongArch64RuntimeLibraryTest, LiteLoadExclusiveWidthsAndTbiMatchInterpreter) {
  // ldaxrb w4,[x0]; ldaxrh w5,[x1]; ldaxr w6,[x2]; ldaxr x7,[x3]
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x085f'fc04, 0x485f'fc25, 0x885f'fc46, 0xc85f'fc67};
  alignas(8) std::array<uint64_t, 4> memory = {
      0x0123'4567'89ab'cdef, 0xfedc'ba98'7654'3210, 0x89ab'cdef'0123'4567, 0};

  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    const std::array<uint32_t, 1> one_insn = {kGuestCode[i]};
    ThreadState interpreted{};
    ThreadState translated{};
    const uint64_t tagged_address = ToGuestAddr(&memory[i]) | 0xab00'0000'0000'0000ULL;
    interpreted.cpu.x[i] = tagged_address;
    translated.cpu.x[i] = tagged_address;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    EXPECT_EQ(translated.cpu.x[4 + i], interpreted.cpu.x[4 + i]) << i;
    EXPECT_EQ(translated.cpu.reservation_address, interpreted.cpu.reservation_address) << i;
    EXPECT_EQ(static_cast<uint64_t>(translated.cpu.reservation_value),
              static_cast<uint64_t>(interpreted.cpu.reservation_value))
        << i;
  }
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
  EXPECT_EQ(unequal_state.cpu.x[1], 2u);
  EXPECT_EQ(GetInsnAddr(unequal_state.cpu), ToGuestAddr(kEqCode.data() + 4));

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

TEST(LoongArch64RuntimeLibraryTest, LiteConditionalCompareMatchesInterpreter) {
  for (uint32_t is_64_bit = 0; is_64_bit < 2; ++is_64_bit) {
    for (uint32_t is_sub = 0; is_sub < 2; ++is_sub) {
      for (uint32_t is_immediate = 0; is_immediate < 2; ++is_immediate) {
        for (uint32_t condition = 0; condition < 16; ++condition) {
          for (uint32_t initial_flags = 0; initial_flags < 16; ++initial_flags) {
            for (uint32_t nzcv : {0u, 5u, 10u, 15u}) {
              constexpr uint32_t kRn = 5;
              constexpr uint32_t kRmOrImm = 6;
              const std::array<uint32_t, 1> guest_code = {
                  0x3a40'0000u | (is_64_bit << 31) | (is_sub << 30) | (kRmOrImm << 16) |
                  (condition << 12) | (is_immediate << 11) | (kRn << 5) | nzcv};
              ThreadState interpreted{};
              interpreted.cpu.x[kRn] = 0x8000'0000'0000'0005u;
              interpreted.cpu.x[kRmOrImm] = 0x7fff'ffff'ffff'fff9u;
              interpreted.cpu.flags = initial_flags;
              SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
              InterpretInsn(&interpreted);

              ThreadState translated{};
              translated.cpu.x[kRn] = interpreted.cpu.x[kRn];
              translated.cpu.x[kRmOrImm] = interpreted.cpu.x[kRmOrImm];
              translated.cpu.flags = initial_flags;
              TranslateAndRun(guest_code, &translated);

              SCOPED_TRACE(testing::Message()
                           << "sf=" << is_64_bit << " sub=" << is_sub << " imm=" << is_immediate
                           << " cond=" << condition << " initial_flags=" << initial_flags
                           << " nzcv=" << nzcv);
              EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
              EXPECT_EQ(translated.cpu.x[kRn], interpreted.cpu.x[kRn]);
              EXPECT_EQ(translated.cpu.x[kRmOrImm], interpreted.cpu.x[kRmOrImm]);
            }
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarFcselPreservesBits) {
  constexpr std::array<uint32_t, 2> kGuestCode = {
      0x1e22'0c20,  // fcsel s0,s1,s2,eq
      0x1e65'4c83,  // fcsel d3,d4,d5,mi
  };
  for (uint32_t flags = 0; flags < 16; ++flags) {
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [flags](ThreadState* state) {
      state->cpu.flags = flags;
      state->cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
      state->cpu.v[1] = MakeUint128(0x7fc1'2345, UINT64_MAX);
      state->cpu.v[2] = MakeUint128(0xffc5'4321, UINT64_MAX);
      state->cpu.v[3] = MakeUint128(UINT64_MAX, UINT64_MAX);
      state->cpu.v[4] = MakeUint128(0x7ff8'1234'5678'9abc, UINT64_MAX);
      state->cpu.v[5] = MakeUint128(0xfff8'cba9'8765'4321, UINT64_MAX);
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));
    InterpretInsn(&interpreted);
    InterpretInsn(&interpreted);
    TranslateAndRun(kGuestCode, &translated);
    SCOPED_TRACE(testing::Message() << "flags=" << flags);
    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
    EXPECT_EQ(translated.cpu.v[3], interpreted.cpu.v[3]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteFcvtDSMatchesInterpreter) {
  constexpr std::array<uint32_t, 1> kGuestCode = {0x1e22'c020};  // fcvt d0,s1
  constexpr std::array<uint32_t, 9> kInputs = {
      0x0000'0000,
      0x8000'0000,
      0x3fc0'0000,
      0xc049'0fdb,
      0x0000'0001,
      0x7f7f'ffff,
      0x7f80'0000,
      0xff80'0000,
      0x7fc1'2345,
  };
  for (uint32_t input : kInputs) {
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
    interpreted.cpu.v[1] = MakeUint128(input, UINT64_MAX);
    translated.cpu.v[0] = interpreted.cpu.v[0];
    translated.cpu.v[1] = interpreted.cpu.v[1];
    SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(kGuestCode, &translated);
    SCOPED_TRACE(testing::Message() << "input=" << std::hex << input);
    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteFcvtzuWSMatchesInterpreter) {
  constexpr std::array<uint32_t, 1> kGuestCode = {0x1e39'0020};  // fcvtzu w0,s1
  constexpr std::array<uint32_t, 13> kInputs = {
      0x0000'0000,
      0x8000'0000,
      0x3f00'0000,
      0x3fc0'0000,
      0x4f00'0000,
      0x4f7f'ffff,
      0x4f80'0000,
      0x7f7f'ffff,
      0x7f80'0000,
      0xff80'0000,
      0xbf80'0000,
      0x7fc1'2345,
      0xffc1'2345,
  };
  for (uint32_t input : kInputs) {
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.x[0] = UINT64_MAX;
    interpreted.cpu.v[1] = MakeUint128(input, UINT64_MAX);
    translated.cpu.x[0] = interpreted.cpu.x[0];
    translated.cpu.v[1] = interpreted.cpu.v[1];
    SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(kGuestCode, &translated);
    SCOPED_TRACE(testing::Message() << "input=" << std::hex << input);
    EXPECT_EQ(translated.cpu.x[0], interpreted.cpu.x[0]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteCntAndHotPermutesMatchInterpreter) {
  constexpr std::array<uint32_t, 9> kGuestCode = {
      0x0e20'5820,  // cnt v0.8b,v1.8b
      0x4e20'5862,  // cnt v2.16b,v3.16b
      0x4e82'1820,  // uzp1 v0.4s,v1.4s,v2.4s
      0x4e85'7883,  // zip2 v3.4s,v4.4s,v5.4s
      0x4e88'28e6,  // trn1 v6.4s,v7.4s,v8.4s
      0x4e80'1800,  // uzp1 v0.4s,v0.4s,v0.4s (all registers alias)
      0x4e88'58e9,  // uzp2 v9.4s,v7.4s,v8.4s
      0x4e88'38ea,  // zip1 v10.4s,v7.4s,v8.4s
      0x4e85'38a5,  // zip1 v5.4s,v5.4s,v5.4s (all registers alias)
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      for (size_t reg = 0; reg < 9; ++reg) {
        state->cpu.v[reg] = MakeUint32x4(
            0x0102'0304u + reg, 0x1020'4080u + reg, 0x55aa'f00fu + reg, 0x8000'0001u + reg);
      }
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteFaddp2SMatchesInterpreter) {
  constexpr std::array<uint32_t, 2> kGuestCode = {
      0x2e22'd423,  // faddp v3.2s,v1.2s,v2.2s
      0x2e20'd4e0,  // faddp v0.2s,v7.2s,v0.2s (destination aliases second source)
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint32x4(0x8000'0000, 0x0000'0000, 0x1111'1111, 0x2222'2222);
      state->cpu.v[1] = MakeUint32x4(0x3f80'0000, 0x4000'0000, 0x3333'3333, 0x4444'4444);
      state->cpu.v[2] = MakeUint32x4(0x4040'0000, 0xc080'0000, 0x5555'5555, 0x6666'6666);
      state->cpu.v[7] = MakeUint32x4(0x7fc1'2345, 0x3f80'0000, 0x7777'7777, 0x8888'8888);
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteHotVectorComparisonsMatchInterpreter) {
  constexpr std::array<uint32_t, 3> kGuestCode = {
      0x4ea0'd800,  // fcmeq v0.4s,v0.4s,#0
      0x6e21'e400,  // fcmge v0.4s,v0.4s,v1.4s
      0x6e27'8e07,  // cmeq v7.16b,v16.16b,v7.16b (destination aliases second source)
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint32x4(0x0000'0000, 0x8000'0000, 0x7fc1'2345, 0xbf80'0000);
      state->cpu.v[1] = MakeUint32x4(0x8000'0000, 0x3f80'0000, 0x4000'0000, 0xc000'0000);
      state->cpu.v[7] = MakeUint32x4(0x0001'0203, 0x0405'0607, 0x0809'0a0b, 0x0c0d'0e0f);
      state->cpu.v[16] = MakeUint32x4(0x1001'0213, 0x1425'3607, 0x4809'5a0b, 0x6c7d'0e8f);
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScvtf4SMatchesInterpreter) {
  constexpr std::array<uint32_t, 1> kGuestCode = {0x4e21'd800};  // scvtf v0.4s,v0.4s
  constexpr std::array<__uint128_t, 3> kInputs = {
      MakeUint32x4(0, 1, 0xffff'ffff, 0x8000'0000),
      MakeUint32x4(0x7fff'ffff, 0x0100'0001, 0xfeff'ffff, 0x4000'0000),
      MakeUint32x4(12345, 0xffff'cfc7, 0x00ff'ffff, 0xff00'0001),
  };
  for (__uint128_t input : kInputs) {
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[0] = input;
    translated.cpu.v[0] = input;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(kGuestCode.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(kGuestCode, &translated);
    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteAdd2DMatchesInterpreter) {
  constexpr std::array<uint32_t, 2> kGuestCode = {
      0x4ee2'8423,  // add v3.2d,v1.2d,v2.2d
      0x4ee1'8400,  // add v0.2d,v0.2d,v1.2d (destination aliases first source)
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint128(UINT64_MAX, 0x7fff'ffff'ffff'ffffULL);
      state->cpu.v[1] = MakeUint128(1, 0x8000'0000'0000'0001ULL);
      state->cpu.v[2] = MakeUint128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
    };
    initialize(&interpreted);
    initialize(&translated);
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    const uint32_t rd = insn & 31;
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]) << std::hex << insn;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteSmulhMatchesInterpreter) {
  constexpr std::array<uint32_t, 2> kGuestCode = {
      0x9b42'7c23,  // smulh x3,x1,x2
      0x9b49'7d08,  // smulh x8,x8,x9 (destination aliases first source)
  };
  constexpr std::array<std::pair<uint64_t, uint64_t>, 4> kInputs = {{
      {0, UINT64_MAX},
      {0x7fff'ffff'ffff'ffffULL, 2},
      {0x8000'0000'0000'0000ULL, UINT64_MAX},
      {0xfedc'ba98'7654'3210ULL, 0x89ab'cdef'0123'4567ULL},
  }};
  for (uint32_t insn : kGuestCode) {
    for (auto [lhs, rhs] : kInputs) {
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      const uint32_t rm = (insn >> 16) & 31;
      const uint32_t rn = (insn >> 5) & 31;
      const uint32_t rd = insn & 31;
      interpreted.cpu.x[rn] = lhs;
      interpreted.cpu.x[rm] = rhs;
      translated.cpu.x[rn] = lhs;
      translated.cpu.x[rm] = rhs;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
      InterpretInsn(&interpreted);
      TranslateAndRun(one_insn, &translated);
      EXPECT_EQ(translated.cpu.x[rd], interpreted.cpu.x[rd])
          << "insn=" << std::hex << insn << " lhs=" << lhs << " rhs=" << rhs;
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteAdcMatchesInterpreter) {
  constexpr std::array<uint32_t, 2> kGuestCode = {
      0x9a0f'020d,  // adc x13,x16,x15
      0x1a02'0023,  // adc w3,w1,w2
  };
  for (uint32_t insn : kGuestCode) {
    for (uint32_t flags : {0u, 2u, 13u, 15u}) {
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      const uint32_t rm = (insn >> 16) & 31;
      const uint32_t rn = (insn >> 5) & 31;
      const uint32_t rd = insn & 31;
      interpreted.cpu.x[rn] = UINT64_MAX;
      interpreted.cpu.x[rm] = 0x8000'0000'0000'0001ULL;
      interpreted.cpu.flags = flags;
      translated.cpu.x[rn] = interpreted.cpu.x[rn];
      translated.cpu.x[rm] = interpreted.cpu.x[rm];
      translated.cpu.flags = flags;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
      InterpretInsn(&interpreted);
      TranslateAndRun(one_insn, &translated);
      EXPECT_EQ(translated.cpu.x[rd], interpreted.cpu.x[rd])
          << "insn=" << std::hex << insn << " flags=" << flags;
      EXPECT_EQ(translated.cpu.flags, flags);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteHotMoviConstantsMatchInterpreter) {
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x6f07'e7e3,  // movi v3.2d,#0xffffffffffffffff
      0x6f00'e5e3,  // movi v3.2d,#0x00000000ffffffff
      0x2f00'e5e3,  // movi d3,#0x00000000ffffffff
      0x0f00'0423,  // movi v3.2s,#1
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[3] = MakeUint128(0x0123'4567'89ab'cdefULL, 0xfedc'ba98'7654'3210ULL);
    translated.cpu.v[3] = interpreted.cpu.v[3];
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    EXPECT_EQ(translated.cpu.v[3], interpreted.cpu.v[3]) << std::hex << insn;
  }
}

}  // namespace
}  // namespace berberis
