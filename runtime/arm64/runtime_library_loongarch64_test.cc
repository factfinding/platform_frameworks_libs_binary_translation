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

#include <signal.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>

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

extern "C" void berberis_HandleNotTranslated(berberis::ThreadState* state);

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

void RunMixedSpillClosure(GuestAddr pc, GuestArgumentBuffer* buffer) {
  float fp_arg;
  memcpy(&fp_arg, &buffer->simd_argv[0], sizeof(fp_arg));

  EXPECT_EQ(pc, 0x1234u);
  EXPECT_EQ(buffer->argc, 8);
  EXPECT_EQ(buffer->simd_argc, 1);
  EXPECT_EQ(buffer->stack_argc, static_cast<int>(10 * sizeof(uint64_t)));
  EXPECT_EQ(buffer->argv[0], 0x1111u);
  EXPECT_EQ(buffer->argv[1], 0x2222u);
  EXPECT_FLOAT_EQ(fp_arg, 18.0f);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(buffer->argv[i + 2], i + 1);
  }
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_EQ(buffer->stack_argv[i], i + 7);
  }
  EXPECT_EQ(buffer->stack_argv[7], 0x1234'5678'9abc'def0ULL);
  EXPECT_EQ(buffer->stack_argv[8], 1u);
  EXPECT_EQ(buffer->stack_argv[9], 0u);
  buffer->argv[0] = 0xfeed'faceULL;
}

void RunConcurrentClosure(GuestAddr pc, GuestArgumentBuffer* buffer) {
  buffer->argv[0] += pc;
}

int g_pretranslation_hook_calls;
GuestAddr g_exact_host_target;

bool HandleExactHostTargetBeforeTranslation(ThreadState* state) {
  if (GetInsnAddr(state->cpu) != g_exact_host_target) {
    return false;
  }
  ++g_pretranslation_hook_calls;
  return true;
}

TEST(LoongArch64RuntimeLibraryTest, ExactHostTargetHookRunsBeforeTranslation) {
  ThreadState state{};
  SetInsnAddr(state.cpu, 0x1234'5000);
  g_exact_host_target = GetInsnAddr(state.cpu);
  g_pretranslation_hook_calls = 0;
  SetHandleNoExecHook(HandleExactHostTargetBeforeTranslation);

  berberis_HandleNotTranslated(&state);

  SetHandleNoExecHook(nullptr);
  EXPECT_EQ(g_pretranslation_hook_calls, 1);
}

TEST(LoongArch64RuntimeLibraryTest, ExactHostTargetHookRunsAfterInterpretedBranch) {
  // BR X0.  The target models an executable host symbol reached while
  // InterpretBatch is already decoding guest instructions.
  alignas(4) constexpr std::array<uint32_t, 1> kCode = {0xd61f0000};
  ThreadState state{};
  g_exact_host_target = 0x1234'5000;
  state.cpu.x[0] = g_exact_host_target;
  SetInsnAddr(state.cpu, ToGuestAddr(kCode.data()));
  g_pretranslation_hook_calls = 0;
  SetHandleNoExecHook(HandleExactHostTargetBeforeTranslation);

  InterpretBatch(&state, 2, nullptr, InterpreterCacheLookupMode::kAll);

  SetHandleNoExecHook(nullptr);
  EXPECT_EQ(g_pretranslation_hook_calls, 1);
  EXPECT_EQ(GetInsnAddr(state.cpu), g_exact_host_target);
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

TEST(LoongArch64RuntimeLibraryTest, StaticClosureTrampolinePreservesMixedSpilledArguments) {
  using MixedCallback = uint64_t (*)(void*,
                                     void*,
                                     float,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int32_t,
                                     int64_t,
                                     bool,
                                     bool);
  MixedCallback mixed_callback =
      AsFuncPtr<MixedCallback>(CreateGuestFunctionWrapper(0x1234,
                                                          "lppfiiiiiiiiiiiiilzz",
                                                          AsHostCode(&RunMixedSpillClosure),
                                                          "mixed_spill_closure_test"));
  EXPECT_EQ(mixed_callback(reinterpret_cast<void*>(0x1111),
                           reinterpret_cast<void*>(0x2222),
                           18.0f,
                           1,
                           2,
                           3,
                           4,
                           5,
                           6,
                           7,
                           8,
                           9,
                           10,
                           11,
                           12,
                           13,
                           0x1234'5678'9abc'def0LL,
                           true,
                           false),
            0xfeed'faceULL);
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

TEST(LoongArch64RuntimeLibraryTest, LiteSelfProfilingExitsAtGearUpThreshold) {
  InitHostEntries();
  constexpr std::array<uint32_t, 1> kGuestCode = {
      0x9100'0400,  // add x0,x0,#1
  };
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  uint32_t counter = 123;
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;
  params.enable_reg_mapping = false;
  params.enable_self_profiling = true;
  params.counter_location = &counter;
  params.counter_threshold = 2;
  params.counter_threshold_callback = AsHostCode(kEntryExitGeneratedCode);
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, params.end_pc);
  ASSERT_EQ(counter, 0u);

  ScopedExecRegion exec(&code);
  ThreadState state{};
  auto run_once = [&]() {
    SetInsnAddr(state.cpu, start_pc);
    SetResidence(state, kOutsideGeneratedCode);
    berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  };
  run_once();
  EXPECT_EQ(counter, 1u);
  EXPECT_EQ(state.cpu.x[0], 1u);
  run_once();
  EXPECT_EQ(counter, 2u);
  EXPECT_EQ(state.cpu.x[0], 2u);
  run_once();
  EXPECT_EQ(counter, 3u);
  EXPECT_EQ(state.cpu.x[0], 2u);
  EXPECT_EQ(GetResidence(state), kOutsideGeneratedCode);
}

TEST(LoongArch64RuntimeLibraryTest, LiteLocalBackedgeRunsLoopAndPollsSignals) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0xd280'0080,  // mov x0,#4
      0xd280'0001,  // mov x1,#0
      0x9100'0421,  // add x1,x1,#1
      0xf100'0400,  // subs x0,x0,#1
      0x54ff'ffc1,  // b.ne add
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;
  params.enable_reg_mapping = true;
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, params.end_pc);
  ScopedExecRegion exec(&code);

  ThreadState state{};
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  EXPECT_EQ(state.cpu.x[0], 0u);
  EXPECT_EQ(state.cpu.x[1], 4u);
  EXPECT_EQ(GetInsnAddr(state.cpu), params.end_pc);

  ThreadState pending_state{};
  pending_state.pending_signals_status.store(kPendingSignalsPresent);
  SetInsnAddr(pending_state.cpu, start_pc);
  SetResidence(pending_state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&pending_state, AsHostCode(exec.GetHostCodeAddr()));
  EXPECT_EQ(pending_state.cpu.x[0], 3u);
  EXPECT_EQ(pending_state.cpu.x[1], 1u);
  EXPECT_EQ(GetInsnAddr(pending_state.cpu), start_pc + 2 * sizeof(uint32_t));
}

TEST(LoongArch64RuntimeLibraryTest, LiteLocalBackedgeContributesToGearUpCounter) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0xd280'0080,  // mov x0,#4
      0xd280'0001,  // mov x1,#0
      0x9100'0421,  // add x1,x1,#1
      0xf100'0400,  // subs x0,x0,#1
      0x54ff'ffc1,  // b.ne add
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  uint32_t counter = 0;
  MachineCode code;
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kGuestCode);
  params.allow_dispatch = false;
  params.enable_reg_mapping = false;
  params.enable_self_profiling = true;
  params.counter_location = &counter;
  params.counter_threshold = 2;
  params.counter_threshold_callback = AsHostCode(kEntryExitGeneratedCode);
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, params.end_pc);
  ScopedExecRegion exec(&code);

  ThreadState state{};
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  EXPECT_EQ(counter, 3u);
  EXPECT_EQ(state.cpu.x[0], 2u);
  EXPECT_EQ(state.cpu.x[1], 2u);
  EXPECT_EQ(GetInsnAddr(state.cpu), start_pc + 2 * sizeof(uint32_t));
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

TEST(LoongArch64RuntimeLibraryTest, OptimizedImmediateLoadsPreserveAll64Bits) {
  InitHostEntries();
  using Assembler = loongarch64::Assembler;
  constexpr std::array<uint64_t, 25> kValues = {
      0,
      1,
      2047,
      2048,
      4095,
      4096,
      4097,
      0x7fff'ffff,
      0x8000'0000,
      0xffff'ffff,
      0x1'0000'0000,
      0x0007'ffff'ffff'ffff,
      0x0008'0000'0000'0000,
      0x000f'ffff'ffff'ffff,
      0x0010'0000'0000'0000,
      0x7fff'ffff'ffff'ffff,
      0x8000'0000'0000'0000,
      0xfff8'0000'0000'0000,
      0xffff'ffff'8000'0000,
      0xffff'ffff'ffff'f000,
      0xffff'ffff'ffff'f7ff,
      0xffff'ffff'ffff'f800,
      0xffff'ffff'ffff'ffff,
      0x0123'4567'89ab'cdef,
      0xfedc'ba98'7654'3210,
  };
  uint64_t random = 0x1234'5678'9abc'def0;
  for (size_t i = 0; i < kValues.size() + 256; ++i) {
    random ^= random << 13;
    random ^= random >> 7;
    random ^= random << 17;
    const uint64_t value = i < kValues.size() ? kValues[i] : random;
    MachineCode code;
    Assembler as(&code);
    // Poison the destination first: shortened loads must define all 64 bits.
    as.Li(Assembler::t0, ~value);
    as.LiOptimized(Assembler::t0, value);
    as.StD(Assembler::t0, Assembler::s8, offsetof(ThreadState, cpu) + offsetof(CPUState, x[0]));
    as.Li(Assembler::t0, kEntryExitGeneratedCode);
    as.Jirl(Assembler::zero, Assembler::t0, 0);
    as.Finalize();
    ScopedExecRegion exec(&code);
    ThreadState state{};
    SetResidence(state, kOutsideGeneratedCode);
    berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
    EXPECT_EQ(state.cpu.x[0], value) << std::hex << value;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteImmediateArithmeticMatchesInterpreterAtBoundaries) {
  constexpr std::array<uint32_t, 5> kImmediates = {0, 1, 2047, 2048, 4095};
  constexpr std::array<uint64_t, 6> kInputs = {
      0, 1, 0x7fff'ffff, 0xffff'ffff, 0x7fff'ffff'ffff'ffff, UINT64_MAX};
  for (uint32_t sf : {0u, 1u}) {
    for (uint32_t sub : {0u, 1u}) {
      for (uint32_t flags : {0u, 1u}) {
        for (uint32_t shift : {0u, 1u}) {
          for (uint32_t imm : kImmediates) {
            for (uint64_t input : kInputs) {
              for (bool use_sp : {false, true}) {
                const uint32_t rn = use_sp ? 31 : 1;
                const uint32_t rd = use_sp ? 31 : 0;
                const std::array<uint32_t, 1> code = {0x1100'0000u | (sf << 31) | (sub << 30) |
                                                      (flags << 29) | (shift << 22) | (imm << 10) |
                                                      (rn << 5) | rd};
                ThreadState interpreted{};
                ThreadState translated{};
                for (ThreadState* state : {&interpreted, &translated}) {
                  state->cpu.x[0] = 0xfeed'face'dead'beef;
                  state->cpu.x[1] = input;
                  state->cpu.sp = input;
                  state->cpu.flags = 0xa;
                  SetInsnAddr(state->cpu, ToGuestAddr(code.data()));
                }
                InterpretInsn(&interpreted);
                TranslateAndRun(code, &translated);
                SCOPED_TRACE(testing::Message() << std::hex << code[0] << " input=" << input);
                EXPECT_EQ(translated.cpu.x[0], interpreted.cpu.x[0]);
                EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
                EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
              }
            }
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteSpImmediateSourcesMatchInterpreterAtBoundaries) {
  constexpr std::array<uint32_t, 5> kImmediates = {0, 1, 2047, 2048, 4095};
  constexpr std::array<uint64_t, 7> kInputs = {
      0,
      1,
      0x7fff'ffff,
      0xffff'ffff,
      0x7fff'ffff'ffff'ffff,
      UINT64_MAX,
      0xab00'1234'ffff'fff0,
  };
  InitHostEntries();
  for (bool mapped : {false, true}) {
    for (uint32_t sf : {0u, 1u}) {
      for (uint32_t sub : {0u, 1u}) {
        for (uint32_t shift : {0u, 1u}) {
          for (uint32_t imm : kImmediates) {
            for (uint32_t rd : {0u, 7u, 31u}) {
              const uint32_t insn = 0x1100'0000u | (sf << 31) | (sub << 30) | (shift << 22) |
                                    (imm << 10) | (31u << 5) | rd;
              const std::array<uint32_t, 6> guest = {
                  insn,
                  0xaa00'0009u | (rd << 16) | (rd << 5),  // orr x9,xD,xD (31 is XZR)
                  0x9100'03ea,                            // mov x10,sp
                  insn,                                   // consume the updated SP
                  0xaa00'000bu | (rd << 16) | (rd << 5),  // orr x11,xD,xD
                  0x9100'03ec,                            // mov x12,sp
              };
              const GuestAddr start = ToGuestAddr(guest.data());
              MachineCode code;
              LiteTranslateParams params;
              params.end_pc = start + sizeof(guest);
              params.enable_reg_mapping = mapped;
              params.allow_dispatch = false;
              auto [success, stop] = TryLiteTranslateRegion(start, &code, params);
              ASSERT_TRUE(success);
              ASSERT_EQ(stop, params.end_pc);
              ScopedExecRegion exec(&code);
              for (uint64_t input : kInputs) {
                SCOPED_TRACE(testing::Message() << "mapped=" << mapped << " insn=" << std::hex
                                                << insn << " input=" << input);
                ThreadState interpreted{};
                ThreadState translated{};
                for (ThreadState* state : {&interpreted, &translated}) {
                  for (size_t reg = 0; reg < 31; ++reg) {
                    state->cpu.x[reg] = 0xfedc'ba98'7654'3210ULL + reg;
                  }
                  state->cpu.sp = input;
                  state->cpu.flags = 0xb;
                  SetInsnAddr(state->cpu, start);
                }
                for (size_t i = 0; i < guest.size(); ++i) {
                  InterpretInsn(&interpreted);
                }
                SetResidence(translated, kOutsideGeneratedCode);
                berberis_RunGeneratedCode(&translated, AsHostCode(exec.GetHostCodeAddr()));
                for (size_t reg = 0; reg < 31; ++reg) {
                  EXPECT_EQ(translated.cpu.x[reg], interpreted.cpu.x[reg]) << "reg=" << reg;
                }
                EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
                EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
                EXPECT_EQ(GetInsnAddr(translated.cpu), GetInsnAddr(interpreted.cpu));

                // Independent modulo arithmetic also checks the destination
                // alias and that W results never truncate an unchanged SP.
                const uint64_t rhs = uint64_t{imm} << (shift * 12);
                const uint64_t mask = sf ? UINT64_MAX : UINT32_MAX;
                const uint64_t first = (sub ? input - rhs : input + rhs) & mask;
                const uint64_t second_input = rd == 31 ? first : input;
                const uint64_t second = (sub ? second_input - rhs : second_input + rhs) & mask;
                EXPECT_EQ(translated.cpu.sp, rd == 31 ? second : input);
                EXPECT_EQ(translated.cpu.x[9], rd == 31 ? 0u : first);
                EXPECT_EQ(translated.cpu.x[10], rd == 31 ? first : input);
                EXPECT_EQ(translated.cpu.x[11], rd == 31 ? 0u : second);
                EXPECT_EQ(translated.cpu.x[12], rd == 31 ? second : input);
                EXPECT_EQ(translated.cpu.flags, 0xbu);
                EXPECT_EQ(GetInsnAddr(translated.cpu), params.end_pc);
              }
            }
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteSmallImmediateRegionsAvoidFullWidthConstants) {
  InitHostEntries();
  for (uint32_t insn : {0x9100'0400u, 0xd100'2000u, 0xd280'0400u}) {
    const std::array<uint32_t, 1> guest_code = {insn};
    MachineCode code;
    LiteTranslateParams params;
    params.end_pc = ToGuestAddr(guest_code.data() + guest_code.size());
    params.allow_dispatch = false;
    auto [success, stop_pc] = TryLiteTranslateRegion(ToGuestAddr(guest_code.data()), &code, params);
    ASSERT_TRUE(success);
    EXPECT_EQ(stop_pc, params.end_pc);
    // At most a load, immediate ALU operation, write-through store and the
    // unchanged six-instruction PC update and exit.
    EXPECT_LE(code.install_size(), 9 * sizeof(uint32_t)) << std::hex << insn;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteUnsignedMemoryOffsetsMatchInterpreterAtBoundaries) {
  alignas(16) std::array<uint64_t, 8194> memory{};
  for (size_t i = 0; i < memory.size(); ++i) {
    memory[i] = 0x8123'4567'89ab'cdefULL ^ i;
  }
  // LDR W0, LDR X0 and LDR Q0. Cover both sides of ADDI.D's signed-12-bit
  // limit and the largest ARM64 scaled unsigned immediate.
  for (uint32_t opcode : {0xb940'0020u, 0xf940'0020u, 0x3dc0'0020u}) {
    for (uint32_t imm : {0u, 127u, 128u, 255u, 256u, 511u, 512u, 4095u}) {
      const std::array<uint32_t, 1> code = {opcode | (imm << 10)};
      ThreadState interpreted{};
      ThreadState translated{};
      for (ThreadState* state : {&interpreted, &translated}) {
        state->cpu.x[1] = ToGuestAddr(memory.data());
        state->cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
        SetInsnAddr(state->cpu, ToGuestAddr(code.data()));
      }
      InterpretInsn(&interpreted);
      TranslateAndRun(code, &translated);
      SCOPED_TRACE(testing::Message() << std::hex << code[0]);
      EXPECT_EQ(translated.cpu.x[0], interpreted.cpu.x[0]);
      EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
    }
  }
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

TEST(LoongArch64RuntimeLibraryTest, LiteLongRegionsAmortizeEntryAndExitCode) {
  std::array<uint32_t, 256> guest_code;
  guest_code.fill(0x9100'0400);  // add x0, x0, #1
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(guest_code.data());
  constexpr std::array<size_t, 3> kRegionLengths = {64, 128, 256};
  std::array<size_t, kRegionLengths.size()> host_sizes{};

  for (size_t index = 0; index < kRegionLengths.size(); ++index) {
    const size_t guest_insns = kRegionLengths[index];
    MachineCode code;
    LiteTranslateParams params;
    params.end_pc = start_pc + guest_insns * sizeof(uint32_t);
    params.allow_dispatch = false;
    params.enable_reg_mapping = true;
    auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
    ASSERT_TRUE(success);
    ASSERT_EQ(stop_pc, params.end_pc);
    host_sizes[index] = code.install_size();

    ThreadState state{};
    ScopedExecRegion exec(&code);
    SetInsnAddr(state.cpu, start_pc);
    SetResidence(state, kOutsideGeneratedCode);
    berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
    EXPECT_EQ(state.cpu.x[0], guest_insns);
    EXPECT_EQ(GetInsnAddr(state.cpu), params.end_pc);
  }

  // A longer region pays for one cache prologue and one dispatcher exit.
  EXPECT_LT(host_sizes[1], 2 * host_sizes[0]);
  EXPECT_LT(host_sizes[2], 2 * host_sizes[1]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteRegisterCacheExcludesDestinationOnlyRegisters) {
  // add x0, x1, x2 repeated.  Only x1 and x2 are read; treating the Rd field
  // as a use would cache x0 and add one needless move after every write.
  constexpr std::array<uint32_t, 8> kGuestCode = {
      0x8b02'0020,
      0x8b02'0020,
      0x8b02'0020,
      0x8b02'0020,
      0x8b02'0020,
      0x8b02'0020,
      0x8b02'0020,
      0x8b02'0020,
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

  // Direct cached operands remove both source loads in every ADD. Only the
  // two source-cache prologue loads remain; destination-only x0 stays uncached.
  EXPECT_EQ(cached_code.install_size() + 2 * kGuestCode.size() * sizeof(uint32_t),
            uncached_code.install_size() + 2 * sizeof(uint32_t));

  ThreadState state{};
  state.cpu.x[1] = 17;
  state.cpu.x[2] = 25;
  ScopedExecRegion exec(&cached_code);
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  EXPECT_EQ(state.cpu.x[0], 42u);
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

TEST(LoongArch64RuntimeLibraryTest, LiteConditionalBranchesShareColdExitStub) {
  // Both branches in kShared target the end of the region.  kDistinct changes
  // only the second target, requiring a second out-of-line dispatch stub.
  constexpr std::array<uint32_t, 3> kShared = {
      0xb400'0060,  // cbz x0,+12
      0xb400'0041,  // cbz x1,+8
      0xd503'201f,
  };
  constexpr std::array<uint32_t, 3> kDistinct = {
      0xb400'0060,  // cbz x0,+12
      0xb400'0061,  // cbz x1,+12
      0xd503'201f,
  };
  auto translate = [](const auto& guest_code, MachineCode* machine_code) {
    GuestAddr start_pc = ToGuestAddr(guest_code.data());
    LiteTranslateParams params;
    params.end_pc = start_pc + sizeof(guest_code);
    params.allow_dispatch = true;
    auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, machine_code, params);
    EXPECT_TRUE(success);
    EXPECT_EQ(stop_pc, params.end_pc);
  };

  MachineCode shared_code;
  MachineCode distinct_code;
  translate(kShared, &shared_code);
  translate(kDistinct, &distinct_code);
  EXPECT_LT(shared_code.install_size(), distinct_code.install_size());
}

TEST(LoongArch64RuntimeLibraryTest, LiteRegisterCacheIgnoresCodeAfterTerminalBranch) {
  constexpr std::array<uint32_t, 16> kGuestCode = {
      0x8b01'0020,  // add x0,x1,x1
      0x1400'0001,  // b +4 (terminates the linear region)
      0x8b02'0042,
      0x8b02'0042,
      0x8b03'0063,
      0x8b03'0063,
      0x8b04'0084,
      0x8b04'0084,
      0x8b05'00a5,
      0x8b05'00a5,
      0x8b06'00c6,
      0x8b06'00c6,
      0x8b07'00e7,
      0x8b07'00e7,
      0x8b08'0108,
      0x8b08'0108,
  };
  GuestAddr start_pc = ToGuestAddr(kGuestCode.data());
  LiteTranslateParams short_params;
  short_params.end_pc = start_pc + 2 * sizeof(uint32_t);
  short_params.allow_dispatch = false;
  MachineCode short_code;
  auto [short_success, short_stop_pc] = TryLiteTranslateRegion(start_pc, &short_code, short_params);
  ASSERT_TRUE(short_success);
  ASSERT_EQ(short_stop_pc, short_params.end_pc);

  LiteTranslateParams long_params = short_params;
  long_params.end_pc = start_pc + sizeof(kGuestCode);
  MachineCode long_code;
  auto [long_success, long_stop_pc] = TryLiteTranslateRegion(start_pc, &long_code, long_params);
  ASSERT_TRUE(long_success);
  ASSERT_EQ(long_stop_pc, short_params.end_pc);
  ASSERT_EQ(long_code.install_size(), short_code.install_size());
  for (size_t offset = 0; offset < short_code.install_size(); offset += sizeof(uint32_t)) {
    EXPECT_EQ(*long_code.AddrAs<const uint32_t>(offset),
              *short_code.AddrAs<const uint32_t>(offset));
  }
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

TEST(LoongArch64RuntimeLibraryTest, LiteTranslatesSimd16UnsignedImmediateLoad) {
  // ldr h1, [x12, #144] — exact opcode observed in the Unity loading profile.
  constexpr std::array<uint32_t, 1> kGuestCode = {0x7d41'2181};
  std::array<uint16_t, 80> memory{};
  memory[72] = 0x8123;

  ThreadState state{};
  state.cpu.x[12] = ToGuestAddr(memory.data());
  state.cpu.v[1] = MakeUint128(UINT64_MAX, UINT64_MAX);
  TranslateAndRun(kGuestCode, &state);

  EXPECT_EQ(state.cpu.v[1], static_cast<__uint128_t>(0x8123));
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

TEST(LoongArch64RuntimeLibraryTest, LiteCachesWriteThroughVectorAccumulator) {
  // Repeated FMLA reads and writes v20.  The mapped value must stay coherent
  // with the write-through ThreadState copy after every guest instruction.
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x4e22'cc34,  // fmla v20.4s, v1.4s, v2.4s
      0x4e22'cc34,
      0x4e22'cc34,
      0x4e22'cc34,
      0x4e22'cc34,
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
  EXPECT_EQ(count_vector_loads(uncached_code), 15u);
  EXPECT_EQ(count_vector_loads(cached_code), 3u);

  ThreadState state{};
  state.cpu.v[1] = MakeUint32x4(0x4000'0000, 0x4000'0000, 0x4000'0000, 0x4000'0000);
  state.cpu.v[2] = MakeUint32x4(0x4040'0000, 0x4040'0000, 0x4040'0000, 0x4040'0000);
  state.cpu.v[20] = MakeUint32x4(0x3f80'0000, 0x3f80'0000, 0x3f80'0000, 0x3f80'0000);
  ScopedExecRegion exec(&cached_code);
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  EXPECT_EQ(state.cpu.v[20], MakeUint32x4(0x41f8'0000, 0x41f8'0000, 0x41f8'0000, 0x41f8'0000));
}

TEST(LoongArch64RuntimeLibraryTest, LiteFabs4SUsesSourceCache) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x4ea0'f822,  // fabs v2.4s, v1.4s
      0x4ea0'f823,  // fabs v3.4s, v1.4s
      0x4ea0'f824,  // fabs v4.4s, v1.4s
      0x4ea0'f825,  // fabs v5.4s, v1.4s
      0x4ea0'f826,  // fabs v6.4s, v1.4s
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
  EXPECT_EQ(count_vector_loads(uncached_code), 5u);
  EXPECT_EQ(count_vector_loads(cached_code), 1u);

  ThreadState state{};
  state.cpu.v[1] = MakeUint32x4(0x8000'0000, 0xffc1'2345, 0xff80'0000, 0xc049'0fdb);
  const __uint128_t expected = MakeUint32x4(0x0000'0000, 0x7fc1'2345, 0x7f80'0000, 0x4049'0fdb);
  ScopedExecRegion exec(&cached_code);
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  for (uint32_t reg = 2; reg <= 6; ++reg) {
    EXPECT_EQ(state.cpu.v[reg], expected) << reg;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LitePermute4SUsesSourceCache) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x4e81'3802,  // zip1 v2.4s, v0.4s, v1.4s
      0x4e81'3803,  // zip1 v3.4s, v0.4s, v1.4s
      0x4e81'3804,  // zip1 v4.4s, v0.4s, v1.4s
      0x4e81'3805,  // zip1 v5.4s, v0.4s, v1.4s
      0x4e81'3806,  // zip1 v6.4s, v0.4s, v1.4s
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
  state.cpu.v[0] = MakeUint32x4(0x0000'0000, 0x1111'1111, 0x2222'2222, 0x3333'3333);
  state.cpu.v[1] = MakeUint32x4(0xaaaa'aaaa, 0xbbbb'bbbb, 0xcccc'cccc, 0xdddd'dddd);
  const __uint128_t expected = MakeUint32x4(0x0000'0000, 0xaaaa'aaaa, 0x1111'1111, 0xbbbb'bbbb);
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

  const __uint128_t expected = MakeUint32x4(0x4000'0000, 0x4080'0000, 0x40c0'0000, 0x4100'0000);
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

TEST(LoongArch64RuntimeLibraryTest, LiteExtUsesSourceCache) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x6e01'3802,  // ext v2.16b, v0.16b, v1.16b, #7
      0x6e01'3803,  // ext v3.16b, v0.16b, v1.16b, #7
      0x6e01'3804,  // ext v4.16b, v0.16b, v1.16b, #7
      0x6e01'3805,  // ext v5.16b, v0.16b, v1.16b, #7
      0x6e01'3806,  // ext v6.16b, v0.16b, v1.16b, #7
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
  state.cpu.v[0] = MakeUint128(0x0706'0504'0302'0100, 0x0f0e'0d0c'0b0a'0908);
  state.cpu.v[1] = MakeUint128(0x1716'1514'1312'1110, 0x1f1e'1d1c'1b1a'1918);
  const __uint128_t expected = MakeUint128(0x0e0d'0c0b'0a09'0807, 0x1615'1413'1211'100f);
  ScopedExecRegion exec(&cached_code);
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  for (uint32_t reg = 2; reg <= 6; ++reg) {
    EXPECT_EQ(state.cpu.v[reg], expected) << reg;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTbl16BMatchesInterpreterWithAliasedIndex) {
  // FFmpeg HEVC uses tbl v19.16b,{v8.16b-v11.16b},v19.16b. Cover every
  // table length while preserving the aliased destination/index operand.
  constexpr std::array<uint8_t, 16> kIndices = {
      0, 15, 16, 31, 32, 47, 48, 63, 64, 255, 7, 23, 39, 55, 1, 62};
  for (uint32_t length = 1; length <= 4; ++length) {
    const std::array<uint32_t, 1> guest_code = {0x4e13'0113u | ((length - 1) << 13)};
    ThreadState interpreted{};
    ThreadState translated{};
    for (uint32_t reg = 8; reg <= 11; ++reg) {
      uint8_t bytes[16];
      for (uint32_t lane = 0; lane < 16; ++lane) {
        bytes[lane] = static_cast<uint8_t>((reg - 8) * 16 + lane + 0x40);
      }
      memcpy(&interpreted.cpu.v[reg], bytes, sizeof(bytes));
    }
    memcpy(&interpreted.cpu.v[19], kIndices.data(), kIndices.size());
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(guest_code.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(guest_code, &translated);
    uint8_t expected[16];
    for (uint32_t lane = 0; lane < 16; ++lane) {
      expected[lane] = kIndices[lane] < length * 16 ? 0x40 + kIndices[lane] : 0;
    }
    __uint128_t expected_vector;
    memcpy(&expected_vector, expected, sizeof(expected_vector));
    EXPECT_EQ(translated.cpu.v[19], expected_vector) << "length=" << length;
    EXPECT_EQ(translated.cpu.v[19], interpreted.cpu.v[19]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteTblRejectsWideningAndNarrowingArithmetic) {
  // The four TBL lengths share all fixed bits except bit 21 with SADDL2,
  // SSUBL2, ADDHN2 and SUBHN2. The old mask translated ADDHN2 0x4e2440a6
  // as a table lookup, corrupting AV1 reconstruction pixels.
  for (uint32_t op : {0x4e20'0000u, 0x4e20'2000u, 0x4e20'4000u, 0x4e20'6000u}) {
    for (uint32_t rd : {4u, 5u, 6u, 31u}) {
      for (bool mapping : {false, true}) {
        const std::array<uint32_t, 1> guest_code = {op | (4 << 16) | (5 << 5) | rd};
        const GuestAddr pc = ToGuestAddr(guest_code.data());
        MachineCode code;
        LiteTranslateParams params;
        params.end_pc = pc + sizeof(guest_code);
        params.allow_dispatch = false;
        params.enable_reg_mapping = mapping;
        auto [success, stop] = TryLiteTranslateRegion(pc, &code, params);
        EXPECT_FALSE(success) << std::hex << guest_code[0];
        EXPECT_EQ(stop, pc);
      }
    }
  }
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

TEST(LoongArch64RuntimeLibraryTest, LiteHotFallbackSimdFormsMatchInterpreter) {
  constexpr std::array<uint32_t, 8> kGuestCode = {
      0x0f08'a420,  // sshll v0.8h, v1.8b, #0
      0x0f0f'a420,  // sshll v0.8h, v1.8b, #7
      0x4f08'a420,  // sshll2 v0.8h, v1.16b, #0
      0x4f0f'a420,  // sshll2 v0.8h, v1.16b, #7
      0x0e82'7820,  // zip2 v0.2s, v1.2s, v2.2s
      0x0e84'7863,  // zip2 v3.2s, v3.2s, v4.2s (observed Unity opcode)
      0x4e22'e420,  // fcmeq v0.4s, v1.4s, v2.4s
      0x4ea1'd820,  // frecpe v0.4s, v1.4s
  };
  constexpr std::array<std::array<__uint128_t, 4>, 3> kInputs = {{
      {MakeUint128(0x7f80'ff00'0180'fe02, 0x55aa'40c0'81ff'807e),
       MakeUint32x4(0x3f80'0000, 0x8000'0000, 0x7f80'0000, 0x7fc0'1234),
       MakeUint32x4(0x3f80'0000, 0x0000'0000, 0xff80'0000, 0x4000'0000),
       MakeUint32x4(0x1111'1111, 0x2222'2222, 0x3333'3333, 0x4444'4444)},
      {MakeUint128(0x0102'0304'8081'feff, 0x1020'3040'7f80'c0e0),
       MakeUint32x4(0x4000'0000, 0x4080'0000, 0xbf00'0000, 0xc100'0000),
       MakeUint32x4(0x4000'0000, 0x4040'0000, 0xbf00'0000, 0xc000'0000),
       MakeUint32x4(0xaaaa'aaaa, 0xbbbb'bbbb, 0xcccc'cccc, 0xdddd'dddd)},
      {MakeUint128(0xfffe'8180'7f02'0100, 0x8001'fe7f'40c0'55aa),
       MakeUint32x4(0x3f00'0000, 0xc000'0000, 0x0080'0000, 0x7f7f'ffff),
       MakeUint32x4(0xbf00'0000, 0xc000'0000, 0x0000'0000, 0x7f7f'ffff),
       MakeUint32x4(0x0123'4567, 0x89ab'cdef, 0x1357'9bdf, 0x2468'ace0)},
  }};

  for (uint32_t insn : kGuestCode) {
    for (const auto& input : kInputs) {
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      auto initialize = [&](ThreadState* state) {
        state->cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
        state->cpu.v[1] = input[0];
        state->cpu.v[2] = input[2];
        state->cpu.v[3] = input[3];
        state->cpu.v[4] = input[1];
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
}

TEST(LoongArch64RuntimeLibraryTest, LiteHotFallbackScalarFpFormsMatchInterpreter) {
  constexpr std::array<uint32_t, 3> kGuestCode = {
      0x7ea9'd501,  // fabd s1, s8, s9
      0x5e61'd800,  // scvtf d0, d0
      0x1f64'8823,  // fnmsub d3, d1, d4, d2
  };

  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint128(0xffff'ffff'ffff'fff9ULL, UINT64_MAX);  // int64_t -7
      state->cpu.v[1] = MakeUint128(0x4008'0000'0000'0000ULL, UINT64_MAX);  // 3.0
      state->cpu.v[2] = MakeUint128(0x4014'0000'0000'0000ULL, UINT64_MAX);  // 5.0
      state->cpu.v[3] = MakeUint128(UINT64_MAX, UINT64_MAX);
      state->cpu.v[4] = MakeUint128(0x4000'0000'0000'0000ULL, UINT64_MAX);              // 2.0
      state->cpu.v[8] = MakeUint32x4(0xc060'0000, UINT32_MAX, UINT32_MAX, UINT32_MAX);  // -3.5f
      state->cpu.v[9] = MakeUint32x4(0x3fa0'0000, UINT32_MAX, UINT32_MAX, UINT32_MAX);  // 1.25f
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

TEST(LoongArch64RuntimeLibraryTest, LiteUcvtfDFromXFixed64MatchesInterpreter) {
  constexpr uint32_t kUcvtfD0X21Fixed64 = 0x9e43'02a0;
  constexpr std::array<uint64_t, 7> kInputs = {
      0,
      1,
      uint64_t{1} << 52,
      (uint64_t{1} << 63) - 1,
      uint64_t{1} << 63,
      (uint64_t{1} << 63) + 1,
      UINT64_MAX,
  };

  for (uint64_t input : kInputs) {
    const std::array<uint32_t, 1> code = {kUcvtfD0X21Fixed64};
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.x[21] = input;
    translated.cpu.x[21] = input;
    interpreted.cpu.v[0] = MakeUint128(UINT64_MAX, UINT64_MAX);
    translated.cpu.v[0] = interpreted.cpu.v[0];
    SetInsnAddr(interpreted.cpu, ToGuestAddr(code.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(code, &translated);

    SCOPED_TRACE(testing::Message() << "input=" << std::hex << input);
    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteLogicalRotateOperandsMatchInterpreter) {
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x4ac1'08a5,  // eor w5, w5, w1, ror #2 (observed Unity opcode)
      0xcac2'4420,  // eor x0, x1, x2, ror #17
      0x0ac5'1c83,  // and w3, w4, w5, ror #7
      0xeac8'fce6,  // ands x6, x7, x8, ror #63
  };

  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    for (uint32_t i = 0; i < 9; ++i) {
      interpreted.cpu.x[i] = 0x0123'4567'89ab'cdefULL ^ (0x1111'1111'1111'1111ULL * i);
      translated.cpu.x[i] = interpreted.cpu.x[i];
    }
    interpreted.cpu.flags = 0xf;
    translated.cpu.flags = interpreted.cpu.flags;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.x[rd], interpreted.cpu.x[rd]);
    EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteVectorFpImmediateMatchesInterpreter) {
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x4f02'f600,  // fmov v0.4s, #0.25
      0x4f00'f604,  // fmov v4.4s, #4.0
      0x0f00'f485,  // fmov v5.2s, #2.0
      0x6f00'f640,  // fmov v0.2d, #2.0
  };

  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[insn & 31] = MakeUint128(UINT64_MAX, UINT64_MAX);
    translated.cpu.v[insn & 31] = interpreted.cpu.v[insn & 31];
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[insn & 31], interpreted.cpu.v[insn & 31]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteVectorFminFormsMatchInterpreter) {
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x0ea2'f420,  // fmin v0.2s, v1.2s, v2.2s
      0x4ea2'f420,  // fmin v0.4s, v1.4s, v2.4s (observed Unity opcode)
      0x4ee2'f420,  // fmin v0.2d, v1.2d, v2.2d
      0x2ea2'f420,  // fminp v0.2s, v1.2s, v2.2s
  };
  constexpr std::array<std::array<__uint128_t, 2>, 3> kInputs = {{
      {MakeUint32x4(0x3f80'0000, 0xc000'0000, 0x40a0'0000, 0xc0e0'0000),
       MakeUint32x4(0x4000'0000, 0xbf80'0000, 0x4080'0000, 0xc100'0000)},
      {MakeUint32x4(0x0000'0000, 0x8000'0000, 0x7fc0'0000, 0x3f80'0000),
       MakeUint32x4(0x8000'0000, 0x0000'0000, 0x4000'0000, 0x7fc0'0000)},
      {MakeUint128(0x3ff0'0000'0000'0000ULL, 0xc008'0000'0000'0000ULL),
       MakeUint128(0x4000'0000'0000'0000ULL, 0xc010'0000'0000'0000ULL)},
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

      SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
      EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteUnsignedScalarVectorConvertMatchesInterpreter) {
  constexpr std::array<uint32_t, 2> kGuestCode = {
      0x7e21'd821,  // ucvtf s1, s1 (observed Unity opcode)
      0x7e61'd820,  // ucvtf d0, d1
  };
  constexpr std::array<uint64_t, 5> kInputs = {
      0,
      1,
      UINT32_MAX,
      0x7fff'ffff'ffff'ffffULL,
      UINT64_MAX,
  };

  for (uint32_t insn : kGuestCode) {
    for (uint64_t input : kInputs) {
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      interpreted.cpu.v[1] = MakeUint128(input, UINT64_MAX);
      translated.cpu.v[1] = interpreted.cpu.v[1];
      SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

      InterpretInsn(&interpreted);
      TranslateAndRun(one_insn, &translated);

      SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn << " input=" << input);
      EXPECT_EQ(translated.cpu.v[insn & 31], interpreted.cpu.v[insn & 31]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarFusedSignFormsMatchInterpreter) {
  constexpr std::array<uint32_t, 8> kGuestCode = {
      0x1f02'0c20,  // fmadd s0, s1, s2, s3
      0x1f02'8c20,  // fmsub s0, s1, s2, s3
      0x1f22'0c20,  // fnmadd s0, s1, s2, s3
      0x1f22'8c20,  // fnmsub s0, s1, s2, s3
      0x1f42'0c20,  // fmadd d0, d1, d2, d3
      0x1f42'8c20,  // fmsub d0, d1, d2, d3 (observed Unity class)
      0x1f62'0c20,  // fnmadd d0, d1, d2, d3
      0x1f62'8c20,  // fnmsub d0, d1, d2, d3
  };

  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[1] = MakeUint128(0x4008'0000'4040'0000ULL, 0);  // 3.0 / 3.0f
      state->cpu.v[2] = MakeUint128(0x4000'0000'4000'0000ULL, 0);  // 2.0 / 2.0f
      state->cpu.v[3] = MakeUint128(0x4014'0000'40a0'0000ULL, 0);  // 5.0 / 5.0f
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

TEST(LoongArch64RuntimeLibraryTest, LiteStructMemoryHasRecoveryPoints) {
  constexpr std::array<uint32_t, 11> kGuestCode = {
      0x4c40'2820,  // ld1 {v0.4s-v3.4s},[x1]
      0x4c9f'2820,  // st1 {v0.4s-v3.4s},[x1],#64
      0x4c40'0820,  // ld4 {v0.4s-v3.4s},[x1]
      0x4d40'cd07,  // ld1r {v7.2d},[x8]
      0x4ddf'c862,  // ld1r {v2.4s},[x3],#4
      0x0d40'8420,  // ld1 {v0.d}[0],[x1]
      0x4ddf'8528,  // ld1 {v8.d}[1],[x9],#8
      0x0d00'84a4,  // st1 {v4.d}[0],[x5]
      0x4d91'860f,  // st1 {v15.d}[1],[x16],x17
      0x4d9f'8080,  // st1 {v0.s}[2],[x4],#4
      0x4dc5'8080,  // ld1 {v0.s}[2],[x4],x5
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
    size_t expected_recovery_points = 1;
    if (insn == 0x4c40'2820 || insn == 0x4c9f'2820) {
      expected_recovery_points = 8;
    } else if (insn == 0x4c40'0820) {
      expected_recovery_points = 4;
    }
    EXPECT_EQ(exec.recovery_map().size(), expected_recovery_points);
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

TEST(LoongArch64RuntimeLibraryTest, LiteInsSFromGeneralMatchesInterpreter) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x4e04'1c20,  // mov v0.s[0],w1
      0x4e0c'1c62,  // mov v2.s[1],w3
      0x4e14'1ca4,  // mov v4.s[2],w5
      0x4e1c'1ce6,  // mov v6.s[3],w7
      0x4e1c'1fe8,  // mov v8.s[3],wzr
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    interpreted.cpu.x[rn] = 0xfeed'face'1234'5678;
    interpreted.cpu.v[rd] = MakeUint128(0x0123'4567'89ab'cdef, 0xfedc'ba98'7654'3210);
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, UnityInsGeneralSequenceMatchesInterpreter) {
  // Real libunity sequence observed immediately after LD1R during the original
  // rendering-corruption diagnosis.  The whole sequence, including its
  // partial general-register-to-vector lane writes, must now stay in Lite JIT.
  constexpr std::array<uint32_t, 9> kFullSequence = {
      0x4e04'0ff2,  // dup v18.4s,wzr
      0x4e04'1ef2,  // mov v18.s[0],w23
      0x4e0c'1f12,  // mov v18.s[1],w24
      0x4e14'1ed2,  // mov v18.s[2],w22
      0x4e31'1e51,  // and v17.16b,v18.16b,v17.16b
      0x0b15'06b5,  // add w21,w21,w21,lsl #1
      0x6e21'da31,  // ucvtf v17.4s,v17.4s
      0x2a1f'03f3,  // mov w19,wzr
      0x6e31'de10,  // fmul v16.4s,v16.4s,v17.4s
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

  ThreadState translated{};
  initialize(&translated);
  {
    GuestAddr start_pc = ToGuestAddr(kFullSequence.data());
    MachineCode code;
    LiteTranslateParams params;
    params.end_pc = start_pc + sizeof(kFullSequence);
    params.allow_dispatch = false;
    auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
    ASSERT_TRUE(success);
    ASSERT_EQ(stop_pc, params.end_pc);
  }
  TranslateAndRun(kFullSequence, &translated);

  EXPECT_EQ(translated.cpu.x[19], interpreted.cpu.x[19]);
  EXPECT_EQ(translated.cpu.x[21], interpreted.cpu.x[21]);
  EXPECT_EQ(translated.cpu.v[16], interpreted.cpu.v[16]);
  EXPECT_EQ(translated.cpu.v[17], interpreted.cpu.v[17]);
  EXPECT_EQ(translated.cpu.v[18], interpreted.cpu.v[18]);
}

TEST(LoongArch64RuntimeLibraryTest, LiteHotStructMemoryMatchesInterpreter) {
  constexpr std::array<uint32_t, 32> kGuestCode = {
      0x4c9f'a820,  // st1 {v0.4s,v1.4s},[x1],#32
      0x0ddf'8464,  // ld1 {v4.d}[0],[x3],#8
      0x4ddf'8464,  // ld1 {v4.d}[1],[x3],#8
      0x4d40'c930,  // ld1r {v16.4s},[x9]
      0x4d40'cd07,  // ld1r {v7.2d},[x8]
      0x4ddf'cd49,  // ld1r {v9.2d},[x10],#8
      0x4dcd'cd8b,  // ld1r {v11.2d},[x12],x13
      0x4ddf'c862,  // ld1r {v2.4s},[x3],#4
      0x4dc6'c8a4,  // ld1r {v4.4s},[x5],x6
      0x0d40'8420,  // ld1 {v0.d}[0],[x1]
      0x4d40'8462,  // ld1 {v2.d}[1],[x3]
      0x0d00'84a4,  // st1 {v4.d}[0],[x5]
      0x4d00'84e6,  // st1 {v6.d}[1],[x7]
      0x4ddf'8528,  // ld1 {v8.d}[1],[x9],#8
      0x4dcc'856a,  // ld1 {v10.d}[1],[x11],x12
      0x0d9f'85cd,  // st1 {v13.d}[0],[x14],#8
      0x4d91'860f,  // st1 {v15.d}[1],[x16],x17
      0x4d00'8121,  // st1 {v1.s}[2],[x9]
      0x0d00'8101,  // st1 {v1.s}[0],[x8]
      0x0d00'9101,  // st1 {v1.s}[1],[x8]
      0x4d00'9101,  // st1 {v1.s}[3],[x8]
      0x0d40'8100,  // ld1 {v0.s}[0],[x8]
      0x0d40'9100,  // ld1 {v0.s}[1],[x8]
      0x4d40'8100,  // ld1 {v0.s}[2],[x8]
      0x4d40'9100,  // ld1 {v0.s}[3],[x8]
      0x4d9f'8080,  // st1 {v0.s}[2],[x4],#4
      0x4d85'8080,  // st1 {v0.s}[2],[x4],x5
      0x4ddf'8080,  // ld1 {v0.s}[2],[x4],#4
      0x4dc5'8080,  // ld1 {v0.s}[2],[x4],x5
      0x4c40'091c,  // ld4 {v28.4s-v31.4s},[x8]
      0x4cdf'0950,  // ld4 {v16.4s-v19.4s},[x10],#64
      0x4cc9'0904,  // ld4 {v4.4s-v7.4s},[x8],x9
  };

  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    const std::array<uint32_t, 1> one_insn = {kGuestCode[i]};
    alignas(16) std::array<uint64_t, 8> interpreted_memory = {0x0123'4567'89ab'cdef,
                                                              0x1122'3344'5566'7788,
                                                              0x99aa'bbcc'ddee'ff00,
                                                              0xfedc'ba98'7654'3210,
                                                              0x0f1e'2d3c'4b5a'6978,
                                                              0x8877'6655'4433'2211,
                                                              0x1357'9bdf'2468'ace0,
                                                              0xdead'beef'cafe'babe};
    alignas(16) std::array<uint64_t, 8> translated_memory = interpreted_memory;
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
    const uint32_t rm = (one_insn[0] >> 16) & 31;
    if ((one_insn[0] & (1u << 23)) != 0 && rm != 31) {
      interpreted.cpu.x[rm] = 64;
      translated.cpu.x[rm] = 64;
    }
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));

    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);

    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << one_insn[0]);
    EXPECT_EQ(translated_memory, interpreted_memory);
    for (uint32_t reg = 0; reg < 32; ++reg) {
      EXPECT_EQ(translated.cpu.v[reg], interpreted.cpu.v[reg]) << reg;
    }
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

TEST(LoongArch64RuntimeLibraryTest, LiteAllBranchConditionsMatchInterpreter) {
  for (uint32_t condition = 0; condition < 16; ++condition) {
    for (uint32_t flags = 0; flags < 16; ++flags) {
      const std::array<uint32_t, 1> code = {0x5400'0040u | condition};  // b.cond +8
      ThreadState interpreted{};
      ThreadState translated{};
      interpreted.cpu.flags = translated.cpu.flags = flags;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(code.data()));
      InterpretInsn(&interpreted);
      TranslateAndRun(code, &translated);
      SCOPED_TRACE(testing::Message() << "cond=" << condition << " flags=" << flags);
      EXPECT_EQ(GetInsnAddr(translated.cpu), GetInsnAddr(interpreted.cpu));
      EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteAllSelectConditionsMatchInterpreter) {
  for (uint32_t sf : {0u, 1u}) {
    for (uint32_t variant = 0; variant < 4; ++variant) {
      for (uint32_t condition = 0; condition < 16; ++condition) {
        for (uint32_t flags = 0; flags < 16; ++flags) {
          // CSEL / CSINC / CSINV / CSNEG W/X0,W/X1,W/X2,cond.
          const std::array<uint32_t, 1> code = {0x1a82'0020u | (sf << 31) | ((variant >> 1) << 30) |
                                                ((variant & 1) << 10) | (condition << 12)};
          ThreadState interpreted{};
          ThreadState translated{};
          for (ThreadState* state : {&interpreted, &translated}) {
            state->cpu.x[1] = 0xfedc'ba98'7654'3210ULL;
            state->cpu.x[2] = 0x8000'0000'ffff'ffffULL;
            state->cpu.flags = flags;
            SetInsnAddr(state->cpu, ToGuestAddr(code.data()));
          }
          InterpretInsn(&interpreted);
          TranslateAndRun(code, &translated);
          SCOPED_TRACE(testing::Message() << std::hex << code[0] << " flags=" << flags);
          EXPECT_EQ(translated.cpu.x[0], interpreted.cpu.x[0]);
          EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteShiftedOperandsPreserveWidthAndFlags) {
  // ADD/ADDS/SUB/SUBS and all logical shifted-register variants.
  constexpr std::array<uint32_t, 12> kOpcodes = {0x0b00'0000,
                                                 0x2b00'0000,
                                                 0x4b00'0000,
                                                 0x6b00'0000,
                                                 0x0a00'0000,
                                                 0x2a00'0000,
                                                 0x4a00'0000,
                                                 0x6a00'0000,
                                                 0x0a20'0000,
                                                 0x2a20'0000,
                                                 0x4a20'0000,
                                                 0x6a20'0000};
  constexpr std::array<uint64_t, 4> kValues = {
      0, UINT64_MAX, 0x8000'0000'8000'0000ULL, 0x0123'4567'ffff'ffffULL};
  for (uint32_t opcode : kOpcodes) {
    for (uint32_t sf : {0u, 1u}) {
      for (uint32_t shift = 0; shift < 4; ++shift) {
        if (shift == 3 && (opcode & 0x0100'0000u)) {
          continue;  // Arithmetic shifted-register forms do not allow ROR.
        }
        for (uint32_t amount : {0u, 1u, 31u, 63u}) {
          if (sf == 0 && amount == 63) {
            continue;
          }
          for (uint64_t value : kValues) {
            const std::array<uint32_t, 1> code = {opcode | (sf << 31) | (shift << 22) | (2u << 16) |
                                                  (amount << 10) | (1u << 5)};
            ThreadState interpreted{};
            ThreadState translated{};
            for (ThreadState* state : {&interpreted, &translated}) {
              state->cpu.x[1] = ~value;
              state->cpu.x[2] = value;
              state->cpu.flags = 0xa;
              SetInsnAddr(state->cpu, ToGuestAddr(code.data()));
            }
            InterpretInsn(&interpreted);
            TranslateAndRun(code, &translated);
            SCOPED_TRACE(testing::Message() << std::hex << code[0] << " value=" << value);
            EXPECT_EQ(translated.cpu.x[0], interpreted.cpu.x[0]);
            EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteDirectCachedArithmeticSourcesPreserveAliasing) {
  InitHostEntries();
  for (bool mapped : {false, true}) {
    for (uint32_t sub : {0u, 1u}) {
      for (uint32_t shift : {0u, 1u, 2u}) {
        for (uint32_t rn : {0u, 1u, 2u, 31u}) {
          for (uint32_t rm : {0u, 1u, 2u, 31u}) {
            for (uint32_t rd : {0u, 1u, 2u, 31u}) {
              std::array<uint32_t, 8> code;
              code.fill(0x8b00'0000u | (sub << 30) | (shift << 22) | (rm << 16) | (rn << 5) | rd);
              GuestAddr start_pc = ToGuestAddr(code.data());
              ThreadState interpreted{};
              ThreadState translated{};
              for (ThreadState* state : {&interpreted, &translated}) {
                for (size_t i = 0; i < 31; ++i) {
                  state->cpu.x[i] = 0xfedc'ba98'7654'3210ULL + i * 0x1111'1111ULL;
                }
                state->cpu.sp = 0x1234'5678;
                state->cpu.flags = 0xb;
                SetInsnAddr(state->cpu, start_pc);
              }
              for (size_t i = 0; i < code.size(); ++i) {
                InterpretInsn(&interpreted);
              }
              MachineCode machine_code;
              LiteTranslateParams params;
              params.end_pc = start_pc + sizeof(code);
              params.allow_dispatch = false;
              params.enable_reg_mapping = mapped;
              auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &machine_code, params);
              ASSERT_TRUE(success);
              ASSERT_EQ(stop_pc, params.end_pc);
              ScopedExecRegion exec(&machine_code);
              SetResidence(translated, kOutsideGeneratedCode);
              berberis_RunGeneratedCode(&translated, AsHostCode(exec.GetHostCodeAddr()));
              SCOPED_TRACE(testing::Message() << std::hex << code[0] << " mapped=" << mapped);
              for (size_t i = 0; i < 31; ++i) {
                EXPECT_EQ(translated.cpu.x[i], interpreted.cpu.x[i]) << "reg=" << i;
              }
              EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
              EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
              EXPECT_EQ(GetInsnAddr(translated.cpu), GetInsnAddr(interpreted.cpu));
            }
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteCachedSpPreservesArithmeticAndZeroRegister) {
  constexpr std::array<uint32_t, 10> kCode = {
      0x9100'83ff,  // add sp,sp,#32
      0x9100'23e0,  // add x0,sp,#8
      0xd100'43ff,  // sub sp,sp,#16
      0x1100'13ff,  // add wsp,wsp,#4 (discard high bits)
      0x9100'03e1,  // mov x1,sp
      0xb27c'003f,  // orr sp,x1,#0x10
      0x8b3f'63ff,  // add sp,sp,xzr (extended register)
      0xaa1f'03e2,  // mov x2,xzr
      0x8b1f'03e3,  // add x3,xzr,xzr
      0x9100'03e5,  // mov x5,sp
  };
  InitHostEntries();
  for (bool mapped : {false, true}) {
    for (uint64_t sp : {0ULL, 0xffff'fff0ULL, 0xab00'1234'ffff'fff0ULL}) {
      // Every prefix also checks architectural state at a region exit.
      for (size_t count = 1; count <= kCode.size(); ++count) {
        GuestAddr start_pc = ToGuestAddr(kCode.data());
        ThreadState interpreted{};
        ThreadState translated{};
        for (ThreadState* state : {&interpreted, &translated}) {
          state->cpu.sp = sp;
          state->cpu.flags = 0xb;
          SetInsnAddr(state->cpu, start_pc);
        }
        for (size_t i = 0; i < count; ++i) {
          InterpretInsn(&interpreted);
        }
        LiteTranslateParams params;
        params.end_pc = start_pc + count * sizeof(uint32_t);
        params.enable_reg_mapping = mapped;
        params.allow_dispatch = false;
        MachineCode code;
        auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
        ASSERT_TRUE(success);
        ASSERT_EQ(stop_pc, params.end_pc);
        ScopedExecRegion exec(&code);
        SetResidence(translated, kOutsideGeneratedCode);
        berberis_RunGeneratedCode(&translated, AsHostCode(exec.GetHostCodeAddr()));
        SCOPED_TRACE(testing::Message() << count << " mapped=" << mapped << " sp=" << sp);
        for (size_t i = 0; i < 31; ++i) {
          EXPECT_EQ(translated.cpu.x[i], interpreted.cpu.x[i]) << "reg=" << i;
        }
        EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
        EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
        EXPECT_EQ(GetInsnAddr(translated.cpu), GetInsnAddr(interpreted.cpu));
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteCachedSpPreservesTaggedMemoryWriteback) {
  constexpr std::array<uint32_t, 10> kCode = {
      0xa9bf'0be1,  // stp x1,x2,[sp,#-16]!
      0xf940'03e0,  // ldr x0,[sp]
      0xa8c1'13e3,  // ldp x3,x4,[sp],#16
      0xf900'0bff,  // str xzr,[sp,#16]
      0xf940'0bff,  // ldr xzr,[sp,#16]
      0xf81f'0fe1,  // str x1,[sp,#-16]!
      0xf841'07e4,  // ldr x4,[sp],#16
      0x3c9f'0fe0,  // str q0,[sp,#-16]!
      0x3cc1'07e1,  // ldr q1,[sp],#16
      0x9100'03e5,  // mov x5,sp
  };
  InitHostEntries();
  for (bool mapped : {false, true}) {
    for (uint64_t tag : {0ULL, 0xab00'0000'0000'0000ULL}) {
      for (size_t count = 1; count <= kCode.size(); ++count) {
        alignas(16) std::array<uint64_t, 16> memory;
        memory.fill(0xa5a5'a5a5'a5a5'a5a5ULL);
        const auto initial_memory = memory;
        GuestAddr start_pc = ToGuestAddr(kCode.data());
        ThreadState interpreted{};
        ThreadState translated{};
        for (ThreadState* state : {&interpreted, &translated}) {
          state->cpu.sp = ToGuestAddr(memory.data() + 8) | tag;
          state->cpu.x[1] = 0x0123'4567'89ab'cdef;
          state->cpu.x[2] = 0xfedc'ba98'7654'3210;
          state->cpu.v[0] = static_cast<__uint128_t>(state->cpu.x[1]) << 64 | state->cpu.x[2];
          state->cpu.flags = 0xb;
          SetInsnAddr(state->cpu, start_pc);
        }
        for (size_t i = 0; i < count; ++i) {
          InterpretInsn(&interpreted);
        }
        const auto expected_memory = memory;
        memory = initial_memory;
        LiteTranslateParams params;
        params.end_pc = start_pc + count * sizeof(uint32_t);
        params.enable_reg_mapping = mapped;
        params.enable_guest_memory = true;
        params.allow_dispatch = false;
        MachineCode code;
        auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
        ASSERT_TRUE(success);
        ASSERT_EQ(stop_pc, params.end_pc);
        ScopedExecRegion exec(&code);
        EXPECT_FALSE(exec.recovery_map().empty());
        SetResidence(translated, kOutsideGeneratedCode);
        berberis_RunGeneratedCode(&translated, AsHostCode(exec.GetHostCodeAddr()));
        SCOPED_TRACE(testing::Message() << count << " mapped=" << mapped << " tag=" << tag);
        EXPECT_EQ(memory, expected_memory);
        for (size_t i = 0; i < 31; ++i) {
          EXPECT_EQ(translated.cpu.x[i], interpreted.cpu.x[i]) << "reg=" << i;
        }
        EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
        EXPECT_EQ(translated.cpu.v[1], interpreted.cpu.v[1]);
        EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
        EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
        EXPECT_EQ(GetInsnAddr(translated.cpu), GetInsnAddr(interpreted.cpu));
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteCachedSpWritesThroughAtSignalPollAndReloadsAtEntry) {
  constexpr std::array<uint32_t, 4> kCode = {
      0xd100'43ff,  // sub sp,sp,#16
      0xd100'0400,  // sub x0,x0,#1
      0xb5ff'ffc0,  // cbnz x0,-8
      0x9100'03e1,  // mov x1,sp
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kCode.data());
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kCode);
  params.enable_reg_mapping = true;
  params.allow_dispatch = false;
  MachineCode code;
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, params.end_pc);
  ScopedExecRegion exec(&code);
  ThreadState state{};
  for (bool pending : {true, false}) {
    state.cpu.sp = pending ? 0x1000 : 0x2000;
    state.cpu.x[0] = 3;
    state.pending_signals_status.store(pending ? kPendingSignalsPresent : 0);
    SetInsnAddr(state.cpu, start_pc);
    SetResidence(state, kOutsideGeneratedCode);
    berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
    EXPECT_EQ(state.cpu.sp, pending ? 0xff0u : 0x1fd0u);
    EXPECT_EQ(state.cpu.x[0], pending ? 2u : 0u);
    EXPECT_EQ(GetInsnAddr(state.cpu), pending ? start_pc : params.end_pc);
    if (!pending) {
      EXPECT_EQ(state.cpu.x[1], state.cpu.sp);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteDirectCachedLogicalSourcesPreserveAliasing) {
  InitHostEntries();
  for (bool mapped : {false, true}) {
    for (uint32_t sf : {0u, 1u}) {
      for (uint32_t opc = 0; opc < 4; ++opc) {
        for (uint32_t shift = 0; shift < 4; ++shift) {
          for (uint32_t rn : {0u, 1u, 2u, 31u}) {
            for (uint32_t rm : {0u, 1u, 2u, 31u}) {
              for (uint32_t rd : {0u, 1u, 2u, 31u}) {
                std::array<uint32_t, 8> code;
                code.fill(0x0a00'0000u | (sf << 31) | (opc << 29) | (shift << 22) | (rm << 16) |
                          (rn << 5) | rd);
                GuestAddr start_pc = ToGuestAddr(code.data());
                ThreadState interpreted{};
                ThreadState translated{};
                for (ThreadState* state : {&interpreted, &translated}) {
                  for (size_t i = 0; i < 31; ++i) {
                    state->cpu.x[i] = 0xfedc'ba98'7654'3210ULL + i * 0x1111'1111ULL;
                  }
                  state->cpu.sp = 0xab00'1234'5678'9000;
                  state->cpu.flags = 0xb;
                  SetInsnAddr(state->cpu, start_pc);
                }
                for (size_t i = 0; i < code.size(); ++i) {
                  InterpretInsn(&interpreted);
                }
                LiteTranslateParams params;
                params.end_pc = start_pc + sizeof(code);
                params.enable_reg_mapping = mapped;
                params.allow_dispatch = false;
                MachineCode machine_code;
                auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &machine_code, params);
                ASSERT_TRUE(success);
                ASSERT_EQ(stop_pc, params.end_pc);
                ScopedExecRegion exec(&machine_code);
                SetResidence(translated, kOutsideGeneratedCode);
                berberis_RunGeneratedCode(&translated, AsHostCode(exec.GetHostCodeAddr()));
                SCOPED_TRACE(testing::Message() << std::hex << code[0] << " mapped=" << mapped);
                for (size_t i = 0; i < 31; ++i) {
                  EXPECT_EQ(translated.cpu.x[i], interpreted.cpu.x[i]) << "reg=" << i;
                }
                EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
                EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
                EXPECT_EQ(GetInsnAddr(translated.cpu), GetInsnAddr(interpreted.cpu));
              }
            }
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, HostBitExtractionAndWordExtensionMatchIntegerReference) {
  using Assembler = loongarch64::Assembler;
  InitHostEntries();
  for (uint32_t msb = 0; msb < 64; ++msb) {
    for (uint32_t lsb = 0; lsb <= msb; ++lsb) {
      MachineCode code;
      Assembler as(&code);
      as.LdD(Assembler::t0, Assembler::s8, offsetof(ThreadState, cpu) + offsetof(CPUState, x[0]));
      as.BstrpickD(Assembler::t1, Assembler::t0, msb, lsb);
      as.StD(Assembler::t1, Assembler::s8, offsetof(ThreadState, cpu) + offsetof(CPUState, x[1]));
      as.AddiW(Assembler::t0, Assembler::t0, 0);
      as.StD(Assembler::t0, Assembler::s8, offsetof(ThreadState, cpu) + offsetof(CPUState, x[2]));
      as.Li(Assembler::t0, kEntryExitGeneratedCode);
      as.Jirl(Assembler::zero, Assembler::t0, 0);
      as.Finalize();
      ScopedExecRegion exec(&code);
      for (uint64_t value : {0ULL,
                             ~0ULL,
                             0x0123'4567'89ab'cdefULL,
                             0xfedc'ba98'7654'3210ULL,
                             0xab00'0000'8000'0000ULL,
                             0x8000'0000'7fff'ffffULL}) {
        ThreadState state{};
        state.cpu.x[0] = value;
        SetResidence(state, kOutsideGeneratedCode);
        berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
        uint32_t width = msb - lsb + 1;
        uint64_t mask = width == 64 ? UINT64_MAX : (uint64_t{1} << width) - 1;
        EXPECT_EQ(state.cpu.x[1], (value >> lsb) & mask) << msb << ':' << lsb;
        uint64_t word = value & UINT32_MAX;
        EXPECT_EQ(state.cpu.x[2], (word & 0x8000'0000) ? word | 0xffff'ffff'0000'0000ULL : word);
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteDemandCacheSkipsWriteFirstLoadsAndHandlesReentry) {
  constexpr std::array<uint32_t, 7> kCode = {
      0xd280'0080,  // mov x0,#4
      0xd280'0061,  // mov x1,#3
      0x9100'0421,  // add x1,x1,#1
      0xd100'0400,  // sub x0,x0,#1
      0xb5ff'ffc0,  // cbnz x0,-8
      0x8b01'0022,  // add x2,x1,x1
      0x8b00'0003,  // add x3,x0,x0
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kCode.data());
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kCode);
  params.allow_dispatch = false;
  MachineCode code;
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, params.end_pc);
  // Both cached values are written before any read. There must be no load
  // from ThreadState into s0-s6 anywhere, including the loop target.
  for (uint32_t offset = 0; offset < code.install_size(); offset += 4) {
    uint32_t insn = *code.AddrAs<const uint32_t>(offset);
    bool cache_load = (insn & 0xffc0'0000) == 0x28c0'0000 && ((insn >> 5) & 31) == 31 &&
                      (insn & 31) >= 23 && (insn & 31) <= 29;
    EXPECT_FALSE(cache_load) << offset;
  }
  ScopedExecRegion exec(&code);
  ThreadState state{};
  for (bool pending : {true, false}) {
    state.cpu.x[0] = UINT64_MAX;
    state.cpu.x[1] = 0xdead'beef;
    state.pending_signals_status.store(pending ? kPendingSignalsPresent : 0);
    SetInsnAddr(state.cpu, start_pc);
    SetResidence(state, kOutsideGeneratedCode);
    berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
    EXPECT_EQ(state.cpu.x[0], pending ? 3u : 0u);
    EXPECT_EQ(state.cpu.x[1], pending ? 4u : 7u);
    EXPECT_EQ(GetInsnAddr(state.cpu), pending ? start_pc + 8 : params.end_pc);
    if (!pending) {
      EXPECT_EQ(state.cpu.x[2], 14u);
      EXPECT_EQ(state.cpu.x[3], 0u);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteDemandCacheHandlesEarlyExitAndReadFirstBackedge) {
  constexpr std::array<uint32_t, 5> kCode = {
      0xb500'00b4,  // cbnz x20,end
      0x9100'0421,  // add x1,x1,#1
      0xd100'0400,  // sub x0,x0,#1
      0xb5ff'ffc0,  // cbnz x0,-8
      0x8b01'0022,  // add x2,x1,x1
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kCode.data());
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kCode);
  params.allow_dispatch = false;
  MachineCode code;
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, params.end_pc);
  int32_t last_cache_load = -1;
  for (uint32_t offset = 0; offset < code.install_size(); offset += 4) {
    uint32_t insn = *code.AddrAs<const uint32_t>(offset);
    if ((insn & 0xffc0'0000) == 0x28c0'0000 && ((insn >> 5) & 31) == 31 && (insn & 31) >= 23 &&
        (insn & 31) <= 29) {
      last_cache_load = offset;
    }
  }
  ASSERT_GE(last_cache_load, 0);
  size_t backedges = 0;
  for (uint32_t offset = 0; offset < code.install_size(); offset += 4) {
    uint32_t insn = *code.AddrAs<const uint32_t>(offset);
    if ((insn & 0xfc00'0000) != 0x5000'0000) {
      continue;
    }
    uint32_t imm26 = ((insn & 0x3ff) << 16) | ((insn >> 10) & 0xffff);
    int64_t displacement = (static_cast<int64_t>(imm26 ^ (1u << 25)) - (1u << 25)) * 4;
    if (displacement < 0) {
      ++backedges;
      EXPECT_GT(static_cast<int64_t>(offset) + displacement, last_cache_load);
    }
  }
  EXPECT_GT(backedges, 0u);
  ScopedExecRegion exec(&code);
  for (bool taken : {true, false}) {
    for (bool pending : {true, false}) {
      ThreadState state{};
      state.cpu.x[0] = 3;
      state.cpu.x[1] = 100;
      state.cpu.x[2] = 0x1234;
      state.cpu.x[20] = taken;
      state.pending_signals_status.store(pending ? kPendingSignalsPresent : 0);
      SetInsnAddr(state.cpu, start_pc);
      SetResidence(state, kOutsideGeneratedCode);
      berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
      EXPECT_EQ(state.cpu.x[0], taken ? 3u : (pending ? 2u : 0u));
      EXPECT_EQ(state.cpu.x[1], taken ? 100u : (pending ? 101u : 103u));
      EXPECT_EQ(state.cpu.x[2], taken || pending ? 0x1234u : 206u);
      EXPECT_EQ(GetInsnAddr(state.cpu), !taken && pending ? start_pc + 4 : params.end_pc);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteDemandCacheHoistsAcrossNestedLoops) {
  constexpr std::array<uint32_t, 9> kCode = {
      0xd280'0040,  // mov x0,#2
      0xd280'0061,  // outer: mov x1,#3
      0xb500'00f4,  // inner: cbnz x20,end
      0x9100'0442,  // add x2,x2,#1
      0xd100'0421,  // sub x1,x1,#1
      0xb5ff'ffa1,  // cbnz x1,inner
      0xd100'0400,  // sub x0,x0,#1
      0xb5ff'ff40,  // cbnz x0,outer
      0x8b02'0043,  // add x3,x2,x2
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kCode.data());
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kCode);
  params.allow_dispatch = false;
  MachineCode code;
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, params.end_pc);
  int32_t last_cache_load = -1;
  size_t loads = 0;
  for (uint32_t offset = 0; offset < code.install_size(); offset += 4) {
    uint32_t insn = *code.AddrAs<const uint32_t>(offset);
    if ((insn & 0xffc0'0000) == 0x28c0'0000 && ((insn >> 5) & 31) == 31 && (insn & 31) >= 23 &&
        (insn & 31) <= 29) {
      last_cache_load = offset;
      ++loads;
    }
  }
  ASSERT_EQ(loads, 1u);  // Only x2 needs its entry value; x0/x1 are write-first.
  size_t backedges = 0;
  for (uint32_t offset = 0; offset < code.install_size(); offset += 4) {
    uint32_t insn = *code.AddrAs<const uint32_t>(offset);
    if ((insn & 0xfc00'0000) != 0x5000'0000) {
      continue;
    }
    uint32_t imm26 = ((insn & 0x3ff) << 16) | ((insn >> 10) & 0xffff);
    int64_t displacement = (static_cast<int64_t>(imm26 ^ (1u << 25)) - (1u << 25)) * 4;
    if (displacement < 0) {
      ++backedges;
      EXPECT_GT(static_cast<int64_t>(offset) + displacement, last_cache_load);
    }
  }
  EXPECT_EQ(backedges, 2u);
  ScopedExecRegion exec(&code);
  for (bool pending : {false, true}) {
    for (bool taken : {false, true}) {
      ThreadState state{};
      state.cpu.x[2] = 10;
      state.cpu.x[20] = taken;
      state.pending_signals_status.store(pending ? kPendingSignalsPresent : 0);
      SetInsnAddr(state.cpu, start_pc);
      SetResidence(state, kOutsideGeneratedCode);
      berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
      EXPECT_EQ(state.cpu.x[2], taken ? 10u : (pending ? 11u : 16u));
      EXPECT_EQ(state.cpu.x[0], taken || pending ? 2u : 0u);
      EXPECT_EQ(GetInsnAddr(state.cpu), !taken && pending ? start_pc + 8 : params.end_pc);
      if (!taken && !pending) {
        EXPECT_EQ(state.cpu.x[3], 32u);
      }
    }
  }
}

volatile sig_atomic_t g_demand_fault_seen;
uintptr_t g_demand_fault_pc;
uintptr_t g_demand_recovery_pc;

void RecoverDemandCacheTestFault(int signo, siginfo_t*, void* context) {
  auto* uc = static_cast<ucontext_t*>(context);
  if (uc->uc_mcontext.sc_pc != g_demand_fault_pc) {
    _exit(128 + signo);
  }
  g_demand_fault_seen = 1;
  uc->uc_mcontext.sc_pc = g_demand_recovery_pc;
}

TEST(LoongArch64RuntimeLibraryTest, LiteDemandCachePreservesStateAtMemoryFault) {
  constexpr std::array<uint32_t, 5> kCode = {
      0xd100'43ff,  // sub sp,sp,#16
      0xf841'07e1,  // ldr x1,[sp],#16: faults before writeback
      0x9100'03e2,  // mov x2,sp
      0x9100'03e2,  // mov x2,sp
      0x8b01'0023,  // add x3,x1,x1
  };
  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(kCode.data());
  LiteTranslateParams params;
  params.end_pc = start_pc + sizeof(kCode);
  params.allow_dispatch = false;
  MachineCode code;
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop_pc, params.end_pc);
  ScopedExecRegion exec(&code);
  ASSERT_EQ(exec.recovery_map().size(), 1u);
  auto [fault, recovery] = *exec.recovery_map().begin();
  g_demand_fault_pc = fault;
  g_demand_recovery_pc = recovery;
  g_demand_fault_seen = 0;
  void* guard = mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(guard, MAP_FAILED);
  ThreadState state{};
  const GuestAddr tagged_guard = ToGuestAddr(guard) | 0xab00'0000'0000'0000ULL;
  state.cpu.sp = tagged_guard + 16;
  state.cpu.x[1] = 0x1234;
  state.cpu.x[2] = 0x5678;
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  struct sigaction action{}, previous{};
  action.sa_sigaction = RecoverDemandCacheTestFault;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  ASSERT_EQ(sigaction(SIGSEGV, &action, &previous), 0);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  int restore_result = sigaction(SIGSEGV, &previous, nullptr);
  int unmap_result = munmap(guard, 4096);
  EXPECT_EQ(restore_result, 0);
  EXPECT_EQ(unmap_result, 0);
  EXPECT_EQ(g_demand_fault_seen, 1);
  EXPECT_EQ(state.cpu.sp, tagged_guard);
  EXPECT_EQ(state.cpu.x[1], 0x1234u);
  EXPECT_EQ(state.cpu.x[2], 0x5678u);
  EXPECT_EQ(GetInsnAddr(state.cpu), start_pc + 4);
  EXPECT_EQ(GetResidence(state), kOutsideGeneratedCode);
}

TEST(LoongArch64RuntimeLibraryTest, LiteDirectCachedWordArithmeticSourcesPreserveAliasing) {
  InitHostEntries();
  for (bool mapped : {false, true}) {
    for (uint32_t sub : {0u, 1u}) {
      for (uint32_t shift : {0u, 1u, 2u}) {
        for (uint32_t rn : {0u, 1u, 2u, 31u}) {
          for (uint32_t rm : {0u, 1u, 2u, 31u}) {
            for (uint32_t rd : {0u, 1u, 2u, 31u}) {
              std::array<uint32_t, 8> code;
              code.fill(0x0b00'0000u | (sub << 30) | (shift << 22) | (rm << 16) | (rn << 5) | rd);
              GuestAddr start_pc = ToGuestAddr(code.data());
              ThreadState interpreted{};
              ThreadState translated{};
              for (ThreadState* state : {&interpreted, &translated}) {
                for (size_t i = 0; i < 31; ++i) {
                  state->cpu.x[i] = 0xfedc'ba98'7654'3210ULL + i * 0x1111'1111ULL;
                }
                state->cpu.sp = 0x1234'5678;
                state->cpu.flags = 0xb;
                SetInsnAddr(state->cpu, start_pc);
              }
              for (size_t i = 0; i < code.size(); ++i) {
                InterpretInsn(&interpreted);
              }
              MachineCode machine_code;
              LiteTranslateParams params;
              params.end_pc = start_pc + sizeof(code);
              params.allow_dispatch = false;
              params.enable_reg_mapping = mapped;
              auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &machine_code, params);
              ASSERT_TRUE(success);
              ASSERT_EQ(stop_pc, params.end_pc);
              ScopedExecRegion exec(&machine_code);
              SetResidence(translated, kOutsideGeneratedCode);
              berberis_RunGeneratedCode(&translated, AsHostCode(exec.GetHostCodeAddr()));
              SCOPED_TRACE(testing::Message() << std::hex << code[0] << " mapped=" << mapped);
              for (size_t i = 0; i < 31; ++i) {
                EXPECT_EQ(translated.cpu.x[i], interpreted.cpu.x[i]) << "reg=" << i;
              }
              EXPECT_EQ(translated.cpu.sp, interpreted.cpu.sp);
              EXPECT_EQ(translated.cpu.flags, interpreted.cpu.flags);
              EXPECT_EQ(GetInsnAddr(translated.cpu), GetInsnAddr(interpreted.cpu));
            }
          }
        }
      }
    }
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
  constexpr std::array<uint32_t, 11> kGuestCode = {
      0x0e20'5820,  // cnt v0.8b,v1.8b
      0x4e20'5862,  // cnt v2.16b,v3.16b
      0x4e82'1820,  // uzp1 v0.4s,v1.4s,v2.4s
      0x4e85'7883,  // zip2 v3.4s,v4.4s,v5.4s
      0x4e88'28e6,  // trn1 v6.4s,v7.4s,v8.4s
      0x4e80'1800,  // uzp1 v0.4s,v0.4s,v0.4s (all registers alias)
      0x4e88'58e9,  // uzp2 v9.4s,v7.4s,v8.4s
      0x4e88'38ea,  // zip1 v10.4s,v7.4s,v8.4s
      0x4e85'38a5,  // zip1 v5.4s,v5.4s,v5.4s (all registers alias)
      0x4e80'6980,  // trn2 v0.4s,v12.4s,v0.4s (destination aliases second source)
      0x4ed3'7a52,  // zip2 v18.2d,v18.2d,v19.2d (destination aliases first source)
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      for (size_t reg = 0; reg < 20; ++reg) {
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

TEST(LoongArch64RuntimeLibraryTest, LiteMovSFromElementMatchesInterpreter) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x5e04'0420,  // mov s0,v1.s[0]
      0x5e0c'0462,  // mov s2,v3.s[1]
      0x5e14'04a4,  // mov s4,v5.s[2]
      0x5e1c'04e6,  // mov s6,v7.s[3]
      0x5e0c'0421,  // mov s1,v1.s[1] (destination aliases source)
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    for (uint32_t reg = 0; reg < 8; ++reg) {
      interpreted.cpu.v[reg] =
          MakeUint32x4(0x0102'0304 + reg, 0x1122'3344 + reg, 0x5566'7788 + reg, 0x99aa'bbcc + reg);
      translated.cpu.v[reg] = interpreted.cpu.v[reg];
    }
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

// All new SIMD paths normalize the full destination before StoreV. Exercise
// mapped and unmapped execution with the same reference and guest code.
template <size_t N>
void RunSimdFallbackCode(const std::array<uint32_t, N>& guest, ThreadState* state, bool mapping) {
  InitHostEntries();
  LiteTranslateParams params;
  GuestAddr start = ToGuestAddr(guest.data());
  params.end_pc = start + sizeof(guest);
  params.allow_dispatch = false;
  params.enable_reg_mapping = mapping;
  MachineCode code;
  auto [success, stop] = TryLiteTranslateRegion(start, &code, params);
  ASSERT_TRUE(success);
  ASSERT_EQ(stop, params.end_pc);
  ScopedExecRegion exec(&code);
  SetInsnAddr(state->cpu, start);
  SetResidence(*state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(state, AsHostCode(exec.GetHostCodeAddr()));
}

TEST(LoongArch64RuntimeLibraryTest, LiteVectorShift32AllImmediatesAndAliases) {
  constexpr std::array<uint32_t, 6> kInputs = {
      0, 1, 0x8000'0000, 0xffff'ffff, 0x8123'4567, 0x5555'aaaa};
  for (bool right : {false, true}) {
    for (uint32_t q : {0u, 1u}) {
      for (uint32_t shift = right ? 1 : 0; shift <= (right ? 32u : 31u); ++shift) {
        for (uint32_t rd : {0u, 1u, 31u}) {
          uint32_t immediate = right ? 64 - shift : 32 + shift;
          const std::array<uint32_t, 1> guest = {(right ? 0x2f00'0400u : 0x0f00'5400u) | (q << 30) |
                                                 (immediate << 16) | (1u << 5) | rd};
          for (uint32_t input : kInputs) {
            ThreadState initial{};
            initial.cpu.v[1] = MakeUint32x4(input, ~input, 0x7f80'0001, 0xffff'ffff);
            initial.cpu.flags = 0xb;
            uint32_t lanes[4];
            memcpy(lanes, &initial.cpu.v[1], sizeof(lanes));
            __uint128_t expected = 0;
            for (uint32_t i = 0; i < (q ? 4u : 2u); ++i) {
              uint32_t value = right ? (shift == 32 ? 0 : lanes[i] >> shift) : lanes[i] << shift;
              expected |= static_cast<__uint128_t>(value) << (32 * i);
            }
            ThreadState interpreted{};
            interpreted.cpu = initial.cpu;
            SetInsnAddr(interpreted.cpu, ToGuestAddr(guest.data()));
            InterpretInsn(&interpreted);
            EXPECT_EQ(interpreted.cpu.v[rd], expected);
            for (bool mapping : {false, true}) {
              ThreadState state{};
              state.cpu = initial.cpu;
              RunSimdFallbackCode(guest, &state, mapping);
              SCOPED_TRACE(testing::Message() << std::hex << guest[0] << " input=" << input);
              for (uint32_t reg = 0; reg < 32; ++reg) {
                EXPECT_EQ(state.cpu.v[reg], interpreted.cpu.v[reg]) << reg;
              }
              EXPECT_EQ(state.cpu.flags, initial.cpu.flags);
            }
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteCmtst32BitMasksAndAliases) {
  for (uint32_t q : {0u, 1u}) {
    for (uint32_t rd : {0u, 1u, 2u, 31u}) {
      for (uint32_t rm : {1u, 2u}) {
        const std::array<uint32_t, 1> guest = {0x0ea0'8c20u | (q << 30) | (rm << 16) | rd};
        for (uint32_t a : {0u, 1u, 0xffff'ffffu, 0x8000'0000u, 0x5555'aaaau}) {
          ThreadState initial{};
          initial.cpu.v[1] = MakeUint32x4(a, ~a, 0x8000'0000, 1);
          initial.cpu.v[2] = MakeUint32x4(~a, a, 0x8000'0000, 0);
          __uint128_t expected = 0;
          for (uint32_t lane = 0; lane < (q ? 4u : 2u); ++lane) {
            uint32_t n = initial.cpu.v[1] >> (32 * lane);
            uint32_t m = initial.cpu.v[rm] >> (32 * lane);
            expected |= static_cast<__uint128_t>((n & m) ? UINT32_MAX : 0) << (32 * lane);
          }
          ThreadState interpreted{};
          interpreted.cpu = initial.cpu;
          SetInsnAddr(interpreted.cpu, ToGuestAddr(guest.data()));
          InterpretInsn(&interpreted);
          EXPECT_EQ(interpreted.cpu.v[rd], expected);
          for (bool mapping : {false, true}) {
            ThreadState state{};
            state.cpu = initial.cpu;
            RunSimdFallbackCode(guest, &state, mapping);
            for (uint32_t reg = 0; reg < 32; ++reg) {
              EXPECT_EQ(state.cpu.v[reg], interpreted.cpu.v[reg]) << reg;
            }
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteFmax4SZeroNanInfinityAndAliases) {
  constexpr std::array<uint32_t, 14> kBits = {0,
                                              0x8000'0000,
                                              1,
                                              0x8000'0001,
                                              0x0080'0000,
                                              0x8080'0000,
                                              0x3f80'0000,
                                              0xbf80'0000,
                                              0x7f80'0000,
                                              0xff80'0000,
                                              0x7fc1'2345,
                                              0xffc1'2345,
                                              0x7f80'0001,
                                              0xff80'0001};
  for (uint32_t rd : {0u, 1u, 2u, 31u}) {
    for (uint32_t a : kBits) {
      for (uint32_t b : kBits) {
        const std::array<uint32_t, 1> guest = {0x4e22'f420u | rd};
        ThreadState initial{};
        initial.cpu.v[1] = MakeUint32x4(a, b, a, b);
        initial.cpu.v[2] = MakeUint32x4(b, a, a, b);
        ThreadState interpreted{};
        interpreted.cpu = initial.cpu;
        SetInsnAddr(interpreted.cpu, ToGuestAddr(guest.data()));
        InterpretInsn(&interpreted);
        // Independently require default NaN and operand-order-independent +0.
        uint32_t expected_low;
        if ((a & 0x7fff'ffff) > 0x7f80'0000 || (b & 0x7fff'ffff) > 0x7f80'0000) {
          expected_low = 0x7fc0'0000;
        } else if ((a & 0x7fff'ffff) == 0 && (b & 0x7fff'ffff) == 0) {
          expected_low = a & b;
        } else {
          float fa, fb;
          memcpy(&fa, &a, 4);
          memcpy(&fb, &b, 4);
          expected_low = fa > fb ? a : b;
        }
        EXPECT_EQ(static_cast<uint32_t>(interpreted.cpu.v[rd]), expected_low);
        for (bool mapping : {false, true}) {
          ThreadState state{};
          state.cpu = initial.cpu;
          RunSimdFallbackCode(guest, &state, mapping);
          SCOPED_TRACE(testing::Message() << std::hex << a << "," << b << " rd=" << rd);
          for (uint32_t reg = 0; reg < 32; ++reg) {
            EXPECT_EQ(state.cpu.v[reg], interpreted.cpu.v[reg]) << reg;
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteSimdFallbackDestinationsStayCoherentInCache) {
  // v0 and v1 are both read and written repeatedly, including a 64-bit
  // result subsequently consumed as a 128-bit value. This detects stale
  // upper lanes in cache even when the ThreadState write looks correct.
  constexpr std::array<uint32_t, 9> guest = {
      0x4e22'f420,  // fmax v0.4s,v1.4s,v2.4s
      0x0f3f'5400,  // shl v0.2s,v0.2s,#31
      0x6e22'1c03,  // eor v3.16b,v0.16b,v2.16b (observe full v0)
      0x4f21'5400,  // shl v0.4s,v0.4s,#1
      0x0ea1'8c00,  // cmtst v0.2s,v0.2s,v1.2s
      0x6e22'1c04,  // eor v4.16b,v0.16b,v2.16b (observe full v0)
      0x6f20'0401,  // ushr v1.4s,v0.4s,#32
      0x4e21'f400,  // fmax v0.4s,v0.4s,v1.4s
      0x4ea0'8c01,  // cmtst v1.4s,v0.4s,v0.4s
  };
  for (uint32_t seed : {0u, 0x7f80'0001u, 0x8000'0000u, 0xffff'ffffu}) {
    ThreadState initial{};
    initial.cpu.v[0] = MakeUint32x4(1, 2, 3, 4);
    initial.cpu.v[1] = MakeUint32x4(seed, 0x3f80'0000, 0x8000'0000, 0x7f80'0000);
    initial.cpu.v[2] = MakeUint32x4(0, 0xbf80'0000, 0, 0x7fc1'2345);
    ThreadState interpreted{};
    interpreted.cpu = initial.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(guest.data()));
    for (size_t i = 0; i < guest.size(); ++i) InterpretInsn(&interpreted);
    for (bool mapping : {false, true}) {
      ThreadState state{};
      state.cpu = initial.cpu;
      RunSimdFallbackCode(guest, &state, mapping);
      for (uint32_t reg = 0; reg < 32; ++reg) {
        EXPECT_EQ(state.cpu.v[reg], interpreted.cpu.v[reg]) << reg;
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteSimdFallbackRejectsUnauditedForms) {
  // Neighboring 8H, 2D, scalar and FP16 encodings must still exit precisely.
  for (uint32_t insn : {0x4f1f'5420u,
                        0x4f7f'5420u,
                        0x5f7f'5420u,
                        0x6f1f'0420u,
                        0x6f7f'0420u,
                        0x4e62'f420u,
                        0x0e22'f420u,
                        0x4e42'f420u,
                        0x4e62'8c20u}) {
    const std::array<uint32_t, 1> guest = {insn};
    for (bool mapping : {false, true}) {
      LiteTranslateParams params;
      GuestAddr start = ToGuestAddr(guest.data());
      params.end_pc = start + sizeof(guest);
      params.enable_reg_mapping = mapping;
      MachineCode code;
      auto [success, stop] = TryLiteTranslateRegion(start, &code, params);
      EXPECT_FALSE(success) << std::hex << insn;
      EXPECT_EQ(stop, start);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteBicAndBsl8AliasesAndCachedWrites) {
  for (uint32_t base : {0x0e60'1c00u, 0x4e60'1c00u, 0x2e60'1c00u}) {
    for (uint32_t rd : {0u, 1u, 2u, 31u}) {
      for (uint32_t rn : {0u, 1u, 2u, 31u}) {
        for (uint32_t rm : {0u, 1u, 2u, 31u}) {
          uint32_t insn = base | (rm << 16) | (rn << 5) | rd;
          // Repetition enables read/write caching, including BSL's old Vd.
          const std::array<uint32_t, 4> guest = {insn, insn, insn, insn};
          for (bool mapping : {false, true}) {
            ThreadState state{};
            ThreadState interpreted{};
            __uint128_t expected[32];
            for (uint32_t reg = 0; reg < 32; ++reg) {
              expected[reg] = MakeUint32x4(0x0123'4567u * (reg + 1),
                                           0x89ab'cdefu ^ reg,
                                           0xfedc'ba98u - reg,
                                           0x7654'3210u + reg);
              state.cpu.v[reg] = interpreted.cpu.v[reg] = expected[reg];
            }
            for (size_t i = 0; i < guest.size(); ++i) {
              __uint128_t value = base == 0x2e60'1c00u ? (expected[rd] & expected[rn]) |
                                                             (~expected[rd] & expected[rm])
                                                       : expected[rn] & ~expected[rm];
              expected[rd] = (base & (1u << 30)) ? value : static_cast<uint64_t>(value);
            }
            SetInsnAddr(interpreted.cpu, ToGuestAddr(guest.data()));
            for (size_t i = 0; i < guest.size(); ++i) InterpretInsn(&interpreted);
            RunSimdFallbackCode(guest, &state, mapping);
            SCOPED_TRACE(testing::Message() << std::hex << insn << " mapping=" << mapping);
            for (uint32_t reg = 0; reg < 32; ++reg) {
              EXPECT_EQ(state.cpu.v[reg], expected[reg]) << reg;
              EXPECT_EQ(interpreted.cpu.v[reg], expected[reg]) << reg;
            }
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteUmovSAllLanesAndZeroRegister) {
  for (uint32_t rn : {0u, 1u, 31u}) {
    for (uint32_t lane = 0; lane < 4; ++lane) {
      for (uint32_t rd : {0u, 1u, 30u, 31u}) {
        const uint32_t insn = 0x0e04'3c00u | (lane << 19) | (rn << 5) | rd;
        const std::array<uint32_t, 4> guest = {insn, insn, insn, insn};
        for (bool mapping : {false, true}) {
          ThreadState state{};
          ThreadState interpreted{};
          for (uint32_t reg = 0; reg < 31; ++reg) state.cpu.x[reg] = UINT64_MAX;
          state.cpu.sp = 0x1234'5678'90ab'cdef;
          state.cpu.v[rn] = MakeUint32x4(0, 0x8000'0000, 0xffff'ffff, 0x1234'5678);
          interpreted.cpu = state.cpu;
          uint64_t expected = static_cast<uint32_t>(state.cpu.v[rn] >> (lane * 32));
          SetInsnAddr(interpreted.cpu, ToGuestAddr(guest.data()));
          for (size_t i = 0; i < guest.size(); ++i) InterpretInsn(&interpreted);
          RunSimdFallbackCode(guest, &state, mapping);
          if (rd != 31) EXPECT_EQ(state.cpu.x[rd], expected);
          EXPECT_EQ(state.cpu.sp, interpreted.cpu.sp);
          for (uint32_t reg = 0; reg < 31; ++reg)
            EXPECT_EQ(state.cpu.x[reg], interpreted.cpu.x[reg]);
          for (uint32_t reg = 0; reg < 32; ++reg)
            EXPECT_EQ(state.cpu.v[reg], interpreted.cpu.v[reg]);
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteBsl8CacheUpperLanesReachUmovAndVectorReader) {
  constexpr std::array<uint32_t, 7> guest = {
      0x2e62'1c20,  // bsl v0.8b,v1.8b,v2.8b
      0x0e14'3c03,  // umov w3,v0.s[2] (must read zero from cache)
      0x4e62'1c04,  // bic v4.16b,v0.16b,v2.16b
      0x2e62'1c20,  // bsl v0.8b,v1.8b,v2.8b
      0x0e1c'3c05,  // umov w5,v0.s[3]
      0x4e60'1c06,  // bic v6.16b,v0.16b,v0.16b
      0x0e04'3c07,  // umov w7,v0.s[0]
  };
  for (bool mapping : {false, true}) {
    ThreadState state{};
    ThreadState interpreted{};
    state.cpu.v[0] = MakeUint128(0xaaaa'5555'aaaa'5555, UINT64_MAX);
    state.cpu.v[1] = MakeUint128(0x1234'5678'9abc'def0, UINT64_MAX);
    state.cpu.v[2] = MakeUint128(0xfedc'ba98'7654'3210, UINT64_MAX);
    interpreted.cpu = state.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(guest.data()));
    for (size_t i = 0; i < guest.size(); ++i) InterpretInsn(&interpreted);
    RunSimdFallbackCode(guest, &state, mapping);
    EXPECT_EQ(state.cpu.x[3], 0u);
    EXPECT_EQ(state.cpu.x[5], 0u);
    for (uint32_t reg = 0; reg < 31; ++reg) EXPECT_EQ(state.cpu.x[reg], interpreted.cpu.x[reg]);
    for (uint32_t reg = 0; reg < 32; ++reg) EXPECT_EQ(state.cpu.v[reg], interpreted.cpu.v[reg]);
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

TEST(LoongArch64RuntimeLibraryTest, LiteFaddp2SUsesSourceCache) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x2e21'd402,  // faddp v2.2s, v0.2s, v1.2s
      0x2e21'd403,  // faddp v3.2s, v0.2s, v1.2s
      0x2e21'd404,  // faddp v4.2s, v0.2s, v1.2s
      0x2e21'd405,  // faddp v5.2s, v0.2s, v1.2s
      0x2e21'd406,  // faddp v6.2s, v0.2s, v1.2s
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
  state.cpu.v[0] = MakeUint32x4(0x3f80'0000, 0x4000'0000, 0x7f80'0000, 0xff80'0000);
  state.cpu.v[1] = MakeUint32x4(0x4040'0000, 0xc080'0000, 0x7fc1'2345, 0xffc5'4321);
  const __uint128_t expected = MakeUint32x4(0x4040'0000, 0xbf80'0000, 0, 0);
  ScopedExecRegion exec(&cached_code);
  SetInsnAddr(state.cpu, start_pc);
  SetResidence(state, kOutsideGeneratedCode);
  berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  for (uint32_t reg = 2; reg <= 6; ++reg) {
    EXPECT_EQ(state.cpu.v[reg], expected) << reg;
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteHotVectorComparisonsMatchInterpreter) {
  constexpr std::array<uint32_t, 6> kGuestCode = {
      0x4ea0'd800,  // fcmeq v0.4s,v0.4s,#0
      0x6e21'e400,  // fcmge v0.4s,v0.4s,v1.4s
      0x6ea7'e462,  // fcmgt v2.4s,v3.4s,v7.4s
      0x6e27'8e07,  // cmeq v7.16b,v16.16b,v7.16b (destination aliases second source)
      0x6e21'3ca3,  // cmhs v3.16b,v5.16b,v1.16b
      0x4ea0'3442,  // cmgt v2.4s,v2.4s,v0.4s
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

TEST(LoongArch64RuntimeLibraryTest, LiteFneg2S4SAndBsl16BMatchInterpreter) {
  constexpr std::array<uint32_t, 6> kGuestCode = {
      0x6ea0'f820,  // fneg v0.4s,v1.4s
      0x6ea0'f821,  // fneg v1.4s,v1.4s (destination aliases source)
      0x2ea0'f820,  // fneg v0.2s,v1.2s
      0x6e63'1c22,  // bsl v2.16b,v1.16b,v3.16b
      0x6e62'1c22,  // bsl v2.16b,v1.16b,v2.16b (destination aliases second source)
      0x6e61'1c21,  // bsl v1.16b,v1.16b,v1.16b (all registers alias)
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    auto initialize = [](ThreadState* state) {
      state->cpu.v[0] = MakeUint32x4(0x0000'0000, 0x8000'0000, 0x7fc1'2345, 0xffc5'4321);
      state->cpu.v[1] = MakeUint32x4(0x3f80'0000, 0xc000'0000, 0xaaaa'5555, 0x0123'4567);
      state->cpu.v[2] = MakeUint32x4(0xffff'0000, 0x00ff'00ff, 0x3333'cccc, 0x8000'0001);
      state->cpu.v[3] = MakeUint32x4(0x1357'9bdf, 0x2468'ace0, 0xffff'ffff, 0x0000'0000);
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

TEST(LoongArch64RuntimeLibraryTest, LiteScalarScvtfMatchesInterpreter) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x5e21'd820,  // scvtf s0,s1
      0x1e22'0062,  // scvtf s2,w3
      0x9e22'00a4,  // scvtf s4,x5
      0x1e62'00e6,  // scvtf d6,w7
      0x9e62'0128,  // scvtf d8,x9
  };
  constexpr std::array<uint64_t, 5> kInputs = {
      0x8000'0000u,
      0x7fff'ffffu,
      0x8000'0000'0000'0000ULL,
      0xffff'ffffu,
      0x7fff'ffff'ffff'ffffULL,
  };
  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    const std::array<uint32_t, 1> one_insn = {kGuestCode[i]};
    ThreadState interpreted{};
    ThreadState translated{};
    const uint32_t rn = (one_insn[0] >> 5) & 31;
    const uint32_t rd = one_insn[0] & 31;
    if (i == 0) {
      interpreted.cpu.v[rn] = kInputs[i];
    } else {
      interpreted.cpu.x[rn] = kInputs[i];
    }
    interpreted.cpu.v[rd] = MakeUint128(UINT64_MAX, UINT64_MAX);
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << one_insn[0]);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarUcvtfMatchesInterpreter) {
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x1e23'0020,  // ucvtf s0,w1
      0x9e23'0062,  // ucvtf s2,x3
      0x1e63'00a4,  // ucvtf d4,w5
      0x9e63'00e6,  // ucvtf d6,x7
  };
  constexpr std::array<uint64_t, 4> kInputs = {
      UINT32_MAX,
      UINT64_MAX,
      UINT32_MAX,
      0x8000'0000'0000'0001ULL,
  };
  for (size_t i = 0; i < kGuestCode.size(); ++i) {
    const std::array<uint32_t, 1> one_insn = {kGuestCode[i]};
    ThreadState interpreted{};
    ThreadState translated{};
    const uint32_t rn = (one_insn[0] >> 5) & 31;
    const uint32_t rd = one_insn[0] & 31;
    interpreted.cpu.x[rn] = kInputs[i];
    interpreted.cpu.v[rd] = MakeUint128(UINT64_MAX, UINT64_MAX);
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << one_insn[0]);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarFabsFnegMatchesInterpreter) {
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x1e21'4020,  // fneg s0,s1
      0x1e61'4062,  // fneg d2,d3
      0x1e20'c0a4,  // fabs s4,s5
      0x1e60'c0e6,  // fabs d6,d7
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    interpreted.cpu.v[rn] = MakeUint128(0xffc1'2345'8000'0000ULL, 0x0123'4567'89ab'cdefULL);
    interpreted.cpu.v[rd] = MakeUint128(UINT64_MAX, UINT64_MAX);
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarMinMaxMatchesInterpreter) {
  constexpr std::array<uint32_t, 8> kGuestCode = {
      0x1e22'4820,  // fmax s0,s1,s2
      0x1e25'5883,  // fmin s3,s4,s5
      0x1e28'68e6,  // fmaxnm s6,s7,s8
      0x1e2b'7949,  // fminnm s9,s10,s11
      0x1e6e'49ac,  // fmax d12,d13,d14
      0x1e71'5a0f,  // fmin d15,d16,d17
      0x1e74'6a72,  // fmaxnm d18,d19,d20
      0x1e77'7ad5,  // fminnm d21,d22,d23
  };
  constexpr std::array<std::array<uint64_t, 2>, 3> kInputs = {{
      {0x4040'0000u, 0xc080'0000u},  // 3.0f, -4.0f
      {0x8000'0000u, 0x0000'0000u},  // -0.0f, +0.0f
      {0x0000'0000u, 0x8000'0000u},  // +0.0f, -0.0f
  }};
  for (uint32_t insn : kGuestCode) {
    const bool is_double = (insn & (1u << 22)) != 0;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rd = insn & 31;
    for (const auto& input : kInputs) {
      uint64_t lhs = input[0];
      uint64_t rhs = input[1];
      if (is_double) {
        lhs = lhs == 0x4040'0000u   ? 0x4008'0000'0000'0000ULL
              : lhs == 0xc080'0000u ? 0xc010'0000'0000'0000ULL
              : lhs == 0x8000'0000u ? 0x8000'0000'0000'0000ULL
                                    : 0;
        rhs = rhs == 0x4040'0000u   ? 0x4008'0000'0000'0000ULL
              : rhs == 0xc080'0000u ? 0xc010'0000'0000'0000ULL
              : rhs == 0x8000'0000u ? 0x8000'0000'0000'0000ULL
                                    : 0;
      }
      const std::array<uint32_t, 1> one_insn = {insn};
      ThreadState interpreted{};
      ThreadState translated{};
      interpreted.cpu.v[rn] = MakeUint128(lhs, UINT64_MAX);
      interpreted.cpu.v[rm] = MakeUint128(rhs, UINT64_MAX);
      interpreted.cpu.v[rd] = MakeUint128(UINT64_MAX, UINT64_MAX);
      translated.cpu = interpreted.cpu;
      SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
      InterpretInsn(&interpreted);
      TranslateAndRun(one_insn, &translated);
      SCOPED_TRACE(testing::Message()
                   << "insn=" << std::hex << insn << " lhs=" << lhs << " rhs=" << rhs);
      EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarMinMaxFallsBackForNaNPayloadRules) {
  constexpr uint32_t kFmaxS = 0x1e22'4820;    // fmax s0,s1,s2
  constexpr uint32_t kFmaxnmS = 0x1e22'6820;  // fmaxnm s0,s1,s2
  constexpr uint64_t kQnan = 0x7fc1'2345u;
  constexpr uint64_t kNumber = 0x4000'0000u;
  constexpr __uint128_t kSentinel = MakeUint128(UINT64_MAX, UINT64_MAX);

  {
    const std::array<uint32_t, 1> code = {kFmaxS};
    ThreadState state{};
    state.cpu.v[0] = kSentinel;
    state.cpu.v[1] = MakeUint128(kQnan, 0);
    state.cpu.v[2] = MakeUint128(kNumber, 0);
    TranslateAndRun(code, &state);
    EXPECT_EQ(GetInsnAddr(state.cpu), ToGuestAddr(code.data()));
    EXPECT_EQ(state.cpu.v[0], kSentinel);
  }

  {
    const std::array<uint32_t, 1> code = {kFmaxnmS};
    ThreadState interpreted{};
    ThreadState translated{};
    interpreted.cpu.v[0] = kSentinel;
    interpreted.cpu.v[1] = MakeUint128(kQnan, 0);
    interpreted.cpu.v[2] = MakeUint128(kNumber, 0);
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(code.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(code, &translated);
    EXPECT_EQ(translated.cpu.v[0], interpreted.cpu.v[0]);
    EXPECT_EQ(GetInsnAddr(translated.cpu), ToGuestAddr(code.data()) + sizeof(uint32_t));
  }

  {
    const std::array<uint32_t, 1> code = {kFmaxnmS};
    ThreadState state{};
    state.cpu.v[0] = kSentinel;
    state.cpu.v[1] = MakeUint128(kQnan, 0);
    state.cpu.v[2] = MakeUint128(0xffc5'4321u, 0);
    TranslateAndRun(code, &state);
    EXPECT_EQ(GetInsnAddr(state.cpu), ToGuestAddr(code.data()));
    EXPECT_EQ(state.cpu.v[0], kSentinel);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarMinMaxNaNFallbackMakesDispatchProgress) {
  // The old isolated fallback test only required returning at the same PC.
  // In a real dispatcher that re-enters the same cached region forever. Use
  // actual dispatch and a cached stop target to require interpreted progress.
  constexpr std::array<uint32_t, 9> kInsns = {
      0x1e22'4820,
      0x1e22'5820,
      0x1e22'6820,
      0x1e22'7820,
      0x1e62'4820,
      0x1e62'5820,
      0x1e62'6820,
      0x1e62'7820,
      0x1e21'4881,  // fmax s1,s4,s1: observed in the stalled Unity worker.
  };
  InitHostEntries();
  auto* cache = TranslationCache::GetInstance();
  for (uint32_t insn : kInsns) {
    for (int nan_operands : {1, 2, 3}) {
      SCOPED_TRACE(testing::Message() << std::hex << insn << " NaNs=" << nan_operands);
      const bool is_double = (insn & (1u << 22)) != 0;
      const uint32_t rn = (insn >> 5) & 31;
      const uint32_t rm = (insn >> 16) & 31;
      const uint32_t rd = insn & 31;
      const uint64_t nan = is_double ? 0x7ff8'0000'0001'2345ULL : 0x7fc1'2345ULL;
      const uint64_t number = is_double ? 0x4000'0000'0000'0000ULL : 0x4000'0000ULL;
      const std::array<uint32_t, 4> guest_code = {
          insn, 0x1400'0002, 0xd503'201f, 0xd503'201f};  // op; b stop; nop; stop
      const GuestAddr pc = ToGuestAddr(guest_code.data());
      const GuestAddr stop = pc + 3 * sizeof(uint32_t);
      auto* entry = cache->AddAndLockForTranslation(stop, 0);
      ASSERT_NE(entry, nullptr);
      cache->SetTranslatedAndUnlock(stop,
                                    entry,
                                    sizeof(uint32_t),
                                    GuestCodeEntry::Kind::kLiteTranslated,
                                    {kEntryExitGeneratedCode, 0});
      ThreadState expected{};
      expected.cpu.v[rd] = MakeUint128(UINT64_MAX, UINT64_MAX);
      expected.cpu.v[rn] = MakeUint128((nan_operands & 1) ? nan : number, UINT64_MAX);
      expected.cpu.v[rm] = MakeUint128((nan_operands & 2) ? nan : number, UINT64_MAX);
      ThreadState actual{};
      actual.cpu = expected.cpu;
      SetInsnAddr(expected.cpu, pc);
      InterpretInsn(&expected);
      MachineCode code;
      LiteTranslateParams params;
      params.end_pc = stop;
      params.allow_dispatch = true;
      auto [success, stop_pc] = TryLiteTranslateRegion(pc, &code, params);
      EXPECT_TRUE(success);
      if (success) {
        ScopedExecRegion exec(&code);
        SetInsnAddr(actual.cpu, pc);
        SetResidence(actual, kOutsideGeneratedCode);
        berberis_RunGeneratedCode(&actual, AsHostCode(exec.GetHostCodeAddr()));
        EXPECT_EQ(actual.cpu.v[rd], expected.cpu.v[rd]);
        EXPECT_EQ(GetInsnAddr(actual.cpu), stop);
        EXPECT_EQ(GetResidence(actual), kOutsideGeneratedCode);
      }
      cache->InvalidateGuestRange(stop, stop + sizeof(uint32_t));
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteHot2SLaneOperationsMatchInterpreter) {
  constexpr std::array<uint32_t, 10> kGuestCode = {
      0x0e0c'0420,  // dup v0.2s,v1.s[1]
      0x0e04'0421,  // dup v1.2s,v1.s[0] (destination aliases source)
      0x0e04'0c62,  // dup v2.2s,w3
      0x0e0e'3ca4,  // umov w4,v5.h[3]
      0x0e88'38e6,  // zip1 v6.2s,v7.2s,v8.2s
      0x0eab'a549,  // smaxp v9.2s,v10.2s,v11.2s
      0x0ea3'a442,  // smaxp v2.2s,v2.2s,v3.2s (destination aliases source)
      0x0eae'adac,  // sminp v12.2s,v13.2s,v14.2s
      0x0e21'da0f,  // scvtf v15.2s,v16.2s
      0x0fa3'9251,  // fmul v17.2s,v18.2s,v3.s[1]
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    for (uint32_t reg = 0; reg < 32; ++reg) {
      interpreted.cpu.x[reg] = 0x8000'0000u + reg * 0x0102'0304u;
      interpreted.cpu.v[reg] =
          MakeUint32x4(0x3f00'0000u + reg, 0xc000'0000u + reg, 0x7000'0000u + reg, reg);
    }
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    if ((insn & 0xffe0'fc00u) == 0x0e00'3c00u) {
      EXPECT_EQ(translated.cpu.x[insn & 31], interpreted.cpu.x[insn & 31]);
    } else {
      EXPECT_EQ(translated.cpu.v[insn & 31], interpreted.cpu.v[insn & 31]);
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteHotModifiedImmediatesMatchInterpreter) {
  constexpr std::array<uint32_t, 7> kGuestCode = {
      0x4f04'8417,  // movi v23.8h,#128
      0x4f04'6406,  // movi v6.4s,#128,lsl#24
      0x4f05'67f4,  // movi v20.4s,#191,lsl#24
      0x4f03'f611,  // fmov v17.4s,#1.0
      0x4f06'f617,  // fmov v23.4s,#-0.25
      0x4f07'f601,  // fmov v1.4s,#-1.0
      0x0f07'f607,  // fmov v7.2s,#-1.0
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    const uint32_t rd = insn & 31;
    interpreted.cpu.v[rd] = MakeUint128(UINT64_MAX, UINT64_MAX);
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteModifiedImmediateDoesNotAliasSshll8H) {
  constexpr std::array<uint32_t, 3> kGuestCode = {
      0x0f00'a400,  // movi v0.4h,#0,lsl#8
      0x4f00'a7e1,  // movi v1.8h,#31,lsl#8
      0x0f07'a7ff,  // movi v31.4h,#255,lsl#8
  };

  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    const GuestAddr start_pc = ToGuestAddr(one_insn.data());
    MachineCode code;
    LiteTranslateParams params;
    params.end_pc = start_pc + sizeof(one_insn);
    params.allow_dispatch = false;
    params.enable_reg_mapping = true;

    auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &code, params);

    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_FALSE(success);
    EXPECT_EQ(stop_pc, start_pc);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteUaddlv8BMatchesInterpreter) {
  constexpr std::array<uint32_t, 2> kGuestCode = {
      0x2e30'3820,  // uaddlv h0,v1.8b
      0x2e30'3842,  // uaddlv h2,v2.8b (destination aliases source)
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    interpreted.cpu.v[rn] = MakeUint128(0xff80'4020'1008'0402ULL, UINT64_MAX);
    if (rd != rn) {
      interpreted.cpu.v[rd] = MakeUint128(UINT64_MAX, UINT64_MAX);
    }
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteScalarFmulByElementMatchesInterpreter) {
  constexpr std::array<uint32_t, 4> kGuestCode = {
      0x5f82'9020,  // fmul s0,s1,v2.s[0]
      0x5fa5'9083,  // fmul s3,s4,v5.s[1]
      0x5f94'98e6,  // fmul s6,s7,v20.s[2]
      0x5fa0'9129,  // fmul s9,s9,v0.s[1]
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    for (uint32_t reg = 0; reg < 32; ++reg) {
      interpreted.cpu.v[reg] = MakeUint32x4(0x3f00'0000 + reg * 0x0001'0000,
                                            0xbf80'0000 - reg * 0x0001'0000,
                                            0x4000'0000 + reg * 0x0000'8000,
                                            0x8000'0000);
    }
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteUshllAndXtnMatchInterpreter) {
  constexpr std::array<uint32_t, 5> kGuestCode = {
      0x2f10'a420,  // ushll v0.4s,v1.4h,#0
      0x2f1f'a462,  // ushll v2.4s,v3.4h,#15
      0x2f10'a484,  // ushll v4.4s,v4.4h,#0 (destination aliases source)
      0x0e61'2820,  // xtn v0.4h,v1.4s
      0x0e61'2bbe,  // xtn v30.4h,v29.4s
  };
  for (uint32_t insn : kGuestCode) {
    const std::array<uint32_t, 1> one_insn = {insn};
    ThreadState interpreted{};
    ThreadState translated{};
    for (uint32_t reg = 0; reg < 32; ++reg) {
      interpreted.cpu.v[reg] = MakeUint32x4(
          0x0123'8000u + reg, 0x4567'ffffu - reg, 0x89ab'0001u + reg, 0xcdef'7fffu - reg);
    }
    translated.cpu = interpreted.cpu;
    SetInsnAddr(interpreted.cpu, ToGuestAddr(one_insn.data()));
    InterpretInsn(&interpreted);
    TranslateAndRun(one_insn, &translated);
    const uint32_t rd = insn & 31;
    SCOPED_TRACE(testing::Message() << "insn=" << std::hex << insn);
    EXPECT_EQ(translated.cpu.v[rd], interpreted.cpu.v[rd]);
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

TEST(LoongArch64RuntimeLibraryTest, LiteFmlaFmls2DDoNotAlias4S) {
  // The checkasm forward transform uses FMLA V0.2D,V4.2D,V5.2D (0x4e65cc80).
  // Interpreting its double lanes as four floats corrupts test coefficients.
  // Keep unaudited double arithmetic in the interpreter, with either cache mode.
  for (uint32_t op : {0x4e60'cc00u, 0x4ee0'cc00u}) {
    for (uint32_t rd : {0u, 4u, 5u, 31u}) {
      for (bool mapping : {false, true}) {
        const std::array<uint32_t, 1> guest_code = {op | (5 << 16) | (4 << 5) | rd};
        const GuestAddr pc = ToGuestAddr(guest_code.data());
        MachineCode code;
        LiteTranslateParams params;
        params.end_pc = pc + sizeof(guest_code);
        params.allow_dispatch = false;
        params.enable_reg_mapping = mapping;
        auto [success, stop] = TryLiteTranslateRegion(pc, &code, params);
        EXPECT_FALSE(success) << std::hex << guest_code[0];
        EXPECT_EQ(stop, pc);
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteInsElementAllWidthsLanesAndAliases) {
  InitHostEntries();
  // imm4's low size bits are architecturally ignored. Exercise all 16
  // encodings as well as every destination lane and same-register copies.
  for (uint32_t size = 0; size < 4; ++size) {
    const uint32_t bytes = 1u << size;
    for (uint32_t dst = 0; dst < 16 / bytes; ++dst) {
      for (uint32_t imm4 = 0; imm4 < 16; ++imm4) {
        for (uint32_t rn : {0u, 1u}) {
          for (bool mapping : {false, true}) {
            const uint32_t imm5 = (dst << (size + 1)) | bytes;
            const uint32_t insn = 0x6e00'0400 | (imm5 << 16) | (imm4 << 11) | (rn << 5);
            const std::array<uint32_t, 5> guest_code = {
                0x4ea0'1c02,  // mov v2.16b,v0.16b
                0x4ea1'1c25,  // mov v5.16b,v1.16b
                insn,
                0x4ea0'1c03,  // mov v3.16b,v0.16b
                0x4ea0'1c04,  // mov v4.16b,v0.16b
            };
            ThreadState interpreted{};
            for (uint32_t r = 0; r < 32; ++r) {
              interpreted.cpu.v[r] =
                  MakeUint128(0x7766'5544'3322'1100ULL + r, 0xffee'ddcc'bbaa'9988ULL - r);
            }
            ThreadState translated{};
            translated.cpu = interpreted.cpu;
            __uint128_t expected = interpreted.cpu.v[0];
            const __uint128_t source = interpreted.cpu.v[rn];
            memcpy(reinterpret_cast<uint8_t*>(&expected) + dst * bytes,
                   reinterpret_cast<const uint8_t*>(&source) + (imm4 >> size) * bytes,
                   bytes);
            const GuestAddr pc = ToGuestAddr(guest_code.data());
            SetInsnAddr(interpreted.cpu, pc);
            for (size_t i = 0; i < guest_code.size(); ++i) InterpretInsn(&interpreted);
            MachineCode code;
            LiteTranslateParams params;
            params.end_pc = pc + sizeof(guest_code);
            params.allow_dispatch = false;
            params.enable_reg_mapping = mapping;
            auto [success, stop] = TryLiteTranslateRegion(pc, &code, params);
            ASSERT_TRUE(success);
            ASSERT_EQ(stop, params.end_pc);
            ScopedExecRegion exec(&code);
            SetInsnAddr(translated.cpu, pc);
            SetResidence(translated, kOutsideGeneratedCode);
            berberis_RunGeneratedCode(&translated, AsHostCode(exec.GetHostCodeAddr()));
            SCOPED_TRACE(testing::Message()
                         << "insn=" << std::hex << insn << " mapping=" << mapping);
            EXPECT_EQ(translated.cpu.v[0], expected);
            for (size_t r = 0; r < 32; ++r) {
              ASSERT_EQ(translated.cpu.v[r], interpreted.cpu.v[r]) << "v" << r;
            }
          }
        }
      }
    }
  }
}

TEST(LoongArch64RuntimeLibraryTest, LiteInsElementRejectsReservedWidths) {
  for (uint32_t imm5 : {0u, 16u}) {
    for (uint32_t imm4 = 0; imm4 < 16; ++imm4) {
      const std::array<uint32_t, 1> guest_code = {0x6e00'0420 | (imm5 << 16) | (imm4 << 11)};
      const GuestAddr pc = ToGuestAddr(guest_code.data());
      MachineCode code;
      LiteTranslateParams params;
      params.end_pc = pc + sizeof(guest_code);
      params.allow_dispatch = false;
      auto [success, stop] = TryLiteTranslateRegion(pc, &code, params);
      EXPECT_FALSE(success);
      EXPECT_EQ(stop, pc);
    }
  }
}



}  // namespace
}  // namespace berberis
