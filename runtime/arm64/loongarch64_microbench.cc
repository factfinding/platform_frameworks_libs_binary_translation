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

#include <time.h>

#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <string_view>
#include <vector>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/arm64/interpreter.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/test_utils/scoped_exec_region.h"

namespace berberis {
namespace {

constexpr uint64_t kDefaultIterations = 200'000;
constexpr uint64_t kWarmupIterations = 2'000;

uint64_t MonotonicNanos() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    std::abort();
  }
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
}

std::vector<uint32_t> MakeRepeated(uint32_t insn) {
  return std::vector<uint32_t>(64, insn);
}

std::vector<uint32_t> MakeConditionalFallthrough() {
  std::vector<uint32_t> code;
  code.reserve(64);
  for (uint32_t index = 0; index < 64; index += 2) {
    // CBZ X20 targets the end of the region.  X20 is non-zero in the
    // benchmark, so execution measures the sequential fallthrough path while
    // still exercising conditional-side-exit code layout.
    uint32_t displacement_in_insns = 64 - index;
    code.push_back(0xb400'0000u | (displacement_in_insns << 5) | 20u);
    code.push_back(0x8b02'0020u);  // add x0,x1,x2
  }
  return code;
}

std::vector<uint32_t> MakeMixed() {
  constexpr uint32_t kPattern[] = {
      0x8b02'0020u,  // add x0,x1,x2
      0x6e22'1c20u,  // eor v0.16b,v1.16b,v2.16b
      0x4e22'cc20u,  // fmla v0.4s,v1.4s,v2.4s
      0x9100'0400u,  // add x0,x0,#1
  };
  std::vector<uint32_t> code;
  code.reserve(64);
  for (size_t i = 0; i < 64; ++i) {
    code.push_back(kPattern[i % std::size(kPattern)]);
  }
  return code;
}

std::vector<uint32_t> MakeBenchmark(std::string_view name) {
  if (name == "simd_bic") return MakeRepeated(0x4e62'1c20u);
  if (name == "simd_bsl8") return MakeRepeated(0x2e62'1c20u);
  if (name == "simd_umovs") return MakeRepeated(0x0e0c'3c20u);
  if (name == "simd_shl32") return MakeRepeated(0x4f3f'5420u);    // shl v0.4s,v1.4s,#31
  if (name == "simd_ushr32") return MakeRepeated(0x6f29'0420u);   // ushr v0.4s,v1.4s,#23
  if (name == "simd_cmtst32") return MakeRepeated(0x4ea2'8c20u);  // cmtst v0.4s,v1.4s,v2.4s
  if (name == "simd_fmax4s") return MakeRepeated(0x4e22'f420u);   // fmax v0.4s,v1.4s,v2.4s
  if (name == "cache_write_first") {
    std::vector<uint32_t> code;
    for (uint32_t reg = 0; reg < 7; ++reg) {
      code.push_back(0xd280'0020u | reg);  // mov xN,#1
    }
    for (uint32_t i = 7; i < 64; ++i) {
      uint32_t reg = i % 7;
      code.push_back(0x9100'0400u | (reg << 5) | reg);  // add xN,xN,#1
    }
    return code;
  }
  if (name == "cache_early_exit") {
    auto code = MakeRepeated(0x8b02'0020u);  // add x0,x1,x2
    code[0] = 0xb500'0814u;                  // cbnz x20,end; remaining 63 instructions do not run
    return code;
  }
  if (name == "integer_w") {
    constexpr std::array<uint32_t, 4> kPattern = {
        0x0a02'0020,  // and w0,w1,w2
        0x2a02'0020,  // orr w0,w1,w2
        0x4a02'0020,  // eor w0,w1,w2
        0x0b02'0020,  // add w0,w1,w2
    };
    std::vector<uint32_t> code;
    for (size_t i = 0; i < 64; ++i) {
      code.push_back(kPattern[i % kPattern.size()]);
    }
    return code;
  }
  if (name == "stack_read" || name == "stack_writeback" || name == "logical_cached") {
    constexpr std::array<uint32_t, 4> kStackRead = {
        0xf940'13e0,  // ldr x0,[sp,#32]
        0xf940'23e3,  // ldr x3,[sp,#64]
        0x9100'0400,  // add x0,x0,#1
        0xa901'0fe0,  // stp x0,x3,[sp,#16]
    };
    constexpr std::array<uint32_t, 4> kStackWriteback = {
        0xa9bf'0be1,  // stp x1,x2,[sp,#-16]!
        0xf940'03e0,  // ldr x0,[sp]
        0xa8c1'13e3,  // ldp x3,x4,[sp],#16
        0xcb03'0000,  // sub x0,x0,x3
    };
    constexpr std::array<uint32_t, 4> kLogical = {
        0x8a02'0020,  // and x0,x1,x2
        0xaa02'0020,  // orr x0,x1,x2
        0xca02'0020,  // eor x0,x1,x2
        0xaa01'03e0,  // mov x0,x1 (orr x0,xzr,x1)
    };
    const auto& pattern = name == "stack_read"        ? kStackRead
                          : name == "stack_writeback" ? kStackWriteback
                                                      : kLogical;
    std::vector<uint32_t> code;
    for (size_t i = 0; i < 64; ++i) {
      code.push_back(pattern[i % pattern.size()]);
    }
    return code;
  }
  if (name == "condition_select") {
    std::vector<uint32_t> code;
    for (uint32_t i = 0; i < 64; ++i) {
      // CSEL X0,X1,X2,cond; cover all condition codes with fixed NZCV.
      code.push_back(0x9a82'0020u | ((i % 16) << 12));
    }
    return code;
  }
  if (name == "flag_conditionals") {
    std::vector<uint32_t> code;
    for (uint32_t i = 0; i < 64; i += 2) {
      code.push_back(0xeb02'003fu);  // cmp x1,x2
      // X1 < X2, so B.GE to the end is not taken. All 64 instructions execute.
      code.push_back(0x5400'000au | ((64 - i - 1) << 5));
    }
    return code;
  }
  // Small constants and fixed offsets occur throughout the sampled Unity
  // main-thread regions. Keep the memory footprint small to measure lowering
  // rather than storage or cache-capacity effects.
  if (name == "small_immediates" || name == "immediate_memory") {
    constexpr std::array<uint32_t, 4> kArithmetic = {
        0x9100'0400,  // add x0,x0,#1
        0xd100'2000,  // sub x0,x0,#8
        0xd280'0403,  // mov x3,#32
        0x8b03'0000,  // add x0,x0,x3
    };
    constexpr std::array<uint32_t, 4> kMemory = {
        0xf940'1140,  // ldr x0,[x10,#32]
        0xf940'2143,  // ldr x3,[x10,#64]
        0x9100'0400,  // add x0,x0,#1
        0xa901'0d40,  // stp x0,x3,[x10,#16]
    };
    const auto& pattern = name == "small_immediates" ? kArithmetic : kMemory;
    std::vector<uint32_t> code;
    for (size_t i = 0; i < 64; ++i) {
      code.push_back(pattern[i % pattern.size()]);
    }
    return code;
  }
  if (name == "gpr_cached") {
    return MakeRepeated(0x8b02'0020u);  // add x0,x1,x2
  }
  if (name == "simd_cached") {
    return MakeRepeated(0x6e22'1c20u);  // eor v0.16b,v1.16b,v2.16b
  }
  if (name == "conditional_fallthrough") {
    return MakeConditionalFallthrough();
  }
  if (name == "mixed") {
    return MakeMixed();
  }
  return {};
}

void InitializeState(ThreadState* state, GuestAddr start_pc) {
  state->cpu.x[1] = 0x0123'4567'89ab'cdefULL;
  state->cpu.x[2] = 0x1111'1111'1111'1111ULL;
  state->cpu.x[20] = 1;
  state->cpu.v[0] = static_cast<__uint128_t>(0x3f00'0000'3f80'0000ULL) |
                    (static_cast<__uint128_t>(0x4000'0000'4040'0000ULL) << 64);
  state->cpu.v[1] = static_cast<__uint128_t>(0x3f80'0000'4000'0000ULL) |
                    (static_cast<__uint128_t>(0x4040'0000'4080'0000ULL) << 64);
  state->cpu.v[2] = static_cast<__uint128_t>(0xbf00'0000'3e80'0000ULL) |
                    (static_cast<__uint128_t>(0x3f00'0000'bf80'0000ULL) << 64);
  SetInsnAddr(state->cpu, start_pc);
  SetResidence(*state, kOutsideGeneratedCode);
}

int RunBenchmark(std::string_view name, uint64_t iterations, bool interpreter) {
  if (interpreter && name != "simd_shl32" && name != "simd_ushr32" && name != "simd_cmtst32" &&
      name != "simd_fmax4s" && name != "simd_bic" && name != "simd_bsl8" && name != "simd_umovs") {
    std::fprintf(stderr, "--interpreter requires a straight-line SIMD fallback benchmark\n");
    return 2;
  }
  std::vector<uint32_t> guest_code = MakeBenchmark(name);
  if (guest_code.empty()) {
    std::fprintf(stderr, "unknown benchmark: %.*s\n", static_cast<int>(name.size()), name.data());
    return 2;
  }

  InitHostEntries();
  GuestAddr start_pc = ToGuestAddr(guest_code.data());
  LiteTranslateParams params;
  params.end_pc = start_pc + guest_code.size() * sizeof(uint32_t);
  params.allow_dispatch = false;
  params.enable_reg_mapping = true;
  params.enable_guest_memory = true;

  MachineCode machine_code;
  uint64_t translate_start = MonotonicNanos();
  auto [success, stop_pc] = interpreter ? std::tuple{true, params.end_pc}
                                        : TryLiteTranslateRegion(start_pc, &machine_code, params);
  uint64_t translate_ns = MonotonicNanos() - translate_start;
  if (!success || stop_pc != params.end_pc) {
    std::fprintf(stderr,
                 "translation failed: benchmark=%.*s stop_insn=%" PRIu64 "\n",
                 static_cast<int>(name.size()),
                 name.data(),
                 (stop_pc - start_pc) / sizeof(uint32_t));
    return 3;
  }

  std::unique_ptr<ScopedExecRegion> exec;
  if (!interpreter) exec = std::make_unique<ScopedExecRegion>(&machine_code);
  ThreadState state{};
  InitializeState(&state, start_pc);
  std::array<uint64_t, 16> memory{};
  memory[4] = 0x0123'4567'89ab'cdefULL;
  memory[8] = 0x1111'1111'1111'1111ULL;
  state.cpu.x[10] = ToGuestAddr(memory.data());
  state.cpu.sp = ToGuestAddr(memory.data() + (name == "stack_writeback" ? 8 : 0));
  auto run_once = [&]() {
    SetInsnAddr(state.cpu, start_pc);
    SetResidence(state, kOutsideGeneratedCode);
    if (interpreter) {
      // These fallback benchmarks contain no branches. Avoid dereferencing a null
      // cache by checking only non-sequential control flow.
      InterpretBatch(
          &state, guest_code.size(), nullptr, InterpreterCacheLookupMode::kNonSequentialOnly);
    } else {
      berberis_RunGeneratedCode(&state, AsHostCode(exec->GetHostCodeAddr()));
    }
  };
  for (uint64_t i = 0; i < kWarmupIterations; ++i) {
    run_once();
  }

  uint64_t start_ns = MonotonicNanos();
  for (uint64_t i = 0; i < iterations; ++i) {
    run_once();
  }
  uint64_t elapsed_ns = MonotonicNanos() - start_ns;
  const size_t executed_insns = name == "cache_early_exit" ? 1 : guest_code.size();
  uint64_t guest_insns = iterations * executed_insns;
  uint64_t checksum = state.cpu.x[0] ^ static_cast<uint64_t>(state.cpu.v[0]) ^
                      static_cast<uint64_t>(state.cpu.v[0] >> 64);
  std::printf("{\"backend\":\"%s\",\"benchmark\":\"%.*s\",\"iterations\":%" PRIu64
              ",\"guest_insns_per_iteration\":%zu,\"host_bytes\":%u,"
              "\"host_bytes_per_guest\":%.3f,\"translate_ns\":%" PRIu64 ",\"elapsed_ns\":%" PRIu64
              ",\"ns_per_guest\":%.4f,"
              "\"checksum\":\"0x%016" PRIx64 "\"}\n",
              interpreter ? "interpreter" : "jit",
              static_cast<int>(name.size()),
              name.data(),
              iterations,
              executed_insns,
              machine_code.install_size(),
              static_cast<double>(machine_code.install_size()) / guest_code.size(),
              translate_ns,
              elapsed_ns,
              static_cast<double>(elapsed_ns) / guest_insns,
              checksum);
  return 0;
}

}  // namespace
}  // namespace berberis

int main(int argc, char** argv) {
  std::string_view benchmark = "all";
  bool interpreter = false;
  uint64_t iterations = berberis::kDefaultIterations;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
      benchmark = argv[++i];
    } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      iterations = std::strtoull(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "--interpreter") == 0) {
      interpreter = true;
    } else {
      std::fprintf(
          stderr, "usage: %s [--benchmark NAME|all] [--iterations N] [--interpreter]\n", argv[0]);
      return 2;
    }
  }

  constexpr std::string_view kBenchmarks[] = {"gpr_cached",
                                              "simd_cached",
                                              "conditional_fallthrough",
                                              "mixed",
                                              "small_immediates",
                                              "immediate_memory",
                                              "condition_select",
                                              "flag_conditionals",
                                              "stack_read",
                                              "stack_writeback",
                                              "logical_cached",
                                              "cache_write_first",
                                              "cache_early_exit",
                                              "integer_w",
                                              "simd_shl32",
                                              "simd_ushr32",
                                              "simd_cmtst32",
                                              "simd_fmax4s",
                                              "simd_bic",
                                              "simd_bsl8",
                                              "simd_umovs"};
  if (benchmark != "all") {
    return berberis::RunBenchmark(benchmark, iterations, interpreter);
  }
  for (std::string_view name : kBenchmarks) {
    int result = berberis::RunBenchmark(name, iterations, interpreter);
    if (result != 0) {
      return result;
    }
  }
  return 0;
}
