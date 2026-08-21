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

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string_view>
#include <vector>

#include "berberis/assembler/machine_code.h"
#include "berberis/guest_state/guest_state.h"
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

int RunBenchmark(std::string_view name, uint64_t iterations) {
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
  auto [success, stop_pc] = TryLiteTranslateRegion(start_pc, &machine_code, params);
  uint64_t translate_ns = MonotonicNanos() - translate_start;
  if (!success || stop_pc != params.end_pc) {
    std::fprintf(stderr,
                 "translation failed: benchmark=%.*s stop_insn=%" PRIu64 "\n",
                 static_cast<int>(name.size()),
                 name.data(),
                 (stop_pc - start_pc) / sizeof(uint32_t));
    return 3;
  }

  ScopedExecRegion exec(&machine_code);
  ThreadState state{};
  InitializeState(&state, start_pc);
  auto run_once = [&]() {
    SetInsnAddr(state.cpu, start_pc);
    SetResidence(state, kOutsideGeneratedCode);
    berberis_RunGeneratedCode(&state, AsHostCode(exec.GetHostCodeAddr()));
  };
  for (uint64_t i = 0; i < kWarmupIterations; ++i) {
    run_once();
  }

  uint64_t start_ns = MonotonicNanos();
  for (uint64_t i = 0; i < iterations; ++i) {
    run_once();
  }
  uint64_t elapsed_ns = MonotonicNanos() - start_ns;
  uint64_t guest_insns = iterations * guest_code.size();
  uint64_t checksum = state.cpu.x[0] ^ static_cast<uint64_t>(state.cpu.v[0]) ^
                      static_cast<uint64_t>(state.cpu.v[0] >> 64);
  std::printf("{\"benchmark\":\"%.*s\",\"iterations\":%" PRIu64
              ",\"guest_insns_per_iteration\":%zu,\"host_bytes\":%u,"
              "\"host_bytes_per_guest\":%.3f,\"translate_ns\":%" PRIu64 ",\"elapsed_ns\":%" PRIu64
              ",\"ns_per_guest\":%.4f,"
              "\"checksum\":\"0x%016" PRIx64 "\"}\n",
              static_cast<int>(name.size()),
              name.data(),
              iterations,
              guest_code.size(),
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
  uint64_t iterations = berberis::kDefaultIterations;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
      benchmark = argv[++i];
    } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      iterations = std::strtoull(argv[++i], nullptr, 10);
    } else {
      std::fprintf(stderr, "usage: %s [--benchmark NAME|all] [--iterations N]\n", argv[0]);
      return 2;
    }
  }

  constexpr std::string_view kBenchmarks[] = {
      "gpr_cached", "simd_cached", "conditional_fallthrough", "mixed"};
  if (benchmark != "all") {
    return berberis::RunBenchmark(benchmark, iterations);
  }
  for (std::string_view name : kBenchmarks) {
    int result = berberis::RunBenchmark(name, iterations);
    if (result != 0) {
      return result;
    }
  }
  return 0;
}
