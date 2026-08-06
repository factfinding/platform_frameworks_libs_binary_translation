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

#include "translator.h"

#include <cstring>
#include <tuple>

#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/logging.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_os_primitives/guest_signal.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/arm64/interpreter.h"
#include "berberis/lite_translator/lite_translate_region.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {
namespace {

enum class TranslationMode {
  kInterpretOnly,
  kLiteTranslateOrInterpret,
};

// Keep the new backend opt-in until its instruction coverage is broad enough
// for arbitrary APK startup code.  Set berberis.mode to
// lite-translate-or-interpret to exercise translated regions.
TranslationMode g_translation_mode = TranslationMode::kInterpretOnly;
uint64_t g_jit_successes;
uint64_t g_jit_fallbacks;

void UpdateTranslationMode() {
  const char* mode = GetTranslationModeConfig();
  if (!mode || strcmp(mode, "interpret-only") == 0) {
    return;
  }
  if (strcmp(mode, "lite-translate-or-interpret") == 0 || strcmp(mode, "two-gear") == 0) {
    g_translation_mode = TranslationMode::kLiteTranslateOrInterpret;
    return;
  }
  LOG_ALWAYS_FATAL("Unsupported LoongArch64 translation mode '%s'", mode);
}

size_t GetExecutableRegionSize(GuestAddr pc) {
  auto [is_executable, size] =
      GuestMapShadow::GetInstance()->GetExecutableRegionSize(pc, config::kGuestPageSize);
  CHECK(is_executable);
  return size;
}

std::tuple<bool, HostCodePiece, size_t> TryLiteTranslateAndInstallRegion(GuestAddr pc) {
  LiteTranslateParams params;
  constexpr size_t kMaxGuestInstructionsPerRegion = 64;
  size_t executable_size = GetExecutableRegionSize(pc);
  size_t max_size = kMaxGuestInstructionsPerRegion * sizeof(uint32_t);
  params.end_pc = pc + (executable_size < max_size ? executable_size : max_size);
  params.allow_dispatch = false;
  params.enable_reg_mapping = false;
  params.enable_guest_memory = true;

  MachineCode machine_code;
  auto [success, stop_pc] = TryLiteTranslateRegion(pc, &machine_code, params);
  size_t size = stop_pc - pc;
  if (success) {
    HostCodePiece piece = InstallTranslated(&machine_code, pc, size, "lite_la64");
    return {piece.code != kNullHostCodeAddr, piece, size};
  }
  if (size == 0) {
    return {false, {}, 0};
  }

  // Preserve a useful supported prefix when the next instruction is not yet
  // implemented.  Re-emitting with a clamped end adds a normal dispatcher
  // exit at the exact unsupported instruction.
  MachineCode prefix_code;
  params.end_pc = stop_pc;
  std::tie(success, stop_pc) = TryLiteTranslateRegion(pc, &prefix_code, params);
  CHECK(success);
  HostCodePiece piece = InstallTranslated(&prefix_code, pc, size, "lite_la64_prefix");
  return {piece.code != kNullHostCodeAddr, piece, size};
}

void TranslateRegion(GuestAddr pc) {
  TranslationCache* cache = TranslationCache::GetInstance();
  GuestCodeEntry* entry = cache->AddAndLockForTranslation(pc, 0);
  if (!entry) {
    return;
  }

  auto [is_executable, insn_size] = IsPcExecutable(pc, GuestMapShadow::GetInstance());
  if (!is_executable) {
    cache->SetTranslatedAndUnlock(
        pc, entry, insn_size, GuestCodeEntry::Kind::kSpecialHandler, {kEntryNoExec, 0});
    return;
  }

  if (g_translation_mode == TranslationMode::kLiteTranslateOrInterpret) {
    auto [success, piece, size] = TryLiteTranslateAndInstallRegion(pc);
    if (success) {
      ++g_jit_successes;
      if (g_jit_successes <= 20 || g_jit_successes % 1000 == 0) {
        TRACE_AND_ALOGD("berberis-la64: JIT #%lu pc=0x%lx insns=%lu",
                        static_cast<unsigned long>(g_jit_successes),
                        static_cast<unsigned long>(pc),
                        static_cast<unsigned long>(size / 4));
      }
      cache->SetTranslatedAndUnlock(pc, entry, size, GuestCodeEntry::Kind::kLiteTranslated, piece);
      return;
    }
  }
  if (g_translation_mode == TranslationMode::kLiteTranslateOrInterpret) {
    ++g_jit_fallbacks;
    if (g_jit_fallbacks <= 20 || g_jit_fallbacks % 1000 == 0) {
      TRACE_AND_ALOGD("berberis-la64: fallback #%lu pc=0x%lx insn=0x%08x",
                      static_cast<unsigned long>(g_jit_fallbacks),
                      static_cast<unsigned long>(pc),
                      *ToHostAddr<const uint32_t>(pc));
    }
  }
  cache->SetTranslatedAndUnlock(
      pc, entry, insn_size, GuestCodeEntry::Kind::kInterpreted, {kEntryInterpret, 0});
}

}  // namespace

void InitTranslatorArch() {
  ClaimHostFaultSignals();
  UpdateTranslationMode();
}

extern "C" __attribute__((used, __visibility__("hidden"))) void berberis_HandleNotTranslated(
    ThreadState* state) {
  TranslateRegion(state->cpu.insn_addr);
}

extern "C" __attribute__((used, __visibility__("hidden"))) void berberis_HandleInterpret(
    ThreadState* state) {
  InterpreterCacheLookupMode lookup_mode =
      g_translation_mode == TranslationMode::kLiteTranslateOrInterpret
          ? InterpreterCacheLookupMode::kAll
          : InterpreterCacheLookupMode::kNonSequentialOnly;
  InterpretBatch(state, 500, TranslationCache::GetInstance(), lookup_mode);
}

extern "C" __attribute__((used, __visibility__("hidden"))) const void* berberis_GetDispatchAddress(
    ThreadState* state) {
  CHECK(state);
  if (ArePendingSignalsPresent(*state)) {
    return AsHostCode(kEntryExitGeneratedCode);
  }
  return AsHostCode(TranslationCache::GetInstance()->GetHostCodePtr(state->cpu.insn_addr)->load());
}

extern "C" __attribute__((used, __visibility__("hidden"))) void
berberis_HandleLiteCounterThresholdReached(ThreadState*) {
  LOG_ALWAYS_FATAL("LoongArch64 bootstrap JIT does not use tiering");
}

}  // namespace berberis
