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

#include "berberis/base/checks.h"
#include "berberis/base/logging.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_os_primitives/guest_signal.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/interpreter/arm64/interpreter.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {

void InitTranslatorArch() {
  ClaimHostFaultSignals();
}

extern "C" __attribute__((used, __visibility__("hidden"))) void berberis_HandleNotTranslated(
    ThreadState* state) {
  TranslationCache* cache = TranslationCache::GetInstance();
  GuestAddr pc = state->cpu.insn_addr;
  GuestCodeEntry* entry = cache->AddAndLockForTranslation(pc, 0);
  if (!entry) {
    return;
  }

  auto [is_executable, insn_size] = IsPcExecutable(pc, GuestMapShadow::GetInstance());
  HostCodePiece target{is_executable ? kEntryInterpret : kEntryNoExec, 0};
  GuestCodeEntry::Kind kind = is_executable ? GuestCodeEntry::Kind::kInterpreted
                                            : GuestCodeEntry::Kind::kSpecialHandler;
  cache->SetTranslatedAndUnlock(pc, entry, insn_size, kind, target);
}

extern "C" __attribute__((used, __visibility__("hidden"))) void berberis_HandleInterpret(
    ThreadState* state) {
  InterpretBatch(
      state, 500, TranslationCache::GetInstance(), InterpreterCacheLookupMode::kNonSequentialOnly);
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
  LOG_ALWAYS_FATAL("LoongArch64 JIT is disabled in interpreter-only mode");
}

}  // namespace berberis
