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

#include "berberis/runtime_primitives/runtime_library.h"

#include "berberis/base/logging.h"
#include "berberis/guest_state/guest_state_opaque.h"

namespace berberis {

extern "C" void berberis_HandleInterpret(ThreadState* state);
extern "C" void berberis_HandleNotTranslated(ThreadState* state);

extern "C" {

// Interpreter-only LoongArch64 dispatch does not enter generated code. These
// functions are distinct address tokens stored in TranslationCache.
#define DEFINE_ENTRY_TOKEN(name)              \
  [[gnu::noinline]] void name() {             \
    asm volatile("" ::: "memory");           \
  }

DEFINE_ENTRY_TOKEN(berberis_entry_Interpret)
DEFINE_ENTRY_TOKEN(berberis_entry_ExitGeneratedCode)
DEFINE_ENTRY_TOKEN(berberis_entry_Stop)
DEFINE_ENTRY_TOKEN(berberis_entry_NoExec)
DEFINE_ENTRY_TOKEN(berberis_entry_NotTranslated)
DEFINE_ENTRY_TOKEN(berberis_entry_Translating)
DEFINE_ENTRY_TOKEN(berberis_entry_Invalidating)
DEFINE_ENTRY_TOKEN(berberis_entry_Wrapping)
DEFINE_ENTRY_TOKEN(berberis_entry_HandleLiteCounterThresholdReached)

#undef DEFINE_ENTRY_TOKEN

void berberis_RunGeneratedCode(ThreadState* state, HostCode code) {
  const HostCodeAddr entry = AsHostCodeAddr(code);
  SetResidence(*state, kOutsideGeneratedCode);

  if (entry == kEntryInterpret) {
    berberis_HandleInterpret(state);
  } else if (entry == kEntryNotTranslated) {
    berberis_HandleNotTranslated(state);
  } else if (entry == kEntryNoExec) {
    berberis_HandleNoExec(state);
  } else if (entry == kEntryStop || entry == kEntryExitGeneratedCode ||
             entry == kEntryTranslating || entry == kEntryInvalidating) {
    return;
  } else if (entry == kEntryWrapping) {
    LOG_ALWAYS_FATAL("LoongArch64 host trampoline dispatch is not implemented");
  } else {
    LOG_ALWAYS_FATAL("LoongArch64 JIT code dispatch is disabled in interpreter-only mode");
  }
}

}  // extern "C"
}  // namespace berberis
