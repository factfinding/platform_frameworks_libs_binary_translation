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

#include <cstddef>

#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/host_function_wrapper_impl.h"

extern "C" void berberis_HandleInterpret(berberis::ThreadState* state);
extern "C" void berberis_HandleNotTranslated(berberis::ThreadState* state);
extern "C" const void* berberis_GetDispatchAddress(berberis::ThreadState* state);
extern "C" void berberis_HandleLiteCounterThresholdReached(berberis::ThreadState* state);

// Inline assembly needs an unmangled entry for the C++ host-call dispatcher.
__attribute__((used, __visibility__("hidden"))) extern "C" void berberis_RunHostCallFromGuest(
    berberis::ThreadState* state) {
  berberis::RunHostCallFromGuest(state);
}

// Generated-code ABI:
//   $s8 - ThreadState pointer
//   $s7 - current guest PC
// All other caller-saved registers may be clobbered by a translated region.
// The frame is 96 bytes so $sp remains 16-byte aligned at calls.
// clang-format off
#define END_GENERATED_CODE(EXIT_INSN)                                   \
  asm(                                                                  \
      "st.d $s7, $s8, %[InsnAddr]\n"                                   \
      "addi.w $t0, $zero, %[OutsideGeneratedCode]\n"                   \
      "st.b $t0, $s8, %[Residence]\n"                                  \
      "move $a0, $s8\n"                                                \
      "ld.d $ra, $sp, 80\n"                                            \
      "ld.d $fp, $sp, 72\n"                                            \
      "ld.d $s0, $sp, 64\n"                                            \
      "ld.d $s1, $sp, 56\n"                                            \
      "ld.d $s2, $sp, 48\n"                                            \
      "ld.d $s3, $sp, 40\n"                                            \
      "ld.d $s4, $sp, 32\n"                                            \
      "ld.d $s5, $sp, 24\n"                                            \
      "ld.d $s6, $sp, 16\n"                                            \
      "ld.d $s7, $sp, 8\n"                                             \
      "ld.d $s8, $sp, 0\n"                                             \
      "addi.d $sp, $sp, 96\n"                                          \
      EXIT_INSN                                                         \
      ::[InsnAddr] "I"(offsetof(berberis::ThreadState, cpu.insn_addr)), \
      [Residence] "I"(offsetof(berberis::ThreadState, residence)),      \
      [OutsideGeneratedCode] "I"(berberis::kOutsideGeneratedCode))
// clang-format on

namespace berberis {

extern "C" {

[[gnu::naked]] [[gnu::noinline]] void berberis_RunGeneratedCode(ThreadState* state, HostCode code) {
  // $a0 contains state and $a1 contains the generated-code entry.
  // clang-format off
  asm(
      "addi.d $sp, $sp, -96\n"
      "st.d $s8, $sp, 0\n"
      "st.d $s7, $sp, 8\n"
      "st.d $s6, $sp, 16\n"
      "st.d $s5, $sp, 24\n"
      "st.d $s4, $sp, 32\n"
      "st.d $s3, $sp, 40\n"
      "st.d $s2, $sp, 48\n"
      "st.d $s1, $sp, 56\n"
      "st.d $s0, $sp, 64\n"
      "st.d $fp, $sp, 72\n"
      "st.d $ra, $sp, 80\n"
      "move $s8, $a0\n"
      "ld.d $s7, $s8, %[InsnAddr]\n"
      "addi.w $t0, $zero, %[InsideGeneratedCode]\n"
      "st.b $t0, $s8, %[Residence]\n"
      "jr $a1\n"
      ::[InsnAddr] "I"(offsetof(ThreadState, cpu.insn_addr)),
      [Residence] "I"(offsetof(ThreadState, residence)),
      [InsideGeneratedCode] "I"(kInsideGeneratedCode));
  // clang-format on
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Interpret() {
  // clang-format off
  asm(
      "st.d $s7, $s8, %[InsnAddr]\n"
      "addi.w $t0, $zero, %[OutsideGeneratedCode]\n"
      "st.b $t0, $s8, %[Residence]\n"
      "move $a0, $s8\n"
      "bl berberis_HandleInterpret\n"
      "move $a0, $s8\n"
      "bl berberis_GetDispatchAddress\n"
      "move $t1, $a0\n"
      "ld.d $s7, $s8, %[InsnAddr]\n"
      "addi.w $t0, $zero, %[InsideGeneratedCode]\n"
      "st.b $t0, $s8, %[Residence]\n"
      "jr $t1\n"
      ::[InsnAddr] "I"(offsetof(ThreadState, cpu.insn_addr)),
      [Residence] "I"(offsetof(ThreadState, residence)),
      [OutsideGeneratedCode] "I"(kOutsideGeneratedCode),
      [InsideGeneratedCode] "I"(kInsideGeneratedCode));
  // clang-format on
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_ExitGeneratedCode() {
  END_GENERATED_CODE("jr $ra\n");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Stop() {
  END_GENERATED_CODE("jr $ra\n");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_NoExec() {
  END_GENERATED_CODE("b berberis_HandleNoExec\n");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_NotTranslated() {
  END_GENERATED_CODE("b berberis_HandleNotTranslated\n");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Translating() {
  END_GENERATED_CODE("jr $ra\n");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Invalidating() {
  END_GENERATED_CODE("jr $ra\n");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_Wrapping() {
  END_GENERATED_CODE("jr $ra\n");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_HandleLiteCounterThresholdReached() {
  END_GENERATED_CODE("b berberis_HandleLiteCounterThresholdReached\n");
}

[[gnu::naked]] [[gnu::noinline]] void berberis_entry_WrappedHostCall() {
  // clang-format off
  asm(
      "st.d $s7, $s8, %[InsnAddr]\n"
      "addi.w $t0, $zero, %[OutsideGeneratedCode]\n"
      "st.b $t0, $s8, %[Residence]\n"
      "move $a0, $s8\n"
      "bl berberis_RunHostCallFromGuest\n"
      "move $a0, $s8\n"
      "bl berberis_GetDispatchAddress\n"
      "move $t1, $a0\n"
      "ld.d $s7, $s8, %[InsnAddr]\n"
      "addi.w $t0, $zero, %[InsideGeneratedCode]\n"
      "st.b $t0, $s8, %[Residence]\n"
      "jr $t1\n"
      ::[InsnAddr] "I"(offsetof(ThreadState, cpu.insn_addr)),
      [Residence] "I"(offsetof(ThreadState, residence)),
      [OutsideGeneratedCode] "I"(kOutsideGeneratedCode),
      [InsideGeneratedCode] "I"(kInsideGeneratedCode));
  // clang-format on
}

}  // extern "C"

}  // namespace berberis
