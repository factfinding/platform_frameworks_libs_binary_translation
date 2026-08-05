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

#include "../faulty_memory_accesses.h"

#include <cstdint>
#include <utility>

#include "berberis/base/checks.h"
#include "berberis/runtime_primitives/recovery_code.h"

namespace berberis {
namespace {

extern "C" FaultyLoadResult FaultyLoad8(const void*);
extern "C" FaultyLoadResult FaultyLoad16(const void*);
extern "C" FaultyLoadResult FaultyLoad32(const void*);
extern "C" FaultyLoadResult FaultyLoad64(const void*);
extern "C" char g_faulty_load_recovery;

__asm__(R"(
  .globl FaultyLoad8
  .balign 16
FaultyLoad8:
  ld.bu $a0, $a0, 0
  move $a1, $zero
  jr $ra

  .globl FaultyLoad16
  .balign 16
FaultyLoad16:
  ld.hu $a0, $a0, 0
  move $a1, $zero
  jr $ra

  .globl FaultyLoad32
  .balign 16
FaultyLoad32:
  ld.wu $a0, $a0, 0
  move $a1, $zero
  jr $ra

  .globl FaultyLoad64
  .balign 16
FaultyLoad64:
  ld.d $a0, $a0, 0
  move $a1, $zero
  jr $ra

  .globl g_faulty_load_recovery
g_faulty_load_recovery:
  li.d $a1, 1
  jr $ra
)");

extern "C" bool FaultyStore8(void*, uint64_t);
extern "C" bool FaultyStore16(void*, uint64_t);
extern "C" bool FaultyStore32(void*, uint64_t);
extern "C" bool FaultyStore64(void*, uint64_t);
extern "C" char g_faulty_store_recovery;

__asm__(R"(
  .globl FaultyStore8
  .balign 16
FaultyStore8:
  st.b $a1, $a0, 0
  move $a0, $zero
  jr $ra

  .globl FaultyStore16
  .balign 16
FaultyStore16:
  st.h $a1, $a0, 0
  move $a0, $zero
  jr $ra

  .globl FaultyStore32
  .balign 16
FaultyStore32:
  st.w $a1, $a0, 0
  move $a0, $zero
  jr $ra

  .globl FaultyStore64
  .balign 16
FaultyStore64:
  st.d $a1, $a0, 0
  move $a0, $zero
  jr $ra

  .globl g_faulty_store_recovery
g_faulty_store_recovery:
  li.d $a0, 1
  jr $ra
)");

template <typename FaultyAccessPointer>
std::pair<uintptr_t, uintptr_t> MakePairAdapter(FaultyAccessPointer fault_addr,
                                                void* recovery_addr) {
  return {reinterpret_cast<uintptr_t>(fault_addr), reinterpret_cast<uintptr_t>(recovery_addr)};
}

}  // namespace

FaultyLoadResult FaultyLoad(const void* addr, uint8_t data_bytes) {
  switch (data_bytes) {
    case 1:
      return FaultyLoad8(addr);
    case 2:
      return FaultyLoad16(addr);
    case 4:
      return FaultyLoad32(addr);
    case 8:
      return FaultyLoad64(addr);
    default:
      LOG_ALWAYS_FATAL("Unexpected FaultyLoad access size");
  }
}

bool FaultyStore(void* addr, uint8_t data_bytes, uint64_t value) {
  switch (data_bytes) {
    case 1:
      return FaultyStore8(addr, value);
    case 2:
      return FaultyStore16(addr, value);
    case 4:
      return FaultyStore32(addr, value);
    case 8:
      return FaultyStore64(addr, value);
    default:
      LOG_ALWAYS_FATAL("Unexpected FaultyStore access size");
  }
}

void AddFaultyMemoryAccessRecoveryCode() {
  InitExtraRecoveryCodeUnsafe({
      MakePairAdapter(&FaultyLoad8, &g_faulty_load_recovery),
      MakePairAdapter(&FaultyLoad16, &g_faulty_load_recovery),
      MakePairAdapter(&FaultyLoad32, &g_faulty_load_recovery),
      MakePairAdapter(&FaultyLoad64, &g_faulty_load_recovery),
      MakePairAdapter(&FaultyStore8, &g_faulty_store_recovery),
      MakePairAdapter(&FaultyStore16, &g_faulty_store_recovery),
      MakePairAdapter(&FaultyStore32, &g_faulty_store_recovery),
      MakePairAdapter(&FaultyStore64, &g_faulty_store_recovery),
  });
}

void* FindFaultyMemoryAccessRecoveryAddrForTesting(void* fault_addr) {
  if (fault_addr == &FaultyLoad8 || fault_addr == &FaultyLoad16 || fault_addr == &FaultyLoad32 ||
      fault_addr == &FaultyLoad64) {
    return &g_faulty_load_recovery;
  }
  if (fault_addr == &FaultyStore8 || fault_addr == &FaultyStore16 || fault_addr == &FaultyStore32 ||
      fault_addr == &FaultyStore64) {
    return &g_faulty_store_recovery;
  }
  return nullptr;
}

}  // namespace berberis
