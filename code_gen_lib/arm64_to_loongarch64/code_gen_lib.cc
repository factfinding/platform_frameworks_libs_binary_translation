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

#include "berberis/code_gen_lib/gen_adaptor.h"
#include "berberis/code_gen_lib/gen_wrapper.h"

#include <ffi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "berberis/assembler/machine_code.h"
#include "berberis/base/checks.h"
#include "berberis/base/logging.h"
#include "berberis/base/macros.h"
#include "berberis/guest_abi/guest_arguments.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/guest_function_wrapper_impl.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

namespace {

// The compiler-rt shipped by the current Android LLVM prebuilt predates
// LoongArch support in __clear_cache and aborts in its fallback path.  Keep
// this definition local to the bridge; upstream compiler-rt implements the
// same operation with ibar 0.
extern "C" __attribute__((visibility("hidden"))) void __clear_cache(void*, void*) {
  __asm__ volatile("ibar 0" ::: "memory");
}

// Each entry is four LoongArch64 instructions.  The table lives in the text
// segment of libberberis_arm64.so, whose CFI shadow is kUncheckedShadow, so a
// CFI-instrumented host library can safely call a wrapper without teaching the
// dynamic linker about anonymous executable mappings.
constexpr size_t kStaticClosureTrampolineSize = 16;
constexpr size_t kStaticClosureTrampolineCount = 16384;

extern "C" {
__attribute__((visibility("hidden"))) extern const uint8_t __berberis_ffi_trampoline_table[];

// Written once before the corresponding trampoline address is published.
// The assembly dispatcher performs the matching acquire barrier after loading
// an entry.
__attribute__((visibility("hidden")))
ffi_closure* __berberis_ffi_closure_slots[kStaticClosureTrampolineCount] = {};
}

std::atomic<size_t> g_next_static_closure_slot{0};

size_t AllocateStaticClosureSlot(void** code) {
  const size_t slot = g_next_static_closure_slot.fetch_add(1, std::memory_order_relaxed);
  LOG_ALWAYS_FATAL_IF(slot >= kStaticClosureTrampolineCount,
                      "Berberis static closure trampoline pool exhausted (%zu entries)",
                      kStaticClosureTrampolineCount);
  *code =
      const_cast<uint8_t*>(__berberis_ffi_trampoline_table) + slot * kStaticClosureTrampolineSize;
  return slot;
}

struct ClosureData {
  ffi_cif cif;
  std::vector<ffi_type*> arg_types;
  std::string signature;
  GuestAddr pc;
  GuestRunnerFunc runner;
  ffi_closure closure;
  void* code;
};

bool IsIntegerType(char type) {
  return type == 'z' || type == 'b' || type == 's' || type == 'c' || type == 'i' || type == 'p' ||
         type == 'l';
}

ffi_type* ToFfiType(char type) {
  switch (type) {
    case 'v':
      return &ffi_type_void;
    case 'z':
      return &ffi_type_uint8;
    case 'b':
      return &ffi_type_sint8;
    case 's':
      return &ffi_type_sint16;
    case 'c':
      return &ffi_type_uint16;
    case 'i':
      return &ffi_type_sint32;
    case 'p':
      return &ffi_type_pointer;
    case 'l':
      return &ffi_type_sint64;
    case 'f':
      return &ffi_type_float;
    case 'd':
      return &ffi_type_double;
    default:
      LOG_ALWAYS_FATAL("Unsupported guest wrapper signature type '%c'", type);
  }
}

uint64_t ReadInteger(char type, void* value) {
  switch (type) {
    case 'z':
      return *static_cast<uint8_t*>(value);
    case 'b':
      return static_cast<uint64_t>(static_cast<int64_t>(*static_cast<int8_t*>(value)));
    case 's':
      return static_cast<uint64_t>(static_cast<int64_t>(*static_cast<int16_t*>(value)));
    case 'c':
      return *static_cast<uint16_t*>(value);
    case 'i':
      return static_cast<uint64_t>(static_cast<int64_t>(*static_cast<int32_t*>(value)));
    case 'p':
      return reinterpret_cast<uintptr_t>(*static_cast<void**>(value));
    case 'l':
      return static_cast<uint64_t>(*static_cast<int64_t*>(value));
    default:
      LOG_ALWAYS_FATAL("Signature type '%c' is not integer-like", type);
  }
}

void WriteResult(char type, void* result, const GuestArgumentBuffer& buffer) {
  const uint64_t value = buffer.argv[0];
  switch (type) {
    case 'v':
      return;
    case 'z':
      *static_cast<uint8_t*>(result) = static_cast<uint8_t>(value);
      return;
    case 'b':
      *static_cast<int8_t*>(result) = static_cast<int8_t>(value);
      return;
    case 's':
      *static_cast<int16_t*>(result) = static_cast<int16_t>(value);
      return;
    case 'c':
      *static_cast<uint16_t*>(result) = static_cast<uint16_t>(value);
      return;
    case 'i':
      *static_cast<int32_t*>(result) = static_cast<int32_t>(value);
      return;
    case 'p':
      *static_cast<void**>(result) = reinterpret_cast<void*>(value);
      return;
    case 'l':
      *static_cast<int64_t*>(result) = static_cast<int64_t>(value);
      return;
    case 'f':
      memcpy(result, &buffer.simd_argv[0], sizeof(float));
      return;
    case 'd':
      memcpy(result, &buffer.simd_argv[0], sizeof(double));
      return;
    default:
      LOG_ALWAYS_FATAL("Unsupported guest wrapper result type '%c'", type);
  }
}

void InvokeGuestClosure(ffi_cif*, void* result, void** args, void* user_data) {
  auto* data = static_cast<ClosureData*>(user_data);
  const char* signature = data->signature.c_str();

  int integer_count = 0;
  int simd_count = 0;
  int stack_count = 0;
  for (size_t i = 1; signature[i] != '\0'; ++i) {
    if (IsIntegerType(signature[i])) {
      if (integer_count++ >= 8) ++stack_count;
    } else {
      CHECK(signature[i] == 'f' || signature[i] == 'd');
      if (simd_count++ >= 8) ++stack_count;
    }
  }

  const size_t buffer_size = sizeof(GuestArgumentBuffer) +
                             static_cast<size_t>(std::max(0, stack_count - 1)) * sizeof(uint64_t);
  std::vector<__uint128_t> storage((buffer_size + sizeof(__uint128_t) - 1) / sizeof(__uint128_t));
  auto* buffer = reinterpret_cast<GuestArgumentBuffer*>(storage.data());
  memset(buffer, 0, buffer_size);

  integer_count = 0;
  simd_count = 0;
  stack_count = 0;
  for (size_t i = 1; signature[i] != '\0'; ++i) {
    const char type = signature[i];
    if (IsIntegerType(type)) {
      uint64_t value = ReadInteger(type, args[i - 1]);
      if (integer_count < 8) {
        buffer->argv[integer_count] = value;
      } else {
        buffer->stack_argv[stack_count++] = value;
      }
      ++integer_count;
    } else {
      const size_t size = type == 'f' ? sizeof(float) : sizeof(double);
      if (simd_count < 8) {
        memcpy(&buffer->simd_argv[simd_count], args[i - 1], size);
      } else {
        memcpy(&buffer->stack_argv[stack_count++], args[i - 1], size);
      }
      ++simd_count;
    }
  }

  buffer->argc = std::min(integer_count, 8);
  buffer->simd_argc = std::min(simd_count, 8);
  buffer->stack_argc = stack_count * sizeof(uint64_t);
  buffer->resc = IsIntegerType(signature[0]) ? 1 : 0;
  buffer->simd_resc = signature[0] == 'f' || signature[0] == 'd' ? 1 : 0;

  data->runner(data->pc, buffer);
  WriteResult(signature[0], result, *buffer);
}

}  // namespace

// LoongArch64 uses direct interpreter dispatch and libffi closures instead of
// generated host trampoline code.
void GenTrampolineAdaptor(MachineCode* mc,
                          GuestAddr pc,
                          HostCode marshall,
                          const void* callee,
                          const char* name) {
  UNUSED(mc, pc, marshall, callee, name);
  LOG_ALWAYS_FATAL("LoongArch64 must use direct host-call dispatch");
}

void GenWrapGuestFunction(MachineCode* mc,
                          GuestAddr pc,
                          const char* signature,
                          HostCode guest_runner,
                          const char* name) {
  UNUSED(mc, pc, signature, guest_runner, name);
  LOG_ALWAYS_FATAL("LoongArch64 must use libffi guest-function wrappers");
}

HostCode CreateGuestFunctionWrapper(GuestAddr pc,
                                    const char* signature,
                                    HostCode guest_runner,
                                    const char* name) {
  UNUSED(name);
  CHECK(signature);
  CHECK(signature[0] != '\0');
  auto data = std::make_unique<ClosureData>();
  data->signature = signature;
  data->pc = pc;
  data->runner = AsFuncPtr<GuestRunnerFunc>(guest_runner);
  data->arg_types.reserve(data->signature.size() - 1);
  for (size_t i = 1; i < data->signature.size(); ++i) {
    data->arg_types.push_back(ToFfiType(data->signature[i]));
  }

  ffi_status status = ffi_prep_cif(&data->cif,
                                   FFI_DEFAULT_ABI,
                                   data->arg_types.size(),
                                   ToFfiType(data->signature[0]),
                                   data->arg_types.data());
  CHECK_EQ(status, FFI_OK);
  const size_t slot = AllocateStaticClosureSlot(&data->code);
  status =
      ffi_prep_closure_loc(&data->closure, &data->cif, InvokeGuestClosure, data.get(), data->code);
  CHECK_EQ(status, FFI_OK);

  // The trampoline becomes callable only after its closure has been fully
  // initialized.  Wrappers are process-lifetime objects, so slots are never
  // cleared or reused.
  __atomic_store_n(&__berberis_ffi_closure_slots[slot], &data->closure, __ATOMIC_RELEASE);

  HostCode result = data->code;
  data.release();  // Wrappers are process-lifetime objects, like the existing code-pool entries.
  return result;
}

}  // namespace berberis
