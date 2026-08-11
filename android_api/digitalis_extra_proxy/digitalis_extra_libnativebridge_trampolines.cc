/*
 * Copyright (C) 2026 utzcoz
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

#if defined(__loongarch__)

#include <dlfcn.h>
#include <signal.h>
#include <stdint.h>
#include <ucontext.h>

#include <cstring>

#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_abi/guest_function_wrapper.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/base/tracing.h"
#include "berberis/runtime_primitives/crash_reporter.h"
#include "berberis/runtime_primitives/runtime_library.h"

namespace berberis {
namespace {

// Some ARM64 protection runtimes patch a host library's GOT with an ARM64 hook
// address.  The next host PLT call then faults while trying to execute the
// non-native bytes.  Recover outside signal context through a deliberately
// integer-only, eight-register ABI bridge.  This covers libc-style hooks such
// as read(2), whose AAPCS64 and LoongArch64 argument registers have the same
// scalar layout; floating-point hooks require a typed wrapper instead.
using GenericIntegerHook = uint64_t(uint64_t,
                                    uint64_t,
                                    uint64_t,
                                    uint64_t,
                                    uint64_t,
                                    uint64_t,
                                    uint64_t,
                                    uint64_t);

thread_local GuestAddr g_direct_guest_call_pc;

uint64_t RunDirectGuestIntegerHook(uint64_t a0,
                                   uint64_t a1,
                                   uint64_t a2,
                                   uint64_t a3,
                                   uint64_t a4,
                                   uint64_t a5,
                                   uint64_t a6,
                                   uint64_t a7) {
  GuestAddr pc = g_direct_guest_call_pc;
  g_direct_guest_call_pc = 0;
  GuestType<GenericIntegerHook*> guest_fn(ToHostAddr<GenericIntegerHook>(pc));
  auto wrapper = WrapGuestFunction(guest_fn, "DirectGuestIntegerHook");
  return wrapper(a0, a1, a2, a3, a4, a5, a6, a7);
}

bool RecoverDirectHostCallIntoGuest(int sig, siginfo_t* info, void* context) {
  if (sig != SIGSEGV || info == nullptr || context == nullptr || info->si_code != SEGV_ACCERR) {
    return false;
  }
  auto* uc = static_cast<ucontext_t*>(context);
  GuestAddr pc = static_cast<GuestAddr>(uc->uc_mcontext.sc_pc);
  if (info->si_addr != ToHostAddr<void>(pc) ||
      !GuestMapShadow::GetInstance()->IsExecutable(pc, 1)) {
    return false;
  }
  g_direct_guest_call_pc = pc;
  uc->uc_mcontext.sc_pc = reinterpret_cast<uintptr_t>(&RunDirectGuestIntegerHook);
  return true;
}

bool HandleNativeBridgeStateQuery(ThreadState* state) {
  void* host_pc = ToHostAddr<void>(GetInsnAddr(GetCPUState(*state)));
  Dl_info info;
  if (dladdr(host_pc, &info) == 0 || info.dli_fname == nullptr || info.dli_sname == nullptr ||
      info.dli_saddr != host_pc) {
    return false;
  }
  const char* slash = std::strrchr(info.dli_fname, '/');
  const char* soname = slash != nullptr ? slash + 1 : info.dli_fname;
  if (std::strcmp(soname, "libnativebridge.so") != 0 ||
      (std::strcmp(info.dli_sname, "NativeBridgeError") != 0 &&
       std::strcmp(info.dli_sname, "NativeBridgeAvailable") != 0 &&
       std::strcmp(info.dli_sname, "NativeBridgeInitialized") != 0)) {
    return false;
  }
  using NativeBridgeStateQuery = bool();
  // This address already has a kEntryNoExec translation: that entry is what
  // brought us here.  AddAndLockForWrapping deliberately refuses to replace
  // an existing translation, so discard the one-byte special-handler entry
  // before installing the typed host trampoline.
  GuestAddr pc = ToGuestAddr(host_pc);
  InvalidateGuestRange(pc, pc + 1);
  WrapHostFunction(reinterpret_cast<NativeBridgeStateQuery*>(host_pc), info.dli_sname);
  return true;
}

bool HandleTypedHostLibcCall(ThreadState* state) {
  void* host_pc = ToHostAddr<void>(GetInsnAddr(GetCPUState(*state)));
  // Hardened code can make dladdr ineffective for an otherwise public libc
  // entry (Hearthstone does this after installing its ART hooks).  Compare
  // the exact addresses returned by the host namespace first.
  void* host_mmap = dlsym(RTLD_DEFAULT, "mmap");
  if (host_pc == host_mmap) {
    using Mmap = void*(void*, size_t, int, int, int, off_t);
    GuestAddr pc = ToGuestAddr(host_pc);
    TRACE_AND_ALOGD("Replacing NoExec entry with typed host wrapper for mmap at %p", host_pc);
    InvalidateGuestRange(pc, pc + 1);
    WrapHostFunction(reinterpret_cast<Mmap*>(host_pc), "mmap");
    return true;
  }

  Dl_info info{};
  int found = dladdr(host_pc, &info);
  if (found == 0 || info.dli_sname == nullptr || info.dli_saddr != host_pc) {
    return false;
  }
  if (std::strcmp(info.dli_sname, "read") == 0) {
    using Read = ssize_t(int, void*, size_t);
    GuestAddr pc = ToGuestAddr(host_pc);
    TRACE_AND_ALOGD("Replacing NoExec entry with typed host wrapper for read at %p", host_pc);
    InvalidateGuestRange(pc, pc + 1);
    WrapHostFunction(reinterpret_cast<Read*>(host_pc), "read");
    return true;
  }
  return false;
}

bool HandleLoongArchNoExec(ThreadState* state) {
  return HandleNativeBridgeStateQuery(state) || HandleTypedHostLibcCall(state);
}

// Hardened ARM64 code may resolve ART's host-only libnativebridge exports by
// parsing the process ELF mappings and branch to an exact symbol address.  No
// ARM64 guest copy exists, but these queries share the no-argument LP64 ABI.
// Register only the no-exec hook during library construction; translation
// cache registration is deferred until Berberis is initialized and a query is
// actually reached.
__attribute__((constructor(102))) void RegisterNativeBridgeStateQueryHook() {
  SetHandleNoExecHook(HandleLoongArchNoExec);
  SetDirectGuestCallHook(RecoverDirectHostCallIntoGuest);
}

}  // namespace
}  // namespace berberis

#endif  // defined(__loongarch__)
