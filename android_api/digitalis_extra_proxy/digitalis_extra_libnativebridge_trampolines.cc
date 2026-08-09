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

#include <cstring>

#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/runtime_library.h"

namespace berberis {
namespace {

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
  WrapHostFunction(reinterpret_cast<NativeBridgeStateQuery*>(host_pc), info.dli_sname);
  return true;
}

// Hardened ARM64 code may resolve ART's host-only libnativebridge exports by
// parsing the process ELF mappings and branch to an exact symbol address.  No
// ARM64 guest copy exists, but these queries share the no-argument LP64 ABI.
// Register only the no-exec hook during library construction; translation
// cache registration is deferred until Berberis is initialized and a query is
// actually reached.
__attribute__((constructor(102))) void RegisterNativeBridgeStateQueryHook() {
  SetHandleNoExecHook(HandleNativeBridgeStateQuery);
}

}  // namespace
}  // namespace berberis

#endif  // defined(__loongarch__)
