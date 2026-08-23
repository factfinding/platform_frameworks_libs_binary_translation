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

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>

#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/runtime_library.h"

namespace berberis {
namespace {

struct AddressRange {
  uintptr_t begin;
  uintptr_t end;

  bool Contains(uintptr_t address) const { return address >= begin && address < end; }
};

AddressRange FindNativeBridgeTextRange() {
  AddressRange range{std::numeric_limits<uintptr_t>::max(), 0};
  // libnativebridge is outside libberberis' Android linker namespace, so both
  // RTLD_DEFAULT and dl_iterate_phdr can omit it even though hardened guest
  // code can discover the mapping through /proc/self/maps.
  FILE* maps = std::fopen("/proc/self/maps", "r");
  if (maps == nullptr) {
    return range;
  }
  char line[4096];
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    if (std::strstr(line, "/libnativebridge.so") == nullptr) {
      continue;
    }
    uintptr_t begin;
    uintptr_t end;
    char permissions[5];
    if (std::sscanf(line, "%lx-%lx %4s", &begin, &end, permissions) != 3 ||
        std::strchr(permissions, 'x') == nullptr) {
      continue;
    }
    range.begin = std::min(range.begin, begin);
    range.end = std::max(range.end, end);
  }
  std::fclose(maps);
  return range;
}

bool HandleNativeBridgeStateQuery(ThreadState* state) {
  void* host_pc = ToHostAddr<void>(GetInsnAddr(GetCPUState(*state)));
  // Reject ordinary guest PCs with a cheap cached range check; dladdr is then
  // needed only for the tiny native-bridge text range.
  static const AddressRange kNativeBridgeText = FindNativeBridgeTextRange();
  if (!kNativeBridgeText.Contains(reinterpret_cast<uintptr_t>(host_pc))) {
    return false;
  }
  Dl_info info;
  if (dladdr(host_pc, &info) == 0 || info.dli_sname == nullptr || info.dli_saddr != host_pc ||
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
// Register only the exact-target hook during library construction; translation
// cache registration is deferred until Berberis is initialized and a query is
// actually reached. The runtime consults this hook before translation because
// host and guest executable mappings may occupy the same GuestMapShadow page.
__attribute__((constructor(102))) void RegisterNativeBridgeStateQueryHook() {
  SetHandleNoExecHook(HandleNativeBridgeStateQuery);
}

}  // namespace
}  // namespace berberis

#endif  // defined(__loongarch__)
