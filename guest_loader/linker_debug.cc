/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include "berberis/guest_loader/guest_loader.h"

#include <link.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "berberis/base/macros.h"
#include "berberis/base/tracing.h"
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#endif
#include "berberis/instrument/loader.h"

#include "guest_loader_impl.h"  // MakeElfSymbolTrampolineCallable

namespace berberis {

namespace {

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
using ExecutableRange = std::pair<GuestAddr, size_t>;

void SyncExecutableSegments(const link_map* link) {
  std::vector<ExecutableRange> current_ranges;
  for (; link != nullptr; link = link->l_next) {
    // The synthetic native-bridge vDSO has a non-zero lowest PT_LOAD virtual
    // address, so l_addr is a load bias rather than an ELF-header address. It
    // was already registered by TinyLoader and does not need mirroring here.
    if (link->l_name != nullptr && strcmp(link->l_name, "[vdso]") == 0) {
      continue;
    }
    auto base = static_cast<uintptr_t>(link->l_addr);
    auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(base);
    if (ehdr->e_ident[EI_MAG0] != ELFMAG0 || ehdr->e_ident[EI_MAG1] != ELFMAG1 ||
        ehdr->e_ident[EI_MAG2] != ELFMAG2 || ehdr->e_ident[EI_MAG3] != ELFMAG3 ||
        ehdr->e_machine != EM_AARCH64 || ehdr->e_phentsize != sizeof(Elf64_Phdr) ||
        ehdr->e_phnum == 0 || ehdr->e_phnum > 128) {
      continue;
    }

    auto* phdrs = reinterpret_cast<const Elf64_Phdr*>(base + ehdr->e_phoff);
    for (size_t i = 0; i < ehdr->e_phnum; ++i) {
      if (phdrs[i].p_type == PT_LOAD && (phdrs[i].p_flags & PF_X) != 0 &&
          phdrs[i].p_memsz != 0) {
        GuestAddr start = static_cast<GuestAddr>(base + phdrs[i].p_vaddr);
        current_ranges.emplace_back(start, phdrs[i].p_memsz);
      }
    }
  }

  std::sort(current_ranges.begin(), current_ranges.end());
  current_ranges.erase(std::unique(current_ranges.begin(), current_ranges.end()),
                       current_ranges.end());

  static std::mutex ranges_mutex;
  static std::vector<ExecutableRange> registered_ranges;
  std::lock_guard<std::mutex> lock(ranges_mutex);
  GuestMapShadow* shadow = GuestMapShadow::GetInstance();
  for (const auto& range : registered_ranges) {
    if (!std::binary_search(current_ranges.begin(), current_ranges.end(), range)) {
      shadow->ClearExecutable(range.first, range.second);
    }
  }
  for (const auto& range : current_ranges) {
    if (!std::binary_search(registered_ranges.begin(), registered_ranges.end(), range)) {
      shadow->SetExecutable(range.first, range.second);
    }
  }
  registered_ranges = std::move(current_ranges);
}
#endif

void DoCustomTrampoline_rtld_db_dlactivity(HostCode callee, ThreadState* state) {
  UNUSED(callee, state);

  // It would be also tempting to bind r_debug to callee, but then we need to know r_debug when
  // creating the trampoline. Also, it seems r_debug might still be 0 when rtld_db_dlactivity is
  // called first couple of times.
  // Thus, search and check.
  const auto* debug = GuestLoader::GetInstance()->FindRDebug();
  if (debug == nullptr) {
    return;
  }

  // ATTENTION: assume struct r_debug and struct link_map are compatible!
  if (debug->r_state == r_debug::RT_CONSISTENT) {
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
    // The guest linker maps shared objects through its host-facing loader
    // path, bypassing MmapForGuest.  Mirror executable PT_LOAD segments into
    // Berberis' guest permission shadow before any ELF constructors run.
    SyncExecutableSegments(debug->r_map);
#endif
    OnConsistentLinkMap(debug->r_map);
  }
}

}  // namespace

void InitLinkerDebug(const LoadedElfFile& linker_elf_file) {
  if (!kInstrumentLoader) {
    return;
  }

  // The correct way to hook linker rtld_db_dlactivity would be to read struct r_debug pointer from
  // main executable's DT_DEBUG and get breakpoint address from there.
  // Unfortunately, DT_DEBUG gets initialized by guest linker, which didn't yet run at this point.
  // Instead, hope breakpoint symbol is exported.
  std::string error_msg;
  if (!MakeElfSymbolTrampolineCallable(linker_elf_file,
                                       "linker",
                                       "rtld_db_dlactivity",
                                       DoCustomTrampoline_rtld_db_dlactivity,
                                       nullptr,
                                       &error_msg)) {
    TRACE("failed to hook rtld_db_dlactivity: %s", error_msg.c_str());
  }
}

}  // namespace berberis
