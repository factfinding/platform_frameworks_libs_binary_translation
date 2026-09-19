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

#include "berberis/runtime_primitives/native_art_hooks.h"

#include <elf.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <mutex>
#include <vector>

#include "berberis/base/scoped_errno.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"

namespace berberis {
namespace {
using ReadFn = ssize_t (*)(int, void*, size_t);
using MmapFn = void* (*)(void*, size_t, int, int, int, off_t);

struct Import {
  uintptr_t* slot;
  NativeArtHookKind kind;
};

bool Enabled() {
  // Both the main process and the TACT download service install the audited
  // synchronous initialization hooks. Keep other processes on the usual path.
  static const bool enabled =
      strcmp(getprogname(), "com.blizzard.wtcg.hearthstone") == 0 ||
      strcmp(getprogname(), "com.blizzard.wtcg.hearthstone:tact_service") == 0;
  return enabled;
}

bool Contains(const dl_phdr_info& info, uintptr_t address, size_t size, bool writable = false) {
  for (size_t i = 0; i < info.dlpi_phnum; ++i) {
    const auto& ph = info.dlpi_phdr[i];
    if (ph.p_type != PT_LOAD || !(ph.p_flags & PF_R) || (writable && !(ph.p_flags & PF_W))) {
      continue;
    }
    const uintptr_t start = info.dlpi_addr + ph.p_vaddr;
    if (address >= start && address - start <= ph.p_memsz &&
        size <= ph.p_memsz - (address - start)) {
      return true;
    }
  }
  return false;
}

int CollectImports(dl_phdr_info* info, size_t, void* opaque) {
  const char* slash = strrchr(info->dlpi_name, '/');
  const char* name = slash ? slash + 1 : info->dlpi_name;
  if (strcmp(name, "libartbase.so") != 0) {
    return 0;
  }
  auto& imports = *static_cast<std::vector<Import>*>(opaque);
  const ElfW(Dyn)* dynamic = nullptr;
  size_t dynamic_count = 0;
  for (size_t i = 0; i < info->dlpi_phnum; ++i) {
    const auto& ph = info->dlpi_phdr[i];
    if (ph.p_type == PT_DYNAMIC) {
      dynamic = reinterpret_cast<const ElfW(Dyn)*>(info->dlpi_addr + ph.p_vaddr);
      dynamic_count = ph.p_memsz / sizeof(*dynamic);
    }
  }
  if (!dynamic ||
      !Contains(*info, reinterpret_cast<uintptr_t>(dynamic), dynamic_count * sizeof(*dynamic))) {
    return 1;
  }
  uintptr_t relocs = 0, symbols = 0, strings = 0;
  size_t relocs_size = 0, strings_size = 0, symbol_size = 0;
  bool rela = false;
  for (size_t i = 0; i < dynamic_count && dynamic[i].d_tag != DT_NULL; ++i) {
    const auto& d = dynamic[i];
    switch (d.d_tag) {
      case DT_JMPREL:
        relocs = info->dlpi_addr + d.d_un.d_ptr;
        break;
      case DT_PLTRELSZ:
        relocs_size = d.d_un.d_val;
        break;
      case DT_PLTREL:
        rela = d.d_un.d_val == DT_RELA;
        break;
      case DT_SYMTAB:
        symbols = info->dlpi_addr + d.d_un.d_ptr;
        break;
      case DT_SYMENT:
        symbol_size = d.d_un.d_val;
        break;
      case DT_STRTAB:
        strings = info->dlpi_addr + d.d_un.d_ptr;
        break;
      case DT_STRSZ:
        strings_size = d.d_un.d_val;
        break;
    }
  }
  if (!rela || symbol_size != sizeof(ElfW(Sym)) || relocs_size % sizeof(ElfW(Rela)) != 0 ||
      !Contains(*info, relocs, relocs_size) || !Contains(*info, strings, strings_size)) {
    return 1;
  }
  for (size_t i = 0; i < relocs_size / sizeof(ElfW(Rela)); ++i) {
    const auto& rel = reinterpret_cast<const ElfW(Rela)*>(relocs)[i];
    if (ELF64_R_TYPE(rel.r_info) != R_LARCH_JUMP_SLOT) {
      continue;
    }
    const uintptr_t sym_addr = symbols + ELF64_R_SYM(rel.r_info) * sizeof(ElfW(Sym));
    if (!Contains(*info, sym_addr, sizeof(ElfW(Sym)))) {
      continue;
    }
    const auto& sym = *reinterpret_cast<const ElfW(Sym)*>(sym_addr);
    if (sym.st_name >= strings_size) {
      continue;
    }
    const char* symbol = reinterpret_cast<const char*>(strings + sym.st_name);
    const size_t remaining = strings_size - sym.st_name;
    NativeArtHookKind kind;
    if (remaining >= 5 && memcmp(symbol, "read", 5) == 0) {
      kind = NativeArtHookKind::kRead;
    } else if (remaining >= 5 && memcmp(symbol, "mmap", 5) == 0) {
      kind = NativeArtHookKind::kMmap;
    } else {
      continue;
    }
    const uintptr_t slot_addr = info->dlpi_addr + rel.r_offset;
    if (slot_addr % alignof(uintptr_t) != 0 ||
        !Contains(*info, slot_addr, sizeof(uintptr_t), true)) {
      continue;
    }
    imports.push_back({reinterpret_cast<uintptr_t*>(slot_addr), kind});
  }
  return 1;
}

std::vector<Import>& Imports() {
  static std::vector<Import> imports;
  return imports;
}

// The guest may have restored RELRO after writing its hook. Restore the exact
// current page protection after publishing the wrapper, never add host X.
int PageProtection(uintptr_t address) {
  FILE* maps = fopen("/proc/self/maps", "re");
  if (!maps) return -1;
  char* line = nullptr;
  size_t capacity = 0;
  int prot = -1;
  while (getline(&line, &capacity, maps) >= 0) {
    unsigned long start, end;
    char permissions[5]{};
    if (sscanf(line, "%lx-%lx %4s", &start, &end, permissions) == 3 && address >= start &&
        address < end) {
      prot = (permissions[0] == 'r' ? PROT_READ : 0) | (permissions[1] == 'w' ? PROT_WRITE : 0) |
             (permissions[2] == 'x' ? PROT_EXEC : 0);
      break;
    }
  }
  free(line);
  fclose(maps);
  return prot;
}
}  // namespace

void RegisterNativeArtHookOriginal(NativeArtHookKind kind, uintptr_t original) {
  if (!original || GuestMapShadow::GetInstance()->IsExecutable(original, 1)) return;
  if (kind == NativeArtHookKind::kRead) {
    WrapHostFunction(reinterpret_cast<ReadFn>(original), "native_art_original_read");
  } else {
    WrapHostFunction(reinterpret_cast<MmapFn>(original), "native_art_original_mmap");
  }
}

uintptr_t WrapNativeArtHook(NativeArtHookKind kind, uintptr_t target) {
  if (!target || !GuestMapShadow::GetInstance()->IsExecutable(target, 1)) return target;
  if (kind == NativeArtHookKind::kRead) {
    return reinterpret_cast<uintptr_t>(
        WrapGuestFunction(GuestType<ReadFn>(target), "native_art_read_hook"));
  }
  return reinterpret_cast<uintptr_t>(
      WrapGuestFunction(GuestType<MmapFn>(target), "native_art_mmap_hook"));
}

void PrepareNativeArtHooks() {
  if (!Enabled()) return;
  ScopedErrno scoped_errno;
  static std::once_flag once;
  std::call_once(once, [] {
    dl_iterate_phdr(CollectImports, &Imports());
    for (const auto& entry : Imports()) {
      RegisterNativeArtHookOriginal(entry.kind, __atomic_load_n(entry.slot, __ATOMIC_ACQUIRE));
    }
    TRACE_AND_ALOGD("Native ART hook bridge: tracking %zu imports", Imports().size());
  });
}

void SyncNativeArtHooks() {
  if (!Enabled()) return;
  ScopedErrno scoped_errno;
  // Serialize temporary RELRO changes across JNI returns on different threads.
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& entry : Imports()) {
    uintptr_t target = __atomic_load_n(entry.slot, __ATOMIC_ACQUIRE);
    const uintptr_t wrapper = WrapNativeArtHook(entry.kind, target);
    if (wrapper == target) continue;
    const uintptr_t address = reinterpret_cast<uintptr_t>(entry.slot);
    const int prot = PageProtection(address);
    if (prot < 0 || !(prot & PROT_READ)) continue;
    const size_t page_size = getpagesize();
    void* page = reinterpret_cast<void*>(address & ~(page_size - 1));
    if (!(prot & PROT_WRITE) && mprotect(page, page_size, prot | PROT_WRITE) != 0) continue;
    bool replaced = __atomic_compare_exchange_n(
        entry.slot, &target, wrapper, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    if (!(prot & PROT_WRITE) && mprotect(page, page_size, prot) != 0) {
      TRACE_AND_ALOGE("Native ART hook bridge: failed to restore RELRO at %p", page);
    }
    if (replaced) {
      TRACE_AND_ALOGD("Native ART hook bridge: %s guest=%p wrapper=%p",
                      entry.kind == NativeArtHookKind::kRead ? "read" : "mmap",
                      reinterpret_cast<void*>(target),
                      reinterpret_cast<void*>(wrapper));
    }
  }
}
}  // namespace berberis
