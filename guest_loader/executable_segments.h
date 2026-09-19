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

#ifndef BERBERIS_GUEST_LOADER_EXECUTABLE_SEGMENTS_H_
#define BERBERIS_GUEST_LOADER_EXECUTABLE_SEGMENTS_H_

#include <elf.h>
#include <link.h>

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"

namespace berberis {

// The caller serializes Sync while the guest link map is consistent.
class LinkerExecutableSegments {
 public:
  void Sync(const link_map* link, GuestMapShadow* shadow) {
    std::vector<Object> current;
    std::vector<Range> added_ranges;
    for (; link != nullptr; link = link->l_next) {
      auto old = std::find_if(objects_.begin(), objects_.end(), [link](const Object& object) {
        return object.Matches(link);
      });
      if (old != objects_.end()) {
        // Loaded objects may rewrite or erase their ELF headers after loading.
        // Re-reading them on an unrelated dlopen must neither simulate an unload
        // nor undo subsequent guest mprotect calls. Snapshot only on first load.
        current.push_back(*old);
        continue;
      }
      if (link->l_addr == 0 || (link->l_name != nullptr && strcmp(link->l_name, "[vdso]") == 0)) {
        continue;
      }
      auto base = static_cast<uintptr_t>(link->l_addr);
      auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(base);
      if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 || ehdr->e_machine != EM_AARCH64 ||
          ehdr->e_phentsize != sizeof(Elf64_Phdr) || ehdr->e_phnum == 0 || ehdr->e_phnum > 128) {
        continue;
      }
      Object object{link, base, {}};
      auto* phdrs = reinterpret_cast<const Elf64_Phdr*>(base + ehdr->e_phoff);
      for (size_t i = 0; i < ehdr->e_phnum; ++i) {
        if (phdrs[i].p_type == PT_LOAD && (phdrs[i].p_flags & PF_X) != 0 && phdrs[i].p_memsz != 0) {
          object.ranges.emplace_back(static_cast<GuestAddr>(base + phdrs[i].p_vaddr),
                                     phdrs[i].p_memsz);
        }
      }
      added_ranges.insert(added_ranges.end(), object.ranges.begin(), object.ranges.end());
      current.push_back(std::move(object));
    }
    for (const auto& old : objects_) {
      if (std::none_of(current.begin(), current.end(), [&](const Object& object) {
            return old.link == object.link && old.base == object.base;
          })) {
        for (const auto& range : old.ranges) {
          shadow->ClearExecutable(range.first, range.second);
        }
      }
    }
    // Clear unloaded objects before registering new objects: their address
    // ranges can overlap when the linker reuses an old mapping.
    for (const auto& range : added_ranges) {
      shadow->SetExecutable(range.first, range.second);
    }
    objects_ = std::move(current);
  }

 private:
  using Range = std::pair<GuestAddr, size_t>;
  struct Object {
    const link_map* link;
    uintptr_t base;
    std::vector<Range> ranges;

    bool Matches(const link_map* other) const {
      return link == other && base == static_cast<uintptr_t>(other->l_addr);
    }
  };
  std::vector<Object> objects_;
};

}  // namespace berberis

#endif  // BERBERIS_GUEST_LOADER_EXECUTABLE_SEGMENTS_H_
