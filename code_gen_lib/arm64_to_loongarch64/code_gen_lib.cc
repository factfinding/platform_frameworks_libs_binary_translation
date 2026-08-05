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

#include "berberis/assembler/machine_code.h"
#include "berberis/base/macros.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/runtime_primitives/host_code.h"

namespace berberis {

// The interpreter and shared-library link do not require generated host code.
// These placeholders make that build boundary explicit while LoongArch64
// trampoline emission is implemented. They must not be enabled in a product.
void GenTrampolineAdaptor(MachineCode* mc,
                          GuestAddr pc,
                          HostCode marshall,
                          const void* callee,
                          const char* name) {
  UNUSED(mc, pc, marshall, callee, name);
}

void GenWrapGuestFunction(MachineCode* mc,
                          GuestAddr pc,
                          const char* signature,
                          HostCode guest_runner,
                          const char* name) {
  UNUSED(mc, pc, signature, guest_runner, name);
}

}  // namespace berberis
