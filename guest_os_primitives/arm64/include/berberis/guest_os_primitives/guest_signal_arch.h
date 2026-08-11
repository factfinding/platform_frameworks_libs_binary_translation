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

#ifndef BERBERIS_GUEST_OS_PRIMITIVES_ARM64_GUEST_SIGNAL_ARCH_H_
#define BERBERIS_GUEST_OS_PRIMITIVES_ARM64_GUEST_SIGNAL_ARCH_H_

namespace berberis {

// Guest struct (__kernel_)sigaction, as expected by rt_sigaction syscall.
// ARM64 uses asm-generic's layout without SA_RESTORER.  Do not confuse this
// with bionic's public struct sigaction, which has a source-compatible
// sa_restorer member that bionic removes before issuing the syscall.
struct Guest_sigaction {
  // Prefix avoids conflict with original 'sa_sigaction' defined as macro.
  GuestAddr guest_sa_sigaction;
  unsigned long sa_flags;
  Guest_sigset_t sa_mask;
};

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
CHECK_STRUCT_LAYOUT(Guest_sigaction, 192, 64);
CHECK_FIELD_LAYOUT(Guest_sigaction, guest_sa_sigaction, 0, 64);
CHECK_FIELD_LAYOUT(Guest_sigaction, sa_flags, 64, 64);
CHECK_FIELD_LAYOUT(Guest_sigaction, sa_mask, 128, 64);
#else
#error "Unexpected guest arch."
#endif

}  // namespace berberis

#endif  // BERBERIS_GUEST_OS_PRIMITIVES_ARM64_GUEST_SIGNAL_ARCH_H_
