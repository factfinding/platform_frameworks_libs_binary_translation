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

#ifndef BERBERIS_RUNTIME_PRIMITIVES_CRASH_REPORTER_H_
#define BERBERIS_RUNTIME_PRIMITIVES_CRASH_REPORTER_H_

#include <csignal>

namespace berberis {

void InitCrashReporter();
void HandleFatalSignal(int sig, siginfo_t* info, void* context);

// Optional last-chance recovery for a host control-flow transfer into guest
// code.  A native bridge may use it to replace the interrupted host PC with a
// host-callable guest wrapper.  The hook runs in signal context and must only
// inspect lock-free state and modify the supplied ucontext.
using DirectGuestCallHook = bool (*)(int sig, siginfo_t* info, void* context);
void SetDirectGuestCallHook(DirectGuestCallHook hook);

}  // namespace berberis

#endif  // BERBERIS_RUNTIME_PRIMITIVES_CRASH_REPORTER_H_
