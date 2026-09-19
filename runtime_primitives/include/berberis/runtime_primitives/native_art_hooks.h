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
#ifndef BERBERIS_RUNTIME_PRIMITIVES_NATIVE_ART_HOOKS_H_
#define BERBERIS_RUNTIME_PRIMITIVES_NATIVE_ART_HOOKS_H_

#include <cstdint>

namespace berberis {

enum class NativeArtHookKind { kRead, kMmap };

// Register the original native target before a guest hook can save its address.
void RegisterNativeArtHookOriginal(NativeArtHookKind kind, uintptr_t original);
// Return a typed native wrapper only for an executable guest target. Native
// pointers and already wrapped targets are returned unchanged.
uintptr_t WrapNativeArtHook(NativeArtHookKind kind, uintptr_t target);

// Hearthstone's synchronous JNI initialization replaces libartbase imports
// with ARM64 pointers. Snapshot originals before JNI entry and bridge those
// pointers before returning to ART. This is not a concurrent GOT-write barrier.
void PrepareNativeArtHooks();
void SyncNativeArtHooks();

}  // namespace berberis
#endif
