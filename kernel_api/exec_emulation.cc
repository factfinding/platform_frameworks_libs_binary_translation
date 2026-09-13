/*
 * Copyright (C) 2018 The Android Open Source Project
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

#include "berberis/kernel_api/exec_emulation.h"

#include <unistd.h>

#include <cstring>
#include <utility>

#include "berberis/base/mmap.h"
#include "berberis/base/strings.h"

namespace berberis {

namespace {

std::pair<const char*, size_t> GetGuestPlatformVarPrefixWithSize() {
  static constexpr char kGuestPlatformVarPrefix[] = "BERBERIS_GUEST_";
  return {kGuestPlatformVarPrefix, sizeof(kGuestPlatformVarPrefix) - 1};
}

bool IsPlatformVar(const char* s) {
  return StartsWith(s, "LD_CONFIG_FILE=") || StartsWith(s, "LD_LIBRARY_PATH=") ||
         StartsWith(s, "LD_DEBUG=") || StartsWith(s, "LD_PRELOAD=");
}

char* const* MangleGuestEnvp(ScopedMmap* dst,
                             char* const* envp,
                             const char* guest_ld_library_path_prefix) {
  if (envp == nullptr) {
    return nullptr;
  }

  int env_count = 0;
  int text_size = 0;
  int mangle_count = 0;
  bool has_ld_library_path = false;

  for (;; ++env_count) {
    char* env = envp[env_count];
    if (env == nullptr) {
      break;
    }

    if (IsPlatformVar(env)) {
      ++mangle_count;
    }
    if (StartsWith(env, "LD_LIBRARY_PATH=")) {
      has_ld_library_path = true;
      if (guest_ld_library_path_prefix != nullptr) {
        text_size += strlen(guest_ld_library_path_prefix) + 1;  // prefix + ':'
      }
    }

    text_size += strlen(env) + 1;  // count terminating '\0'
  }

  const bool add_ld_library_path =
      guest_ld_library_path_prefix != nullptr && !has_ld_library_path;
  if (mangle_count == 0 && !add_ld_library_path) {
    return envp;
  }

  auto [guest_prefix, guest_prefix_size] = GetGuestPlatformVarPrefixWithSize();
  static constexpr char kLdLibraryPath[] = "LD_LIBRARY_PATH=";

  if (add_ld_library_path) {
    text_size += guest_prefix_size + sizeof(kLdLibraryPath) - 1 +
                 strlen(guest_ld_library_path_prefix) + 1;
  }

  const int new_env_count = env_count + static_cast<int>(add_ld_library_path);
  size_t array_size = sizeof(char*) * (new_env_count + 1);  // pointers + terminating nullptr
  dst->Init(array_size + text_size +                      // array + orig text
            guest_prefix_size * mangle_count);            // added prefixes

  char** new_array = static_cast<char**>(dst->data());
  char* new_text = static_cast<char*>(dst->data()) + array_size;

  for (int i = 0; i < env_count; ++i) {
    char* env = envp[i];
    new_array[i] = new_text;

    if (IsPlatformVar(env)) {
      strcpy(new_text, guest_prefix);
      new_text += guest_prefix_size;
    }

    if (guest_ld_library_path_prefix != nullptr && StartsWith(env, kLdLibraryPath)) {
      const char* old_path = env + sizeof(kLdLibraryPath) - 1;
      strcpy(new_text, kLdLibraryPath);
      new_text += sizeof(kLdLibraryPath) - 1;
      strcpy(new_text, guest_ld_library_path_prefix);
      new_text += strlen(guest_ld_library_path_prefix);
      *new_text++ = ':';
      strcpy(new_text, old_path);
      new_text += strlen(old_path) + 1;
    } else {
      strcpy(new_text, env);
      new_text += strlen(env) + 1;  // count terminating '\0'
    }
  }

  if (add_ld_library_path) {
    new_array[env_count] = new_text;
    strcpy(new_text, guest_prefix);
    new_text += guest_prefix_size;
    strcpy(new_text, kLdLibraryPath);
    new_text += sizeof(kLdLibraryPath) - 1;
    strcpy(new_text, guest_ld_library_path_prefix);
  }

  new_array[new_env_count] = nullptr;  // add terminating nullptr

  return new_array;
}

}  // namespace

char** DemangleGuestEnvp(char** dst, char** envp) {
  auto [guest_prefix, guest_prefix_size] = GetGuestPlatformVarPrefixWithSize();

  for (; *envp; ++envp) {
    char* env = *envp;
    if (IsPlatformVar(env)) {
      continue;
    }
    if (StartsWith(env, guest_prefix) && IsPlatformVar(env + guest_prefix_size)) {
      env += guest_prefix_size;
    }
    *dst++ = env;
  }

  *dst++ = nullptr;
  return dst;
}

int ExecveForGuest(const char* filename, char* const argv[], char* const envp[]) {
  const char* host_filename = filename;
  char* const* host_argv = argv;
  const char* guest_ld_library_path_prefix = nullptr;
  ScopedMmap new_argv;

#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64) && defined(__loongarch__)
  // Android executables often launch another ELF explicitly through
  // /system/bin/linker64.  In an ARM64 guest process that path resolves to the
  // LoongArch64 host linker, which rejects the guest ELF before the native
  // bridge can see it.  Route such invocations through the standalone
  // Berberis runner; it selects /system/bin/arm64/linker64 for the guest and
  // preserves the original linker arguments after argv[0].
  static constexpr char kSystemLinker64[] = "/system/bin/linker64";
  static constexpr char kArm64ProgramRunner[] = "/system/bin/berberis_program_runner_arm64";
  if (filename != nullptr && strcmp(filename, kSystemLinker64) == 0 && argv != nullptr &&
      argv[0] != nullptr && argv[1] != nullptr) {
    size_t argc = 0;
    while (argv[argc] != nullptr) {
      ++argc;
    }
    new_argv.Init((argc + 1) * sizeof(char*));
    auto** rewritten_argv = static_cast<char**>(new_argv.data());
    rewritten_argv[0] = const_cast<char*>(kArm64ProgramRunner);
    for (size_t i = 1; i <= argc; ++i) {
      rewritten_argv[i] = argv[i];
    }
    host_filename = kArm64ProgramRunner;
    host_argv = rewritten_argv;
    guest_ld_library_path_prefix = "/system/lib64/arm64";
  }
#endif

  ScopedMmap new_envp;
  return execve(host_filename,
                host_argv,
                MangleGuestEnvp(&new_envp, envp, guest_ld_library_path_prefix));
}

}  // namespace berberis
