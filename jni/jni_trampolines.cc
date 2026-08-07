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

#include "berberis/jni/jni_trampolines.h"

#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string_view>
#include <vector>

#include <jni.h>  // NOLINT [build/include_order]
#include <sys/system_properties.h>

#include "berberis/base/checks.h"
#include "berberis/base/gettid.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_abi/function_wrappers.h"
#include "berberis/guest_abi/guest_arguments.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/guest_abi/guest_type.h"
#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/native_bridge/jmethod_shorty.h"
#include "berberis/runtime_primitives/host_code.h"
#include "berberis/runtime_primitives/known_guest_function_wrapper.h"
#include "berberis/runtime_primitives/runtime_library.h"

#include "guest_jni_trampolines.h"

// #define LOG_JNI(...) TRACE_AND_ALOGE(__VA_ARGS__)
#define LOG_JNI(...)

namespace berberis {

namespace {

char ConvertDalvikTypeCharToWrapperTypeChar(char c) {
  switch (c) {
    case 'V':  // void
      return 'v';
    case 'Z':  // boolean
      return 'z';
    case 'B':  // byte
      return 'b';
    case 'S':  // short
      return 's';
    case 'C':  // char
      return 'c';
    case 'I':  // int
      return 'i';
    case 'L':  // class object - pointer
      return 'p';
    case 'J':  // long
      return 'l';
    case 'F':  // float
      return 'f';
    case 'D':  // double
      return 'd';
    default:
      FATAL("Failed to convert Dalvik char '%c'", c);
  }
}

void ConvertDalvikShortyToWrapperSignature(char* dst,
                                           int size,
                                           const char* src,
                                           bool add_jnienv_and_jobject) {
  // return type, env and clazz.
  CHECK_GT(size, 3);
  char* cur = dst;
  *cur++ = ConvertDalvikTypeCharToWrapperTypeChar(*src++);

  if (add_jnienv_and_jobject) {
    *cur++ = 'p';
    *cur++ = 'p';
  }

  while (*src) {
    CHECK_LT(cur, dst + (size - 1));
    *cur++ = ConvertDalvikTypeCharToWrapperTypeChar(*src++);
  }

  *cur = '\0';
}

void RunGuestJNIFunction(GuestAddr pc, GuestArgumentBuffer* buf) {
  auto [host_jni_env] = HostArgumentsValues<void(JNIEnv*)>(buf);
  {
    auto&& [guest_jni_env] = GuestArgumentsReferences<void(JNIEnv*)>(buf);
    guest_jni_env = ToGuestJNIEnv(host_jni_env);
  }
  RunGuestCall(pc, buf);
}

void RunGuestJNIOnLoad(GuestAddr pc, GuestArgumentBuffer* buf) {
  auto [host_java_vm, reserved] = HostArgumentsValues<decltype(JNI_OnLoad)>(buf);
  {
    auto&& [guest_java_vm, reserved] = GuestArgumentsReferences<decltype(JNI_OnLoad)>(buf);
    guest_java_vm = ToGuestJavaVM(host_java_vm);
  }
  RunGuestCall(pc, buf);
}

}  // namespace

HostCode WrapGuestJNIFunction(GuestAddr pc,
                              const char* shorty,
                              const char* name,
                              bool has_jnienv_and_jobject) {
  const size_t size = strlen(shorty);
  char signature[size + /* env, clazz and trailing zero */ 3];
  ConvertDalvikShortyToWrapperSignature(
      signature, sizeof(signature), shorty, has_jnienv_and_jobject);
  auto guest_runner = has_jnienv_and_jobject ? RunGuestJNIFunction : RunGuestCall;
  return WrapGuestFunctionImpl(pc, signature, guest_runner, name);
}

HostCode WrapGuestJNIOnLoad(GuestAddr pc) {
  return WrapGuestFunctionImpl(pc, "ipp", RunGuestJNIOnLoad, "JNI_OnLoad");
}

namespace {

constexpr uint64_t kJniProfileReportIntervalNs = 5'000'000'000ULL;
constexpr uint64_t kJniProfileCheckMask = 63;
constexpr size_t kJniProfileTopCount = 20;

struct JniProfileCounter;

std::mutex g_jni_profile_mutex;
std::vector<JniProfileCounter*> g_jni_profile_counters;
std::atomic<uint64_t> g_jni_profile_calls{0};
std::atomic<uint64_t> g_jni_profile_next_report_ns{0};
std::atomic<pid_t> g_jni_profile_config_pid{-1};
std::atomic<bool> g_jni_profile_enabled{false};

uint64_t JniProfileNowNs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
}

bool IsJniProfileEnabled() {
  const pid_t pid = getpid();
  if (g_jni_profile_config_pid.load(std::memory_order_acquire) == pid) {
    return g_jni_profile_enabled.load(std::memory_order_relaxed);
  }

  std::lock_guard<std::mutex> lock(g_jni_profile_mutex);
  if (g_jni_profile_config_pid.load(std::memory_order_relaxed) != pid) {
    char value[PROP_VALUE_MAX] = {};
    __system_property_get("debug.berberis.jni_profile", value);
    const std::string_view setting(value);
    const bool enabled = setting == "1" || setting == "true" || setting == getprogname();
    g_jni_profile_counters.clear();
    g_jni_profile_calls.store(0, std::memory_order_relaxed);
    g_jni_profile_next_report_ns.store(0, std::memory_order_relaxed);
    g_jni_profile_enabled.store(enabled, std::memory_order_relaxed);
    g_jni_profile_config_pid.store(pid, std::memory_order_release);
    if (enabled) {
      TRACE_AND_ALOGI("berberis-jni-profile: enabled process=%s", getprogname());
    }
  }
  return g_jni_profile_enabled.load(std::memory_order_relaxed);
}

struct JniProfileCounter {
  explicit JniProfileCounter(const char* counter_name) : name(counter_name) {}

  void EnsureRegistered() {
    const pid_t pid = getpid();
    if (registered_pid.load(std::memory_order_acquire) != pid) {
      std::lock_guard<std::mutex> lock(g_jni_profile_mutex);
      if (registered_pid.load(std::memory_order_relaxed) != pid) {
        calls.store(0, std::memory_order_relaxed);
        total_ns.store(0, std::memory_order_relaxed);
        host_ns.store(0, std::memory_order_relaxed);
        max_bridge_ns.store(0, std::memory_order_relaxed);
        g_jni_profile_counters.push_back(this);
        registered_pid.store(pid, std::memory_order_release);
      }
    }
  }

  const char* name;
  std::atomic<pid_t> registered_pid{-1};
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> total_ns{0};
  std::atomic<uint64_t> host_ns{0};
  std::atomic<uint64_t> max_bridge_ns{0};
};

struct JniProfileSnapshot {
  const char* name;
  uint64_t calls;
  uint64_t total_ns;
  uint64_t host_ns;
  uint64_t max_bridge_ns;
};

void DumpJniProfile() {
  std::vector<JniProfileSnapshot> snapshots;
  {
    std::lock_guard<std::mutex> lock(g_jni_profile_mutex);
    snapshots.reserve(g_jni_profile_counters.size());
    for (const JniProfileCounter* counter : g_jni_profile_counters) {
      snapshots.push_back({counter->name,
                           counter->calls.load(std::memory_order_relaxed),
                           counter->total_ns.load(std::memory_order_relaxed),
                           counter->host_ns.load(std::memory_order_relaxed),
                           counter->max_bridge_ns.load(std::memory_order_relaxed)});
    }
  }

  std::sort(snapshots.begin(), snapshots.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.total_ns - lhs.host_ns > rhs.total_ns - rhs.host_ns;
  });

  uint64_t all_calls = 0;
  uint64_t all_bridge_ns = 0;
  for (const auto& snapshot : snapshots) {
    all_calls += snapshot.calls;
    all_bridge_ns += snapshot.total_ns - snapshot.host_ns;
  }

  TRACE_AND_ALOGI("berberis-jni-profile: total_calls=%llu bridge_ms=%.3f counters=%zu",
                  static_cast<unsigned long long>(all_calls),
                  static_cast<double>(all_bridge_ns) / 1'000'000.0,
                  snapshots.size());
  const size_t limit = std::min(kJniProfileTopCount, snapshots.size());
  for (size_t i = 0; i < limit; ++i) {
    const auto& snapshot = snapshots[i];
    if (snapshot.calls == 0) {
      break;
    }
    const uint64_t bridge_ns = snapshot.total_ns - snapshot.host_ns;
    const double share = all_bridge_ns == 0 ? 0.0 : 100.0 * bridge_ns / all_bridge_ns;
    TRACE_AND_ALOGI(
        "berberis-jni-profile: rank=%zu name=%s calls=%llu bridge_ms=%.3f host_ms=%.3f "
        "avg_bridge_ns=%.1f max_bridge_us=%.3f share=%.2f%%",
        i + 1,
        snapshot.name,
        static_cast<unsigned long long>(snapshot.calls),
        static_cast<double>(bridge_ns) / 1'000'000.0,
        static_cast<double>(snapshot.host_ns) / 1'000'000.0,
        static_cast<double>(bridge_ns) / snapshot.calls,
        static_cast<double>(snapshot.max_bridge_ns) / 1'000.0,
        share);
  }
}

class ScopedJniProfile {
 public:
  explicit ScopedJniProfile(JniProfileCounter* counter)
      : counter_(IsJniProfileEnabled() ? counter : nullptr),
        start_ns_(counter_ == nullptr ? 0 : JniProfileNowNs()) {
    if (counter_ != nullptr) {
      counter_->EnsureRegistered();
    }
  }

  void StartHostCall() {
    if (counter_ != nullptr) {
      host_start_ns_ = JniProfileNowNs();
    }
  }

  void EndHostCall() {
    if (counter_ != nullptr && host_start_ns_ != 0) {
      host_elapsed_ns_ += JniProfileNowNs() - host_start_ns_;
      host_start_ns_ = 0;
    }
  }

  ~ScopedJniProfile() {
    if (counter_ == nullptr) {
      return;
    }

    const uint64_t now_ns = JniProfileNowNs();
    if (host_start_ns_ != 0) {
      host_elapsed_ns_ += now_ns - host_start_ns_;
      host_start_ns_ = 0;
    }
    const uint64_t elapsed_ns = now_ns - start_ns_;
    const uint64_t bridge_ns = elapsed_ns - host_elapsed_ns_;
    counter_->calls.fetch_add(1, std::memory_order_relaxed);
    counter_->total_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    counter_->host_ns.fetch_add(host_elapsed_ns_, std::memory_order_relaxed);
    uint64_t old_max = counter_->max_bridge_ns.load(std::memory_order_relaxed);
    while (bridge_ns > old_max && !counter_->max_bridge_ns.compare_exchange_weak(
                                      old_max, bridge_ns, std::memory_order_relaxed)) {
    }

    const uint64_t call = g_jni_profile_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (call <= 64 && (call & (call - 1)) == 0) {
      DumpJniProfile();
      if (call == 64) {
        g_jni_profile_next_report_ns.store(now_ns + kJniProfileReportIntervalNs,
                                           std::memory_order_relaxed);
      }
      return;
    }
    if ((call & kJniProfileCheckMask) != 0) {
      return;
    }

    uint64_t next_report = g_jni_profile_next_report_ns.load(std::memory_order_relaxed);
    if (next_report == 0) {
      g_jni_profile_next_report_ns.compare_exchange_strong(
          next_report, now_ns + kJniProfileReportIntervalNs, std::memory_order_relaxed);
      return;
    }
    if (now_ns < next_report ||
        !g_jni_profile_next_report_ns.compare_exchange_strong(
            next_report, now_ns + kJniProfileReportIntervalNs, std::memory_order_relaxed)) {
      return;
    }
    DumpJniProfile();
  }

 private:
  JniProfileCounter* counter_;
  uint64_t start_ns_;
  uint64_t host_start_ns_ = 0;
  uint64_t host_elapsed_ns_ = 0;
};

std::vector<jvalue> ConvertVAList(JNIEnv* env, jmethodID methodID, GuestVAListParams&& params) {
  std::vector<jvalue> result;
  const char* short_signature = GetJMethodShorty(env, methodID);
  CHECK(short_signature);
  short_signature++;  // skip return value
  int len = strlen(short_signature);
  result.resize(len);
  for (int i = 0; i < len; i++) {
    jvalue& arg = result[i];
    char c = short_signature[i];
    switch (c) {
      case 'Z':  // boolean (u8)
        arg.z = params.GetParam<uint8_t>();
        break;
      case 'B':  // byte (i8)
        arg.b = params.GetParam<int8_t>();
        break;
      case 'S':  // short (i16)
        arg.s = params.GetParam<int16_t>();
        break;
      case 'C':  // char (u16)
        arg.c = params.GetParam<uint16_t>();
        break;
      case 'I':  // int (i32)
        arg.i = params.GetParam<int32_t>();
        break;
      case 'J':  // long (i64)
        arg.j = params.GetParam<int64_t>();
        break;
      case 'F':  // float - passed as double
        arg.f = params.GetParam<double>();
        break;
      case 'D':  // double
        arg.d = params.GetParam<double>();
        break;
      case 'L':  // class object (pointer)
        arg.l = params.GetParam<jobject>();
        break;
      default:
        FATAL("Failed to convert Dalvik char '%c'", c);
        break;
    }
  }
  return result;
}

// jint RegisterNatives(
//     JNIEnv *env, jclass clazz,
//     const JNINativeMethod *methods, jint nMethods);
void DoTrampoline_JNIEnv_RegisterNatives(HostCode /* callee */, ProcessState* state) {
  static JniProfileCounter profile_counter("RegisterNatives");
  ScopedJniProfile profile(&profile_counter);
  using PFN_callee = decltype(std::declval<JNIEnv>().functions->RegisterNatives);
  auto [guest_env, arg_clazz, arg_methods, arg_n] = GuestParamsValues<PFN_callee>(state);
  JNIEnv* arg_env = ToHostJNIEnv(guest_env);

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  profile.StartHostCall();
  ret = (arg_env->functions)->RegisterNatives(arg_env, arg_clazz, arg_methods, arg_n);
  profile.EndHostCall();
}

// jint GetJavaVM(
//     JNIEnv *env, JavaVM **vm);
void DoTrampoline_JNIEnv_GetJavaVM(HostCode /* callee */, ProcessState* state) {
  static JniProfileCounter profile_counter("GetJavaVM");
  ScopedJniProfile profile(&profile_counter);
  using PFN_callee = decltype(std::declval<JNIEnv>().functions->GetJavaVM);
  auto [guest_env, arg_vm] = GuestParamsValues<PFN_callee>(state);
  JNIEnv* arg_env = ToHostJNIEnv(guest_env);
  JavaVM* host_vm;

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  profile.StartHostCall();
  ret = (arg_env->functions)->GetJavaVM(arg_env, &host_vm);
  profile.EndHostCall();
  if (ret == 0) {
    *bit_cast<GuestType<JavaVM*>*>(arg_vm) = ToGuestJavaVM(host_vm);
  }
}

void DoTrampoline_JNIEnv_CallStaticVoidMethodV(HostCode /* callee */, ProcessState* state) {
  static JniProfileCounter profile_counter("CallStaticVoidMethodV");
  ScopedJniProfile profile(&profile_counter);
  using PFN_callee = decltype(std::declval<JNIEnv>().functions->CallStaticVoidMethodV);
  auto [arg_env, arg_1, arg_2, arg_va] = GuestParamsValues<PFN_callee>(state);
  JNIEnv* arg_0 = ToHostJNIEnv(arg_env);
  std::vector<jvalue> arg_vector = ConvertVAList(arg_0, arg_2, ToGuestAddr(arg_va));
  jvalue* arg_3 = &arg_vector[0];

  // Note, this call is the only difference from the auto-generated trampoline.
  profile.StartHostCall();
  JNIEnv_CallStaticVoidMethodV_ForGuest(arg_0, arg_1, arg_2, arg_3);

  (arg_0->functions)->CallStaticVoidMethodA(arg_0, arg_1, arg_2, arg_3);
  profile.EndHostCall();
}

// region digitalis
// jfieldID GetStaticFieldID(JNIEnv*, jclass, const char* name, const char* sig);
//
// A guest jclass is an opaque JNI reference passed through unchanged. Heavily
// obfuscated anti-tamper SDKs (Baidu Maps' sofire / libsofiresec) resolve a
// custom-loaded class via FindClass from a host-spawned worker thread whose
// managed class-loader context can't see that class, so FindClass returns null;
// the SDK then feeds that null straight into GetStaticFieldID. Under the
// emulator's host ART CheckJNI (enabled on userdebug builds; OFF on production
// devices) a null jclass is a process-fatal abort ("java_class == null in call
// to GetStaticFieldID"), which on Baidu Maps kills the :SandBoxProcess. Mirror
// the production (CheckJNI-off) path: on a null jclass, return a null jfieldID
// to the guest without entering host ART, so the abort can't fire and the SDK
// takes its own (null-tolerant) failure branch. arm64-guest only; the riscv64
// build reproduces the original auto-generated forwarding (byte-identical).
void DoTrampoline_JNIEnv_GetStaticFieldID(HostCode /* callee */, ProcessState* state) {
  static JniProfileCounter profile_counter("GetStaticFieldID");
  ScopedJniProfile profile(&profile_counter);
  using PFN_callee = decltype(std::declval<JNIEnv>().functions->GetStaticFieldID);
  auto [guest_env, arg_clazz, arg_name, arg_sig] = GuestParamsValues<PFN_callee>(state);
  JNIEnv* arg_env = ToHostJNIEnv(guest_env);

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64)
  if (arg_clazz == nullptr) {
    TRACE("GetStaticFieldID: guest passed null jclass; returning null jfieldID "
          "instead of aborting under host CheckJNI");
    ret = nullptr;
    return;
  }
#endif
  profile.StartHostCall();
  ret = (arg_env->functions)->GetStaticFieldID(arg_env, arg_clazz, arg_name, arg_sig);
  profile.EndHostCall();
}
// endregion

struct KnownMethodTrampoline {
  unsigned index;
  TrampolineFunc marshal_and_call;
};

#include "jni_trampolines-inl.h"  // NOLINT(build/include)

// According to our observations there is only one instance of JavaVM
// and there are 1 or sometimes more instances of JNIEnv per thread created
// by Java Runtime (JNIEnv instances are not shared between different threads).
//
// This is why we store one global mapping for JavaVM for the app.
// And multiple mappings of JNIEnv per thread. There is often only one JNIEnv
// per thread, but we have seen examples where 2 instances where created.
//
// It is likely that the new JNIEnv instance for the thread supersedes the
// previous one but the code below does not make this assumption.
struct JNIEnvMapping {
  std::deque<JNIEnv> guest_jni_envs;
  std::map<GuestType<JNIEnv*>, JNIEnv*> guest_to_host_jni_env;
  std::map<JNIEnv*, GuestType<JNIEnv*>> host_to_guest_jni_env;
};

std::mutex g_jni_guard_mutex;

JavaVM g_guest_java_vm;
JavaVM* g_host_java_vm;

// TODO(b/399909631): Add a callback from GuestThread::Destroy to remove entries
// from this map.
std::map<pid_t, JNIEnvMapping> g_jni_env_mappings;

void RemoveJNIEnvMappingForTid(pid_t tid) {
  std::lock_guard<std::mutex> lock(g_jni_guard_mutex);
  g_jni_env_mappings.erase(tid);
}

void DoJavaVMTrampoline_DestroyJavaVM(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->DestroyJavaVM);
  auto [arg_vm] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->DestroyJavaVM(arg_java_vm);
}

// jint AttachCurrentThread(JavaVM*, JNIEnv**, void*);
void DoJavaVMTrampoline_AttachCurrentThread(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->AttachCurrentThread);
  auto [arg_vm, arg_env_ptr, arg_args] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);
  JNIEnv* env = nullptr;

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->AttachCurrentThread(arg_java_vm, &env, arg_args);

  GuestType<JNIEnv*> guest_jni_env = ToGuestJNIEnv(env);
  memcpy(arg_env_ptr, &guest_jni_env, sizeof(guest_jni_env));
}

// jint DetachCurrentThread(JavaVM*);
void DoJavaVMTrampoline_DetachCurrentThread(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->DetachCurrentThread);
  auto [arg_vm] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);

  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->DetachCurrentThread(arg_java_vm);
}

// jint GetEnv(JavaVM*, void**, jint);
void DoJavaVMTrampoline_GetEnv(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->GetEnv);
  auto [arg_vm, arg_env_ptr, arg_version] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);

  LOG_JNI("JavaVM::GetEnv(%p, %p, %d)", arg_java_vm, arg_env_ptr, arg_version);

  void* env = nullptr;
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->GetEnv(arg_java_vm, &env, arg_version);

  GuestType<JNIEnv*> guest_jni_env = ToGuestJNIEnv(static_cast<JNIEnv*>(env));
  memcpy(arg_env_ptr, &guest_jni_env, sizeof(guest_jni_env));

  LOG_JNI("= jint(%d)", ret);
}

// jint AttachCurrentThreadAsDaemon(JavaVM* vm, void** penv, void* args);
void DoJavaVMTrampoline_AttachCurrentThreadAsDaemon(HostCode /* callee */, ProcessState* state) {
  using PFN_callee = decltype(std::declval<JavaVM>().functions->AttachCurrentThreadAsDaemon);
  auto [arg_vm, arg_env_ptr, arg_args] = GuestParamsValues<PFN_callee>(state);
  JavaVM* arg_java_vm = ToHostJavaVM(arg_vm);

  JNIEnv* env = nullptr;
  auto&& [ret] = GuestReturnReference<PFN_callee>(state);
  ret = (arg_java_vm->functions)->AttachCurrentThreadAsDaemon(arg_java_vm, &env, arg_args);

  GuestType<JNIEnv*> guest_jni_env = ToGuestJNIEnv(env);
  memcpy(arg_env_ptr, &guest_jni_env, sizeof(guest_jni_env));
}

void WrapJavaVM(void* java_vm) {
  HostCode* vtable = *reinterpret_cast<HostCode**>(java_vm);
  // vtable[0] is NULL
  // vtable[1] is NULL
  // vtable[2] is NULL

  WrapHostFunctionImpl(vtable[3], DoJavaVMTrampoline_DestroyJavaVM, "JavaVM::DestroyJavaVM");

  WrapHostFunctionImpl(
      vtable[4], DoJavaVMTrampoline_AttachCurrentThread, "JavaVM::AttachCurrentThread");

  WrapHostFunctionImpl(
      vtable[5], DoJavaVMTrampoline_DetachCurrentThread, "JavaVM::DetachCurrentThread");

  WrapHostFunctionImpl(vtable[6], DoJavaVMTrampoline_GetEnv, "JavaVM::GetEnv");

  WrapHostFunctionImpl(vtable[7],
                       DoJavaVMTrampoline_AttachCurrentThreadAsDaemon,
                       "JavaVM::AttachCurrentThreadAsDaemon");
}

// We set this to 1 when host JavaVM functions are wrapped.
std::atomic<uint32_t> g_java_vm_wrapped = {0};

// region digitalis
// Wrap each distinct host JNIEnv function table exactly once, keyed by the
// JNINativeInterface* the env points at. A single process-global "wrapped once"
// flag is insufficient for a guest that hosts more than one Java runtime /
// function table: Chromium's sandboxed renderer is forked from its own
// app-zygote and runs with a FRESH host JNIEnv whose table was never registered
// as guest trampolines, and an inherited global flag then suppresses wrapping.
// The renderer's first JNI call (NewStringUTF) then branches straight into host
// libart and trips berberis_HandleNoExec (SIGSEGV). Observed with Brave's
// sandboxed renderer. WrapJNIEnv (-> MakeTrampolineCallable) is idempotent per
// host function address, so wrapping additional tables only registers new
// addresses.
std::mutex g_jni_wrap_mutex;

void WrapJNIEnvTableOnce(JNIEnv* host_jni_env) {
  if (host_jni_env == nullptr) {
    return;
  }
  static auto* g_wrapped_jni_tables = new std::map<const void*, char>();
  const void* table = *reinterpret_cast<const void* const*>(host_jni_env);
  std::lock_guard<std::mutex> lock(g_jni_wrap_mutex);
  if (g_wrapped_jni_tables->emplace(table, 0).second) {
    WrapJNIEnv(host_jni_env);
  }
}
// endregion

}  // namespace

GuestType<JNIEnv*> ToGuestJNIEnv(JNIEnv* host_jni_env) {
  if (!host_jni_env) {
    return 0;
  }
  // region digitalis
  // Wrap this host env's JNINativeInterface table the first time we see it.
  // Per-table (not a single process-global flag) so a second Java runtime's
  // fresh table — e.g. Chromium's forked sandboxed renderer — is wrapped too.
  WrapJNIEnvTableOnce(host_jni_env);
  // endregion

  std::lock_guard<std::mutex> lock(g_jni_guard_mutex);
  pid_t thread_id = GettidSyscall();
  JNIEnvMapping& mapping = g_jni_env_mappings[thread_id];

  auto it = mapping.host_to_guest_jni_env.find(host_jni_env);
  if (it != mapping.host_to_guest_jni_env.end()) {
    return it->second;
  }

  mapping.guest_jni_envs.emplace_back(*host_jni_env);
  JNIEnv* guest_jni_env = &mapping.guest_jni_envs.back();
  auto [unused_it1, host_to_guest_inserted] =
      mapping.host_to_guest_jni_env.try_emplace(host_jni_env, guest_jni_env);
  CHECK(host_to_guest_inserted);

  auto [unused_it2, guest_to_host_inserted] =
      mapping.guest_to_host_jni_env.try_emplace(guest_jni_env, host_jni_env);
  CHECK(guest_to_host_inserted);

  return guest_jni_env;
}

JNIEnv* ToHostJNIEnv(GuestType<JNIEnv*> guest_jni_env) {
  std::lock_guard<std::mutex> lock(g_jni_guard_mutex);
  pid_t thread_id = GettidSyscall();
  JNIEnvMapping& mapping = g_jni_env_mappings[thread_id];

  auto it = mapping.guest_to_host_jni_env.find(guest_jni_env);

  if (it == mapping.guest_to_host_jni_env.end()) {
    TRACE_AND_ALOGE(
        "Unexpected guest JNIEnv: %p (it was never passed to guest), passing to host 'as is'",
        ToHostAddr(guest_jni_env));
    return ToHostAddr(guest_jni_env);
  }

  return it->second;
}

// region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64) && defined(__ANDROID__)
// On a real device an app's own native code is implicitly exempt from non-SDK
// (hidden) API restrictions. Under NativeBridge the guest app's native code is
// translated guest code, which host ART classifies in a domain the app's
// exemptions don't cover, so a JNI lookup of a hidden method from guest native
// code is blocked. Gecko (Firefox) hits exactly this: AndroidBridge::GetMethodID
// for android.os.MessageQueue.next() returns null and MOZ_CRASH()es the process.
// Grant the process a blanket hidden-API exemption via host ART's VMRuntime as
// soon as we have the host JavaVM, mirroring real-device behaviour. Best-effort:
// any failure is traced and ignored. Called once, under g_jni_guard_mutex.
void GrantHiddenApiExemptions(JavaVM* host_java_vm) {
  JNIEnv* env = nullptr;
  if (host_java_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK ||
      env == nullptr) {
    TRACE("GrantHiddenApiExemptions: no host JNIEnv, skipping");
    return;
  }
  // Clear any pending exception after EACH JNI lookup before making the next
  // JNI call. On a target-SDK-35 app whose calling context host ART attributes
  // to the app domain (e.g. bugsnag's NativeBridge on the stack), the
  // GetMethodID for the core-platform method setHiddenApiExemptions is denied
  // by hidden-API enforcement: it returns nullptr AND leaves a pending
  // NoSuchMethodError. Making any further JNI call (the FindClass below, or a
  // class resolution it triggers) while that exception is pending makes host
  // ART abort with "No pending exception expected". Clearing after every lookup
  // keeps this best-effort path able to bail cleanly instead of crashing the
  // process.
  jclass vmruntime_class = env->FindClass("dalvik/system/VMRuntime");
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  }
  jclass string_class = env->FindClass("java/lang/String");
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  }
  jmethodID get_runtime =
      vmruntime_class ? env->GetStaticMethodID(
                            vmruntime_class, "getRuntime", "()Ldalvik/system/VMRuntime;")
                      : nullptr;
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  }
  jmethodID set_exemptions =
      vmruntime_class
          ? env->GetMethodID(vmruntime_class, "setHiddenApiExemptions", "([Ljava/lang/String;)V")
          : nullptr;
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  }
  if (vmruntime_class == nullptr || get_runtime == nullptr || set_exemptions == nullptr ||
      string_class == nullptr) {
    TRACE("GrantHiddenApiExemptions: VMRuntime API not resolvable, skipping");
    return;
  }
  jobject runtime = env->CallStaticObjectMethod(vmruntime_class, get_runtime);
  // A single "L" entry matches the prefix of every signature, exempting all.
  jstring all = env->NewStringUTF("L");
  jobjectArray exemptions = env->NewObjectArray(1, string_class, all);
  if (runtime != nullptr && exemptions != nullptr) {
    env->CallVoidMethod(runtime, set_exemptions, exemptions);
  }
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    TRACE("GrantHiddenApiExemptions: setHiddenApiExemptions threw, ignoring");
  } else {
    TRACE("GrantHiddenApiExemptions: granted blanket hidden-API exemption");
  }
}
#endif
// endregion

GuestType<JavaVM*> ToGuestJavaVM(JavaVM* host_java_vm) {
  CHECK(host_java_vm);
  if (std::atomic_load_explicit(&g_java_vm_wrapped, std::memory_order_acquire) == 0U) {
    WrapJavaVM(host_java_vm);
    std::atomic_store_explicit(&g_java_vm_wrapped, 1U, std::memory_order_release);
  }

  std::lock_guard<std::mutex> lock(g_jni_guard_mutex);
  if (g_host_java_vm == nullptr) {
    g_guest_java_vm = *host_java_vm;
    g_host_java_vm = host_java_vm;
    // region digitalis
#if defined(NATIVE_BRIDGE_GUEST_ARCH_ARM64) && defined(__ANDROID__)
    GrantHiddenApiExemptions(host_java_vm);
#endif
    // endregion
  }

  if (g_host_java_vm != host_java_vm) {
    TRACE("Warning: Unexpected host JavaVM: %p (expecting %p), passing as is",
          host_java_vm,
          g_host_java_vm);
    return host_java_vm;
  }

  return &g_guest_java_vm;
}

JavaVM* ToHostJavaVM(GuestType<JavaVM*> guest_java_vm) {
  std::lock_guard<std::mutex> lock(g_jni_guard_mutex);
  if (ToHostAddr(guest_java_vm) == &g_guest_java_vm) {
    return g_host_java_vm;
  }

  TRACE("Warning: Unexpected guest JavaVM: %p (expecting %p), passing as is",
        ToHostAddr(guest_java_vm),
        &g_guest_java_vm);

  return ToHostAddr(guest_java_vm);
}

// region digitalis
JavaVM* GetHostJavaVM() {
  return g_host_java_vm;
}
// endregion

namespace {

GuestThreadExitListenerFn g_next_guest_thread_exit_listener = nullptr;

void JNIGuestThreadListener(pid_t tid) {
  RemoveJNIEnvMappingForTid(tid);
  if (g_next_guest_thread_exit_listener != nullptr) {
    g_next_guest_thread_exit_listener(tid);
  }
}

}  // namespace

void InitializeJNI() {
  RegisterKnownGuestFunctionWrapper("JNI_OnLoad", WrapGuestJNIOnLoad);
  CHECK(g_next_guest_thread_exit_listener == nullptr);
  g_next_guest_thread_exit_listener = RegisterGuestThreadExitListener(JNIGuestThreadListener);
}

}  // namespace berberis
