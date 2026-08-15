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

#include "berberis/guest_os_primitives/guest_setjmp.h"

#include <csetjmp>
#include <cstdint>
#include <cstring>  // memcpy

#include "berberis/base/host_signal.h"
#include "berberis/guest_state/guest_state.h"

namespace berberis {

namespace {

// jmp_buf format is totally Bionic-private (see bionic/libc/arch-arm64/bionic/setjmp.S)
// We don't have to use the original format as save/restore is only done here.
// Still, let's keep it compatible with some release as it might help debugging...
//
// ARM64 jmp_buf layout (bionic):
// word   name            description
// 0      sigflag/cookie  setjmp cookie in top 63 bits, signal mask flag in low bit
// 1      sigmask         64-bit signal mask
// 2      x30 (LR)
// 3      SP
// 4      x28
// 5      x29 (FP)
// 6      x26
// 7      x27
// 8      x24
// 9      x25
// 10     x22
// 11     x23
// 12     x20
// 13     x21
// 14     low bits of x18 (shadow-call-stack pointer)
// 15     x19
// 16     d14
// 17     d15
// 18     d12
// 19     d13
// 20     d10
// 21     d11
// 22     d8
// 23     d9
// 24     checksum
// _JBLEN: defined in bionic/libc/include/setjmp.h (32 for arm64)

const int kJmpBufSigFlagAndCookieWord = 0;
const int kJmpBufSigMaskWord = 1;
const int kJmpBufLRWord = 2;
const int kJmpBufSPWord = 3;
const int kJmpBufX28Word = 4;
const int kJmpBufX26Word = 6;
const int kJmpBufX24Word = 8;
const int kJmpBufX22Word = 10;
const int kJmpBufX20Word = 12;
const int kJmpBufSCSWord = 14;
const int kJmpBufX19Word = 15;
const int kJmpBufFloatingPointBaseWord = 16;
const int kJmpBufChecksumWord = 24;
// jmp_buf should be at least 32 words long.
// Use the last word to store the address of the host jmp_buf.
const int kJmpBufHostBufWord = 31;

// jmp_buf cookie can be anything but 0 (see bionic/tests/setjmp_test.cpp: setjmp_cookie)
// ATTENTION: Keep low bit 0 for signal mask flag.
const uint64_t kJmpBufCookie = 0x12'3210ULL;
const uint64_t kShadowCallStackMask = 16 * 1024 - 1;

uint64_t CalcJumpBufChecksum(const uint64_t* buf) {
  uint64_t res = 0;
  for (int i = 0; i < kJmpBufChecksumWord; ++i) {
    res ^= buf[i];
  }
  return res;
}

}  // namespace

void SaveRegsToJumpBuf(const ThreadState* state, void* guest_jmp_buf, int save_sig_mask) {
  uint64_t* buf = reinterpret_cast<uint64_t*>(guest_jmp_buf);

  // Clear the buffer in case the format has gaps.
  memset(buf, 0, kJmpBufChecksumWord * sizeof(uint64_t));

  // Cookie, signal flag, signal mask
  buf[kJmpBufSigFlagAndCookieWord] = kJmpBufCookie;
  if (save_sig_mask) {
    buf[kJmpBufSigFlagAndCookieWord] |= 0x1;
    RTSigprocmaskSyscallOrDie(
        SIG_SETMASK, nullptr, reinterpret_cast<HostSigset*>(buf + kJmpBufSigMaskWord));
  }

  // Match Android 16 bionic's layout and pointer mangling.  Keeping this
  // layout exact matters for code that inspects or copies jmp_buf itself.
  const uint64_t cookie = buf[kJmpBufSigFlagAndCookieWord] & ~0x1ULL;
  buf[kJmpBufLRWord] = state->cpu.x[30] ^ cookie;
  buf[kJmpBufSPWord] = state->cpu.sp ^ cookie;
  buf[kJmpBufX28Word] = state->cpu.x[28] ^ cookie;
  buf[kJmpBufX28Word + 1] = state->cpu.x[29] ^ cookie;
  buf[kJmpBufX26Word] = state->cpu.x[26] ^ cookie;
  buf[kJmpBufX26Word + 1] = state->cpu.x[27] ^ cookie;
  buf[kJmpBufX24Word] = state->cpu.x[24] ^ cookie;
  buf[kJmpBufX24Word + 1] = state->cpu.x[25] ^ cookie;
  buf[kJmpBufX22Word] = state->cpu.x[22] ^ cookie;
  buf[kJmpBufX22Word + 1] = state->cpu.x[23] ^ cookie;
  buf[kJmpBufX20Word] = state->cpu.x[20] ^ cookie;
  buf[kJmpBufX20Word + 1] = state->cpu.x[21] ^ cookie;
  buf[kJmpBufSCSWord] = (state->cpu.x[18] & kShadowCallStackMask) ^ cookie;
  buf[kJmpBufX19Word] = state->cpu.x[19] ^ cookie;

  // d14, d15, d12, d13, d10, d11, d8, d9.
  constexpr int kFloatingPointRegisterOrder[] = {14, 15, 12, 13, 10, 11, 8, 9};
  for (int i = 0; i < 8; ++i) {
    uint64_t low64;
    memcpy(&low64, &state->cpu.v[kFloatingPointRegisterOrder[i]], sizeof(low64));
    buf[kJmpBufFloatingPointBaseWord + i] = low64;
  }

  // Checksum
  buf[kJmpBufChecksumWord] = CalcJumpBufChecksum(buf);
}

void RestoreRegsFromJumpBuf(ThreadState* state, void* guest_jmp_buf, int retval) {
  const uint64_t* buf = reinterpret_cast<const uint64_t*>(guest_jmp_buf);

  // Checksum
  if (buf[kJmpBufChecksumWord] != CalcJumpBufChecksum(buf)) {
    LOG_ALWAYS_FATAL("setjmp checksum mismatch");
  }

  // Cookie
  if ((buf[kJmpBufSigFlagAndCookieWord] & ~0x1ULL) != kJmpBufCookie) {
    LOG_ALWAYS_FATAL("setjmp cookie mismatch");
  }

  // Signal mask
  if (buf[kJmpBufSigFlagAndCookieWord] & 0x1) {
    RTSigprocmaskSyscallOrDie(
        SIG_SETMASK, reinterpret_cast<const HostSigset*>(buf + kJmpBufSigMaskWord), nullptr);
  }

  const uint64_t cookie = buf[kJmpBufSigFlagAndCookieWord] & ~0x1ULL;
  state->cpu.x[30] = buf[kJmpBufLRWord] ^ cookie;
  state->cpu.sp = buf[kJmpBufSPWord] ^ cookie;
  state->cpu.x[28] = buf[kJmpBufX28Word] ^ cookie;
  state->cpu.x[29] = buf[kJmpBufX28Word + 1] ^ cookie;
  state->cpu.x[26] = buf[kJmpBufX26Word] ^ cookie;
  state->cpu.x[27] = buf[kJmpBufX26Word + 1] ^ cookie;
  state->cpu.x[24] = buf[kJmpBufX24Word] ^ cookie;
  state->cpu.x[25] = buf[kJmpBufX24Word + 1] ^ cookie;
  state->cpu.x[22] = buf[kJmpBufX22Word] ^ cookie;
  state->cpu.x[23] = buf[kJmpBufX22Word + 1] ^ cookie;
  state->cpu.x[20] = buf[kJmpBufX20Word] ^ cookie;
  state->cpu.x[21] = buf[kJmpBufX20Word + 1] ^ cookie;
  state->cpu.x[18] =
      (state->cpu.x[18] & ~kShadowCallStackMask) |
      ((buf[kJmpBufSCSWord] ^ cookie) & kShadowCallStackMask);
  state->cpu.x[19] = buf[kJmpBufX19Word] ^ cookie;

  // Restore d14, d15, d12, d13, d10, d11, d8, d9 (lower 64 bits, zero upper).
  constexpr int kFloatingPointRegisterOrder[] = {14, 15, 12, 13, 10, 11, 8, 9};
  for (int i = 0; i < 8; ++i) {
    __uint128_t val = buf[kJmpBufFloatingPointBaseWord + i];
    state->cpu.v[kFloatingPointRegisterOrder[i]] = val;
  }

  // Function return: set x0 = retval, pc = lr
  CPUState& cpu = state->cpu;
  SetInsnAddr(cpu, GetLinkRegister(cpu));
  SetReturnValueRegister(cpu, retval);
}

jmp_buf** GetHostJmpBufPtr(void* guest_jmp_buf) {
  uint64_t* buf = reinterpret_cast<uint64_t*>(guest_jmp_buf);
  return reinterpret_cast<jmp_buf**>(buf + kJmpBufHostBufWord);
}

}  // namespace berberis
