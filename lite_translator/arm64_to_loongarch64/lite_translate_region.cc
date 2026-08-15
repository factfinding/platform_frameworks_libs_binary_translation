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

#include "berberis/lite_translator/lite_translate_region.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "berberis/assembler/loongarch64.h"
#include "berberis/base/checks.h"
#include "berberis/base/config.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/runtime_library.h"
#include "berberis/runtime_primitives/translation_cache.h"

namespace berberis {
namespace {

using Assembler = loongarch64::Assembler;
using Register = loongarch64::Register;
using SimdRegister = loongarch64::SimdRegister;

constexpr int32_t kSpOffset = offsetof(ThreadState, cpu) + offsetof(CPUState, sp);
constexpr int32_t kFlagsOffset = offsetof(ThreadState, cpu) + offsetof(CPUState, flags);
constexpr int32_t kTlsOffset = offsetof(ThreadState, tls);
// Logical-immediate operations are enabled independently.  Validate each
// opcode in production workloads before adding it to this mask.
constexpr uint32_t kLogicalImmediateOpcMask = (1u << 0) | (1u << 1) | (1u << 3);  // AND, ORR, ANDS

constexpr int64_t SignExtend(uint64_t value, uint32_t width) {
  uint64_t sign = uint64_t{1} << (width - 1);
  uint64_t mask = (uint64_t{1} << width) - 1;
  return static_cast<int64_t>((value & mask) ^ sign) - static_cast<int64_t>(sign);
}

bool DecodeLogicalImmediate(uint32_t n,
                            uint32_t immr,
                            uint32_t imms,
                            uint32_t data_size,
                            uint64_t* result) {
  uint32_t len_source = (n << 6) | (~imms & 0x3f);
  int32_t len = -1;
  for (int32_t bit = 6; bit >= 0; --bit) {
    if ((len_source & (uint32_t{1} << bit)) != 0) {
      len = bit;
      break;
    }
  }
  if (len < 1) {
    return false;
  }
  uint32_t levels = (uint32_t{1} << len) - 1;
  uint32_t s = imms & levels;
  uint32_t r = immr & levels;
  if (s == levels) {
    return false;
  }

  uint32_t element_size = uint32_t{1} << len;
  if (element_size > data_size) {
    return false;
  }
  uint64_t element_mask = element_size == 64 ? UINT64_MAX : (uint64_t{1} << element_size) - 1;
  uint64_t element = (uint64_t{1} << (s + 1)) - 1;
  if (r != 0) {
    element = ((element >> r) | (element << (element_size - r))) & element_mask;
  }

  uint64_t value = 0;
  for (uint32_t offset = 0; offset < data_size; offset += element_size) {
    value |= element << offset;
  }
  *result = value;
  return true;
}

class LiteTranslator {
 public:
  explicit LiteTranslator(MachineCode* machine_code, bool enable_guest_memory, bool allow_dispatch)
      : as_(machine_code),
        enable_guest_memory_(enable_guest_memory),
        allow_dispatch_(allow_dispatch) {}

  bool Translate(uint32_t insn, GuestAddr pc) {
    if ((insn & 0x1f80'0000u) == 0x1280'0000u) {
      return TranslateMoveWide(insn);
    }
    if ((insn & 0x1f80'0000u) == 0x1100'0000u) {
      return TranslateAddSubImmediate(insn);
    }
    if ((insn & 0x1f20'0000u) == 0x0b00'0000u) {
      return TranslateAddSubShiftedRegister(insn);
    }
    if ((insn & 0x1fe0'0000u) == 0x0b20'0000u) {
      return TranslateAddSubExtendedRegister(insn);
    }
    if ((insn & 0x1f00'0000u) == 0x0a00'0000u) {
      return TranslateLogicalShiftedRegister(insn);
    }
    if ((insn & 0x1f80'0000u) == 0x1200'0000u &&
        (kLogicalImmediateOpcMask & (1u << ((insn >> 29) & 3))) != 0) {
      return TranslateLogicalImmediate(insn);
    }
    if ((insn & 0x1f80'0000u) == 0x1300'0000u) {
      return TranslateBitfieldExtract(insn);
    }
    if ((insn & 0x1f80'0000u) == 0x1380'0000u) {
      return TranslateExtract(insn);
    }
    if ((insn & 0x1f00'0000u) == 0x1000'0000u) {
      return TranslatePcRelativeAddress(insn, pc);
    }
    // Armv8.5 MTE tag loads/stores. Android code uses these instructions even
    // when the host has no MTE backing. Keep their data and writeback effects
    // in Lite JIT so a tag operation does not split an otherwise translatable
    // memory-heavy region.
    if ((insn & 0xff20'0000u) == 0xd920'0000u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateMteLoadStore(insn, pc);
    }
    if ((insn & 0x3b00'0000u) == 0x3900'0000u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateLoadStoreUnsignedImmediate(insn, pc);
    }
    if ((insn & 0x3b20'0000u) == 0x3800'0000u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateLoadStoreIndexed(insn, pc);
    }
    if ((insn & 0x3b20'0c00u) == 0x3820'0800u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateLoadStoreRegisterOffset(insn, pc);
    }
    if ((insn & 0x3e00'0000u) == 0x2800'0000u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateLoadStorePair(insn, pc);
    }
    if ((insn & 0x3e00'0000u) == 0x2c00'0000u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateSimdLoadStorePair(insn, pc);
    }
    // Armv8.1 LSE compare-and-swap.  A/R select acquire/release semantics;
    // byte through doubleword widths share the same LL/SC lowering.
    if ((insn & 0x3fa0'7c00u) == 0x08a0'7c00u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateAtomicMemory(insn, pc, AtomicMemoryOp::kCas);
    }
    // The current game profile is dominated by LDADD and SWP.  They differ
    // only in o3 (bit 15), so keep one audited LL/SC implementation for both.
    if ((insn & 0x3f20'7c00u) == 0x3820'0000u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateAtomicMemory(
          insn, pc, (insn & 0x8000u) ? AtomicMemoryOp::kSwp : AtomicMemoryOp::kLdadd);
    }
    // LDXR/LDAXR, including byte and halfword forms.  Preserve the guest
    // reservation in ThreadState so a following interpreted STXR can consume
    // it when the rest of the exclusive sequence is not yet translated.
    if ((insn & 0x3f7f'7c00u) == 0x085f'7c00u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateLoadExclusive(insn, pc);
    }
    // STLR[BH] uses Rs==31 in the exclusive/ordered load-store encoding.
    if ((insn & 0x3fff'fc00u) == 0x089f'fc00u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateStoreRelease(insn, pc);
    }
    // Rs==31 selects STLR rather than the store-exclusive family.  Leave it
    // to the interpreter until it has its own release-store lowering.
    if ((insn & 0x3f60'7c00u) == 0x0800'7c00u && ((insn >> 16) & 31) != 31) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateStoreExclusive(insn, pc);
    }
    // LD1/ST1 of one through four complete 4S vectors.  Unlike LD2/LD3/LD4,
    // these forms transfer consecutive vectors without interleaving, so all
    // list lengths share one lowering.  Accept both no-writeback and
    // immediate/register post-index encodings.
    if ((insn & 0xffbf'0c00u) == 0x4c00'0800u ||
        (insn & 0xffa0'0c00u) == 0x4c80'0800u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateLdSt1Multiple4S(insn, pc);
    }
    if ((insn & 0xbfff'fc00u) == 0x0ddf'8400u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateLd1D1PostIndex(insn, pc);
    }
    if ((insn & 0xffff'fc00u) == 0x4d40'c800u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateLd1r4S(insn, pc);
    }
    if ((insn & 0xffff'fc00u) == 0x4d00'8000u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateSt1S2(insn, pc);
    }
    if ((insn & 0xffe0'8400u) == 0x6e00'0400u) {
      return TranslateInsElement(insn);
    }
    if ((insn & 0xffe0'fc00u) == 0x4e00'0400u) {
      return TranslateDupElement4S(insn);
    }
    // DUP Vd.16B, Wn.  This is the hottest SIMD-copy form in the current
    // Unity workload.  Keep the initial LA64 implementation deliberately
    // narrow until the other lane widths have independent validation.
    if ((insn & 0xffff'fc00u) == 0x4e01'0c00u) {
      return TranslateDup16B(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x4e04'0400u) {
      return TranslateDup4SFromElement(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x4e08'0c00u) {
      return TranslateDup2D(insn);
    }
    if ((insn & 0xbfe0'8400u) == 0x2e00'0000u) {
      return TranslateExt(insn);
    }
    if ((insn & 0xbf3f'fc00u) == 0x0e20'0800u) {
      return TranslateVectorRev64(insn);
    }
    // TBL Vd.16B, {Vn.16B ... V(n+len).16B}, Vm.16B.  HEVC's NEON
    // deblocking filters use the four-register form in their innermost byte
    // permutation loop.  TBX and the 8-byte form remain in the interpreter.
    if ((insn & 0xffc0'9c00u) == 0x4e00'0000u) {
      return TranslateTbl16B(insn);
    }
    if ((insn & 0xffe0'fc00u) == 0x4e20'1c00u) {
      return TranslateVectorLogical(insn, 0);
    }
    if ((insn & 0xffe0'fc00u) == 0x4ea0'1c00u) {
      return TranslateVectorLogical(insn, 1);
    }
    if ((insn & 0xffe0'fc00u) == 0x6e20'1c00u) {
      return TranslateVectorLogical(insn, 2);
    }
    if ((insn & 0xffe0'fc00u) == 0x4ea0'8400u) {
      return TranslateVectorAddSub4S(insn, false);  // ADD
    }
    if ((insn & 0xffe0'fc00u) == 0x4ee0'8400u) {
      return TranslateVectorAdd2D(insn);
    }
    if ((insn & 0xffe0'fc00u) == 0x6ea0'8400u) {
      return TranslateVectorAddSub4S(insn, true);  // SUB
    }
    if ((insn & 0xffe0'fc00u) == 0x4ea0'9c00u) {
      return TranslateVectorMulAcc4S(insn, 0);  // MUL
    }
    if ((insn & 0xffe0'fc00u) == 0x4ea0'9400u) {
      return TranslateVectorMulAcc4S(insn, 1);  // MLA
    }
    if ((insn & 0xffe0'fc00u) == 0x6ea0'9400u) {
      return TranslateVectorMulAcc4S(insn, 2);  // MLS
    }
    // SSHLL/SSHLL2 Vd.4S, Vn.4H/8H, #shift.  These widening shifts are hot in
    // HEVC inverse transforms.  immh=001x selects the 16-to-32-bit form.
    if ((insn & 0xbf80'fc00u) == 0x0f00'a400u &&
        (((insn >> 19) & 0xeu) == 0x2u)) {
      return TranslateSshll4S(insn);
    }
    // SQRSHRN/SQRSHRN2 Vd.4H/8H, Vn.4S, #shift.  LSX performs the same
    // rounding signed-saturating narrow; the Q form additionally preserves
    // Vd's low 64 bits and inserts the narrowed lanes into the high half.
    if ((insn & 0xbf80'fc00u) == 0x0f00'9c00u &&
        (((insn >> 19) & 0xeu) == 0x2u)) {
      return TranslateSqrshrn4H(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x4ea0'f800u) {
      return TranslateFabs4S(insn);
    }
    if ((insn & 0xbfff'fc00u) == 0x0e20'5800u) {
      return TranslateCnt8B16B(insn);
    }
    // 32-bit lane permutations.  These are common in Unity math and texture
    // conversion code.  The scalar implementation preserves all source lanes
    // before writing Vd, so register aliases have the architectural behavior.
    if ((insn & 0xffe0'fc00u) == 0x4e80'1800u) {
      return TranslatePermute4S(insn, 0);  // UZP1
    }
    if ((insn & 0xffe0'fc00u) == 0x4e80'7800u) {
      return TranslatePermute4S(insn, 1);  // ZIP2
    }
    if ((insn & 0xffe0'fc00u) == 0x4e80'2800u) {
      return TranslatePermute4S(insn, 2);  // TRN1
    }
    if ((insn & 0xffe0'fc00u) == 0x4e80'5800u) {
      return TranslatePermute4S(insn, 3);  // UZP2
    }
    if ((insn & 0xffe0'fc00u) == 0x4e80'3800u) {
      return TranslatePermute4S(insn, 4);  // ZIP1
    }
    // MOVI Vd.2D, #0.  Compilers use this reserved modified-immediate form
    // as the canonical full-width vector clear.
    if ((insn & 0xffff'ffe0u) == 0x6f00'e400u) {
      return TranslateMovi2DZero(insn);
    }
    if ((insn & 0xffff'ffe0u) == 0x2f00'e400u) {
      return TranslateMovi2DZero(insn);
    }
    if ((insn & 0xffff'ffe0u) == 0x4f00'e420u) {
      return TranslateMovi16BOne(insn);
    }
    if ((insn & 0xffff'ffe0u) == 0x6f07'e7e0u) {
      return TranslateMoviConstant(insn, UINT64_MAX, UINT64_MAX);
    }
    if ((insn & 0xffff'ffe0u) == 0x6f00'e5e0u) {
      return TranslateMoviConstant(insn, UINT32_MAX, UINT32_MAX);
    }
    if ((insn & 0xffff'ffe0u) == 0x2f00'e5e0u) {
      return TranslateMoviConstant(insn, UINT32_MAX, 0);
    }
    if ((insn & 0xffff'ffe0u) == 0x0f00'0420u) {
      return TranslateMoviConstant(insn, 0x0000'0001'0000'0001ULL, 0);
    }
    // Bit 23 distinguishes FMLA (0) from FMLS (1).
    if ((insn & 0xffa0'fc00u) == 0x4e20'cc00u ||
        (insn & 0xffa0'fc00u) == 0x4ea0'cc00u) {
      return TranslateFmlaFmls4S(insn);
    }
    // Scalar FP arithmetic is pervasive in Unity startup code.  Use scalar
    // LA64 operations so inactive SIMD lanes cannot raise spurious FP flags.
    if ((insn & 0xff20'0c00u) == 0x1e20'0800u) {
      return TranslateScalarFpBinary(insn);
    }
    if ((insn & 0xff20'8000u) == 0x1f00'0000u) {
      return TranslateScalarFmadd(insn);
    }
    if ((insn & 0xff20'0c00u) == 0x1e20'0c00u) {
      return TranslateScalarFcsel(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x1e22'c000u) {
      return TranslateFcvtDS(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x1e39'0000u) {
      return TranslateFcvtzuWS(insn);
    }
    if ((insn & 0xffe0'fc07u) == 0x1e20'2000u) {
      return TranslateFcmpS(insn);
    }
    if ((insn & 0xffe0'1fe0u) == 0x1e20'1000u) {
      return TranslateFmovSImmediate(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x1e38'0000u) {
      return TranslateFcvtzsWS(insn);
    }
    // FADDP shares the broad FADD opcode pattern, so recognize its 2S form
    // before the generic vector floating-point matcher.
    if ((insn & 0xffe0'fc00u) == 0x2e20'd400u) {
      return TranslateFaddp2S(insn);
    }
    // AdvSIMD floating-point add/sub and mul/div.  Q and size select 2S/4S/2D;
    // half-precision encodings remain in the interpreter.
    if ((insn & 0xbfa0'fc00u) == 0x0e20'd400u) {
      return TranslateVectorFpBinary(insn, 2);  // FADD
    }
    if ((insn & 0xbfa0'fc00u) == 0x0ea0'd400u) {
      return TranslateVectorFpBinary(insn, 3);  // FSUB
    }
    if ((insn & 0xffff'fc00u) == 0x4ea0'd800u) {
      return TranslateFcmeq4SZero(insn);
    }
    if ((insn & 0xffe0'fc00u) == 0x6e20'e400u) {
      return TranslateFcmge4S(insn);
    }
    if ((insn & 0xffe0'fc00u) == 0x6e20'8c00u) {
      return TranslateCmeq16B(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x4e21'd800u) {
      return TranslateScvtf4S(insn);
    }
    if ((insn & 0xbf20'fc00u) == 0x2e20'dc00u) {
      return TranslateVectorFpBinary(insn, 0);  // FMUL
    }
    if ((insn & 0xbf20'fc00u) == 0x2e20'fc00u) {
      return TranslateVectorFpBinary(insn, 1);  // FDIV
    }
    if ((insn & 0xffff'fc00u) == 0x6e21'd800u) {
      return TranslateUcvtf4S(insn);
    }
    if ((insn & 0xffc0'f400u) == 0x4f80'1000u) {
      return TranslateFmla4SByElement(insn);
    }
    if ((insn & 0xffc0'f400u) == 0x4f80'9000u) {
      return TranslateFmul4SByElement(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x1e20'4000u) {
      return TranslateFmovS(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x1e60'4000u) {
      return TranslateFmovD(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x1e27'0000u) {
      return TranslateFmovSFromW(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x9e67'0000u) {
      return TranslateFmovDFromX(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x1e26'0000u) {
      return TranslateFmovWFromS(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x9e66'0000u) {
      return TranslateFmovXFromD(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x4e04'0c00u) {
      return TranslateDup4S(insn);
    }
    if ((insn & 0x1fe0'0800u) == 0x1a80'0000u) {
      return TranslateConditionalSelect(insn);
    }
    if ((insn & 0x7fe0'fc00u) == 0x1a00'0000u) {
      return TranslateAdc(insn);
    }
    if ((insn & 0x3fe0'0410u) == 0x3a40'0000u) {
      return TranslateConditionalCompare(insn);
    }
    if ((insn & 0x7fe0'0000u) == 0x1b00'0000u) {
      return TranslateMultiplyAddSub(insn);
    }
    if ((insn & 0xff60'0000u) == 0x9b20'0000u) {
      return TranslateMultiplyAddSubLong(insn);
    }
    if ((insn & 0xffe0'fc00u) == 0x9bc0'7c00u) {
      return TranslateUmulh(insn);
    }
    if ((insn & 0xffe0'fc00u) == 0x9b40'7c00u) {
      return TranslateSmulh(insn);
    }
    if ((insn & 0xffe0'fc00u) == 0x9ba0'7c00u) {
      return TranslateUmull(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x5ac0'1000u || (insn & 0xffff'fc00u) == 0xdac0'1000u) {
      return TranslateClz(insn);
    }
    if ((insn & 0xffff'fc00u) == 0x5ac0'0800u) {
      return TranslateRev32(insn);
    }
    if ((insn & 0xffff'fc00u) == 0xdac0'0c00u) {
      return TranslateRev64(insn);
    }
    if ((insn & 0x7fe0'fc00u) == 0x1ac0'2000u || (insn & 0x7fe0'fc00u) == 0x1ac0'2400u ||
        (insn & 0x7fe0'fc00u) == 0x1ac0'2800u || (insn & 0x7fe0'fc00u) == 0x1ac0'2c00u ||
        (insn & 0x7fe0'fc00u) == 0x1ac0'0800u || (insn & 0x7fe0'fc00u) == 0x1ac0'0c00u) {
      return TranslateDataProcessingTwoSource(insn);
    }
    if ((insn & 0x7c00'0000u) == 0x1400'0000u) {
      TranslateBranchImmediate(insn, pc);
      region_end_reached_ = true;
      return true;
    }
    if ((insn & 0x7e00'0000u) == 0x3400'0000u) {
      TranslateCompareAndBranch(insn, pc);
      region_end_reached_ = true;
      return true;
    }
    if ((insn & 0x7e00'0000u) == 0x3600'0000u) {
      TranslateTestAndBranch(insn, pc);
      region_end_reached_ = true;
      return true;
    }
    if ((insn & 0xff00'0010u) == 0x5400'0000u) {
      TranslateConditionalBranch(insn, pc);
      region_end_reached_ = true;
      return true;
    }
    if ((insn & 0xffff'fc1fu) == 0xd61f'0000u || (insn & 0xffff'fc1fu) == 0xd63f'0000u ||
        (insn & 0xffff'fc1fu) == 0xd65f'0000u) {
      TranslateBranchRegister(insn, pc);
      region_end_reached_ = true;
      return true;
    }
    // MRS Xt, TPIDR_EL0 reads the guest thread pointer, which Berberis keeps
    // explicitly in ThreadState instead of the host's architectural TP.
    if ((insn & 0xffff'ffe0u) == 0xd53b'd040u) {
      uint32_t rt = insn & 31;
      as_.LdD(Assembler::t0, Assembler::s8, kTlsOffset);
      StoreXOrDiscard(rt, Assembler::t0);
      return true;
    }
    // ARM barriers become LoongArch DBAR.  DMB options only narrow the shareability
    // domain; DBAR 0 is the conservative full-system ordering required here.
    if ((insn & 0xffff'f0ffu) == 0xd503'30bfu) {
      as_.Dbar(0);
      return true;
    }
    // Berberis models the guest exclusive monitor in ThreadState.  Generated
    // LDXR does not leave a host LL reservation live across guest instructions.
    if ((insn & 0xffff'f0ffu) == 0xd503'305fu) {
      as_.StD(Assembler::zero,
              Assembler::s8,
              offsetof(ThreadState, cpu) + offsetof(CPUState, reservation_address));
      return true;
    }
    // AArch64 HINT instructions are architectural no-ops for binary
    // translation.  This includes NOP, YIELD/WFE/WFI, PAC/AUT hints and BTI.
    if ((insn & 0xffff'f01fu) == 0xd503'201fu) {
      return true;
    }
    return false;
  }

  void Exit(GuestAddr pc) {
    as_.Li(Assembler::s7, pc);
    if (allow_dispatch_ && config::kLinkJumpsBetweenRegions) {
      Assembler::Label* pending_signal = as_.MakeLabel();
      as_.LdBU(Assembler::t0,
               Assembler::s8,
               static_cast<int32_t>(offsetof(ThreadState, pending_signals_status)));
      as_.AddiD(Assembler::t1, Assembler::zero, kPendingSignalsPresent);
      as_.Beq(Assembler::t0, Assembler::t1, *pending_signal);

      as_.Li(Assembler::t0,
             reinterpret_cast<uint64_t>(TranslationCache::GetInstance()->GetHostCodePtr(pc)));
      as_.LdD(Assembler::t0, Assembler::t0, 0);
      as_.Jirl(Assembler::zero, Assembler::t0, 0);

      as_.Bind(pending_signal);
    }
    ExitGeneratedCode();
  }

  void ExitGeneratedCode(GuestAddr pc) {
    as_.Li(Assembler::s7, pc);
    ExitGeneratedCode();
  }

  void ExitGeneratedCode() {
    as_.Li(Assembler::t0, kEntryExitGeneratedCode);
    as_.Jirl(Assembler::zero, Assembler::t0, 0);
  }

  void ExitIndirect(Register target) {
    if (target != Assembler::s7) {
      as_.Move(Assembler::s7, target);
    }
    if (!allow_dispatch_ || !config::kLinkJumpsBetweenRegions) {
      ExitGeneratedCode();
      return;
    }

    Assembler::Label* pending_signal = as_.MakeLabel();
    as_.LdBU(Assembler::t0,
             Assembler::s8,
             static_cast<int32_t>(offsetof(ThreadState, pending_signals_status)));
    as_.AddiD(Assembler::t1, Assembler::zero, kPendingSignalsPresent);
    as_.Beq(Assembler::t0, Assembler::t1, *pending_signal);

    // TranslationCache is a two-level table indexed by the low 48 bits of
    // the guest PC: bits 47:24 select a child table and bits 23:0 select its
    // HostCodeAddr slot.  Masking both halves matches TableOfTables::SplitKey
    // and also ignores PAC/TBI bits carried by an indirect branch target.
    constexpr uint64_t kTableIndexMask = (uint64_t{1} << 24) - 1;
    as_.Li(Assembler::t1, kTableIndexMask);
    as_.SrliD(Assembler::t2, Assembler::s7, 24);
    as_.And(Assembler::t2, Assembler::t2, Assembler::t1);
    as_.SlliD(Assembler::t2, Assembler::t2, 3);
    as_.Li(Assembler::t3,
           reinterpret_cast<uint64_t>(TranslationCache::GetInstance()->main_table_ptr()));
    as_.AddD(Assembler::t2, Assembler::t2, Assembler::t3);
    as_.LdD(Assembler::t2, Assembler::t2, 0);

    as_.And(Assembler::t1, Assembler::s7, Assembler::t1);
    as_.SlliD(Assembler::t1, Assembler::t1, 3);
    as_.AddD(Assembler::t1, Assembler::t1, Assembler::t2);
    as_.LdD(Assembler::t0, Assembler::t1, 0);
    as_.Jirl(Assembler::zero, Assembler::t0, 0);

    as_.Bind(pending_signal);
    ExitGeneratedCode();
  }

  void Finalize() { as_.Finalize(); }
  bool region_end_reached() const { return region_end_reached_; }

 private:
  enum class AtomicMemoryOp { kCas, kSwp, kLdadd };

  static constexpr int32_t XOffset(uint32_t reg) {
    return offsetof(ThreadState, cpu) + offsetof(CPUState, x) + reg * sizeof(uint64_t);
  }

  static constexpr int32_t VOffset(uint32_t reg) {
    return offsetof(ThreadState, cpu) + offsetof(CPUState, v) + reg * sizeof(__uint128_t);
  }

  void LoadV(uint32_t reg, SimdRegister dst) { as_.Vld(dst, Assembler::s8, VOffset(reg)); }

  void StoreV(uint32_t reg, SimdRegister src) { as_.Vst(src, Assembler::s8, VOffset(reg)); }

  void LoadXOrZero(uint32_t reg, Register dst) {
    if (reg == 31) {
      as_.Move(dst, Assembler::zero);
    } else {
      as_.LdD(dst, Assembler::s8, XOffset(reg));
    }
  }

  void StoreXOrDiscard(uint32_t reg, Register src) {
    if (reg != 31) {
      as_.StD(src, Assembler::s8, XOffset(reg));
    }
  }

  void LoadXOrSp(uint32_t reg, Register dst) {
    as_.LdD(dst, Assembler::s8, reg == 31 ? kSpOffset : XOffset(reg));
  }

  void StoreXOrSp(uint32_t reg, Register src) {
    as_.StD(src, Assembler::s8, reg == 31 ? kSpOffset : XOffset(reg));
  }

  void ZeroExtend32(Register reg) {
    as_.SlliD(reg, reg, 32);
    as_.SrliD(reg, reg, 32);
  }

  void SignExtend32(Register reg) {
    as_.SlliD(reg, reg, 32);
    as_.SraiD(reg, reg, 32);
  }

  // Android enables top-byte-ignore for userspace pointers.  Guest pointers
  // may therefore carry an allocation tag in bits 63:56, while the host
  // address used by generated memory instructions must not.  Keep the tagged
  // value in guest registers (including writeback results) and strip it only
  // from the temporary effective address used for the actual access.
  void ApplyTbi(Register reg) {
    as_.SlliD(reg, reg, 8);
    as_.SrliD(reg, reg, 8);
  }

  bool ShiftOperand(Register reg, uint32_t shift_kind, uint32_t amount, bool is_64_bit) {
    if (shift_kind == 3 || (!is_64_bit && amount >= 32)) {
      return false;
    }
    if (!is_64_bit) {
      if (shift_kind == 2) {
        SignExtend32(reg);
      } else {
        ZeroExtend32(reg);
      }
    }
    switch (shift_kind) {
      case 0:
        as_.SlliD(reg, reg, amount);
        break;
      case 1:
        as_.SrliD(reg, reg, amount);
        break;
      case 2:
        as_.SraiD(reg, reg, amount);
        break;
    }
    return true;
  }

  bool TranslateMoveWide(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    uint32_t opc = (insn >> 29) & 3;
    uint32_t hw = (insn >> 21) & 3;
    uint32_t rd = insn & 31;
    uint64_t shift = hw * 16;
    uint64_t imm = (insn >> 5) & 0xffff;
    if (opc == 1 || (!is_64_bit && hw >= 2)) {
      return false;
    }

    uint64_t width_mask = is_64_bit ? UINT64_MAX : UINT32_MAX;
    if (opc == 3) {  // MOVK
      LoadXOrZero(rd, Assembler::t0);
      uint64_t keep_mask = ~(uint64_t{0xffff} << shift) & width_mask;
      as_.Li(Assembler::t1, keep_mask);
      as_.And(Assembler::t0, Assembler::t0, Assembler::t1);
      as_.Li(Assembler::t1, imm << shift);
      as_.Or(Assembler::t0, Assembler::t0, Assembler::t1);
    } else {
      uint64_t value = imm << shift;
      if (opc == 0) {  // MOVN
        value = ~value;
      }
      as_.Li(Assembler::t0, value & width_mask);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateAddSubImmediate(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    bool is_sub = ((insn >> 30) & 1) != 0;
    bool set_flags = ((insn >> 29) & 1) != 0;
    uint32_t shift = (insn >> 22) & 1;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    uint64_t imm = static_cast<uint64_t>((insn >> 10) & 0xfff) << (shift * 12);
    LoadXOrSp(rn, Assembler::t0);
    if (set_flags) {
      as_.Move(Assembler::t4, Assembler::t0);
    }
    as_.Li(Assembler::t1, imm);
    if (is_sub) {
      as_.SubD(Assembler::t0, Assembler::t0, Assembler::t1);
    } else {
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
      if (set_flags) {
        ZeroExtend32(Assembler::t4);
        ZeroExtend32(Assembler::t1);
      }
    }
    if (set_flags) {
      if (is_sub) {
        ComputeSubFlags(Assembler::t4, Assembler::t1, Assembler::t0, is_64_bit ? 64 : 32);
      } else {
        ComputeAddFlags(Assembler::t4, Assembler::t1, Assembler::t0, is_64_bit ? 64 : 32);
      }
    }
    if (set_flags) {
      StoreXOrDiscard(rd, Assembler::t0);
    } else {
      StoreXOrSp(rd, Assembler::t0);
    }
    return true;
  }

  void ComputeSubFlags(Register lhs, Register rhs, Register result, uint32_t width) {
    as_.Move(Assembler::t3, Assembler::zero);

    // N is the sign bit of the result.
    as_.SrliD(Assembler::t5, result, width - 1);
    as_.SlliD(Assembler::t5, Assembler::t5, 3);
    as_.Or(Assembler::t3, Assembler::t3, Assembler::t5);

    // V = sign((lhs ^ rhs) & (lhs ^ result)) for subtraction.
    as_.Xor(Assembler::t5, lhs, rhs);
    as_.Xor(Assembler::t6, lhs, result);
    as_.And(Assembler::t5, Assembler::t5, Assembler::t6);
    as_.SrliD(Assembler::t5, Assembler::t5, width - 1);
    as_.Or(Assembler::t3, Assembler::t3, Assembler::t5);

    Assembler::Label* no_carry = as_.MakeLabel();
    as_.Bltu(lhs, rhs, *no_carry);
    as_.AddiD(Assembler::t3, Assembler::t3, 2);
    as_.Bind(no_carry);

    Assembler::Label* nonzero = as_.MakeLabel();
    as_.Bnez(result, *nonzero);
    as_.AddiD(Assembler::t3, Assembler::t3, 4);
    as_.Bind(nonzero);
    as_.StW(Assembler::t3, Assembler::s8, kFlagsOffset);
  }

  void ComputeAddFlags(Register lhs, Register rhs, Register result, uint32_t width) {
    as_.Move(Assembler::t3, Assembler::zero);
    as_.SrliD(Assembler::t5, result, width - 1);
    as_.SlliD(Assembler::t5, Assembler::t5, 3);
    as_.Or(Assembler::t3, Assembler::t3, Assembler::t5);

    // V = sign(~(lhs ^ rhs) & (lhs ^ result)) for addition.
    as_.Xor(Assembler::t5, lhs, rhs);
    as_.Li(Assembler::t7, UINT64_MAX);
    as_.Xor(Assembler::t5, Assembler::t5, Assembler::t7);
    as_.Xor(Assembler::t6, lhs, result);
    as_.And(Assembler::t5, Assembler::t5, Assembler::t6);
    as_.SrliD(Assembler::t5, Assembler::t5, width - 1);
    as_.Or(Assembler::t3, Assembler::t3, Assembler::t5);

    // Carry is an unsigned wrap: result < lhs.
    Assembler::Label* carry = as_.MakeLabel();
    Assembler::Label* carry_done = as_.MakeLabel();
    as_.Bltu(result, lhs, *carry);
    as_.B(*carry_done);
    as_.Bind(carry);
    as_.AddiD(Assembler::t3, Assembler::t3, 2);
    as_.Bind(carry_done);

    Assembler::Label* nonzero = as_.MakeLabel();
    as_.Bnez(result, *nonzero);
    as_.AddiD(Assembler::t3, Assembler::t3, 4);
    as_.Bind(nonzero);
    as_.StW(Assembler::t3, Assembler::s8, kFlagsOffset);
  }

  bool TranslateAddSubShiftedRegister(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    bool is_sub = ((insn >> 30) & 1) != 0;
    bool set_flags = ((insn >> 29) & 1) != 0;
    uint32_t shift_kind = (insn >> 22) & 3;
    uint32_t rm = (insn >> 16) & 31;
    uint32_t amount = (insn >> 10) & 0x3f;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    if (!ShiftOperand(Assembler::t1, shift_kind, amount, is_64_bit)) {
      return false;
    }
    if (set_flags) {
      as_.Move(Assembler::t4, Assembler::t0);
      if (!is_64_bit) {
        ZeroExtend32(Assembler::t4);
      }
    }
    if (is_sub) {
      as_.SubD(Assembler::t0, Assembler::t0, Assembler::t1);
    } else {
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    if (set_flags) {
      if (is_sub) {
        ComputeSubFlags(Assembler::t4, Assembler::t1, Assembler::t0, is_64_bit ? 64 : 32);
      } else {
        ComputeAddFlags(Assembler::t4, Assembler::t1, Assembler::t0, is_64_bit ? 64 : 32);
      }
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateAddSubExtendedRegister(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    bool is_sub = ((insn >> 30) & 1) != 0;
    bool set_flags = ((insn >> 29) & 1) != 0;
    uint32_t rm = (insn >> 16) & 31;
    uint32_t option = (insn >> 13) & 7;
    uint32_t amount = (insn >> 10) & 7;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    if (amount > 4) {
      return false;
    }

    LoadXOrSp(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    uint32_t source_width = 8u << (option & 3);
    if (source_width < 64) {
      uint32_t shift = 64 - source_width;
      as_.SlliD(Assembler::t1, Assembler::t1, shift);
      if ((option & 4) != 0) {
        as_.SraiD(Assembler::t1, Assembler::t1, shift);
      } else {
        as_.SrliD(Assembler::t1, Assembler::t1, shift);
      }
    }
    if (amount != 0) {
      as_.SlliD(Assembler::t1, Assembler::t1, amount);
    }
    if (set_flags) {
      as_.Move(Assembler::t4, Assembler::t0);
      if (!is_64_bit) {
        ZeroExtend32(Assembler::t4);
      }
    }
    if (is_sub) {
      as_.SubD(Assembler::t0, Assembler::t0, Assembler::t1);
    } else {
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
      ZeroExtend32(Assembler::t1);
    }
    if (set_flags) {
      if (is_sub) {
        ComputeSubFlags(Assembler::t4, Assembler::t1, Assembler::t0, is_64_bit ? 64 : 32);
      } else {
        ComputeAddFlags(Assembler::t4, Assembler::t1, Assembler::t0, is_64_bit ? 64 : 32);
      }
    }
    if (set_flags) {
      StoreXOrDiscard(rd, Assembler::t0);
    } else {
      StoreXOrSp(rd, Assembler::t0);
    }
    return true;
  }

  bool TranslateLogicalShiftedRegister(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    uint32_t opc = (insn >> 29) & 3;
    uint32_t shift_kind = (insn >> 22) & 3;
    bool invert = ((insn >> 21) & 1) != 0;
    uint32_t rm = (insn >> 16) & 31;
    uint32_t amount = (insn >> 10) & 0x3f;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    if (!ShiftOperand(Assembler::t1, shift_kind, amount, is_64_bit)) {
      return false;
    }
    if (invert) {
      as_.Li(Assembler::t2, UINT64_MAX);
      as_.Xor(Assembler::t1, Assembler::t1, Assembler::t2);
    }
    switch (opc) {
      case 0:
      case 3:
        as_.And(Assembler::t0, Assembler::t0, Assembler::t1);
        break;
      case 1:
        as_.Or(Assembler::t0, Assembler::t0, Assembler::t1);
        break;
      case 2:
        as_.Xor(Assembler::t0, Assembler::t0, Assembler::t1);
        break;
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    if (opc == 3) {
      ComputeLogicalFlags(Assembler::t0, is_64_bit ? 64 : 32);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateLogicalImmediate(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    uint32_t opc = (insn >> 29) & 3;
    uint32_t n = (insn >> 22) & 1;
    uint32_t immr = (insn >> 16) & 0x3f;
    uint32_t imms = (insn >> 10) & 0x3f;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    uint64_t immediate;
    if ((!is_64_bit && n != 0) ||
        !DecodeLogicalImmediate(n, immr, imms, is_64_bit ? 64 : 32, &immediate)) {
      return false;
    }

    LoadXOrZero(rn, Assembler::t0);
    as_.Li(Assembler::t1, immediate);
    switch (opc) {
      case 0:
      case 3:
        as_.And(Assembler::t0, Assembler::t0, Assembler::t1);
        break;
      case 1:
        as_.Or(Assembler::t0, Assembler::t0, Assembler::t1);
        break;
      case 2:
        as_.Xor(Assembler::t0, Assembler::t0, Assembler::t1);
        break;
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    if (opc == 3) {
      ComputeLogicalFlags(Assembler::t0, is_64_bit ? 64 : 32);
    }
    // Unlike logical shifted-register instructions, non-flag-setting logical
    // immediate instructions use register 31 as SP/WSP for the destination.
    // ANDS still uses XZR/WZR and discards the result.
    if (opc == 3) {
      StoreXOrDiscard(rd, Assembler::t0);
    } else {
      StoreXOrSp(rd, Assembler::t0);
    }
    return true;
  }

  bool TranslateBitfieldExtract(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    uint32_t opc = (insn >> 29) & 3;
    bool n = ((insn >> 22) & 1) != 0;
    uint32_t immr = (insn >> 16) & 0x3f;
    uint32_t imms = (insn >> 10) & 0x3f;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    uint32_t data_size = is_64_bit ? 64 : 32;
    if (opc > 2 || n != is_64_bit || immr >= data_size || imms >= data_size) {
      return false;
    }

    if (opc == 1) {  // BFM / BFI / BFXIL
      uint32_t field_width;
      uint32_t destination_lsb;
      if (immr > imms) {
        field_width = imms + 1;
        destination_lsb = data_size - immr;
      } else {
        field_width = imms - immr + 1;
        destination_lsb = 0;
      }
      const uint64_t field_mask = field_width == 64 ? UINT64_MAX : (uint64_t{1} << field_width) - 1;
      const uint64_t destination_mask = field_mask << destination_lsb;

      LoadXOrZero(rd, Assembler::t0);
      LoadXOrZero(rn, Assembler::t1);
      if (immr > imms) {
        as_.Li(Assembler::t2, field_mask);
        as_.And(Assembler::t1, Assembler::t1, Assembler::t2);
        as_.SlliD(Assembler::t1, Assembler::t1, destination_lsb);
      } else {
        if (immr != 0) {
          as_.SrliD(Assembler::t1, Assembler::t1, immr);
        }
        as_.Li(Assembler::t2, field_mask);
        as_.And(Assembler::t1, Assembler::t1, Assembler::t2);
      }
      as_.Li(Assembler::t2, ~destination_mask);
      as_.And(Assembler::t0, Assembler::t0, Assembler::t2);
      as_.Or(Assembler::t0, Assembler::t0, Assembler::t1);
      if (!is_64_bit) {
        ZeroExtend32(Assembler::t0);
      }
      StoreXOrDiscard(rd, Assembler::t0);
      return true;
    }

    if (immr > imms) {
      // SBFIZ/UBFIZ are the wrapping SBFM/UBFM forms.  Extract the low
      // imms+1 source bits, extend them according to the opcode, then insert
      // at data_size-immr.  LSL is simply the UBFIZ case where those two
      // widths consume the full destination.
      uint32_t field_width = imms + 1;
      uint32_t destination_lsb = data_size - immr;
      uint64_t mask = field_width == 64 ? UINT64_MAX : (uint64_t{1} << field_width) - 1;
      LoadXOrZero(rn, Assembler::t0);
      as_.Li(Assembler::t1, mask);
      as_.And(Assembler::t0, Assembler::t0, Assembler::t1);
      if (opc == 0) {
        as_.SlliD(Assembler::t0, Assembler::t0, 64 - field_width);
        as_.SraiD(Assembler::t0, Assembler::t0, 64 - field_width);
      }
      as_.SlliD(Assembler::t0, Assembler::t0, destination_lsb);
      if (!is_64_bit) {
        ZeroExtend32(Assembler::t0);
      }
      StoreXOrDiscard(rd, Assembler::t0);
      return true;
    }

    uint32_t field_width = imms - immr + 1;
    LoadXOrZero(rn, Assembler::t0);
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    if (immr != 0) {
      as_.SrliD(Assembler::t0, Assembler::t0, immr);
    }

    if (opc == 0) {  // SBFM / SBFX / SXT[BHW]
      as_.SlliD(Assembler::t0, Assembler::t0, 64 - field_width);
      as_.SraiD(Assembler::t0, Assembler::t0, 64 - field_width);
    } else {  // UBFM / UBFX / UXT[BH]
      uint64_t mask = field_width == 64 ? UINT64_MAX : (uint64_t{1} << field_width) - 1;
      as_.Li(Assembler::t1, mask);
      as_.And(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateExtract(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    bool n = ((insn >> 22) & 1) != 0;
    uint32_t rm = (insn >> 16) & 31;
    uint32_t lsb = (insn >> 10) & 0x3f;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    uint32_t width = is_64_bit ? 64 : 32;
    if (n != is_64_bit || lsb >= width) {
      return false;
    }

    LoadXOrZero(rm, Assembler::t0);
    if (rn == rm && lsb != 0) {
      as_.Li(Assembler::t1, lsb);
      if (is_64_bit) {
        as_.RotrD(Assembler::t0, Assembler::t0, Assembler::t1);
      } else {
        as_.RotrW(Assembler::t0, Assembler::t0, Assembler::t1);
      }
    } else if (lsb != 0) {
      LoadXOrZero(rn, Assembler::t1);
      if (is_64_bit) {
        as_.SrliD(Assembler::t0, Assembler::t0, lsb);
        as_.SlliD(Assembler::t1, Assembler::t1, width - lsb);
      } else {
        ZeroExtend32(Assembler::t0);
        ZeroExtend32(Assembler::t1);
        as_.SrliD(Assembler::t0, Assembler::t0, lsb);
        as_.SlliD(Assembler::t1, Assembler::t1, width - lsb);
      }
      as_.Or(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateDataProcessingTwoSource(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    uint32_t opcode = insn & 0x7fe0'fc00u;
    uint32_t rm = (insn >> 16) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;

    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    switch (opcode) {
      case 0x1ac0'2000u:  // LSLV
        if (is_64_bit) {
          as_.SllD(Assembler::t0, Assembler::t0, Assembler::t1);
        } else {
          as_.SllW(Assembler::t0, Assembler::t0, Assembler::t1);
        }
        break;
      case 0x1ac0'2400u:  // LSRV
        if (is_64_bit) {
          as_.SrlD(Assembler::t0, Assembler::t0, Assembler::t1);
        } else {
          as_.SrlW(Assembler::t0, Assembler::t0, Assembler::t1);
        }
        break;
      case 0x1ac0'2800u:  // ASRV
        if (is_64_bit) {
          as_.SraD(Assembler::t0, Assembler::t0, Assembler::t1);
        } else {
          as_.SraW(Assembler::t0, Assembler::t0, Assembler::t1);
        }
        break;
      case 0x1ac0'2c00u:  // RORV
        if (is_64_bit) {
          as_.RotrD(Assembler::t0, Assembler::t0, Assembler::t1);
        } else {
          as_.RotrW(Assembler::t0, Assembler::t0, Assembler::t1);
        }
        break;
      case 0x1ac0'0800u: {  // UDIV
        Assembler::Label* divisor_zero = as_.MakeLabel();
        Assembler::Label* done = as_.MakeLabel();
        if (!is_64_bit) {
          ZeroExtend32(Assembler::t0);
          ZeroExtend32(Assembler::t1);
        }
        as_.Beqz(Assembler::t1, *divisor_zero);
        as_.DivDU(Assembler::t0, Assembler::t0, Assembler::t1);
        as_.B(*done);
        as_.Bind(divisor_zero);
        as_.Move(Assembler::t0, Assembler::zero);
        as_.Bind(done);
        break;
      }
      case 0x1ac0'0c00u: {  // SDIV
        Assembler::Label* divisor_zero = as_.MakeLabel();
        Assembler::Label* do_divide = as_.MakeLabel();
        Assembler::Label* done = as_.MakeLabel();
        if (!is_64_bit) {
          SignExtend32(Assembler::t0);
          SignExtend32(Assembler::t1);
        }
        as_.Beqz(Assembler::t1, *divisor_zero);
        if (is_64_bit) {
          as_.Li(Assembler::t2, uint64_t{1} << 63);
          as_.Bne(Assembler::t0, Assembler::t2, *do_divide);
          as_.Li(Assembler::t2, UINT64_MAX);
          as_.Beq(Assembler::t1, Assembler::t2, *done);
          as_.Bind(do_divide);
        }
        as_.DivD(Assembler::t0, Assembler::t0, Assembler::t1);
        as_.B(*done);
        as_.Bind(divisor_zero);
        as_.Move(Assembler::t0, Assembler::zero);
        as_.Bind(done);
        break;
      }
      default:
        return false;
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateMultiplyAddSub(uint32_t insn) {
    const bool is_64_bit = (insn >> 31) != 0;
    const bool subtract = ((insn >> 15) & 1) != 0;
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t ra = (insn >> 10) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;

    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    as_.MulD(Assembler::t0, Assembler::t0, Assembler::t1);
    LoadXOrZero(ra, Assembler::t2);
    if (subtract) {
      as_.SubD(Assembler::t0, Assembler::t2, Assembler::t0);
    } else {
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t2);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateMultiplyAddSubLong(uint32_t insn) {
    const bool is_unsigned = ((insn >> 23) & 1) != 0;
    const bool subtract = ((insn >> 15) & 1) != 0;
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t ra = (insn >> 10) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;

    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    if (is_unsigned) {
      ZeroExtend32(Assembler::t0);
      ZeroExtend32(Assembler::t1);
    } else {
      SignExtend32(Assembler::t0);
      SignExtend32(Assembler::t1);
    }
    as_.MulD(Assembler::t0, Assembler::t0, Assembler::t1);
    LoadXOrZero(ra, Assembler::t2);
    if (subtract) {
      as_.SubD(Assembler::t0, Assembler::t2, Assembler::t0);
    } else {
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t2);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateUmulh(uint32_t insn) {
    uint32_t rm = (insn >> 16) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    as_.MulhDU(Assembler::t0, Assembler::t0, Assembler::t1);
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateSmulh(uint32_t insn) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    as_.MulhD(Assembler::t0, Assembler::t0, Assembler::t1);
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateAdc(uint32_t insn) {
    const bool is_64_bit = (insn >> 31) != 0;
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
      ZeroExtend32(Assembler::t1);
    }
    as_.LdWU(Assembler::t2, Assembler::s8, kFlagsOffset);
    as_.SrliD(Assembler::t2, Assembler::t2, 1);
    as_.AddiD(Assembler::t3, Assembler::zero, 1);
    as_.And(Assembler::t2, Assembler::t2, Assembler::t3);
    as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    as_.AddD(Assembler::t0, Assembler::t0, Assembler::t2);
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateUmull(uint32_t insn) {
    uint32_t rm = (insn >> 16) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    ZeroExtend32(Assembler::t0);
    ZeroExtend32(Assembler::t1);
    as_.MulD(Assembler::t0, Assembler::t0, Assembler::t1);
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateClz(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    if (is_64_bit) {
      as_.ClzD(Assembler::t0, Assembler::t0);
    } else {
      as_.ClzW(Assembler::t0, Assembler::t0);
      ZeroExtend32(Assembler::t0);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateRev32(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    as_.Revb2W(Assembler::t0, Assembler::t0);
    ZeroExtend32(Assembler::t0);
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateRev64(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    as_.RevbD(Assembler::t0, Assembler::t0);
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  void ComputeLogicalFlags(Register result, uint32_t width) {
    as_.SrliD(Assembler::t3, result, width - 1);
    as_.SlliD(Assembler::t3, Assembler::t3, 3);
    Assembler::Label* nonzero = as_.MakeLabel();
    as_.Bnez(result, *nonzero);
    as_.AddiD(Assembler::t3, Assembler::t3, 4);
    as_.Bind(nonzero);
    as_.StW(Assembler::t3, Assembler::s8, kFlagsOffset);
  }

  bool TranslatePcRelativeAddress(uint32_t insn, GuestAddr pc) {
    bool page_relative = (insn >> 31) != 0;
    uint32_t rd = insn & 31;
    uint64_t encoded_imm = ((insn >> 5) & 0x7ffff) << 2 | ((insn >> 29) & 3);
    int64_t immediate = SignExtend(encoded_imm, 21);
    GuestAddr value;
    if (page_relative) {
      value = (pc & ~GuestAddr{0xfff}) + immediate * 4096;
    } else {
      value = pc + immediate;
    }
    as_.Li(Assembler::t0, value);
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateLoadStoreUnsignedImmediate(uint32_t insn, GuestAddr pc) {
    if ((insn & 0x0400'0000u) != 0) {
      uint32_t size = insn >> 30;
      uint32_t opc = (insn >> 22) & 3;
      bool is_simd32 = size == 2 && opc <= 1;
      bool is_simd64 = size == 3 && opc <= 1;
      bool is_simd128 = size == 0 && (opc == 2 || opc == 3);
      if (!is_simd32 && !is_simd64 && !is_simd128) {
        return false;
      }
      uint32_t imm12 = (insn >> 10) & 0xfff;
      uint32_t rn = (insn >> 5) & 31;
      uint32_t rt = insn & 31;
      LoadXOrSp(rn, Assembler::t0);
      uint32_t scale = is_simd32 ? 2 : (is_simd64 ? 3 : 4);
      as_.Li(Assembler::t1, static_cast<uint64_t>(imm12) << scale);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
      ApplyTbi(Assembler::t0);
      if (is_simd32) {
        EmitSimd32LoadStore(opc == 0, rt, pc);
      } else if (is_simd64) {
        EmitSimd64LoadStore(opc == 0, rt, pc);
      } else {
        EmitSimd128LoadStore(opc == 2, rt, pc);
      }
      return true;
    }
    uint32_t size = insn >> 30;
    uint32_t opc = (insn >> 22) & 3;
    uint32_t imm12 = (insn >> 10) & 0xfff;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    if (size == 3 && opc == 2) {
      return true;
    }
    const bool signed_load = opc > 1 && size <= 2;
    if (opc > 1 && !signed_load) {
      return false;
    }

    LoadXOrSp(rn, Assembler::t0);
    as_.Li(Assembler::t1, static_cast<uint64_t>(imm12) << size);
    as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    ApplyTbi(Assembler::t0);
    EmitLoadStore(size, opc, rt, pc);
    return true;
  }

  bool TranslateMteLoadStore(uint32_t insn, GuestAddr pc) {
    const uint32_t opc = (insn >> 22) & 3;
    const int64_t offset = SignExtend((insn >> 12) & 0x1ff, 9) * 16;
    const uint32_t mode = (insn >> 10) & 3;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rt = insn & 31;
    const bool writeback = mode == 1 || mode == 3;

    // Preserve the tagged architectural base. Post-index accesses the old
    // base; offset and pre-index forms access base + immediate.
    LoadXOrSp(rn, Assembler::t0);
    as_.Move(Assembler::t1, Assembler::t0);
    if (mode != 1) {
      as_.Li(Assembler::t2, offset);
      as_.AddD(Assembler::t1, Assembler::t1, Assembler::t2);
    }

    const bool is_ldg = opc == 1 && mode == 0;
    const bool is_stzg = opc == 1 && mode != 0;
    const bool is_stz2g = opc == 3 && mode != 0;
    const bool is_ldgm = opc == 3 && mode == 0;

    if (is_ldg) {
      // No MTE backing means the loaded allocation tag is zero. LDG only
      // replaces Rt[59:56], preserving every other bit of Rt.
      if (rt != 31) {
        LoadXOrZero(rt, Assembler::t2);
        as_.Li(Assembler::t3, ~UINT64_C(0x0f00'0000'0000'0000));
        as_.And(Assembler::t2, Assembler::t2, Assembler::t3);
        StoreXOrDiscard(rt, Assembler::t2);
      }
    } else if (is_ldgm) {
      // All packed tags read as zero without MTE backing.
      StoreXOrDiscard(rt, Assembler::zero);
    } else if (is_stzg || is_stz2g) {
      // Tag-and-zero forms retain their data side effect. Align to the tag
      // granule, strip TBI only from the effective host address, and attach a
      // recovery entry to every store. Architectural writeback happens only
      // after all stores succeed.
      const uint64_t alignment_mask = is_stz2g ? ~UINT64_C(0x1f) : ~UINT64_C(0x0f);
      as_.Li(Assembler::t2, alignment_mask);
      as_.And(Assembler::t1, Assembler::t1, Assembler::t2);
      ApplyTbi(Assembler::t1);

      Assembler::Label* recovery = as_.MakeLabel();
      Assembler::Label* done = as_.MakeLabel();
      const uint32_t byte_count = is_stz2g ? 32 : 16;
      for (uint32_t byte_offset = 0; byte_offset < byte_count; byte_offset += 8) {
        as_.SetRecoveryPoint(recovery);
        as_.StD(Assembler::zero, Assembler::t1, byte_offset);
      }
      if (writeback) {
        as_.Li(Assembler::t2, offset);
        as_.AddD(Assembler::t0, Assembler::t0, Assembler::t2);
        StoreXOrSp(rn, Assembler::t0);
      }
      as_.B(*done);
      as_.Bind(recovery);
      ExitGeneratedCode(pc);
      as_.Bind(done);
      return true;
    }
    // STG/ST2G/STGM/STZGM are tag-only operations and become NOPs without
    // MTE backing. Their pre/post-index forms still update the base register.

    if (writeback) {
      as_.Li(Assembler::t2, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t2);
      StoreXOrSp(rn, Assembler::t0);
    }
    return true;
  }

  bool TranslateLoadStoreIndexed(uint32_t insn, GuestAddr pc) {
    if ((insn & 0x0400'0000u) != 0) {
      uint32_t size = insn >> 30;
      uint32_t opc = (insn >> 22) & 3;
      int64_t offset = SignExtend((insn >> 12) & 0x1ff, 9);
      uint32_t mode = (insn >> 10) & 3;
      uint32_t rn = (insn >> 5) & 31;
      uint32_t rt = insn & 31;
      bool writeback = mode == 1 || mode == 3;
      bool is_simd32 = size == 2 && opc <= 1;
      bool is_simd64 = size == 3 && opc <= 1;
      bool is_simd128 = size == 0 && (opc == 2 || opc == 3);
      if ((!is_simd32 && !is_simd64 && !is_simd128) || mode == 2) {
        return false;
      }

      LoadXOrSp(rn, Assembler::t0);
      if (mode == 0 || mode == 3) {
        as_.Li(Assembler::t1, offset);
        as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
      }
      ApplyTbi(Assembler::t0);
      if (is_simd32) {
        EmitSimd32LoadStore(opc == 0, rt, pc);
      } else if (is_simd64) {
        EmitSimd64LoadStore(opc == 0, rt, pc);
      } else {
        EmitSimd128LoadStore(opc == 2, rt, pc);
      }
      if (writeback) {
        LoadXOrSp(rn, Assembler::t0);
        as_.Li(Assembler::t1, offset);
        as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
        StoreXOrSp(rn, Assembler::t0);
      }
      return true;
    }
    uint32_t size = insn >> 30;
    uint32_t opc = (insn >> 22) & 3;
    int64_t offset = SignExtend((insn >> 12) & 0x1ff, 9);
    uint32_t mode = (insn >> 10) & 3;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    bool writeback = mode == 1 || mode == 3;
    if (size == 3 && opc == 2) {
      return mode == 0;
    }
    const bool signed_load = opc > 1 && size <= 2;
    if ((opc > 1 && !signed_load) || mode == 2 || (writeback && rn != 31 && rn == rt)) {
      return false;
    }

    LoadXOrSp(rn, Assembler::t0);
    if (mode == 0 || mode == 3) {
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    ApplyTbi(Assembler::t0);
    EmitLoadStore(size, opc, rt, pc);
    if (writeback) {
      LoadXOrSp(rn, Assembler::t0);
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
      StoreXOrSp(rn, Assembler::t0);
    }
    return true;
  }

  bool TranslateLoadStoreRegisterOffset(uint32_t insn, GuestAddr pc) {
    if ((insn & 0x0400'0000u) != 0) {
      uint32_t size = insn >> 30;
      uint32_t opc = (insn >> 22) & 3;
      uint32_t rm = (insn >> 16) & 31;
      uint32_t option = (insn >> 13) & 7;
      bool scaled = ((insn >> 12) & 1) != 0;
      uint32_t rn = (insn >> 5) & 31;
      uint32_t rt = insn & 31;
      bool is_simd32 = size == 2 && opc <= 1;
      bool is_simd64 = size == 3 && opc <= 1;
      bool is_simd128 = size == 0 && (opc == 2 || opc == 3);
      if ((!is_simd32 && !is_simd64 && !is_simd128) ||
          (option != 2 && option != 3 && option != 6 && option != 7)) {
        return false;
      }

      LoadXOrSp(rn, Assembler::t0);
      LoadXOrZero(rm, Assembler::t1);
      if (option == 2) {
        ZeroExtend32(Assembler::t1);
      } else if (option == 6) {
        SignExtend32(Assembler::t1);
      }
      if (scaled) {
        as_.SlliD(Assembler::t1, Assembler::t1, is_simd32 ? 2 : (is_simd64 ? 3 : 4));
      }
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
      ApplyTbi(Assembler::t0);
      if (is_simd32) {
        EmitSimd32LoadStore(opc == 0, rt, pc);
      } else if (is_simd64) {
        EmitSimd64LoadStore(opc == 0, rt, pc);
      } else {
        EmitSimd128LoadStore(opc == 2, rt, pc);
      }
      return true;
    }
    uint32_t size = insn >> 30;
    uint32_t opc = (insn >> 22) & 3;
    uint32_t rm = (insn >> 16) & 31;
    uint32_t option = (insn >> 13) & 7;
    bool scaled = ((insn >> 12) & 1) != 0;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    if (size == 3 && opc == 2) {
      return true;
    }
    // The integer register-offset form accepts UXTW, UXTX/LSL, SXTW and
    // SXTX.  Other option encodings are reserved for this instruction class.
    const bool signed_load = opc > 1 && size <= 2;
    if ((opc > 1 && !signed_load) || (option != 2 && option != 3 && option != 6 && option != 7)) {
      return false;
    }

    LoadXOrSp(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    if (option == 2) {
      ZeroExtend32(Assembler::t1);
    } else if (option == 6) {
      SignExtend32(Assembler::t1);
    }
    if (scaled && size != 0) {
      as_.SlliD(Assembler::t1, Assembler::t1, size);
    }
    as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    ApplyTbi(Assembler::t0);
    EmitLoadStore(size, opc, rt, pc);
    return true;
  }

  bool TranslateLoadStorePair(uint32_t insn, GuestAddr pc) {
    uint32_t opc = insn >> 30;
    uint32_t mode = (insn >> 23) & 3;
    bool load = ((insn >> 22) & 1) != 0;
    int64_t imm7 = SignExtend((insn >> 15) & 0x7f, 7);
    uint32_t rt2 = (insn >> 10) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    if ((opc != 0 && opc != 2) || mode == 0 || (load && rt == rt2) ||
        (mode != 2 && rn != 31 && (rn == rt || rn == rt2))) {
      return false;
    }
    uint32_t size = opc == 2 ? 3 : 2;
    int64_t offset = imm7 * (int64_t{1} << size);

    LoadXOrSp(rn, Assembler::t0);
    if (mode != 1) {
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    ApplyTbi(Assembler::t0);
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (load) {
      as_.SetRecoveryPoint(recovery);
      if (size == 3) {
        as_.LdD(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.LdD(Assembler::t2, Assembler::t0, 8);
      } else {
        as_.LdWU(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.LdWU(Assembler::t2, Assembler::t0, 4);
      }
      StoreXOrDiscard(rt, Assembler::t1);
      StoreXOrDiscard(rt2, Assembler::t2);
    } else {
      LoadXOrZero(rt, Assembler::t1);
      LoadXOrZero(rt2, Assembler::t2);
      as_.SetRecoveryPoint(recovery);
      if (size == 3) {
        as_.StD(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.StD(Assembler::t2, Assembler::t0, 8);
      } else {
        as_.StW(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.StW(Assembler::t2, Assembler::t0, 4);
      }
    }
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
    if (mode == 1 || mode == 3) {
      LoadXOrSp(rn, Assembler::t0);
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
      StoreXOrSp(rn, Assembler::t0);
    }
    return true;
  }

  bool TranslateLoadExclusive(uint32_t insn, GuestAddr pc) {
    const uint32_t size = insn >> 30;
    const bool acquire = ((insn >> 15) & 1) != 0;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rt = insn & 31;

    LoadXOrSp(rn, Assembler::t0);
    as_.Move(Assembler::t3, Assembler::t0);
    ApplyTbi(Assembler::t0);

    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    as_.SetRecoveryPoint(recovery);
    switch (size) {
      case 0:
        as_.LdBU(Assembler::t1, Assembler::t0, 0);
        break;
      case 1:
        as_.LdHU(Assembler::t1, Assembler::t0, 0);
        break;
      case 2:
        as_.LdWU(Assembler::t1, Assembler::t0, 0);
        break;
      case 3:
        as_.LdD(Assembler::t1, Assembler::t0, 0);
        break;
    }
    if (acquire) {
      as_.Dbar(0x14);
    }
    as_.StD(Assembler::t3,
            Assembler::s8,
            offsetof(ThreadState, cpu) + offsetof(CPUState, reservation_address));
    as_.StD(Assembler::t1,
            Assembler::s8,
            offsetof(ThreadState, cpu) + offsetof(CPUState, reservation_value));
    StoreXOrDiscard(rt, Assembler::t1);
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
    return true;
  }

  bool TranslateStoreRelease(uint32_t insn, GuestAddr pc) {
    const uint32_t size = insn >> 30;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rt = insn & 31;

    LoadXOrSp(rn, Assembler::t0);
    ApplyTbi(Assembler::t0);
    LoadXOrZero(rt, Assembler::t1);

    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    as_.SetRecoveryPoint(recovery);
    switch (size) {
      case 0:
        as_.StB(Assembler::t1, Assembler::t0, 0);
        break;
      case 1:
        as_.StH(Assembler::t1, Assembler::t0, 0);
        break;
      case 2:
        as_.StW(Assembler::t1, Assembler::t0, 0);
        break;
      case 3:
        as_.StD(Assembler::t1, Assembler::t0, 0);
        break;
    }
    // LoongArch's release barrier hint is emitted after the store, matching
    // the platform's established atomic lowering (for example Go's loong64
    // runtime).  It is lighter than a full DBAR 0 in this very hot path.
    as_.Dbar(0x12);
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
    return true;
  }

  bool TranslateStoreExclusive(uint32_t insn, GuestAddr pc) {
    const uint32_t size = insn >> 30;
    if (size < 2) {
      return false;
    }
    const bool release = ((insn >> 15) & 1) != 0;
    const uint32_t rs = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rt = insn & 31;
    constexpr int32_t kReservationAddressOffset =
        offsetof(ThreadState, cpu) + offsetof(CPUState, reservation_address);
    constexpr int32_t kReservationValueOffset =
        offsetof(ThreadState, cpu) + offsetof(CPUState, reservation_value);

    Assembler::Label* retry = as_.MakeLabel();
    Assembler::Label* fail = as_.MakeLabel();
    Assembler::Label* success = as_.MakeLabel();
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();

    // Check the architectural reservation before touching memory.  Keep it
    // intact until the potentially faulting LL/SC completes, so recovery can
    // re-enter the interpreter at this instruction without swallowing a fault.
    LoadXOrSp(rn, Assembler::t0);
    as_.LdD(Assembler::t1, Assembler::s8, kReservationAddressOffset);
    as_.Bne(Assembler::t0, Assembler::t1, *fail);
    ApplyTbi(Assembler::t0);
    as_.LdD(Assembler::t1, Assembler::s8, kReservationValueOffset);
    LoadXOrZero(rt, Assembler::t2);
    if (size == 2) {
      ZeroExtend32(Assembler::t1);
      ZeroExtend32(Assembler::t2);
    }

    as_.Bind(retry);
    as_.SetRecoveryPoint(recovery);
    if (size == 2) {
      as_.LlW(Assembler::t4, Assembler::t0);
      ZeroExtend32(Assembler::t4);
    } else {
      as_.LlD(Assembler::t4, Assembler::t0);
    }
    as_.Bne(Assembler::t4, Assembler::t1, *fail);
    as_.Move(Assembler::t3, Assembler::t2);
    as_.SetRecoveryPoint(recovery);
    if (size == 2) {
      as_.ScW(Assembler::t3, Assembler::t0);
    } else {
      as_.ScD(Assembler::t3, Assembler::t0);
    }
    as_.Beqz(Assembler::t3, *retry);
    as_.B(*success);

    as_.Bind(fail);
    as_.StD(Assembler::zero, Assembler::s8, kReservationAddressOffset);
    as_.AddiD(Assembler::t0, Assembler::zero, 1);
    StoreXOrDiscard(rs, Assembler::t0);
    as_.B(*done);

    as_.Bind(success);
    if (release) {
      as_.Dbar(0x12);
    }
    as_.StD(Assembler::zero, Assembler::s8, kReservationAddressOffset);
    StoreXOrDiscard(rs, Assembler::zero);
    as_.B(*done);

    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
    return true;
  }

  bool TranslateAtomicMemory(uint32_t insn, GuestAddr pc, AtomicMemoryOp operation) {
    const uint32_t size = insn >> 30;
    const bool acquire = operation == AtomicMemoryOp::kCas ? ((insn >> 22) & 1) != 0
                                                           : ((insn >> 23) & 1) != 0;
    const bool release = operation == AtomicMemoryOp::kCas ? ((insn >> 15) & 1) != 0
                                                           : ((insn >> 22) & 1) != 0;
    const uint32_t rs = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rt = insn & 31;

    Assembler::Label* retry = as_.MakeLabel();
    Assembler::Label* compare_failed = as_.MakeLabel();
    Assembler::Label* success = as_.MakeLabel();
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();

    LoadXOrSp(rn, Assembler::t0);
    ApplyTbi(Assembler::t0);
    LoadXOrZero(rs, Assembler::t1);
    if (operation == AtomicMemoryOp::kCas && size >= 2) {
      LoadXOrZero(rt, Assembler::t2);
    }

    // LA64 has native full-barrier atomic exchange/add for W/D.  Besides
    // avoiding an LL/SC retry loop, these instructions provide the ordering
    // used by Clang's own __atomic lowering and are essential for pthread and
    // libc++ synchronization.  B/H still require a containing-word LL/SC.
    if (size >= 2 && operation != AtomicMemoryOp::kCas) {
      Assembler::Label* recovery = as_.MakeLabel();
      Assembler::Label* done = as_.MakeLabel();
      as_.SetRecoveryPoint(recovery);
      if (operation == AtomicMemoryOp::kSwp) {
        size == 2 ? as_.AmswapDbW(Assembler::t5, Assembler::t1, Assembler::t0)
                  : as_.AmswapDbD(Assembler::t5, Assembler::t1, Assembler::t0);
      } else {
        size == 2 ? as_.AmaddDbW(Assembler::t5, Assembler::t1, Assembler::t0)
                  : as_.AmaddDbD(Assembler::t5, Assembler::t1, Assembler::t0);
      }
      if (size == 2) {
        ZeroExtend32(Assembler::t5);
      }
      StoreXOrDiscard(rt, Assembler::t5);
      as_.B(*done);
      as_.Bind(recovery);
      ExitGeneratedCode(pc);
      as_.Bind(done);
      return true;
    }

    if (size < 2) {
      // LA64 LL/SC operates on W/D values.  For B/H atomics reserve the
      // naturally containing word, then extract and merge only the selected
      // field.  Architecturally valid LSE atomics are naturally aligned, so a
      // halfword never crosses that containing word.
      as_.Li(Assembler::t7, size == 0 ? 0xffu : 0xffffu);
      as_.And(Assembler::t1, Assembler::t1, Assembler::t7);
      as_.Li(Assembler::t3, 3);
      as_.And(Assembler::t8, Assembler::t0, Assembler::t3);
      as_.SlliD(Assembler::t8, Assembler::t8, 3);
      as_.Li(Assembler::t3, UINT64_MAX - 3);
      as_.And(Assembler::t0, Assembler::t0, Assembler::t3);

      as_.Bind(retry);
      as_.SetRecoveryPoint(recovery);
      as_.LlW(Assembler::t4, Assembler::t0);
      ZeroExtend32(Assembler::t4);
      as_.SrlW(Assembler::t5, Assembler::t4, Assembler::t8);
      as_.And(Assembler::t5, Assembler::t5, Assembler::t7);
      if (operation == AtomicMemoryOp::kCas) {
        as_.Bne(Assembler::t5, Assembler::t1, *compare_failed);
        // Mask construction below consumes t2, and SC failure branches back
        // here.  Reload the architectural desired value on every attempt so
        // an ABA-style retry cannot store the former mask temporary.
        LoadXOrZero(rt, Assembler::t6);
        as_.And(Assembler::t6, Assembler::t6, Assembler::t7);
      } else if (operation == AtomicMemoryOp::kSwp) {
        as_.Move(Assembler::t6, Assembler::t1);
      } else {
        as_.AddD(Assembler::t6, Assembler::t5, Assembler::t1);
        as_.And(Assembler::t6, Assembler::t6, Assembler::t7);
      }

      as_.SllW(Assembler::t6, Assembler::t6, Assembler::t8);
      as_.SllW(Assembler::t3, Assembler::t7, Assembler::t8);
      as_.Li(Assembler::t2, UINT64_MAX);
      as_.Xor(Assembler::t3, Assembler::t3, Assembler::t2);
      as_.And(Assembler::t3, Assembler::t4, Assembler::t3);
      as_.Or(Assembler::t3, Assembler::t3, Assembler::t6);
      as_.SetRecoveryPoint(recovery);
      as_.ScW(Assembler::t3, Assembler::t0);
      as_.Beqz(Assembler::t3, *retry);
    } else {
      if (size == 2) {
        ZeroExtend32(Assembler::t1);
        if (operation == AtomicMemoryOp::kCas) {
          ZeroExtend32(Assembler::t2);
        }
      }
      as_.Bind(retry);
      as_.SetRecoveryPoint(recovery);
      if (size == 2) {
        as_.LlW(Assembler::t4, Assembler::t0);
        ZeroExtend32(Assembler::t4);
      } else {
        as_.LlD(Assembler::t4, Assembler::t0);
      }
      as_.Move(Assembler::t5, Assembler::t4);
      if (operation == AtomicMemoryOp::kCas) {
        as_.Bne(Assembler::t5, Assembler::t1, *compare_failed);
        as_.Move(Assembler::t3, Assembler::t2);
      } else if (operation == AtomicMemoryOp::kSwp) {
        as_.Move(Assembler::t3, Assembler::t1);
      } else {
        as_.AddD(Assembler::t3, Assembler::t5, Assembler::t1);
      }
      as_.SetRecoveryPoint(recovery);
      if (size == 2) {
        as_.ScW(Assembler::t3, Assembler::t0);
      } else {
        as_.ScD(Assembler::t3, Assembler::t0);
      }
      as_.Beqz(Assembler::t3, *retry);
    }

    as_.B(*success);
    as_.Bind(compare_failed);
    if (acquire) {
      as_.Dbar(0x14);
    }
    StoreXOrDiscard(rs, Assembler::t5);
    as_.B(*done);

    as_.Bind(success);
    if (release) {
      as_.Dbar(0x12);
    }
    if (acquire) {
      as_.Dbar(0x14);
    }
    StoreXOrDiscard(operation == AtomicMemoryOp::kCas ? rs : rt, Assembler::t5);
    as_.B(*done);

    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
    return true;
  }

  bool TranslateSimdLoadStorePair(uint32_t insn, GuestAddr pc) {
    uint32_t opc = insn >> 30;
    uint32_t mode = (insn >> 23) & 3;
    bool load = ((insn >> 22) & 1) != 0;
    int64_t imm7 = SignExtend((insn >> 15) & 0x7f, 7);
    uint32_t rt2 = (insn >> 10) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    if (opc > 2 || mode == 0 || (load && rt == rt2)) {
      return false;
    }
    uint32_t element_size = 4u << opc;
    int64_t offset = imm7 * element_size;

    LoadXOrSp(rn, Assembler::t0);
    if (mode != 1) {
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    ApplyTbi(Assembler::t0);
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (load) {
      if (element_size == 16) {
        as_.SetRecoveryPoint(recovery);
        as_.LdD(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.LdD(Assembler::t2, Assembler::t0, 8);
        as_.SetRecoveryPoint(recovery);
        as_.LdD(Assembler::t3, Assembler::t0, 16);
        as_.SetRecoveryPoint(recovery);
        as_.LdD(Assembler::t4, Assembler::t0, 24);
        as_.StD(Assembler::t1, Assembler::s8, VOffset(rt));
        as_.StD(Assembler::t2, Assembler::s8, VOffset(rt) + 8);
        as_.StD(Assembler::t3, Assembler::s8, VOffset(rt2));
        as_.StD(Assembler::t4, Assembler::s8, VOffset(rt2) + 8);
      } else if (element_size == 8) {
        as_.SetRecoveryPoint(recovery);
        as_.LdD(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.LdD(Assembler::t2, Assembler::t0, 8);
        as_.StD(Assembler::t1, Assembler::s8, VOffset(rt));
        as_.StD(Assembler::zero, Assembler::s8, VOffset(rt) + 8);
        as_.StD(Assembler::t2, Assembler::s8, VOffset(rt2));
        as_.StD(Assembler::zero, Assembler::s8, VOffset(rt2) + 8);
      } else {
        as_.SetRecoveryPoint(recovery);
        as_.LdWU(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.LdWU(Assembler::t2, Assembler::t0, 4);
        as_.StW(Assembler::t1, Assembler::s8, VOffset(rt));
        as_.StW(Assembler::zero, Assembler::s8, VOffset(rt) + 4);
        as_.StD(Assembler::zero, Assembler::s8, VOffset(rt) + 8);
        as_.StW(Assembler::t2, Assembler::s8, VOffset(rt2));
        as_.StW(Assembler::zero, Assembler::s8, VOffset(rt2) + 4);
        as_.StD(Assembler::zero, Assembler::s8, VOffset(rt2) + 8);
      }
    } else {
      if (element_size == 16) {
        as_.LdD(Assembler::t1, Assembler::s8, VOffset(rt));
        as_.LdD(Assembler::t2, Assembler::s8, VOffset(rt) + 8);
        as_.LdD(Assembler::t3, Assembler::s8, VOffset(rt2));
        as_.LdD(Assembler::t4, Assembler::s8, VOffset(rt2) + 8);
        as_.SetRecoveryPoint(recovery);
        as_.StD(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.StD(Assembler::t2, Assembler::t0, 8);
        as_.SetRecoveryPoint(recovery);
        as_.StD(Assembler::t3, Assembler::t0, 16);
        as_.SetRecoveryPoint(recovery);
        as_.StD(Assembler::t4, Assembler::t0, 24);
      } else if (element_size == 8) {
        as_.LdD(Assembler::t1, Assembler::s8, VOffset(rt));
        as_.LdD(Assembler::t2, Assembler::s8, VOffset(rt2));
        as_.SetRecoveryPoint(recovery);
        as_.StD(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.StD(Assembler::t2, Assembler::t0, 8);
      } else {
        as_.LdWU(Assembler::t1, Assembler::s8, VOffset(rt));
        as_.LdWU(Assembler::t2, Assembler::s8, VOffset(rt2));
        as_.SetRecoveryPoint(recovery);
        as_.StW(Assembler::t1, Assembler::t0, 0);
        as_.SetRecoveryPoint(recovery);
        as_.StW(Assembler::t2, Assembler::t0, 4);
      }
    }
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
    if (mode == 1 || mode == 3) {
      LoadXOrSp(rn, Assembler::t0);
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
      StoreXOrSp(rn, Assembler::t0);
    }
    return true;
  }

  bool TranslateLdSt1Multiple4S(uint32_t insn, GuestAddr pc) {
    const bool load = ((insn >> 22) & 1) != 0;
    const bool post_index = ((insn >> 23) & 1) != 0;
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t opcode = (insn >> 12) & 15;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rt = insn & 31;
    uint32_t register_count;
    switch (opcode) {
      case 7:
        register_count = 1;
        break;
      case 10:
        register_count = 2;
        break;
      case 6:
        register_count = 3;
        break;
      case 2:
        register_count = 4;
        break;
      default:
        // Interleaved LD2/LD3/LD4 encodings share this instruction class but
        // require lane deinterleaving and are intentionally handled later.
        return false;
    }

    constexpr std::array<Register, 8> kValues = {
        Assembler::t1, Assembler::t2, Assembler::t3, Assembler::t4,
        Assembler::t5, Assembler::t6, Assembler::t7, Assembler::t8};
    const uint32_t value_count = register_count * 2;
    LoadXOrSp(rn, Assembler::t0);
    ApplyTbi(Assembler::t0);
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();

    if (load) {
      // Delay every architectural V-register write until all potentially
      // faulting loads succeed.  Recovery can therefore re-enter the
      // interpreter at the original instruction without partial register
      // updates.
      for (uint32_t value = 0; value < value_count; ++value) {
        as_.SetRecoveryPoint(recovery);
        as_.LdD(kValues[value], Assembler::t0, value * 8);
      }
      for (uint32_t reg = 0; reg < register_count; ++reg) {
        const uint32_t destination = (rt + reg) & 31;
        as_.StD(kValues[reg * 2], Assembler::s8, VOffset(destination));
        as_.StD(kValues[reg * 2 + 1], Assembler::s8, VOffset(destination) + 8);
      }
    } else {
      // Snapshot all vector sources before touching guest memory.  This also
      // handles lists that wrap from V31 to V0.
      for (uint32_t reg = 0; reg < register_count; ++reg) {
        const uint32_t source = (rt + reg) & 31;
        as_.LdD(kValues[reg * 2], Assembler::s8, VOffset(source));
        as_.LdD(kValues[reg * 2 + 1], Assembler::s8, VOffset(source) + 8);
      }
      for (uint32_t value = 0; value < value_count; ++value) {
        as_.SetRecoveryPoint(recovery);
        as_.StD(kValues[value], Assembler::t0, value * 8);
      }
    }
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);

    if (post_index) {
      LoadXOrSp(rn, Assembler::t0);
      if (rm == 31) {
        as_.AddiD(Assembler::t0, Assembler::t0, register_count * 16);
      } else {
        LoadXOrZero(rm, Assembler::t1);
        as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
      }
      StoreXOrSp(rn, Assembler::t0);
    }
    return true;
  }

  bool TranslateLd1D1PostIndex(uint32_t insn, GuestAddr pc) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rt = insn & 31;
    LoadXOrSp(rn, Assembler::t0);
    ApplyTbi(Assembler::t0);
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    as_.SetRecoveryPoint(recovery);
    as_.LdD(Assembler::t1, Assembler::t0, 0);
    const int32_t lane_offset = ((insn >> 30) & 1) * 8;
    as_.StD(Assembler::t1, Assembler::s8, VOffset(rt) + lane_offset);
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
    LoadXOrSp(rn, Assembler::t0);
    as_.AddiD(Assembler::t0, Assembler::t0, 8);
    StoreXOrSp(rn, Assembler::t0);
    return true;
  }

  bool TranslateLd1r4S(uint32_t insn, GuestAddr pc) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rt = insn & 31;
    LoadXOrSp(rn, Assembler::t0);
    ApplyTbi(Assembler::t0);
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    as_.SetRecoveryPoint(recovery);
    as_.LdWU(Assembler::t1, Assembler::t0, 0);
    as_.Vreplgr2vrW(Assembler::vr0, Assembler::t1);
    StoreV(rt, Assembler::vr0);
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
    return true;
  }

  bool TranslateSt1S2(uint32_t insn, GuestAddr pc) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rt = insn & 31;
    LoadXOrSp(rn, Assembler::t0);
    ApplyTbi(Assembler::t0);
    as_.LdWU(Assembler::t1, Assembler::s8, VOffset(rt) + 8);
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    as_.SetRecoveryPoint(recovery);
    as_.StW(Assembler::t1, Assembler::t0, 0);
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
    return true;
  }

  bool TranslateDup16B(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    as_.Li(Assembler::t1, 0xff);
    as_.And(Assembler::t0, Assembler::t0, Assembler::t1);
    as_.Li(Assembler::t1, 0x0101'0101'0101'0101ULL);
    as_.MulD(Assembler::t0, Assembler::t0, Assembler::t1);
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateMovi2DZero(uint32_t insn) {
    uint32_t rd = insn & 31;
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateDup4SFromElement(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    as_.VreplveiW(Assembler::vr0, Assembler::vr1, 0);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateInsElement(uint32_t insn) {
    const uint32_t imm5 = (insn >> 16) & 31;
    const uint32_t imm4 = (insn >> 11) & 15;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    uint32_t element_size;
    uint32_t dst_index;
    uint32_t src_index;
    if (imm5 & 4) {
      element_size = 4;
      dst_index = imm5 >> 3;
      src_index = imm4 >> 2;
    } else if (imm5 & 8) {
      element_size = 8;
      dst_index = imm5 >> 4;
      src_index = imm4 >> 3;
    } else {
      return false;
    }
    if (element_size == 4) {
      as_.LdWU(Assembler::t0, Assembler::s8, VOffset(rn) + src_index * 4);
      as_.StW(Assembler::t0, Assembler::s8, VOffset(rd) + dst_index * 4);
    } else {
      as_.LdD(Assembler::t0, Assembler::s8, VOffset(rn) + src_index * 8);
      as_.StD(Assembler::t0, Assembler::s8, VOffset(rd) + dst_index * 8);
    }
    return true;
  }

  bool TranslateDupElement4S(uint32_t insn) {
    const uint32_t imm5 = (insn >> 16) & 31;
    if ((imm5 & 7) != 4) {
      return false;
    }
    const uint32_t index = imm5 >> 3;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    as_.VreplveiW(Assembler::vr0, Assembler::vr1, index);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateUcvtf4S(uint32_t insn) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    as_.VffintSWu(Assembler::vr0, Assembler::vr1);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateDup2D(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateExt(uint32_t insn) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    const uint32_t byte_offset = (insn >> 11) & 15;
    const bool is_128_bit = (insn & 0x4000'0000u) != 0;
    if (!is_128_bit && byte_offset >= 8) {
      return false;
    }

    // Load every source before writing Vd so Vd may alias either input.  EXT
    // extracts from the 16/32-byte concatenation Vn:Vm at a constant byte
    // offset; lowering the two 64-bit output chunks avoids a memory helper and
    // works on hosts with or without LSX enabled at runtime.
    as_.LdD(Assembler::t0, Assembler::s8, VOffset(rn));
    as_.LdD(Assembler::t1, Assembler::s8, VOffset(rn) + 8);
    as_.LdD(Assembler::t2, Assembler::s8, VOffset(rm));
    as_.LdD(Assembler::t3, Assembler::s8, VOffset(rm) + 8);
    const std::array<Register, 4> chunks = {
        Assembler::t0, Assembler::t1, Assembler::t2, Assembler::t3};
    const uint32_t chunk = byte_offset / 8;
    const uint32_t shift = (byte_offset & 7) * 8;

    if (shift == 0) {
      as_.Move(Assembler::t4, chunks[chunk]);
    } else {
      as_.SrliD(Assembler::t4, chunks[chunk], shift);
      // The 8-byte form concatenates Vn.low64:Vm.low64 rather than using
      // Vn.high64 as the second chunk.
      const Register next_chunk = is_128_bit ? chunks[chunk + 1] : Assembler::t2;
      as_.SlliD(Assembler::t6, next_chunk, 64 - shift);
      as_.Or(Assembler::t4, Assembler::t4, Assembler::t6);
    }
    if (is_128_bit) {
      if (shift == 0) {
        as_.Move(Assembler::t5, chunks[chunk + 1]);
      } else {
        as_.SrliD(Assembler::t5, chunks[chunk + 1], shift);
        as_.SlliD(Assembler::t6, chunks[chunk + 2], 64 - shift);
        as_.Or(Assembler::t5, Assembler::t5, Assembler::t6);
      }
      as_.StD(Assembler::t5, Assembler::s8, VOffset(rd) + 8);
    } else {
      as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    }
    as_.StD(Assembler::t4, Assembler::s8, VOffset(rd));
    return true;
  }

  bool TranslateVectorRev64(uint32_t insn) {
    const uint32_t size = (insn >> 22) & 3;
    const bool is_128_bit = (insn & 0x4000'0000u) != 0;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    if (size == 3) {
      return false;
    }

    LoadV(rn, Assembler::vr1);
    switch (size) {
      case 0:
        // Reverse bytes in each word, then swap each pair of words.  Together
        // these reverse the bytes independently in both 64-bit lanes.
        as_.Vshuf4iB(Assembler::vr0, Assembler::vr1, 0x1b);
        as_.Vshuf4iW(Assembler::vr0, Assembler::vr0, 0xb1);
        break;
      case 1:
        as_.Vshuf4iH(Assembler::vr0, Assembler::vr1, 0x1b);
        break;
      case 2:
        as_.Vshuf4iW(Assembler::vr0, Assembler::vr1, 0xb1);
        break;
    }
    StoreV(rd, Assembler::vr0);
    if (!is_128_bit) {
      as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    }
    return true;
  }

  bool TranslateTbl16B(uint32_t insn) {
    const uint32_t rd = insn & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t table_regs = ((insn >> 13) & 3) + 1;
    const uint32_t table_bytes = table_regs * 16;

    // Consecutive vector registers wrap at V31.  The hot FFmpeg form does not
    // wrap; leave the uncommon wrapping case to the interpreter rather than
    // adding a branch to every generated byte lookup.
    if (rn + table_regs > 32) {
      return false;
    }

    // Preserve the complete index vector before writing Vd.  In FFmpeg's hot
    // TBL form Vd == Vm, so writing lanes one by one would otherwise corrupt
    // indices that have not been consumed yet.  Build both output halves in
    // registers and commit them only after all sixteen lookups are complete.
    as_.LdD(Assembler::t7, Assembler::s8, VOffset(rm));
    as_.LdD(Assembler::t8, Assembler::s8, VOffset(rm) + 8);
    as_.Move(Assembler::t0, Assembler::zero);
    as_.Move(Assembler::t1, Assembler::zero);
    as_.AddiD(Assembler::t3, Assembler::zero, table_bytes);
    as_.AddiD(Assembler::t6, Assembler::zero, 0xff);

    for (uint32_t lane = 0; lane < 16; ++lane) {
      const Register index_half = lane < 8 ? Assembler::t7 : Assembler::t8;
      const uint32_t shift = (lane & 7) * 8;
      if (shift == 0) {
        as_.Move(Assembler::t2, index_half);
      } else {
        as_.SrliD(Assembler::t2, index_half, shift);
      }
      as_.And(Assembler::t2, Assembler::t2, Assembler::t6);
      as_.Move(Assembler::t5, Assembler::zero);

      Assembler::Label* in_range = as_.MakeLabel();
      Assembler::Label* lookup_done = as_.MakeLabel();
      as_.Bltu(Assembler::t2, Assembler::t3, *in_range);
      as_.B(*lookup_done);
      as_.Bind(in_range);
      as_.AddiD(Assembler::t4, Assembler::s8, VOffset(rn));
      as_.AddD(Assembler::t4, Assembler::t4, Assembler::t2);
      as_.LdBU(Assembler::t5, Assembler::t4, 0);
      as_.Bind(lookup_done);

      if (shift != 0) {
        as_.SlliD(Assembler::t5, Assembler::t5, shift);
      }
      Register output_half = lane < 8 ? Assembler::t0 : Assembler::t1;
      as_.Or(output_half, output_half, Assembler::t5);
    }

    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::t1, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateVectorLogical(uint32_t insn, uint32_t operation) {
    uint32_t rm = (insn >> 16) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    as_.LdD(Assembler::t0, Assembler::s8, VOffset(rn));
    as_.LdD(Assembler::t1, Assembler::s8, VOffset(rn) + 8);
    as_.LdD(Assembler::t2, Assembler::s8, VOffset(rm));
    as_.LdD(Assembler::t3, Assembler::s8, VOffset(rm) + 8);
    if (operation == 0) {
      as_.And(Assembler::t0, Assembler::t0, Assembler::t2);
      as_.And(Assembler::t1, Assembler::t1, Assembler::t3);
    } else if (operation == 1) {
      as_.Or(Assembler::t0, Assembler::t0, Assembler::t2);
      as_.Or(Assembler::t1, Assembler::t1, Assembler::t3);
    } else {
      as_.Xor(Assembler::t0, Assembler::t0, Assembler::t2);
      as_.Xor(Assembler::t1, Assembler::t1, Assembler::t3);
    }
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::t1, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateVectorMulAcc4S(uint32_t insn, uint32_t operation) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    if (operation == 0) {
      as_.VmulW(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    } else {
      LoadV(rd, Assembler::vr0);
      if (operation == 1) {
        as_.VmaddW(Assembler::vr0, Assembler::vr1, Assembler::vr2);
      } else {
        as_.VmsubW(Assembler::vr0, Assembler::vr1, Assembler::vr2);
      }
    }
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateVectorAddSub4S(uint32_t insn, bool subtract) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    if (subtract) {
      as_.VsubW(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    } else {
      as_.VaddW(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    }
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateVectorAdd2D(uint32_t insn) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    as_.VaddD(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateFaddp2S(uint32_t insn) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;

    // Snapshot both sources before writing Vd so every register-aliasing
    // combination behaves like ARM.  Use scalar additions: the 2S form has
    // only two architectural results, and inactive lanes must not raise FP
    // exceptions.
    as_.LdD(Assembler::t0, Assembler::s8, VOffset(rn));
    as_.LdD(Assembler::t4, Assembler::s8, VOffset(rm));
    as_.Move(Assembler::t1, Assembler::t0);
    ZeroExtend32(Assembler::t1);
    as_.SrliD(Assembler::t2, Assembler::t0, 32);
    as_.Movgr2frW(Assembler::vr1, Assembler::t1);
    as_.Movgr2frW(Assembler::vr2, Assembler::t2);
    as_.FaddS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    as_.Movfr2grS(Assembler::t3, Assembler::vr0);

    as_.Move(Assembler::t1, Assembler::t4);
    ZeroExtend32(Assembler::t1);
    as_.SrliD(Assembler::t2, Assembler::t4, 32);
    as_.Movgr2frW(Assembler::vr1, Assembler::t1);
    as_.Movgr2frW(Assembler::vr2, Assembler::t2);
    as_.FaddS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    as_.Movfr2grS(Assembler::t5, Assembler::vr0);

    as_.StW(Assembler::t3, Assembler::s8, VOffset(rd));
    as_.StW(Assembler::t5, Assembler::s8, VOffset(rd) + 4);
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateFcmeq4SZero(uint32_t insn) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    as_.VxorV(Assembler::vr2, Assembler::vr2, Assembler::vr2);
    as_.VfcmpCeqS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateFcmge4S(uint32_t insn) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    // ARM Vn >= Vm is the ordered comparison Vm <= Vn.
    as_.VfcmpSleS(Assembler::vr0, Assembler::vr2, Assembler::vr1);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateCmeq16B(uint32_t insn) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    as_.VseqB(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateScvtf4S(uint32_t insn) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    as_.VffintSW(Assembler::vr0, Assembler::vr1);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateSshll4S(uint32_t insn) {
    const uint32_t immh_immb = (insn >> 16) & 0x7f;
    const uint32_t shift = immh_immb - 16;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    if ((insn & 0x4000'0000u) != 0) {
      // SSHLL2 widens Vn's upper four halfwords.  Move that 64-bit half down
      // before LSX widens the low four signed halfwords.
      as_.VbsrlV(Assembler::vr1, Assembler::vr1, 8);
    }
    as_.VsllwilWH(Assembler::vr0, Assembler::vr1, shift);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateSqrshrn4H(uint32_t insn) {
    const uint32_t immh_immb = (insn >> 16) & 0x7f;
    const uint32_t shift = 32 - immh_immb;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    as_.VxorV(Assembler::vr0, Assembler::vr0, Assembler::vr0);
    as_.VssrarniHW(Assembler::vr0, Assembler::vr1, shift);
    if ((insn & 0x4000'0000u) != 0) {
      // vpickev.d places its third operand's low 64 bits below its second
      // operand's low 64 bits: {narrowed, old_vd.low64} in ARM lane order.
      LoadV(rd, Assembler::vr2);
      as_.VpickevD(Assembler::vr0, Assembler::vr0, Assembler::vr2);
    }
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateFabs4S(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    constexpr uint64_t kClearFloatSign = 0x7fff'ffff'7fff'ffffULL;
    as_.LdD(Assembler::t0, Assembler::s8, VOffset(rn));
    as_.LdD(Assembler::t1, Assembler::s8, VOffset(rn) + 8);
    as_.Li(Assembler::t2, kClearFloatSign);
    as_.And(Assembler::t0, Assembler::t0, Assembler::t2);
    as_.And(Assembler::t1, Assembler::t1, Assembler::t2);
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::t1, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateCnt8B16B(uint32_t insn) {
    const bool is_128_bit = (insn & 0x4000'0000u) != 0;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    as_.VpcntB(Assembler::vr0, Assembler::vr1);
    if (is_128_bit) {
      StoreV(rd, Assembler::vr0);
    } else {
      as_.Vst(Assembler::vr0, Assembler::s8, VOffset(rd));
      as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    }
    return true;
  }

  bool TranslatePermute4S(uint32_t insn, uint32_t operation) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    const Register outputs[4] = {Assembler::t0, Assembler::t1, Assembler::t2, Assembler::t3};
    uint32_t source_regs[4];
    uint32_t source_lanes[4];

    if (operation == 0) {  // UZP1: n[0], n[2], m[0], m[2]
      source_regs[0] = rn;
      source_lanes[0] = 0;
      source_regs[1] = rn;
      source_lanes[1] = 2;
      source_regs[2] = rm;
      source_lanes[2] = 0;
      source_regs[3] = rm;
      source_lanes[3] = 2;
    } else if (operation == 1) {  // ZIP2: n[2], m[2], n[3], m[3]
      source_regs[0] = rn;
      source_lanes[0] = 2;
      source_regs[1] = rm;
      source_lanes[1] = 2;
      source_regs[2] = rn;
      source_lanes[2] = 3;
      source_regs[3] = rm;
      source_lanes[3] = 3;
    } else if (operation == 2) {  // TRN1: n[0], m[0], n[2], m[2]
      source_regs[0] = rn;
      source_lanes[0] = 0;
      source_regs[1] = rm;
      source_lanes[1] = 0;
      source_regs[2] = rn;
      source_lanes[2] = 2;
      source_regs[3] = rm;
      source_lanes[3] = 2;
    } else if (operation == 3) {  // UZP2: n[1], n[3], m[1], m[3]
      source_regs[0] = rn;
      source_lanes[0] = 1;
      source_regs[1] = rn;
      source_lanes[1] = 3;
      source_regs[2] = rm;
      source_lanes[2] = 1;
      source_regs[3] = rm;
      source_lanes[3] = 3;
    } else {  // ZIP1: n[0], m[0], n[1], m[1]
      source_regs[0] = rn;
      source_lanes[0] = 0;
      source_regs[1] = rm;
      source_lanes[1] = 0;
      source_regs[2] = rn;
      source_lanes[2] = 1;
      source_regs[3] = rm;
      source_lanes[3] = 1;
    }
    for (size_t lane = 0; lane < 4; ++lane) {
      as_.LdWU(outputs[lane],
               Assembler::s8,
               VOffset(source_regs[lane]) + static_cast<int32_t>(source_lanes[lane] * 4));
    }
    for (size_t lane = 0; lane < 4; ++lane) {
      as_.StW(outputs[lane], Assembler::s8, VOffset(rd) + static_cast<int32_t>(lane * 4));
    }
    return true;
  }

  bool TranslateMovi16BOne(uint32_t insn) {
    uint32_t rd = insn & 31;
    as_.Li(Assembler::t0, 0x0101'0101'0101'0101ULL);
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateMoviConstant(uint32_t insn, uint64_t low, uint64_t high) {
    const uint32_t rd = insn & 31;
    as_.Li(Assembler::t0, low);
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.Li(Assembler::t0, high);
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateFmlaFmls4S(uint32_t insn) {
    uint32_t rm = (insn >> 16) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    LoadV(rd, Assembler::vr0);
    if ((insn & 0x0080'0000u) != 0) {
      // ARM FMLS is fma(-Vn, Vm, Vd).  LoongArch VFNMSUB is arithmetically
      // equivalent for ordinary values but differs for signed zero and NaN
      // sign propagation.  Flip Vn's sign bit explicitly, then use VFMADD to
      // preserve ARM's fused operation and exceptional-value behavior.
      as_.Li(Assembler::t0, 0x8000'0000u);
      as_.Vreplgr2vrW(Assembler::vr3, Assembler::t0);
      as_.VxorV(Assembler::vr1, Assembler::vr1, Assembler::vr3);
    }
    as_.VfmaddS(Assembler::vr0, Assembler::vr1, Assembler::vr2, Assembler::vr0);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateFmul4S(uint32_t insn) {
    uint32_t rm = (insn >> 16) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    as_.VfmulS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateScalarFpBinary(uint32_t insn) {
    uint32_t ftype = (insn >> 22) & 3;
    uint32_t operation = (insn >> 12) & 15;
    uint32_t rm = (insn >> 16) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    if (ftype > 1 || operation > 3) {
      return false;
    }

    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    bool is_double = ftype == 1;
    switch (operation) {
      case 0:
        is_double ? as_.FmulD(Assembler::vr0, Assembler::vr1, Assembler::vr2)
                  : as_.FmulS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
        break;
      case 1:
        is_double ? as_.FdivD(Assembler::vr0, Assembler::vr1, Assembler::vr2)
                  : as_.FdivS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
        break;
      case 2:
        is_double ? as_.FaddD(Assembler::vr0, Assembler::vr1, Assembler::vr2)
                  : as_.FaddS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
        break;
      case 3:
        is_double ? as_.FsubD(Assembler::vr0, Assembler::vr1, Assembler::vr2)
                  : as_.FsubS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
        break;
    }
    as_.Vst(Assembler::vr0, Assembler::s8, VOffset(rd));
    if (!is_double) {
      as_.StW(Assembler::zero, Assembler::s8, VOffset(rd) + 4);
    }
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateScalarFmadd(uint32_t insn) {
    const uint32_t ftype = (insn >> 22) & 3;
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t ra = (insn >> 10) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    if (ftype > 1) {
      return false;
    }

    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    LoadV(ra, Assembler::vr3);
    if (ftype == 0) {
      as_.FmaddS(Assembler::vr0, Assembler::vr1, Assembler::vr2, Assembler::vr3);
      as_.Vst(Assembler::vr0, Assembler::s8, VOffset(rd));
      as_.StW(Assembler::zero, Assembler::s8, VOffset(rd) + 4);
    } else {
      as_.FmaddD(Assembler::vr0, Assembler::vr1, Assembler::vr2, Assembler::vr3);
      as_.Vst(Assembler::vr0, Assembler::s8, VOffset(rd));
    }
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateScalarFcsel(uint32_t insn) {
    const uint32_t ftype = (insn >> 22) & 3;
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t condition = (insn >> 12) & 15;
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    if (ftype > 1) {
      return false;
    }

    EmitCondition(condition);
    Assembler::Label* use_false = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    as_.Beqz(Assembler::t2, *use_false);
    if (ftype == 0) {
      as_.LdWU(Assembler::t0, Assembler::s8, VOffset(rn));
    } else {
      as_.LdD(Assembler::t0, Assembler::s8, VOffset(rn));
    }
    as_.B(*done);
    as_.Bind(use_false);
    if (ftype == 0) {
      as_.LdWU(Assembler::t0, Assembler::s8, VOffset(rm));
    } else {
      as_.LdD(Assembler::t0, Assembler::s8, VOffset(rm));
    }
    as_.Bind(done);
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateFcvtDS(uint32_t insn) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    as_.FcvtDS(Assembler::vr0, Assembler::vr1);
    as_.Vst(Assembler::vr0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateFcvtzuWS(uint32_t insn) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    Assembler::Label* fast_path = as_.MakeLabel();
    Assembler::Label* nan_or_negative = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();

    // ARM FCVTZU returns zero for negative inputs and NaNs, saturates positive
    // overflow to UINT32_MAX, and truncates ordinary positive values.  LA64
    // has no scalar unsigned single-to-integer conversion, but every finite
    // input below 2^32 fits in FTINTRZ.L.S and can then be narrowed exactly.
    as_.LdWU(Assembler::t0, Assembler::s8, VOffset(rn));
    as_.SrliD(Assembler::t1, Assembler::t0, 31);
    as_.Bnez(Assembler::t1, *nan_or_negative);
    as_.Li(Assembler::t1, 0x4f80'0000u);  // 2^32
    as_.Bltu(Assembler::t0, Assembler::t1, *fast_path);
    as_.Li(Assembler::t1, 0x7f80'0000u);  // +infinity
    as_.Bltu(Assembler::t1, Assembler::t0, *nan_or_negative);
    as_.Li(Assembler::t0, UINT32_MAX);
    as_.B(*done);

    as_.Bind(nan_or_negative);
    as_.Move(Assembler::t0, Assembler::zero);
    as_.B(*done);

    as_.Bind(fast_path);
    as_.Movgr2frW(Assembler::vr0, Assembler::t0);
    as_.FtintrzLS(Assembler::vr0, Assembler::vr0);
    as_.Movfr2grD(Assembler::t0, Assembler::vr0);
    ZeroExtend32(Assembler::t0);

    as_.Bind(done);
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateFcmpS(uint32_t insn) {
    const uint32_t rm = (insn >> 16) & 31;
    const uint32_t rn = (insn >> 5) & 31;
    const bool with_zero = ((insn >> 3) & 1) != 0;
    Assembler::Label* unordered = as_.MakeLabel();
    Assembler::Label* equal = as_.MakeLabel();
    Assembler::Label* less = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();

    LoadV(rn, Assembler::vr1);
    if (with_zero) {
      as_.Movgr2frW(Assembler::vr2, Assembler::zero);
    } else {
      LoadV(rm, Assembler::vr2);
    }
    as_.FcmpCunS(Assembler::vr1, Assembler::vr2);
    as_.Movcf2gr(Assembler::t0);
    as_.Bnez(Assembler::t0, *unordered);
    as_.FcmpCeqS(Assembler::vr1, Assembler::vr2);
    as_.Movcf2gr(Assembler::t0);
    as_.Bnez(Assembler::t0, *equal);
    as_.FcmpCltS(Assembler::vr1, Assembler::vr2);
    as_.Movcf2gr(Assembler::t0);
    as_.Bnez(Assembler::t0, *less);

    as_.Li(Assembler::t0, CPUState::kFlagCarry);
    as_.B(*done);
    as_.Bind(less);
    as_.Li(Assembler::t0, CPUState::kFlagNegative);
    as_.B(*done);
    as_.Bind(equal);
    as_.Li(Assembler::t0, CPUState::kFlagZero | CPUState::kFlagCarry);
    as_.B(*done);
    as_.Bind(unordered);
    as_.Li(Assembler::t0, CPUState::kFlagCarry | CPUState::kFlagOverflow);
    as_.Bind(done);
    as_.StW(Assembler::t0, Assembler::s8, kFlagsOffset);
    return true;
  }

  bool TranslateFmovSImmediate(uint32_t insn) {
    const uint32_t imm8 = (insn >> 13) & 0xff;
    const uint32_t rd = insn & 31;
    const uint32_t b6 = (imm8 >> 6) & 1;
    const uint32_t bits = ((imm8 & 0x80) << 24) | ((b6 ^ 1) << 30) |
                          (b6 ? 0x3e00'0000u : 0) | ((imm8 & 0x3f) << 19);
    as_.Li(Assembler::t0, bits);
    as_.StW(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StW(Assembler::zero, Assembler::s8, VOffset(rd) + 4);
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateFcvtzsWS(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    Assembler::Label* fast_path = as_.MakeLabel();
    Assembler::Label* nan = as_.MakeLabel();
    Assembler::Label* negative_overflow = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();

    // ARM FCVTZS saturates finite overflows and infinities, and produces zero
    // for NaNs.  LA64 FTINTRZ.W.S does not provide those architectural result
    // guarantees, so classify the IEEE-754 bits before using it for the common
    // finite range (-2^31, 2^31).
    as_.LdWU(Assembler::t0, Assembler::s8, VOffset(rn));
    as_.Li(Assembler::t1, 0x7fff'ffffu);
    as_.And(Assembler::t2, Assembler::t0, Assembler::t1);
    as_.Li(Assembler::t1, 0x4f00'0000u);
    as_.Bltu(Assembler::t2, Assembler::t1, *fast_path);

    as_.Li(Assembler::t1, 0x7f80'0000u);
    as_.Bltu(Assembler::t1, Assembler::t2, *nan);
    as_.SrliD(Assembler::t1, Assembler::t0, 31);
    as_.Bnez(Assembler::t1, *negative_overflow);
    as_.Li(Assembler::t0, 0x7fff'ffffu);
    as_.B(*done);

    as_.Bind(negative_overflow);
    as_.Li(Assembler::t0, 0x8000'0000u);
    as_.B(*done);

    as_.Bind(nan);
    as_.Move(Assembler::t0, Assembler::zero);
    as_.B(*done);

    as_.Bind(fast_path);
    as_.Movgr2frW(Assembler::vr0, Assembler::t0);
    as_.FtintrzWS(Assembler::vr0, Assembler::vr0);
    as_.Movfr2grS(Assembler::t0, Assembler::vr0);

    as_.Bind(done);
    ZeroExtend32(Assembler::t0);
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateVectorFpBinary(uint32_t insn, uint32_t operation) {
    bool is_128_bit = ((insn >> 30) & 1) != 0;
    bool is_double = ((insn >> 22) & 1) != 0;
    if (is_double && !is_128_bit) {
      return false;
    }
    uint32_t rm = (insn >> 16) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    switch (operation) {
      case 0:
        is_double ? as_.VfmulD(Assembler::vr0, Assembler::vr1, Assembler::vr2)
                  : as_.VfmulS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
        break;
      case 1:
        is_double ? as_.VfdivD(Assembler::vr0, Assembler::vr1, Assembler::vr2)
                  : as_.VfdivS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
        break;
      case 2:
        is_double ? as_.VfaddD(Assembler::vr0, Assembler::vr1, Assembler::vr2)
                  : as_.VfaddS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
        break;
      case 3:
        is_double ? as_.VfsubD(Assembler::vr0, Assembler::vr1, Assembler::vr2)
                  : as_.VfsubS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
        break;
      default:
        return false;
    }
    StoreV(rd, Assembler::vr0);
    if (!is_128_bit) {
      as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    }
    return true;
  }

  bool TranslateFadd4S(uint32_t insn) {
    uint32_t rm = (insn >> 16) & 31;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    as_.VfaddS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateFmla4SByElement(uint32_t insn) {
    uint32_t rm = (((insn >> 20) & 1) << 4) | ((insn >> 16) & 15);
    uint32_t index = (((insn >> 11) & 1) << 1) | ((insn >> 21) & 1);
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    LoadV(rd, Assembler::vr0);
    as_.VreplveiW(Assembler::vr2, Assembler::vr2, index);
    as_.VfmaddS(Assembler::vr0, Assembler::vr1, Assembler::vr2, Assembler::vr0);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateFmul4SByElement(uint32_t insn) {
    uint32_t rm = (((insn >> 20) & 1) << 4) | ((insn >> 16) & 15);
    uint32_t index = (((insn >> 11) & 1) << 1) | ((insn >> 21) & 1);
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadV(rn, Assembler::vr1);
    LoadV(rm, Assembler::vr2);
    as_.VreplveiW(Assembler::vr2, Assembler::vr2, index);
    as_.VfmulS(Assembler::vr0, Assembler::vr1, Assembler::vr2);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  bool TranslateFmovS(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    as_.LdWU(Assembler::t0, Assembler::s8, VOffset(rn));
    as_.StW(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StW(Assembler::zero, Assembler::s8, VOffset(rd) + 4);
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateFmovD(uint32_t insn) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    as_.LdD(Assembler::t0, Assembler::s8, VOffset(rn));
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateFmovSFromW(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    as_.StW(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StW(Assembler::zero, Assembler::s8, VOffset(rd) + 4);
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateFmovWFromS(uint32_t insn) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    as_.LdWU(Assembler::t0, Assembler::s8, VOffset(rn));
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateFmovXFromD(uint32_t insn) {
    const uint32_t rn = (insn >> 5) & 31;
    const uint32_t rd = insn & 31;
    as_.LdD(Assembler::t0, Assembler::s8, VOffset(rn));
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  bool TranslateFmovDFromX(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    as_.StD(Assembler::t0, Assembler::s8, VOffset(rd));
    as_.StD(Assembler::zero, Assembler::s8, VOffset(rd) + 8);
    return true;
  }

  bool TranslateDup4S(uint32_t insn) {
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;
    LoadXOrZero(rn, Assembler::t0);
    as_.Vreplgr2vrW(Assembler::vr0, Assembler::t0);
    StoreV(rd, Assembler::vr0);
    return true;
  }

  void EmitSimd32LoadStore(bool is_store, uint32_t rt, GuestAddr pc) {
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (!is_store) {
      as_.SetRecoveryPoint(recovery);
      as_.LdWU(Assembler::t1, Assembler::t0, 0);
      as_.StW(Assembler::t1, Assembler::s8, VOffset(rt));
      as_.StW(Assembler::zero, Assembler::s8, VOffset(rt) + 4);
      as_.StD(Assembler::zero, Assembler::s8, VOffset(rt) + 8);
    } else {
      as_.LdWU(Assembler::t1, Assembler::s8, VOffset(rt));
      as_.SetRecoveryPoint(recovery);
      as_.StW(Assembler::t1, Assembler::t0, 0);
    }
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
  }

  void EmitSimd64LoadStore(bool is_store, uint32_t rt, GuestAddr pc) {
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (!is_store) {
      as_.SetRecoveryPoint(recovery);
      as_.LdD(Assembler::t1, Assembler::t0, 0);
      as_.StD(Assembler::t1, Assembler::s8, VOffset(rt));
      as_.StD(Assembler::zero, Assembler::s8, VOffset(rt) + 8);
    } else {
      as_.LdD(Assembler::t1, Assembler::s8, VOffset(rt));
      as_.SetRecoveryPoint(recovery);
      as_.StD(Assembler::t1, Assembler::t0, 0);
    }
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
  }

  void EmitSimd128LoadStore(bool is_store, uint32_t rt, GuestAddr pc) {
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (!is_store) {
      as_.SetRecoveryPoint(recovery);
      as_.LdD(Assembler::t1, Assembler::t0, 0);
      as_.SetRecoveryPoint(recovery);
      as_.LdD(Assembler::t2, Assembler::t0, 8);
      as_.StD(Assembler::t1, Assembler::s8, VOffset(rt));
      as_.StD(Assembler::t2, Assembler::s8, VOffset(rt) + 8);
    } else {
      as_.LdD(Assembler::t1, Assembler::s8, VOffset(rt));
      as_.LdD(Assembler::t2, Assembler::s8, VOffset(rt) + 8);
      as_.SetRecoveryPoint(recovery);
      as_.StD(Assembler::t1, Assembler::t0, 0);
      as_.SetRecoveryPoint(recovery);
      as_.StD(Assembler::t2, Assembler::t0, 8);
    }
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
  }

  void EmitLoadStore(uint32_t size, uint32_t opc, uint32_t rt, GuestAddr pc) {
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (opc != 0) {
      as_.SetRecoveryPoint(recovery);
      if (opc >= 2) {
        switch (size) {
          case 0:
            as_.LdBU(Assembler::t1, Assembler::t0, 0);
            as_.SlliD(Assembler::t1, Assembler::t1, 56);
            as_.SraiD(Assembler::t1, Assembler::t1, 56);
            break;
          case 1:
            as_.LdHU(Assembler::t1, Assembler::t0, 0);
            as_.SlliD(Assembler::t1, Assembler::t1, 48);
            as_.SraiD(Assembler::t1, Assembler::t1, 48);
            break;
          case 2:
            as_.LdW(Assembler::t1, Assembler::t0, 0);
            break;
        }
        if (opc == 3) {
          // LDRSB/LDRSH to a W register sign-extends within 32 bits, then
          // applies normal W-register upper-half zeroing.
          ZeroExtend32(Assembler::t1);
        }
      } else {
        switch (size) {
          case 0:
            as_.LdBU(Assembler::t1, Assembler::t0, 0);
            break;
          case 1:
            as_.LdHU(Assembler::t1, Assembler::t0, 0);
            break;
          case 2:
            as_.LdWU(Assembler::t1, Assembler::t0, 0);
            break;
          case 3:
            as_.LdD(Assembler::t1, Assembler::t0, 0);
            break;
        }
      }
      StoreXOrDiscard(rt, Assembler::t1);
    } else {
      LoadXOrZero(rt, Assembler::t1);
      as_.SetRecoveryPoint(recovery);
      switch (size) {
        case 0:
          as_.StB(Assembler::t1, Assembler::t0, 0);
          break;
        case 1:
          as_.StH(Assembler::t1, Assembler::t0, 0);
          break;
        case 2:
          as_.StW(Assembler::t1, Assembler::t0, 0);
          break;
        case 3:
          as_.StD(Assembler::t1, Assembler::t0, 0);
          break;
      }
    }
    as_.B(*done);
    as_.Bind(recovery);
    ExitGeneratedCode(pc);
    as_.Bind(done);
  }

  void TranslateBranchImmediate(uint32_t insn, GuestAddr pc) {
    bool link = (insn >> 31) != 0;
    int64_t displacement = SignExtend(insn & 0x03ff'ffff, 26) * 4;
    if (link) {
      as_.Li(Assembler::t0, pc + 4);
      as_.StD(Assembler::t0, Assembler::s8, XOffset(30));
    }
    Exit(pc + displacement);
  }

  void TranslateCompareAndBranch(uint32_t insn, GuestAddr pc) {
    bool is_64_bit = (insn >> 31) != 0;
    bool nonzero = ((insn >> 24) & 1) != 0;
    int64_t displacement = SignExtend((insn >> 5) & 0x7ffff, 19) * 4;
    uint32_t rt = insn & 31;

    LoadXOrZero(rt, Assembler::t1);
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t1);
    }
    Assembler::Label* taken = as_.MakeLabel();
    if (nonzero) {
      as_.Bnez(Assembler::t1, *taken);
    } else {
      as_.Beqz(Assembler::t1, *taken);
    }
    Exit(pc + 4);
    as_.Bind(taken);
    Exit(pc + displacement);
  }

  void TranslateTestAndBranch(uint32_t insn, GuestAddr pc) {
    uint32_t bit = ((insn >> 26) & 0x20) | ((insn >> 19) & 0x1f);
    bool nonzero = ((insn >> 24) & 1) != 0;
    int64_t displacement = SignExtend((insn >> 5) & 0x3fff, 14) * 4;
    uint32_t rt = insn & 31;

    LoadXOrZero(rt, Assembler::t1);
    as_.SrliD(Assembler::t1, Assembler::t1, bit);
    // TBZ/TBNZ tests exactly one bit.  Testing the shifted value directly
    // would incorrectly include every bit above the requested position.
    as_.AddiD(Assembler::t0, Assembler::zero, 1);
    as_.And(Assembler::t1, Assembler::t1, Assembler::t0);
    Assembler::Label* taken = as_.MakeLabel();
    if (nonzero) {
      as_.Bnez(Assembler::t1, *taken);
    } else {
      as_.Beqz(Assembler::t1, *taken);
    }
    Exit(pc + 4);
    as_.Bind(taken);
    Exit(pc + displacement);
  }

  void TranslateConditionalBranch(uint32_t insn, GuestAddr pc) {
    int64_t displacement = SignExtend((insn >> 5) & 0x7ffff, 19) * 4;
    EmitCondition(insn & 0xf);

    Assembler::Label* taken = as_.MakeLabel();
    as_.Bnez(Assembler::t2, *taken);
    Exit(pc + 4);
    as_.Bind(taken);
    Exit(pc + displacement);
  }

  void EmitCondition(uint32_t condition) {
    as_.LdWU(Assembler::t1, Assembler::s8, kFlagsOffset);

    // Normalize N/Z/C/V into individual zero-or-one registers.
    as_.SrliD(Assembler::t2, Assembler::t1, 3);  // N
    as_.SrliD(Assembler::t3, Assembler::t1, 2);  // Z
    as_.SrliD(Assembler::t4, Assembler::t1, 1);  // C
    as_.Li(Assembler::t5, 1);
    as_.And(Assembler::t2, Assembler::t2, Assembler::t5);
    as_.And(Assembler::t3, Assembler::t3, Assembler::t5);
    as_.And(Assembler::t4, Assembler::t4, Assembler::t5);
    as_.And(Assembler::t1, Assembler::t1, Assembler::t5);  // V

    switch (condition >> 1) {
      case 0:  // EQ/NE: Z
        as_.Move(Assembler::t2, Assembler::t3);
        break;
      case 1:  // CS/CC: C
        as_.Move(Assembler::t2, Assembler::t4);
        break;
      case 2:  // MI/PL: N
        break;
      case 3:  // VS/VC: V
        as_.Move(Assembler::t2, Assembler::t1);
        break;
      case 4:  // HI/LS: C && !Z
        as_.Xor(Assembler::t3, Assembler::t3, Assembler::t5);
        as_.And(Assembler::t2, Assembler::t4, Assembler::t3);
        break;
      case 5:  // GE/LT: N == V
        as_.Xor(Assembler::t2, Assembler::t2, Assembler::t1);
        as_.Xor(Assembler::t2, Assembler::t2, Assembler::t5);
        break;
      case 6:  // GT/LE: !Z && N == V
        as_.Xor(Assembler::t2, Assembler::t2, Assembler::t1);
        as_.Xor(Assembler::t2, Assembler::t2, Assembler::t5);
        as_.Xor(Assembler::t3, Assembler::t3, Assembler::t5);
        as_.And(Assembler::t2, Assembler::t2, Assembler::t3);
        break;
      case 7:  // AL/NV
        as_.Move(Assembler::t2, Assembler::t5);
        break;
    }
    // Odd conditions are the inverse, except NV which ARM treats as always.
    if ((condition & 1) != 0 && condition != 0xf) {
      as_.Xor(Assembler::t2, Assembler::t2, Assembler::t5);
    }
  }

  bool TranslateConditionalCompare(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    bool is_sub = ((insn >> 30) & 1) != 0;
    bool is_immediate = ((insn >> 11) & 1) != 0;
    uint32_t rm_or_imm = (insn >> 16) & 31;
    uint32_t condition = (insn >> 12) & 15;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t nzcv = insn & 15;

    EmitCondition(condition);
    Assembler::Label* use_immediate_flags = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    as_.Beqz(Assembler::t2, *use_immediate_flags);

    LoadXOrZero(rn, Assembler::t0);
    if (is_immediate) {
      as_.Li(Assembler::t1, rm_or_imm);
    } else {
      LoadXOrZero(rm_or_imm, Assembler::t1);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
      ZeroExtend32(Assembler::t1);
    }
    as_.Move(Assembler::t4, Assembler::t0);
    if (is_sub) {
      as_.SubD(Assembler::t0, Assembler::t0, Assembler::t1);
    } else {
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    if (is_sub) {
      ComputeSubFlags(Assembler::t4, Assembler::t1, Assembler::t0, is_64_bit ? 64 : 32);
    } else {
      ComputeAddFlags(Assembler::t4, Assembler::t1, Assembler::t0, is_64_bit ? 64 : 32);
    }
    as_.B(*done);

    as_.Bind(use_immediate_flags);
    as_.Li(Assembler::t0, nzcv);
    as_.StW(Assembler::t0, Assembler::s8, kFlagsOffset);
    as_.Bind(done);
    return true;
  }

  bool TranslateConditionalSelect(uint32_t insn) {
    bool is_64_bit = (insn >> 31) != 0;
    bool invert_or_negate = ((insn >> 30) & 1) != 0;
    bool increment_or_negate = ((insn >> 10) & 1) != 0;
    uint32_t rm = (insn >> 16) & 31;
    uint32_t condition = (insn >> 12) & 0xf;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rd = insn & 31;

    EmitCondition(condition);
    Assembler::Label* use_false = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    as_.Beqz(Assembler::t2, *use_false);
    LoadXOrZero(rn, Assembler::t0);
    as_.B(*done);
    as_.Bind(use_false);
    LoadXOrZero(rm, Assembler::t0);
    if (invert_or_negate && increment_or_negate) {  // CSNEG
      as_.SubD(Assembler::t0, Assembler::zero, Assembler::t0);
    } else if (invert_or_negate) {  // CSINV
      as_.Li(Assembler::t1, UINT64_MAX);
      as_.Xor(Assembler::t0, Assembler::t0, Assembler::t1);
    } else if (increment_or_negate) {  // CSINC
      as_.AddiD(Assembler::t0, Assembler::t0, 1);
    }
    as_.Bind(done);
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
  }

  void TranslateBranchRegister(uint32_t insn, GuestAddr pc) {
    uint32_t rn = (insn >> 5) & 31;
    bool link = (insn & 0xffff'fc1fu) == 0xd63f'0000u;
    LoadXOrZero(rn, Assembler::t1);
    if (link) {
      as_.Li(Assembler::t0, pc + 4);
      as_.StD(Assembler::t0, Assembler::s8, XOffset(30));
    }
    ExitIndirect(Assembler::t1);
  }

  Assembler as_;
  bool enable_guest_memory_;
  bool allow_dispatch_;
  bool region_end_reached_ = false;
};

}  // namespace

std::tuple<bool, GuestAddr> TryLiteTranslateRegion(GuestAddr start_pc,
                                                   MachineCode* machine_code,
                                                   LiteTranslateParams params) {
  CHECK_LT(start_pc, params.end_pc);
  // The bootstrap backend deliberately returns to the dispatcher at region
  // boundaries.  Direct chaining and self-profiling are added once the base
  // instruction set has device differential coverage.
  if (params.enable_self_profiling) {
    return {false, start_pc};
  }

  LiteTranslator translator(machine_code, params.enable_guest_memory, params.allow_dispatch);
  GuestAddr pc = start_pc;
  while (pc < params.end_pc && !translator.region_end_reached()) {
    uint32_t insn = *ToHostAddr<const uint32_t>(pc);
    if (!translator.Translate(insn, pc)) {
      return {false, pc};
    }
    pc += sizeof(insn);
  }
  if (!translator.region_end_reached()) {
    translator.Exit(pc);
  }
  translator.Finalize();
  return {true, pc};
}

}  // namespace berberis
