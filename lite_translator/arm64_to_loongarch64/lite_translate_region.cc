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

constexpr int32_t kSpOffset = offsetof(ThreadState, cpu) + offsetof(CPUState, sp);
constexpr int32_t kFlagsOffset = offsetof(ThreadState, cpu) + offsetof(CPUState, flags);

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
    // Keep logical immediates interpreted until exhaustive execution tests
    // isolate the late application crash caused by this generated path.
    if (false && (insn & 0x1f80'0000u) == 0x1200'0000u) {
      return TranslateLogicalImmediate(insn);
    }
    if ((insn & 0x1f80'0000u) == 0x1300'0000u) {
      return TranslateBitfieldExtract(insn);
    }
    if ((insn & 0x1f00'0000u) == 0x1000'0000u) {
      return TranslatePcRelativeAddress(insn, pc);
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
    if ((insn & 0x1fe0'0800u) == 0x1a80'0000u) {
      return TranslateConditionalSelect(insn);
    }
    if ((insn & 0x7fe0'0000u) == 0x1b00'0000u) {
      return TranslateMultiplyAddSub(insn);
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
  static constexpr int32_t XOffset(uint32_t reg) {
    return offsetof(ThreadState, cpu) + offsetof(CPUState, x) + reg * sizeof(uint64_t);
  }

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
    if (invert) {
      return false;
    }

    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    if (!ShiftOperand(Assembler::t1, shift_kind, amount, is_64_bit)) {
      return false;
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
    StoreXOrDiscard(rd, Assembler::t0);
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
      if (opc != 2 || immr != imms + 1) {
        return false;
      }
      LoadXOrZero(rn, Assembler::t0);
      as_.SlliD(Assembler::t0, Assembler::t0, data_size - immr);
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
    // Bit 26 selects the SIMD&FP register bank.  The top-level class mask is
    // shared with integer loads/stores, but this bootstrap backend only
    // synchronizes general-purpose registers in CPUState.
    if ((insn & 0x0400'0000u) != 0) {
      return false;
    }
    uint32_t size = insn >> 30;
    uint32_t opc = (insn >> 22) & 3;
    uint32_t imm12 = (insn >> 10) & 0xfff;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    const bool ldrsw = size == 2 && opc == 2;
    if (opc > 1 && !ldrsw) {
      return false;
    }

    LoadXOrSp(rn, Assembler::t0);
    as_.Li(Assembler::t1, static_cast<uint64_t>(imm12) << size);
    as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    ApplyTbi(Assembler::t0);
    EmitLoadStore(size, opc, rt, pc);
    return true;
  }

  bool TranslateLoadStoreIndexed(uint32_t insn, GuestAddr pc) {
    if ((insn & 0x0400'0000u) != 0) {
      return false;
    }
    uint32_t size = insn >> 30;
    uint32_t opc = (insn >> 22) & 3;
    int64_t offset = SignExtend((insn >> 12) & 0x1ff, 9);
    uint32_t mode = (insn >> 10) & 3;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    bool writeback = mode == 1 || mode == 3;
    const bool ldrsw = size == 2 && opc == 2;
    if ((opc > 1 && !ldrsw) || mode == 2 || (writeback && rn != 31 && rn == rt)) {
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
      return false;
    }
    uint32_t size = insn >> 30;
    uint32_t opc = (insn >> 22) & 3;
    uint32_t rm = (insn >> 16) & 31;
    uint32_t option = (insn >> 13) & 7;
    bool scaled = ((insn >> 12) & 1) != 0;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    // The integer register-offset form accepts UXTW, UXTX/LSL, SXTW and
    // SXTX.  Other option encodings are reserved for this instruction class.
    const bool ldrsw = size == 2 && opc == 2;
    if ((opc > 1 && !ldrsw) || (option != 2 && option != 3 && option != 6 && option != 7)) {
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

  void EmitLoadStore(uint32_t size, uint32_t opc, uint32_t rt, GuestAddr pc) {
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (opc != 0) {
      as_.SetRecoveryPoint(recovery);
      if (opc == 2) {
        // Callers only admit LDRSW for opc == 2.
        as_.LdW(Assembler::t1, Assembler::t0, 0);
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
