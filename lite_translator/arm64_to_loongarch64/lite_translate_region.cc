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
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/runtime_primitives/runtime_library.h"

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

class LiteTranslator {
 public:
  explicit LiteTranslator(MachineCode* machine_code, bool enable_guest_memory)
      : as_(machine_code), enable_guest_memory_(enable_guest_memory) {}

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
    if ((insn & 0x1f00'0000u) == 0x0a00'0000u) {
      return TranslateLogicalShiftedRegister(insn);
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
    if ((insn & 0x3e00'0000u) == 0x2800'0000u) {
      if (!enable_guest_memory_) {
        return false;
      }
      return TranslateLoadStorePair(insn, pc);
    }
    if ((insn & 0x1fe0'0800u) == 0x1a80'0000u) {
      return TranslateConditionalSelect(insn);
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
    if (insn == 0xd503'201f) {  // NOP
      return true;
    }
    return false;
  }

  void Exit(GuestAddr pc) {
    as_.Li(Assembler::s7, pc);
    as_.Li(Assembler::t0, kEntryExitGeneratedCode);
    as_.Jirl(Assembler::zero, Assembler::t0, 0);
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
    uint32_t size = insn >> 30;
    uint32_t opc = (insn >> 22) & 3;
    uint32_t imm12 = (insn >> 10) & 0xfff;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    if ((size != 2 && size != 3) || opc > 1) {
      return false;
    }

    LoadXOrSp(rn, Assembler::t0);
    as_.Li(Assembler::t1, static_cast<uint64_t>(imm12) << size);
    as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);

    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (opc == 0) {  // STR
      LoadXOrZero(rt, Assembler::t1);
      as_.SetRecoveryPoint(recovery);
      if (size == 3) {
        as_.StD(Assembler::t1, Assembler::t0, 0);
      } else {
        as_.StW(Assembler::t1, Assembler::t0, 0);
      }
    } else {  // LDR
      as_.SetRecoveryPoint(recovery);
      if (size == 3) {
        as_.LdD(Assembler::t1, Assembler::t0, 0);
      } else {
        as_.LdWU(Assembler::t1, Assembler::t0, 0);
      }
      StoreXOrDiscard(rt, Assembler::t1);
    }
    as_.B(*done);
    as_.Bind(recovery);
    Exit(pc);
    as_.Bind(done);
    return true;
  }

  bool TranslateLoadStoreIndexed(uint32_t insn, GuestAddr pc) {
    uint32_t size = insn >> 30;
    uint32_t opc = (insn >> 22) & 3;
    int64_t offset = SignExtend((insn >> 12) & 0x1ff, 9);
    uint32_t mode = (insn >> 10) & 3;
    uint32_t rn = (insn >> 5) & 31;
    uint32_t rt = insn & 31;
    bool writeback = mode == 1 || mode == 3;
    if ((size != 2 && size != 3) || opc > 1 || mode == 2 || (writeback && rn != 31 && rn == rt)) {
      return false;
    }

    LoadXOrSp(rn, Assembler::t0);
    if (mode == 0 || mode == 3) {
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    EmitLoadStore(size, opc == 1, rt, pc);
    if (writeback) {
      LoadXOrSp(rn, Assembler::t0);
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
      StoreXOrSp(rn, Assembler::t0);
    }
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
    if ((opc != 0 && opc != 2) || mode == 0 || (mode != 2 && rn != 31 && (rn == rt || rn == rt2))) {
      return false;
    }
    uint32_t size = opc == 2 ? 3 : 2;
    int64_t offset = imm7 * (int64_t{1} << size);

    LoadXOrSp(rn, Assembler::t0);
    if (mode != 1) {
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
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
    Exit(pc);
    as_.Bind(done);
    if (mode == 1 || mode == 3) {
      LoadXOrSp(rn, Assembler::t0);
      as_.Li(Assembler::t1, offset);
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
      StoreXOrSp(rn, Assembler::t0);
    }
    return true;
  }

  void EmitLoadStore(uint32_t size, bool load, uint32_t rt, GuestAddr pc) {
    Assembler::Label* recovery = as_.MakeLabel();
    Assembler::Label* done = as_.MakeLabel();
    if (load) {
      as_.SetRecoveryPoint(recovery);
      if (size == 3) {
        as_.LdD(Assembler::t1, Assembler::t0, 0);
      } else {
        as_.LdWU(Assembler::t1, Assembler::t0, 0);
      }
      StoreXOrDiscard(rt, Assembler::t1);
    } else {
      LoadXOrZero(rt, Assembler::t1);
      as_.SetRecoveryPoint(recovery);
      if (size == 3) {
        as_.StD(Assembler::t1, Assembler::t0, 0);
      } else {
        as_.StW(Assembler::t1, Assembler::t0, 0);
      }
    }
    as_.B(*done);
    as_.Bind(recovery);
    Exit(pc);
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
    as_.Move(Assembler::s7, Assembler::t1);
    as_.Li(Assembler::t0, kEntryExitGeneratedCode);
    as_.Jirl(Assembler::zero, Assembler::t0, 0);
  }

  Assembler as_;
  bool enable_guest_memory_;
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

  LiteTranslator translator(machine_code, params.enable_guest_memory);
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
