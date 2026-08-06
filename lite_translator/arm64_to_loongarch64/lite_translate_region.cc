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

constexpr int64_t SignExtend(uint64_t value, uint32_t width) {
  uint64_t sign = uint64_t{1} << (width - 1);
  uint64_t mask = (uint64_t{1} << width) - 1;
  return static_cast<int64_t>((value & mask) ^ sign) - static_cast<int64_t>(sign);
}

class LiteTranslator {
 public:
  explicit LiteTranslator(MachineCode* machine_code) : as_(machine_code) {}

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
    if (set_flags) {
      return false;
    }

    uint64_t imm = static_cast<uint64_t>((insn >> 10) & 0xfff) << (shift * 12);
    LoadXOrSp(rn, Assembler::t0);
    as_.Li(Assembler::t1, imm);
    if (is_sub) {
      as_.SubD(Assembler::t0, Assembler::t0, Assembler::t1);
    } else {
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
    }
    StoreXOrSp(rd, Assembler::t0);
    return true;
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
    if (set_flags) {
      return false;
    }

    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    if (!ShiftOperand(Assembler::t1, shift_kind, amount, is_64_bit)) {
      return false;
    }
    if (is_sub) {
      as_.SubD(Assembler::t0, Assembler::t0, Assembler::t1);
    } else {
      as_.AddD(Assembler::t0, Assembler::t0, Assembler::t1);
    }
    if (!is_64_bit) {
      ZeroExtend32(Assembler::t0);
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
    if (opc == 3 || invert) {
      return false;
    }

    LoadXOrZero(rn, Assembler::t0);
    LoadXOrZero(rm, Assembler::t1);
    if (!ShiftOperand(Assembler::t1, shift_kind, amount, is_64_bit)) {
      return false;
    }
    switch (opc) {
      case 0:
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
    StoreXOrDiscard(rd, Assembler::t0);
    return true;
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

  LiteTranslator translator(machine_code);
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
