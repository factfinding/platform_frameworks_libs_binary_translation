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

#ifndef BERBERIS_ASSEMBLER_LOONGARCH64_H_
#define BERBERIS_ASSEMBLER_LOONGARCH64_H_

#include <cstdint>

#include "berberis/assembler/common.h"
#include "berberis/base/arena_vector.h"
#include "berberis/base/checks.h"
#include "berberis/base/macros.h"

namespace berberis::loongarch64 {

class Assembler;

class Register {
 public:
  constexpr bool operator==(const Register& other) const { return num_ == other.num_; }
  constexpr bool operator!=(const Register& other) const { return num_ != other.num_; }
  constexpr uint8_t GetPhysicalIndex() const { return num_; }

 private:
  friend class Assembler;
  explicit constexpr Register(uint8_t num) : num_(num) {}
  uint8_t num_;
};

// Minimal LA64 assembler used to bootstrap the ARM64-to-LoongArch64 lite JIT.
// Keep encoders explicit until the supported instruction set is large enough
// to justify extending the generated assembler tables.
class Assembler : public AssemblerBase {
 public:
#define BERBERIS_DEFINE_LOONGARCH_REGISTER(name, number) static constexpr Register name{number}
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r0, 0);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r1, 1);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r2, 2);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r3, 3);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r4, 4);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r5, 5);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r6, 6);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r7, 7);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r8, 8);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r9, 9);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r10, 10);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r11, 11);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r12, 12);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r13, 13);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r14, 14);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r15, 15);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r16, 16);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r17, 17);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r18, 18);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r19, 19);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r20, 20);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r21, 21);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r22, 22);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r23, 23);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r24, 24);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r25, 25);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r26, 26);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r27, 27);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r28, 28);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r29, 29);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r30, 30);
  BERBERIS_DEFINE_LOONGARCH_REGISTER(r31, 31);
#undef BERBERIS_DEFINE_LOONGARCH_REGISTER

  // LoongArch ELF psABI aliases.
  static constexpr Register zero = r0;
  static constexpr Register ra = r1;
  static constexpr Register tp = r2;
  static constexpr Register sp = r3;
  static constexpr Register a0 = r4;
  static constexpr Register a1 = r5;
  static constexpr Register a2 = r6;
  static constexpr Register a3 = r7;
  static constexpr Register a4 = r8;
  static constexpr Register a5 = r9;
  static constexpr Register a6 = r10;
  static constexpr Register a7 = r11;
  static constexpr Register t0 = r12;
  static constexpr Register t1 = r13;
  static constexpr Register t2 = r14;
  static constexpr Register t3 = r15;
  static constexpr Register t4 = r16;
  static constexpr Register t5 = r17;
  static constexpr Register t6 = r18;
  static constexpr Register t7 = r19;
  static constexpr Register t8 = r20;
  static constexpr Register fp = r22;
  static constexpr Register s0 = r23;
  static constexpr Register s1 = r24;
  static constexpr Register s2 = r25;
  static constexpr Register s3 = r26;
  static constexpr Register s4 = r27;
  static constexpr Register s5 = r28;
  static constexpr Register s6 = r29;
  static constexpr Register s7 = r30;
  static constexpr Register s8 = r31;

  explicit Assembler(MachineCode* code) : AssemblerBase(code), fixups_(code->arena()) {}

  void AddD(Register rd, Register rj, Register rk) { Emit3R(0x0010'8000, rd, rj, rk); }
  void SubD(Register rd, Register rj, Register rk) { Emit3R(0x0011'8000, rd, rj, rk); }
  void And(Register rd, Register rj, Register rk) { Emit3R(0x0014'8000, rd, rj, rk); }
  void Or(Register rd, Register rj, Register rk) { Emit3R(0x0015'0000, rd, rj, rk); }
  void Xor(Register rd, Register rj, Register rk) { Emit3R(0x0015'8000, rd, rj, rk); }

  void AddiD(Register rd, Register rj, int32_t imm12) {
    Emit2RI12(0x02c0'0000, rd, rj, EncodeSigned(imm12, 12));
  }
  void Ori(Register rd, Register rj, uint32_t imm12) {
    CHECK_LT(imm12, 1u << 12);
    Emit2RI12(0x0380'0000, rd, rj, imm12);
  }
  void Lu52iD(Register rd, Register rj, int32_t imm12) {
    Emit2RI12(0x0300'0000, rd, rj, EncodeSigned(imm12, 12));
  }
  void Lu12iW(Register rd, int32_t imm20) { Emit1RI20(0x1400'0000, rd, imm20); }
  void Lu32iD(Register rd, int32_t imm20) { Emit1RI20(0x1600'0000, rd, imm20); }

  void LdD(Register rd, Register rj, int32_t imm12) {
    Emit2RI12(0x28c0'0000, rd, rj, EncodeSigned(imm12, 12));
  }
  void StD(Register rd, Register rj, int32_t imm12) {
    Emit2RI12(0x29c0'0000, rd, rj, EncodeSigned(imm12, 12));
  }

  void SlliD(Register rd, Register rj, uint32_t shift) { Emit2RI6(0x0041'0000, rd, rj, shift); }
  void SrliD(Register rd, Register rj, uint32_t shift) { Emit2RI6(0x0045'0000, rd, rj, shift); }
  void SraiD(Register rd, Register rj, uint32_t shift) { Emit2RI6(0x0049'0000, rd, rj, shift); }

  void Beq(Register rj, Register rd, int32_t offset) {
    Emit32(0x5800'0000 | EncodeOffset16(offset) | EncodeRj(rj) | EncodeRd(rd));
  }
  void Bne(Register rj, Register rd, int32_t offset) {
    Emit32(0x5c00'0000 | EncodeOffset16(offset) | EncodeRj(rj) | EncodeRd(rd));
  }
  void Beqz(Register rj, int32_t offset) {
    Emit32(0x4000'0000 | EncodeOffset21(offset) | EncodeRj(rj));
  }
  void Bnez(Register rj, int32_t offset) {
    Emit32(0x4400'0000 | EncodeOffset21(offset) | EncodeRj(rj));
  }
  void B(int32_t offset) { Emit32(0x5000'0000 | EncodeOffset26(offset)); }
  void Bl(int32_t offset) { Emit32(0x5400'0000 | EncodeOffset26(offset)); }
  void Jirl(Register rd, Register rj, int32_t offset) {
    Emit32(0x4c00'0000 | EncodeOffset16(offset) | EncodeRj(rj) | EncodeRd(rd));
  }

  void Beq(Register rj, Register rd, const Label& label) {
    AddFixup(label, FixupKind::kOffset16);
    Beq(rj, rd, 0);
  }
  void Bne(Register rj, Register rd, const Label& label) {
    AddFixup(label, FixupKind::kOffset16);
    Bne(rj, rd, 0);
  }
  void Beqz(Register rj, const Label& label) {
    AddFixup(label, FixupKind::kOffset21);
    Beqz(rj, 0);
  }
  void Bnez(Register rj, const Label& label) {
    AddFixup(label, FixupKind::kOffset21);
    Bnez(rj, 0);
  }
  void B(const Label& label) {
    AddFixup(label, FixupKind::kOffset26);
    B(0);
  }
  void Bl(const Label& label) {
    AddFixup(label, FixupKind::kOffset26);
    Bl(0);
  }

  void Move(Register rd, Register rj) { Or(rd, rj, zero); }
  void Ret() { Jirl(zero, ra, 0); }

  void Finalize() {
    for (const Fixup& fixup : fixups_) {
      CHECK(fixup.label->IsBound());
      int64_t offset64 = static_cast<int64_t>(fixup.label->position()) - fixup.pc;
      CHECK_GE(offset64, INT32_MIN);
      CHECK_LE(offset64, INT32_MAX);
      int32_t offset = static_cast<int32_t>(offset64);
      uint32_t encoded = 0;
      switch (fixup.kind) {
        case FixupKind::kOffset16:
          encoded = EncodeOffset16(offset);
          break;
        case FixupKind::kOffset21:
          encoded = EncodeOffset21(offset);
          break;
        case FixupKind::kOffset26:
          encoded = EncodeOffset26(offset);
          break;
      }
      *AddrAs<uint32_t>(fixup.pc) |= encoded;
    }
  }

 private:
  enum class FixupKind { kOffset16, kOffset21, kOffset26 };
  struct Fixup {
    const Label* label;
    uint32_t pc;
    FixupKind kind;
  };

  static uint32_t EncodeRd(Register rd) { return rd.num_; }
  static uint32_t EncodeRj(Register rj) { return static_cast<uint32_t>(rj.num_) << 5; }
  static uint32_t EncodeRk(Register rk) { return static_cast<uint32_t>(rk.num_) << 10; }

  static uint32_t EncodeSigned(int32_t value, uint32_t width) {
    int32_t min = -(1 << (width - 1));
    int32_t max = (1 << (width - 1)) - 1;
    CHECK_GE(value, min);
    CHECK_LE(value, max);
    return static_cast<uint32_t>(value) & ((1u << width) - 1);
  }

  static uint32_t EncodeScaledOffset(int32_t offset, uint32_t width) {
    CHECK_EQ(offset & 3, 0);
    int64_t scaled = offset / 4;
    int64_t min = -(int64_t{1} << (width - 1));
    int64_t max = (int64_t{1} << (width - 1)) - 1;
    CHECK_GE(scaled, min);
    CHECK_LE(scaled, max);
    return static_cast<uint32_t>(scaled) & ((uint32_t{1} << width) - 1);
  }

  static uint32_t EncodeOffset16(int32_t offset) { return EncodeScaledOffset(offset, 16) << 10; }
  static uint32_t EncodeOffset21(int32_t offset) {
    uint32_t scaled = EncodeScaledOffset(offset, 21);
    return ((scaled & 0xffff) << 10) | (scaled >> 16);
  }
  static uint32_t EncodeOffset26(int32_t offset) {
    uint32_t scaled = EncodeScaledOffset(offset, 26);
    return ((scaled & 0xffff) << 10) | (scaled >> 16);
  }

  void Emit3R(uint32_t opcode, Register rd, Register rj, Register rk) {
    Emit32(opcode | EncodeRk(rk) | EncodeRj(rj) | EncodeRd(rd));
  }
  void Emit2RI12(uint32_t opcode, Register rd, Register rj, uint32_t imm12) {
    Emit32(opcode | (imm12 << 10) | EncodeRj(rj) | EncodeRd(rd));
  }
  void Emit2RI6(uint32_t opcode, Register rd, Register rj, uint32_t imm6) {
    CHECK_LT(imm6, 64u);
    Emit32(opcode | (imm6 << 10) | EncodeRj(rj) | EncodeRd(rd));
  }
  void Emit1RI20(uint32_t opcode, Register rd, int32_t imm20) {
    Emit32(opcode | (EncodeSigned(imm20, 20) << 5) | EncodeRd(rd));
  }
  void AddFixup(const Label& label, FixupKind kind) { fixups_.push_back({&label, pc(), kind}); }

  ArenaVector<Fixup> fixups_;

  DISALLOW_COPY_AND_ASSIGN(Assembler);
};

}  // namespace berberis::loongarch64

#endif  // BERBERIS_ASSEMBLER_LOONGARCH64_H_
