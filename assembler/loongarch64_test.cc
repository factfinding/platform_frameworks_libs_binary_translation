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

#include "berberis/assembler/loongarch64.h"

#include <array>
#include <cstdint>

#include "gtest/gtest.h"

#include "berberis/assembler/machine_code.h"

namespace berberis::loongarch64 {
namespace {

TEST(LoongArch64AssemblerTest, EncodesBootstrapInstructions) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.AddD(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.SubD(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.And(Assembler::s0, Assembler::s1, Assembler::s2);
  assembler.Or(Assembler::a0, Assembler::a1, Assembler::zero);
  assembler.Xor(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.MulD(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.AddiD(Assembler::a0, Assembler::a1, -16);
  assembler.Lu12iW(Assembler::a0, 0x12345);
  assembler.Ori(Assembler::a0, Assembler::a0, 0x678);
  assembler.Lu32iD(Assembler::a0, 0x12345);
  assembler.Lu52iD(Assembler::a0, Assembler::a0, 0x123);
  assembler.LdD(Assembler::a0, Assembler::a1, 24);
  assembler.LdW(Assembler::a0, Assembler::a1, 20);
  assembler.LdWU(Assembler::a0, Assembler::a1, 20);
  assembler.LdBU(Assembler::a0, Assembler::a1, 16);
  assembler.LdHU(Assembler::a0, Assembler::a1, 12);
  assembler.StD(Assembler::a0, Assembler::a1, -8);
  assembler.StW(Assembler::a0, Assembler::a1, -4);
  assembler.StH(Assembler::a0, Assembler::a1, -2);
  assembler.StB(Assembler::a0, Assembler::a1, -1);
  assembler.Beq(Assembler::a0, Assembler::a1, 8);
  assembler.Bne(Assembler::a0, Assembler::a1, 8);
  assembler.Bltu(Assembler::a0, Assembler::a1, 8);
  assembler.Beqz(Assembler::a0, 8);
  assembler.Bnez(Assembler::a0, 8);
  assembler.B(8);
  assembler.Bl(8);
  assembler.Jirl(Assembler::ra, Assembler::a0, 0);
  assembler.SlliD(Assembler::a0, Assembler::a1, 7);
  assembler.SrliD(Assembler::a0, Assembler::a1, 7);
  assembler.SraiD(Assembler::a0, Assembler::a1, 7);

  // Values are generated independently with LLVM 21 llvm-mc for the
  // loongarch64 target. MachineCode stores words in target little endian.
  constexpr std::array<uint32_t, 31> kExpected = {
      0x001098a4, 0x0011b9ac, 0x0014e717, 0x001500a4, 0x001598a4, 0x001db9ac, 0x02ffc0a4,
      0x142468a4, 0x0399e084, 0x162468a4, 0x03048c84, 0x28c060a4, 0x288050a4, 0x2a8050a4,
      0x2a0040a4, 0x2a4030a4, 0x29ffe0a4, 0x29bff0a4, 0x297ff8a4, 0x293ffca4, 0x58000885,
      0x5c000885, 0x68000885, 0x40000880, 0x44000880, 0x50000800, 0x54000800, 0x4c000081,
      0x00411ca4, 0x00451ca4, 0x00491ca4,
  };

  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, ResolvesForwardAndBackwardLabels) {
  MachineCode code;
  Assembler assembler(&code);
  Assembler::Label start;
  Assembler::Label end;

  assembler.Bind(&start);
  assembler.B(end);
  assembler.AddD(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.Bind(&end);
  assembler.Beqz(Assembler::a0, start);
  assembler.Finalize();

  EXPECT_EQ(*code.AddrAs<const uint32_t>(0), 0x50000800u);
  EXPECT_EQ(*code.AddrAs<const uint32_t>(8), 0x43fff89fu);
}

TEST(LoongArch64AssemblerTest, EncodesVariableShiftsAndDivision) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.SllW(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.SrlW(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.SraW(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.RotrW(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.SllD(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.SrlD(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.SraD(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.RotrD(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.DivD(Assembler::a0, Assembler::a1, Assembler::a2);
  assembler.DivDU(Assembler::a0, Assembler::a1, Assembler::a2);

  // Values are generated independently with GNU as 2.47 for LoongArch64.
  constexpr std::array<uint32_t, 10> kExpected = {
      0x0017'18a4,
      0x0017'98a4,
      0x0018'18a4,
      0x001b'18a4,
      0x0018'98a4,
      0x0019'18a4,
      0x0019'98a4,
      0x001b'98a4,
      0x0022'18a4,
      0x0023'18a4,
  };
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, EncodesMultiplyHighAndByteReverse) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.MulhDU(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.RevbD(Assembler::t0, Assembler::t1);

  // Generated independently with LLVM's LoongArch assembler.
  constexpr std::array<uint32_t, 2> kExpected = {0x001e'b9ac, 0x0000'3dac};
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, EncodesLsxFloatingPointInstructions) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.Vld(Assembler::vr0, Assembler::s8, 0);
  assembler.Vst(Assembler::vr1, Assembler::s8, 16);
  assembler.VfmulS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfaddS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfmaddS(Assembler::vr5, Assembler::vr6, Assembler::vr7, Assembler::vr8);
  assembler.Vreplgr2vrW(Assembler::vr1, Assembler::t0);
  assembler.VreplveiW(Assembler::vr2, Assembler::vr3, 0);
  assembler.VreplveiW(Assembler::vr4, Assembler::vr5, 3);

  // Generated independently with LLVM's LoongArch assembler and -mlsx.
  constexpr std::array<uint32_t, 8> kExpected = {0x2c00'03e0,
                                                  0x2c40'43e1,
                                                  0x7138'9062,
                                                  0x7130'9062,
                                                  0x0914'1cc5,
                                                  0x729f'0981,
                                                  0x72f7'e062,
                                                  0x72f7'eca4};
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, Materializes64BitImmediate) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.Li(Assembler::a0, 0x1234'5678'9abc'def0);
  assembler.Li(Assembler::a1, 0xffff'ffff'ffff'ffff);

  // Generated independently with LLVM 21 llvm-mc.
  constexpr std::array<uint32_t, 8> kExpected = {
      0x1535'79a4,
      0x03bb'c084,
      0x168a'cf04,
      0x0304'8c84,
      0x15ff'ffe5,
      0x03bf'fca5,
      0x17ff'ffe5,
      0x033f'fca5,
  };
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, RecordsMemoryRecoveryPoint) {
  MachineCode code;
  Assembler assembler(&code);
  Assembler::Label* recovery = assembler.MakeLabel();

  assembler.SetRecoveryPoint(recovery);
  assembler.LdD(Assembler::a0, Assembler::a1, 0);
  assembler.Bind(recovery);
  assembler.Ret();
  assembler.Finalize();

  std::array<uint8_t, 64> installed{};
  RecoveryMap recovery_map;
  code.InstallUnsafe(installed.data(), &recovery_map);
  uintptr_t fault_address = reinterpret_cast<uintptr_t>(installed.data());
  uintptr_t recovery_address = fault_address + sizeof(uint32_t);
  EXPECT_EQ(recovery_address, recovery_map[fault_address]);
}

}  // namespace
}  // namespace berberis::loongarch64
