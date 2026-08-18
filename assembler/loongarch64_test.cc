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
  assembler.Dbar(0x14);
  assembler.LlW(Assembler::a0, Assembler::a1);
  assembler.ScW(Assembler::a0, Assembler::a1);
  assembler.LlD(Assembler::a0, Assembler::a1);
  assembler.ScD(Assembler::a0, Assembler::a1);
  assembler.AmswapDbW(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.AmswapDbD(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.AmaddDbW(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.AmaddDbD(Assembler::t0, Assembler::t1, Assembler::t2);

  // Values are generated independently with LLVM 21 llvm-mc for the
  // loongarch64 target. MachineCode stores words in target little endian.
  constexpr std::array<uint32_t, 40> kExpected = {
      0x001098a4, 0x0011b9ac, 0x0014e717, 0x001500a4, 0x001598a4, 0x001db9ac,
      0x02ffc0a4, 0x142468a4, 0x0399e084, 0x162468a4, 0x03048c84, 0x28c060a4,
      0x288050a4, 0x2a8050a4, 0x2a0040a4, 0x2a4030a4, 0x29ffe0a4, 0x29bff0a4,
      0x297ff8a4, 0x293ffca4, 0x58000885, 0x5c000885, 0x68000885, 0x40000880,
      0x44000880, 0x50000800, 0x54000800, 0x4c000081, 0x00411ca4, 0x00451ca4,
      0x00491ca4, 0x38720014, 0x200000a4, 0x210000a4, 0x220000a4, 0x230000a4,
      0x386935cc, 0x3869b5cc, 0x386a35cc, 0x386ab5cc,
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

TEST(LoongArch64AssemblerTest, EncodesVectorLogical) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.VandV(Assembler::vr4, Assembler::vr1, Assembler::vr2);
  assembler.VorV(Assembler::vr4, Assembler::vr1, Assembler::vr1);
  assembler.VorV(Assembler::vr8, Assembler::vr3, Assembler::vr3);
  assembler.Finalize();

  ASSERT_EQ(code.install_size(), 3 * sizeof(uint32_t));
  EXPECT_EQ(*code.AddrAs<const uint32_t>(0), 0x7126'0824u);
  EXPECT_EQ(*code.AddrAs<const uint32_t>(4), 0x7126'8424u);
  EXPECT_EQ(*code.AddrAs<const uint32_t>(8), 0x7126'8c68u);
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

TEST(LoongArch64AssemblerTest, EncodesCountLeadingZerosAndWordReverse) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.ClzW(Assembler::t0, Assembler::t1);
  assembler.ClzD(Assembler::t0, Assembler::t1);
  assembler.Revb2W(Assembler::t0, Assembler::t1);

  // Generated independently with LLVM's LoongArch assembler.
  constexpr std::array<uint32_t, 3> kExpected = {0x0000'15ac, 0x0000'25ac, 0x0000'39ac};
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, EncodesScalarFmaAndLsxImmediateShuffles) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.FmaddS(Assembler::vr2, Assembler::vr3, Assembler::vr4, Assembler::vr5);
  assembler.FmaddD(Assembler::vr2, Assembler::vr3, Assembler::vr4, Assembler::vr5);
  assembler.Vshuf4iB(Assembler::vr2, Assembler::vr3, 0x1b);
  assembler.Vshuf4iH(Assembler::vr2, Assembler::vr3, 0x1b);
  assembler.Vshuf4iW(Assembler::vr2, Assembler::vr3, 0xb1);

  // Generated independently with LLVM 21 llvm-mc for LoongArch64 with LSX.
  constexpr std::array<uint32_t, 5> kExpected = {
      0x0812'9062, 0x0822'9062, 0x7390'6c62, 0x7394'6c62, 0x739a'c462};
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, EncodesHotScalarConversions) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.FcvtDS(Assembler::vr2, Assembler::vr3);
  assembler.FtintrzLS(Assembler::vr2, Assembler::vr3);
  assembler.Movfr2grD(Assembler::t0, Assembler::vr2);
  assembler.VpcntB(Assembler::vr2, Assembler::vr3);

  // Generated independently with LLVM 21 llvm-mc for LoongArch64 and LSX.
  constexpr std::array<uint32_t, 4> kExpected = {
      0x0119'2462, 0x011a'a462, 0x0114'b84c, 0x729c'2062};
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, EncodesHotLsxConversionsAndComparisons) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.VffintSW(Assembler::vr2, Assembler::vr3);
  assembler.VfcmpCeqS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfcmpSleS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VseqB(Assembler::vr2, Assembler::vr3, Assembler::vr4);

  // Generated independently with LLVM 21 llvm-mc for LoongArch64 and LSX.
  constexpr std::array<uint32_t, 4> kExpected = {
      0x729e'0062, 0x0c52'1062, 0x0c53'9062, 0x7000'1062};
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, EncodesHotSignedMultiplyAndVectorAdd) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.MulhD(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.VaddD(Assembler::vr2, Assembler::vr3, Assembler::vr4);

  // Generated independently with LLVM 21 llvm-mc for LoongArch64 and LSX.
  constexpr std::array<uint32_t, 2> kExpected = {0x001e'39ac, 0x700b'9062};
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, EncodesLamSubwordCompareExchange) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.AmcasB(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.AmcasH(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.AmcasDbB(Assembler::t0, Assembler::t1, Assembler::t2);
  assembler.AmcasDbH(Assembler::t0, Assembler::t1, Assembler::t2);

  // Generated independently with LLVM 21 llvm-mc for LoongArch64 LAM_BH.
  constexpr std::array<uint32_t, 4> kExpected = {
      0x3858'35cc, 0x3858'b5cc, 0x385a'35cc, 0x385a'b5cc};
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  for (size_t i = 0; i < kExpected.size(); ++i) {
    EXPECT_EQ(*code.AddrAs<const uint32_t>(i * sizeof(uint32_t)), kExpected[i]) << i;
  }
}

TEST(LoongArch64AssemblerTest, EncodesStoreConditionalQuadword) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.ScQ(Assembler::t0, Assembler::t1, Assembler::t2);

  // Generated independently with LLVM 21 llvm-mc for LoongArch64 SCQ.
  constexpr uint32_t kExpected = 0x3857'35cc;
  ASSERT_EQ(code.install_size(), sizeof(kExpected));
  EXPECT_EQ(*code.AddrAs<const uint32_t>(0), kExpected);
}

TEST(LoongArch64AssemblerTest, EncodesLsxFloatingPointInstructions) {
  MachineCode code;
  Assembler assembler(&code);

  assembler.Vld(Assembler::vr0, Assembler::s8, 0);
  assembler.Vst(Assembler::vr1, Assembler::s8, 16);
  assembler.VfmulS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfmulD(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfaddS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfaddD(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfsubS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfsubD(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfdivS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VfdivD(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.VffintSWu(Assembler::vr2, Assembler::vr3);
  assembler.FmulS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.FmulD(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.FdivS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.FdivD(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.FaddS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.FaddD(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.FsubS(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.FsubD(Assembler::vr2, Assembler::vr3, Assembler::vr4);
  assembler.FtintrzWS(Assembler::vr0, Assembler::vr1);
  assembler.Movfr2grS(Assembler::t0, Assembler::vr0);
  assembler.Movgr2frW(Assembler::vr1, Assembler::t1);
  assembler.FcmpCunS(Assembler::vr1, Assembler::vr2);
  assembler.FcmpCeqS(Assembler::vr3, Assembler::vr4);
  assembler.FcmpCltS(Assembler::vr5, Assembler::vr6);
  assembler.Movcf2gr(Assembler::t0);
  assembler.VfmaddS(Assembler::vr5, Assembler::vr6, Assembler::vr7, Assembler::vr8);
  assembler.VxorV(Assembler::vr5, Assembler::vr6, Assembler::vr7);
  assembler.VaddW(Assembler::vr0, Assembler::vr1, Assembler::vr2);
  assembler.VsubW(Assembler::vr3, Assembler::vr4, Assembler::vr5);
  assembler.VmulW(Assembler::vr0, Assembler::vr1, Assembler::vr2);
  assembler.VmaddW(Assembler::vr3, Assembler::vr4, Assembler::vr5);
  assembler.VmsubW(Assembler::vr6, Assembler::vr7, Assembler::vr8);
  assembler.VbsrlV(Assembler::vr0, Assembler::vr1, 8);
  assembler.VsllwilWH(Assembler::vr2, Assembler::vr3, 15);
  assembler.VssrarniHW(Assembler::vr0, Assembler::vr1, 12);
  assembler.VpickevD(Assembler::vr3, Assembler::vr4, Assembler::vr5);
  assembler.Vreplgr2vrW(Assembler::vr1, Assembler::t0);
  assembler.VreplveiW(Assembler::vr2, Assembler::vr3, 0);
  assembler.VreplveiW(Assembler::vr4, Assembler::vr5, 3);

  // Generated independently with LLVM's LoongArch assembler and -mlsx.
  constexpr std::array<uint32_t, 40> kExpected = {
      0x2c00'03e0, 0x2c40'43e1, 0x7138'9062, 0x7139'1062, 0x7130'9062, 0x7131'1062,
      0x7132'9062, 0x7133'1062, 0x713a'9062, 0x713b'1062, 0x729e'0462, 0x0104'9062,
      0x0105'1062,
      0x0106'9062, 0x0107'1062, 0x0100'9062, 0x0101'1062, 0x0102'9062, 0x0103'1062,
      0x011a'8420, 0x0114'b40c, 0x0114'a5a1, 0x0c14'0820, 0x0c12'1060, 0x0c11'18a0,
      0x0114'dc0c, 0x0914'1cc5, 0x7127'1cc5, 0x700b'0820, 0x700d'1483, 0x7085'0820,
      0x70a9'1483, 0x70ab'20e6,
      0x728e'a020, 0x7308'7c62, 0x7368'b020, 0x711f'9483, 0x729f'0981, 0x72f7'e062,
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
