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
  assembler.AddiD(Assembler::a0, Assembler::a1, -16);
  assembler.Lu12iW(Assembler::a0, 0x12345);
  assembler.Ori(Assembler::a0, Assembler::a0, 0x678);
  assembler.Lu32iD(Assembler::a0, 0x12345);
  assembler.Lu52iD(Assembler::a0, Assembler::a0, 0x123);
  assembler.LdD(Assembler::a0, Assembler::a1, 24);
  assembler.StD(Assembler::a0, Assembler::a1, -8);
  assembler.Beq(Assembler::a0, Assembler::a1, 8);
  assembler.Bne(Assembler::a0, Assembler::a1, 8);
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
  constexpr std::array<uint32_t, 22> kExpected = {
      0x001098a4, 0x0011b9ac, 0x0014e717, 0x001500a4, 0x001598a4, 0x02ffc0a4,
      0x142468a4, 0x0399e084, 0x162468a4, 0x03048c84, 0x28c060a4, 0x29ffe0a4,
      0x58000885, 0x5c000885, 0x40000880, 0x44000880, 0x50000800, 0x54000800,
      0x4c000081, 0x00411ca4, 0x00451ca4, 0x00491ca4,
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

}  // namespace
}  // namespace berberis::loongarch64
