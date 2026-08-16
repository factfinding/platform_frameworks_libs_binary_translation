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

    .text

    // Keep the complete table in CFI shadow slots owned by
    // libberberis_arm64.so.  Each entry is exactly 16 bytes.
    .p2align 18
    .globl __berberis_ffi_trampoline_table
    .hidden __berberis_ffi_trampoline_table
    .type __berberis_ffi_trampoline_table, @function
__berberis_ffi_trampoline_table:
    .rept 16384
        pcaddi  $t0, 0
        b       __berberis_ffi_closure_dispatch
        nop
        nop
    .endr
    .size __berberis_ffi_trampoline_table, .-__berberis_ffi_trampoline_table

    .p2align 2
    .hidden __berberis_ffi_closure_dispatch
    .type __berberis_ffi_closure_dispatch, @function
__berberis_ffi_closure_dispatch:
    // t0 contains the address of the selected trampoline.  Preserve all
    // callback argument registers and use only the ABI scratch registers t0
    // and t1 while resolving the corresponding ffi_closure.
    la.local $t1, __berberis_ffi_trampoline_table
    sub.d    $t0, $t0, $t1
    srli.d   $t0, $t0, 4

    la.local $t1, __berberis_ffi_closure_slots
    slli.d   $t0, $t0, 3
    add.d    $t1, $t1, $t0
    ld.d     $t0, $t1, 0
    dbar     0x14
    beqz     $t0, 1f

    // libffi's LoongArch64 closure entry expects t0 to point at the
    // initialized ffi_closure and leaves the original callback arguments in
    // their ABI-defined registers.
    b        ffi_closure_asm

1:
    break    0
    .size __berberis_ffi_closure_dispatch, .-__berberis_ffi_closure_dispatch

    .section .note.GNU-stack,"",@progbits
