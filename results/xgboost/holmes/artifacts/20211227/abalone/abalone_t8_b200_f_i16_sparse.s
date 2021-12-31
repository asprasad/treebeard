	.text
	.file	"LLVMDialectModule"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function Prediction_Function
.LCPI0_0:
	.long	0x3f000000                      # float 0.5
	.text
	.globl	Prediction_Function
	.p2align	4, 0x90
	.type	Prediction_Function,@function
Prediction_Function:                    # @Prediction_Function
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdi, -16(%rsp)                 # 8-byte Spill
	xorl	%ebp, %ebp
	vmovss	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero,zero,zero
	leaq	.Loffsets(%rip), %r12
	leaq	.Llengths(%rip), %r13
	leaq	.LleavesOffsets(%rip), %rsi
	leaq	.Lmodel(%rip), %rcx
	leaq	.LlookupTable(%rip), %rdi
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_13:                               #   in Loop: Header=BB0_1 Depth=1
	movq	80(%rsp), %rax
	movq	-8(%rsp), %rbp                  # 8-byte Reload
	vmovss	%xmm1, (%rax,%rbp,4)
	incq	%rbp
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_7 Depth 3
	cmpq	$199, %rbp
	jg	.LBB0_14
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	movq	%rbp, -8(%rsp)                  # 8-byte Spill
	leaq	(,%rbp,8), %rbp
	xorl	%r8d, %r8d
	vmovaps	%xmm0, %xmm1
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_11:                               #   in Loop: Header=BB0_3 Depth=2
	addq	%rax, %r10
	leaq	.Lleaves(%rip), %rax
	vmovss	(%rax,%r10,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
.LBB0_12:                               #   in Loop: Header=BB0_3 Depth=2
	vaddss	%xmm2, %xmm1, %xmm1
	incq	%r8
.LBB0_3:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_7 Depth 3
	cmpq	$999, %r8                       # imm = 0x3E7
	jg	.LBB0_13
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	movq	(%r12,%r8,8), %r9
	movq	(%r13,%r8,8), %r11
	movq	(%rsi,%r8,8), %r10
	xorl	%r14d, %r14d
	movb	$1, %al
	cmpq	%r11, %r14
	jge	.LBB0_7
	.p2align	4, 0x90
.LBB0_6:                                #   in Loop: Header=BB0_3 Depth=2
	leaq	(%r9,%r14), %rax
	shlq	$6, %rax
	movl	32(%rax,%rcx), %eax
	cmpw	$-1, %ax
	sete	%al
.LBB0_7:                                #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	testb	%al, %al
	jne	.LBB0_9
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=3
	addq	%r9, %r14
	shlq	$6, %r14
	vpmovsxwd	32(%r14,%rcx), %ymm2
	leaq	(%rdx,%rbp,4), %rax
	kxnorw	%k0, %k0, %k1
	vxorps	%xmm3, %xmm3, %xmm3
	vgatherdps	(%rax,%ymm2,4), %ymm3 {%k1}
	movswq	48(%r14,%rcx), %rax
	vcmpngtps	(%r14,%rcx), %ymm3, %k0
	movswq	50(%r14,%rcx), %r15
	kmovb	%k0, %ebx
	shlq	$8, %rax
	orq	%rbx, %rax
	movsbq	(%rdi,%rax), %r14
	addq	%r15, %r14
	movb	$1, %al
	cmpq	%r11, %r14
	jl	.LBB0_6
	jmp	.LBB0_7
	.p2align	4, 0x90
.LBB0_9:                                #   in Loop: Header=BB0_3 Depth=2
	movq	%r14, %rax
	subq	%r11, %rax
	jge	.LBB0_11
# %bb.10:                               #   in Loop: Header=BB0_3 Depth=2
	addq	%r14, %r9
	shlq	$6, %r9
	vmovss	(%r9,%rcx), %xmm2               # xmm2 = mem[0],zero,zero,zero
	jmp	.LBB0_12
.LBB0_14:
	movq	-16(%rsp), %rax                 # 8-byte Reload
	movq	72(%rsp), %rcx
	movq	%rcx, (%rax)
	movq	80(%rsp), %rcx
	movq	%rcx, 8(%rax)
	movq	88(%rsp), %rcx
	movq	%rcx, 16(%rax)
	movq	96(%rsp), %rcx
	movq	%rcx, 24(%rax)
	movq	104(%rsp), %rcx
	movq	%rcx, 32(%rax)
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.Lfunc_end0:
	.size	Prediction_Function, .Lfunc_end0-Prediction_Function
	.cfi_endproc
                                        # -- End function
	.globl	Get_model                       # -- Begin function Get_model
	.p2align	4, 0x90
	.type	Get_model,@function
Get_model:                              # @Get_model
	.cfi_startproc
# %bb.0:
	movq	%rdi, %rax
	leaq	.Lmodel(%rip), %rcx
	movq	%rcx, 8(%rdi)
	movl	$3735928559, %ecx               # imm = 0xDEADBEEF
	movq	%rcx, (%rdi)
	movq	$1, 32(%rdi)
	movq	$10000, 24(%rdi)                # imm = 0x2710
	movq	$0, 16(%rdi)
	retq
.Lfunc_end1:
	.size	Get_model, .Lfunc_end1-Get_model
	.cfi_endproc
                                        # -- End function
	.globl	Get_offsets                     # -- Begin function Get_offsets
	.p2align	4, 0x90
	.type	Get_offsets,@function
Get_offsets:                            # @Get_offsets
	.cfi_startproc
# %bb.0:
	movq	%rdi, %rax
	leaq	.Loffsets(%rip), %rcx
	movq	%rcx, 8(%rdi)
	movl	$3735928559, %ecx               # imm = 0xDEADBEEF
	movq	%rcx, (%rdi)
	movq	$1, 32(%rdi)
	movq	$1000, 24(%rdi)                 # imm = 0x3E8
	movq	$0, 16(%rdi)
	retq
.Lfunc_end2:
	.size	Get_offsets, .Lfunc_end2-Get_offsets
	.cfi_endproc
                                        # -- End function
	.globl	Get_lengths                     # -- Begin function Get_lengths
	.p2align	4, 0x90
	.type	Get_lengths,@function
Get_lengths:                            # @Get_lengths
	.cfi_startproc
# %bb.0:
	movq	%rdi, %rax
	leaq	.Llengths(%rip), %rcx
	movq	%rcx, 8(%rdi)
	movl	$3735928559, %ecx               # imm = 0xDEADBEEF
	movq	%rcx, (%rdi)
	movq	$1, 32(%rdi)
	movq	$1000, 24(%rdi)                 # imm = 0x3E8
	movq	$0, 16(%rdi)
	retq
.Lfunc_end3:
	.size	Get_lengths, .Lfunc_end3-Get_lengths
	.cfi_endproc
                                        # -- End function
	.globl	Get_leaves                      # -- Begin function Get_leaves
	.p2align	4, 0x90
	.type	Get_leaves,@function
Get_leaves:                             # @Get_leaves
	.cfi_startproc
# %bb.0:
	movq	%rdi, %rax
	leaq	.Lleaves(%rip), %rcx
	movq	%rcx, 8(%rdi)
	movl	$3735928559, %ecx               # imm = 0xDEADBEEF
	movq	%rcx, (%rdi)
	movq	$1, 32(%rdi)
	movq	$72243, 24(%rdi)                # imm = 0x11A33
	movq	$0, 16(%rdi)
	retq
.Lfunc_end4:
	.size	Get_leaves, .Lfunc_end4-Get_leaves
	.cfi_endproc
                                        # -- End function
	.globl	Get_leavesOffsets               # -- Begin function Get_leavesOffsets
	.p2align	4, 0x90
	.type	Get_leavesOffsets,@function
Get_leavesOffsets:                      # @Get_leavesOffsets
	.cfi_startproc
# %bb.0:
	movq	%rdi, %rax
	leaq	.LleavesOffsets(%rip), %rcx
	movq	%rcx, 8(%rdi)
	movl	$3735928559, %ecx               # imm = 0xDEADBEEF
	movq	%rcx, (%rdi)
	movq	$1, 32(%rdi)
	movq	$1000, 24(%rdi)                 # imm = 0x3E8
	movq	$0, 16(%rdi)
	retq
.Lfunc_end5:
	.size	Get_leavesOffsets, .Lfunc_end5-Get_leavesOffsets
	.cfi_endproc
                                        # -- End function
	.globl	Get_leavesLengths               # -- Begin function Get_leavesLengths
	.p2align	4, 0x90
	.type	Get_leavesLengths,@function
Get_leavesLengths:                      # @Get_leavesLengths
	.cfi_startproc
# %bb.0:
	movq	%rdi, %rax
	leaq	.LleavesLengths(%rip), %rcx
	movq	%rcx, 8(%rdi)
	movl	$3735928559, %ecx               # imm = 0xDEADBEEF
	movq	%rcx, (%rdi)
	movq	$1, 32(%rdi)
	movq	$1000, 24(%rdi)                 # imm = 0x3E8
	movq	$0, 16(%rdi)
	retq
.Lfunc_end6:
	.size	Get_leavesLengths, .Lfunc_end6-Get_leavesLengths
	.cfi_endproc
                                        # -- End function
	.globl	Init_model                      # -- Begin function Init_model
	.p2align	4, 0x90
	.type	Init_model,@function
Init_model:                             # @Init_model
	.cfi_startproc
# %bb.0:
	movq	88(%rsp), %r8
	movq	48(%rsp), %r9
	movq	8(%rsp), %r10
	xorl	%edi, %edi
	leaq	.Lmodel(%rip), %r11
	xorl	%ecx, %ecx
	cmpq	$9999, %rcx                     # imm = 0x270F
	jg	.LBB7_3
	.p2align	4, 0x90
.LBB7_2:                                # =>This Inner Loop Header: Depth=1
	vmovups	(%rsi,%rdi,2), %ymm0
	vmovups	(%r10,%rdi), %xmm1
	movzwl	(%r9,%rcx,2), %edx
	movzwl	(%r8,%rcx,2), %eax
	vmovaps	%ymm0, (%r11,%rdi,4)
	vmovaps	%xmm1, 32(%r11,%rdi,4)
	movw	%dx, 48(%r11,%rdi,4)
	movw	%ax, 50(%r11,%rdi,4)
	incq	%rcx
	addq	$16, %rdi
	cmpq	$9999, %rcx                     # imm = 0x270F
	jle	.LBB7_2
.LBB7_3:
	movl	$640000, %eax                   # imm = 0x9C400
	vzeroupper
	retq
.Lfunc_end7:
	.size	Init_model, .Lfunc_end7-Init_model
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5                               # -- Begin function Get_lookupTable
.LCPI8_0:
	.quad	0                               # 0x0
	.quad	1430                            # 0x596
	.quad	256                             # 0x100
	.quad	256                             # 0x100
	.text
	.globl	Get_lookupTable
	.p2align	4, 0x90
	.type	Get_lookupTable,@function
Get_lookupTable:                        # @Get_lookupTable
	.cfi_startproc
# %bb.0:
	movq	%rdi, %rax
	vmovaps	.LCPI8_0(%rip), %ymm0           # ymm0 = [0,1430,256,256]
	vmovups	%ymm0, 16(%rdi)
	leaq	.LlookupTable(%rip), %rcx
	movq	%rcx, 8(%rdi)
	movl	$3735928559, %ecx               # imm = 0xDEADBEEF
	movq	%rcx, (%rdi)
	movq	$1, 48(%rdi)
	vzeroupper
	retq
.Lfunc_end8:
	.size	Get_lookupTable, .Lfunc_end8-Get_lookupTable
	.cfi_endproc
                                        # -- End function
	.type	.LlookupTable,@object           # @lookupTable
	.local	.LlookupTable
	.comm	.LlookupTable,366080,16
	.type	.Lmodel,@object                 # @model
	.local	.Lmodel
	.comm	.Lmodel,640000,32
	.type	.Loffsets,@object               # @offsets
	.local	.Loffsets
	.comm	.Loffsets,8000,16
	.type	.Llengths,@object               # @lengths
	.local	.Llengths
	.comm	.Llengths,8000,16
	.type	.Lleaves,@object                # @leaves
	.local	.Lleaves
	.comm	.Lleaves,288972,16
	.type	.LleavesOffsets,@object         # @leavesOffsets
	.local	.LleavesOffsets
	.comm	.LleavesOffsets,8000,16
	.type	.LleavesLengths,@object         # @leavesLengths
	.local	.LleavesLengths
	.comm	.LleavesLengths,8000,16
	.section	".note.GNU-stack","",@progbits
