	.text
	.file	"LLVMDialectModule"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function Prediction_Function
.LCPI0_0:
	.long	0x80000000                      # float -0
.LCPI0_1:
	.long	0x3f800000                      # float 1
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
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %rbx
	movq	%rdi, (%rsp)                    # 8-byte Spill
	xorl	%eax, %eax
	leaq	.Loffsets(%rip), %r10
	leaq	.Llengths(%rip), %r15
	leaq	.LleavesOffsets(%rip), %r14
	leaq	.Lmodel(%rip), %r12
	leaq	.LlookupTable(%rip), %r13
	leaq	.Lleaves(%rip), %r11
	vbroadcastss	.LCPI0_0(%rip), %xmm0   # xmm0 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	vmovaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_13:                               #   in Loop: Header=BB0_1 Depth=1
	vxorps	16(%rsp), %xmm0, %xmm0          # 16-byte Folded Reload
	movq	%r11, %rbp
	vzeroupper
	callq	expf@PLT
	movq	%rbp, %r11
	leaq	.Loffsets(%rip), %r10
	vmovss	.LCPI0_1(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vaddss	%xmm1, %xmm0, %xmm0
	vdivss	%xmm0, %xmm1, %xmm0
	movq	120(%rsp), %rax
	movq	8(%rsp), %rcx                   # 8-byte Reload
	vmovss	%xmm0, (%rax,%rcx,4)
	movq	%rcx, %rax
	incq	%rax
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_7 Depth 3
	cmpq	$199, %rax
	jg	.LBB0_14
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	imulq	$692, %rax, %rax                # imm = 0x2B4
	xorl	%r9d, %r9d
	vmovss	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero,zero,zero
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_11:                               #   in Loop: Header=BB0_3 Depth=2
	addq	%rcx, %r8
	vmovss	(%r11,%r8,4), %xmm1             # xmm1 = mem[0],zero,zero,zero
.LBB0_12:                               #   in Loop: Header=BB0_3 Depth=2
	vaddss	%xmm1, %xmm0, %xmm0
	incq	%r9
.LBB0_3:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_7 Depth 3
	cmpq	$999, %r9                       # imm = 0x3E7
	jg	.LBB0_13
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	movq	(%r10,%r9,8), %rdx
	movq	(%r15,%r9,8), %rdi
	movq	(%r14,%r9,8), %r8
	xorl	%ebp, %ebp
	movb	$1, %cl
	cmpq	%rdi, %rbp
	jge	.LBB0_7
	.p2align	4, 0x90
.LBB0_6:                                #   in Loop: Header=BB0_3 Depth=2
	leaq	(%rdx,%rbp), %rcx
	shlq	$6, %rcx
	movl	32(%rcx,%r12), %ecx
	cmpw	$-1, %cx
	sete	%cl
.LBB0_7:                                #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	testb	%cl, %cl
	jne	.LBB0_9
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=3
	addq	%rdx, %rbp
	shlq	$6, %rbp
	vpmovsxwd	32(%rbp,%r12), %ymm1
	leaq	(%rbx,%rax,4), %rcx
	kxnorw	%k0, %k0, %k1
	vxorps	%xmm2, %xmm2, %xmm2
	vgatherdps	(%rcx,%ymm1,4), %ymm2 {%k1}
	movswq	48(%rbp,%r12), %rcx
	vcmpngtps	(%rbp,%r12), %ymm2, %k0
	movswq	50(%rbp,%r12), %rsi
	kmovb	%k0, %ebp
	shlq	$8, %rcx
	orq	%rbp, %rcx
	movsbq	(%r13,%rcx), %rbp
	addq	%rsi, %rbp
	movb	$1, %cl
	cmpq	%rdi, %rbp
	jl	.LBB0_6
	jmp	.LBB0_7
	.p2align	4, 0x90
.LBB0_9:                                #   in Loop: Header=BB0_3 Depth=2
	movq	%rbp, %rcx
	subq	%rdi, %rcx
	jge	.LBB0_11
# %bb.10:                               #   in Loop: Header=BB0_3 Depth=2
	addq	%rbp, %rdx
	shlq	$6, %rdx
	vmovss	(%rdx,%r12), %xmm1              # xmm1 = mem[0],zero,zero,zero
	jmp	.LBB0_12
.LBB0_14:
	movq	(%rsp), %rax                    # 8-byte Reload
	movq	112(%rsp), %rcx
	movq	%rcx, (%rax)
	movq	120(%rsp), %rcx
	movq	%rcx, 8(%rax)
	movq	128(%rsp), %rcx
	movq	%rcx, 16(%rax)
	movq	136(%rsp), %rcx
	movq	%rcx, 24(%rax)
	movq	144(%rsp), %rcx
	movq	%rcx, 32(%rax)
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
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
	movq	$65755, 24(%rdi)                # imm = 0x100DB
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
	movq	$351702, 24(%rdi)               # imm = 0x55DD6
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
	cmpq	$65754, %rcx                    # imm = 0x100DA
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
	cmpq	$65754, %rcx                    # imm = 0x100DA
	jle	.LBB7_2
.LBB7_3:
	movl	$4208320, %eax                  # imm = 0x4036C0
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
	.comm	.Lmodel,4208320,32
	.type	.Loffsets,@object               # @offsets
	.local	.Loffsets
	.comm	.Loffsets,8000,16
	.type	.Llengths,@object               # @lengths
	.local	.Llengths
	.comm	.Llengths,8000,16
	.type	.Lleaves,@object                # @leaves
	.local	.Lleaves
	.comm	.Lleaves,1406808,16
	.type	.LleavesOffsets,@object         # @leavesOffsets
	.local	.LleavesOffsets
	.comm	.LleavesOffsets,8000,16
	.type	.LleavesLengths,@object         # @leavesLengths
	.local	.LleavesLengths
	.comm	.LleavesLengths,8000,16
	.section	".note.GNU-stack","",@progbits
