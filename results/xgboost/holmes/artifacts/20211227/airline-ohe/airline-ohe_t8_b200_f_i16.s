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
	movq	%rdi, 8(%rsp)                   # 8-byte Spill
	xorl	%r13d, %r13d
	leaq	.Loffsets(%rip), %r12
	leaq	.Lmodel(%rip), %r15
	leaq	.LlookupTable(%rip), %r14
	vbroadcastss	.LCPI0_0(%rip), %xmm0   # xmm0 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	vmovaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_8:                                #   in Loop: Header=BB0_1 Depth=1
	vxorps	16(%rsp), %xmm0, %xmm0          # 16-byte Folded Reload
	vzeroupper
	callq	expf@PLT
	vmovss	.LCPI0_1(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vaddss	%xmm1, %xmm0, %xmm0
	vdivss	%xmm0, %xmm1, %xmm0
	movq	120(%rsp), %rax
	vmovss	%xmm0, (%rax,%r13,4)
	incq	%r13
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_5 Depth 3
	cmpq	$199, %r13
	jg	.LBB0_9
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	imulq	$692, %r13, %rax                # imm = 0x2B4
	xorl	%ecx, %ecx
	vmovss	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero,zero,zero
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_7:                                #   in Loop: Header=BB0_3 Depth=2
	vaddss	(%rdi,%r15), %xmm0, %xmm0
	incq	%rcx
.LBB0_3:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_5 Depth 3
	cmpq	$999, %rcx                      # imm = 0x3E7
	jg	.LBB0_8
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	movq	(%r12,%rcx,8), %rdx
	xorl	%esi, %esi
	.p2align	4, 0x90
.LBB0_5:                                #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	leaq	(%rdx,%rsi), %rdi
	shlq	$6, %rdi
	movl	32(%rdi,%r15), %ebp
	cmpw	$-1, %bp
	je	.LBB0_7
# %bb.6:                                #   in Loop: Header=BB0_5 Depth=3
	vpmovsxwd	32(%rdi,%r15), %ymm1
	leaq	(%rbx,%rax,4), %rbp
	kxnorw	%k0, %k0, %k1
	vxorps	%xmm2, %xmm2, %xmm2
	vgatherdps	(%rbp,%ymm1,4), %ymm2 {%k1}
	movswq	48(%rdi,%r15), %rbp
	vcmpngtps	(%rdi,%r15), %ymm2, %k0
	leaq	(%rsi,%rsi,8), %rsi
	kmovb	%k0, %edi
	shlq	$8, %rbp
	orq	%rdi, %rbp
	movsbq	(%r14,%rbp), %rdi
	addq	%rdi, %rsi
	incq	%rsi
	jmp	.LBB0_5
.LBB0_9:
	movq	8(%rsp), %rax                   # 8-byte Reload
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
	movq	$820000, 24(%rdi)               # imm = 0xC8320
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
	.globl	Init_model                      # -- Begin function Init_model
	.p2align	4, 0x90
	.type	Init_model,@function
Init_model:                             # @Init_model
	.cfi_startproc
# %bb.0:
	movq	48(%rsp), %r8
	movq	8(%rsp), %r9
	movl	$6, %edx
	xorl	%edi, %edi
	leaq	.Lmodel(%rip), %rax
	cmpq	$819999, %rdi                   # imm = 0xC831F
	jg	.LBB4_3
	.p2align	4, 0x90
.LBB4_2:                                # =>This Inner Loop Header: Depth=1
	vmovups	-24(%rsi,%rdx,4), %ymm0
	vmovups	-12(%r9,%rdx,2), %xmm1
	movzwl	(%r8,%rdi,2), %ecx
	vmovaps	%ymm0, -48(%rax,%rdx,8)
	vmovaps	%xmm1, -16(%rax,%rdx,8)
	movw	%cx, (%rax,%rdx,8)
	incq	%rdi
	addq	$8, %rdx
	cmpq	$819999, %rdi                   # imm = 0xC831F
	jle	.LBB4_2
.LBB4_3:
	movl	$52480000, %eax                 # imm = 0x320C800
	vzeroupper
	retq
.Lfunc_end4:
	.size	Init_model, .Lfunc_end4-Init_model
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5                               # -- Begin function Get_lookupTable
.LCPI5_0:
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
	vmovaps	.LCPI5_0(%rip), %ymm0           # ymm0 = [0,1430,256,256]
	vmovups	%ymm0, 16(%rdi)
	leaq	.LlookupTable(%rip), %rcx
	movq	%rcx, 8(%rdi)
	movl	$3735928559, %ecx               # imm = 0xDEADBEEF
	movq	%rcx, (%rdi)
	movq	$1, 48(%rdi)
	vzeroupper
	retq
.Lfunc_end5:
	.size	Get_lookupTable, .Lfunc_end5-Get_lookupTable
	.cfi_endproc
                                        # -- End function
	.type	.LlookupTable,@object           # @lookupTable
	.local	.LlookupTable
	.comm	.LlookupTable,366080,16
	.type	.Lmodel,@object                 # @model
	.local	.Lmodel
	.comm	.Lmodel,52480000,32
	.type	.Loffsets,@object               # @offsets
	.local	.Loffsets
	.comm	.Loffsets,8000,16
	.type	.Llengths,@object               # @lengths
	.local	.Llengths
	.comm	.Llengths,8000,16
	.section	".note.GNU-stack","",@progbits
