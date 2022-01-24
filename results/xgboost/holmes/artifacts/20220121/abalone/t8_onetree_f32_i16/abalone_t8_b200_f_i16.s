	.text
	.file	"LLVMDialectModule"
	.globl	Prediction_Function             # -- Begin function Prediction_Function
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
	movq	%rdi, %rax
	movq	80(%rsp), %r12
	xorl	%ecx, %ecx
	cmpq	$199, %rcx
	jg	.LBB0_3
	.p2align	4, 0x90
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	movl	$1056964608, (%r12,%rcx,4)      # imm = 0x3F000000
	incq	%rcx
	cmpq	$199, %rcx
	jle	.LBB0_2
.LBB0_3:
	xorl	%r15d, %r15d
	leaq	.Loffsets(%rip), %r14
	leaq	.Lmodel(%rip), %rbx
	leaq	.LlookupTable(%rip), %r10
	jmp	.LBB0_4
	.p2align	4, 0x90
.LBB0_11:                               #   in Loop: Header=BB0_4 Depth=1
	incq	%r15
.LBB0_4:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
                                        #       Child Loop BB0_8 Depth 3
	cmpq	$999, %r15                      # imm = 0x3E7
	jg	.LBB0_12
# %bb.5:                                #   in Loop: Header=BB0_4 Depth=1
	movq	(%r14,%r15,8), %r11
	xorl	%r13d, %r13d
	jmp	.LBB0_6
	.p2align	4, 0x90
.LBB0_10:                               #   in Loop: Header=BB0_6 Depth=2
	vmovss	(%r8,%rbx), %xmm0               # xmm0 = mem[0],zero,zero,zero
	vaddss	(%r12,%r13,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r12,%r13,4)
	incq	%r13
.LBB0_6:                                #   Parent Loop BB0_4 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_8 Depth 3
	cmpq	$199, %r13
	jg	.LBB0_11
# %bb.7:                                #   in Loop: Header=BB0_6 Depth=2
	leaq	(,%r13,8), %rsi
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_8:                                #   Parent Loop BB0_4 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	leaq	(%r11,%rbp), %r8
	shlq	$6, %r8
	movl	32(%r8,%rbx), %r9d
	cmpw	$-1, %r9w
	je	.LBB0_10
# %bb.9:                                #   in Loop: Header=BB0_8 Depth=3
	vpmovsxwd	32(%r8,%rbx), %ymm0
	leaq	(%rdx,%rsi,4), %rdi
	vxorps	%xmm1, %xmm1, %xmm1
	kxnorw	%k0, %k0, %k1
	vgatherdps	(%rdi,%ymm0,4), %ymm1 {%k1}
	movswq	48(%r8,%rbx), %rdi
	vcmpngeps	(%r8,%rbx), %ymm1, %k0
	leaq	(%rbp,%rbp,8), %rbp
	kmovb	%k0, %ecx
	shlq	$8, %rdi
	orq	%rcx, %rdi
	movsbq	(%r10,%rdi), %rcx
	addq	%rcx, %rbp
	incq	%rbp
	jmp	.LBB0_8
.LBB0_12:
	movq	72(%rsp), %rcx
	movq	%rcx, (%rax)
	movq	%r12, 8(%rax)
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
	movq	$91000, 24(%rdi)                # imm = 0x16378
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
	cmpq	$90999, %rdi                    # imm = 0x16377
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
	cmpq	$90999, %rdi                    # imm = 0x16377
	jle	.LBB4_2
.LBB4_3:
	movl	$5824000, %eax                  # imm = 0x58DE00
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
	.comm	.Lmodel,5824000,32
	.type	.Loffsets,@object               # @offsets
	.local	.Loffsets
	.comm	.Loffsets,8000,16
	.type	.Llengths,@object               # @lengths
	.local	.Llengths
	.comm	.Llengths,8000,16
	.section	".note.GNU-stack","",@progbits
