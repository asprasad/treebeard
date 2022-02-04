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
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %r15
	movq	104(%rsp), %r12
	xorl	%eax, %eax
	cmpq	$199, %rax
	jg	.LBB0_3
	.p2align	4, 0x90
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	movl	$-2147483648, (%r12,%rax,4)     # imm = 0x80000000
	incq	%rax
	cmpq	$199, %rax
	jle	.LBB0_2
.LBB0_3:
	xorl	%r10d, %r10d
	leaq	.Loffsets(%rip), %r9
	leaq	.Lmodel(%rip), %rsi
	leaq	.LlookupTable(%rip), %r14
	jmp	.LBB0_4
	.p2align	4, 0x90
.LBB0_14:                               #   in Loop: Header=BB0_4 Depth=1
	addq	$10, %r10
.LBB0_4:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
                                        #       Child Loop BB0_8 Depth 3
                                        #         Child Loop BB0_10 Depth 4
	cmpq	$99, %r10
	jg	.LBB0_15
# %bb.5:                                #   in Loop: Header=BB0_4 Depth=1
	xorl	%r8d, %r8d
	jmp	.LBB0_6
	.p2align	4, 0x90
.LBB0_13:                               #   in Loop: Header=BB0_6 Depth=2
	vaddss	(%r12,%r8,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r12,%r8,4)
	incq	%r8
.LBB0_6:                                #   Parent Loop BB0_4 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_8 Depth 3
                                        #         Child Loop BB0_10 Depth 4
	cmpq	$199, %r8
	jg	.LBB0_14
# %bb.7:                                #   in Loop: Header=BB0_6 Depth=2
	leaq	(%r8,%r8,2), %rax
	leaq	(%r8,%rax,4), %rbx
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%r11d, %r11d
	jmp	.LBB0_8
	.p2align	4, 0x90
.LBB0_12:                               #   in Loop: Header=BB0_8 Depth=3
	vaddss	(%rbp,%rsi), %xmm0, %xmm0
	incq	%r11
.LBB0_8:                                #   Parent Loop BB0_4 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_10 Depth 4
	cmpq	$9, %r11
	jg	.LBB0_13
# %bb.9:                                #   in Loop: Header=BB0_8 Depth=3
	leaq	(%r10,%r11), %rax
	movq	(%r9,%rax,8), %rcx
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_10:                               #   Parent Loop BB0_4 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        #       Parent Loop BB0_8 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	leaq	(%rcx,%rax), %rbp
	shlq	$6, %rbp
	movl	32(%rbp,%rsi), %r13d
	cmpw	$-1, %r13w
	je	.LBB0_12
# %bb.11:                               #   in Loop: Header=BB0_10 Depth=4
	vpmovsxwd	32(%rbp,%rsi), %ymm1
	leaq	(%rdx,%rbx,4), %rdi
	kxnorw	%k0, %k0, %k1
	vxorps	%xmm2, %xmm2, %xmm2
	vgatherdps	(%rdi,%ymm1,4), %ymm2 {%k1}
	movswq	48(%rbp,%rsi), %rdi
	vcmpngeps	(%rbp,%rsi), %ymm2, %k0
	leaq	(%rax,%rax,8), %rax
	kmovb	%k0, %ebp
	shlq	$8, %rdi
	orq	%rbp, %rdi
	movsbq	(%r14,%rdi), %rdi
	addq	%rdi, %rax
	incq	%rax
	jmp	.LBB0_10
.LBB0_15:
	xorl	%ebx, %ebx
	vbroadcastss	.LCPI0_0(%rip), %xmm0   # xmm0 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	vmovaps	%xmm0, (%rsp)                   # 16-byte Spill
	cmpq	$199, %rbx
	jg	.LBB0_18
	.p2align	4, 0x90
.LBB0_17:                               # =>This Inner Loop Header: Depth=1
	vmovss	(%r12,%rbx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	vxorps	(%rsp), %xmm0, %xmm0            # 16-byte Folded Reload
	vzeroupper
	callq	expf@PLT
	vmovss	.LCPI0_1(%rip), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vaddss	%xmm1, %xmm0, %xmm0
	vdivss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, (%r12,%rbx,4)
	incq	%rbx
	cmpq	$199, %rbx
	jle	.LBB0_17
.LBB0_18:
	movq	96(%rsp), %rax
	movq	%rax, (%r15)
	movq	%r12, 8(%r15)
	movq	112(%rsp), %rax
	movq	%rax, 16(%r15)
	movq	120(%rsp), %rax
	movq	%rax, 24(%r15)
	movq	128(%rsp), %rax
	movq	%rax, 32(%r15)
	movq	%r15, %rax
	addq	$24, %rsp
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
	movq	$82000, 24(%rdi)                # imm = 0x14050
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
	movq	$100, 24(%rdi)
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
	movq	$100, 24(%rdi)
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
	cmpq	$81999, %rdi                    # imm = 0x1404F
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
	cmpq	$81999, %rdi                    # imm = 0x1404F
	jle	.LBB4_2
.LBB4_3:
	movl	$5248000, %eax                  # imm = 0x501400
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
	.comm	.Lmodel,5248000,32
	.type	.Loffsets,@object               # @offsets
	.local	.Loffsets
	.comm	.Loffsets,800,16
	.type	.Llengths,@object               # @lengths
	.local	.Llengths
	.comm	.Llengths,800,16
	.section	".note.GNU-stack","",@progbits
