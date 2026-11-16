00000000000037c0 <offset4_compat_dot(int, offset4_iq3s_block const*, block_q8_K const*)>:
    37c0:	55                   	push   %rbp
    37c1:	44 8d 87 ff 00 00 00 	lea    0xff(%rdi),%r8d
    37c8:	48 89 e5             	mov    %rsp,%rbp
    37cb:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    37cf:	48 83 ec 08          	sub    $0x8,%rsp
    37d3:	85 ff                	test   %edi,%edi
    37d5:	44 0f 49 c7          	cmovns %edi,%r8d
    37d9:	41 c1 f8 08          	sar    $0x8,%r8d
    37dd:	81 ff ff 00 00 00    	cmp    $0xff,%edi
    37e3:	0f 8e 57 04 00 00    	jle    3c40 <offset4_compat_dot(int, offset4_iq3s_block const*, block_q8_K const*)+0x480>
    37e9:	48 89 d0             	mov    %rdx,%rax
    37ec:	ba 0f 0f 0f 0f       	mov    $0xf0f0f0f,%edx
    37f1:	c5 c0 57 ff          	vxorps %xmm7,%xmm7,%xmm7
    37f5:	48 89 f1             	mov    %rsi,%rcx
    37f8:	c5 79 6f 2d 50 2b 00 	vmovdqa 0x2b50(%rip),%xmm13        # 6350 <iq3s_grid+0x850>
    37ff:	00 
    3800:	c5 fc 29 7c 24 e8    	vmovaps %ymm7,-0x18(%rsp)
    3806:	31 f6                	xor    %esi,%esi
    3808:	c5 f9 6e fa          	vmovd  %edx,%xmm7
    380c:	c5 79 6f 35 fc 2a 00 	vmovdqa 0x2afc(%rip),%xmm14        # 6310 <iq3s_grid+0x810>
    3813:	00 
    3814:	c5 79 70 ff 00       	vpshufd $0x0,%xmm7,%xmm15
    3819:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3820:	0f b6 b9 84 00 00 00 	movzbl 0x84(%rcx),%edi
    3827:	83 c6 01             	add    $0x1,%esi
    382a:	48 81 c1 90 00 00 00 	add    $0x90,%rcx
    3831:	48 05 24 01 00 00    	add    $0x124,%rax
    3837:	c5 fa 6f 81 70 ff ff 	vmovdqu -0x90(%rcx),%xmm0
    383e:	ff 
    383f:	c5 fa 6f a0 e0 fe ff 	vmovdqu -0x120(%rax),%xmm4
    3846:	ff 
    3847:	c5 fa 6f 98 f0 fe ff 	vmovdqu -0x110(%rax),%xmm3
    384e:	ff 
    384f:	c4 c1 79 db cf       	vpand  %xmm15,%xmm0,%xmm1
    3854:	c5 f9 71 d0 04       	vpsrlw $0x4,%xmm0,%xmm0
    3859:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    385e:	89 fa                	mov    %edi,%edx
    3860:	40 c0 ef 04          	shr    $0x4,%dil
    3864:	c4 e2 59 08 e1       	vpsignb %xmm1,%xmm4,%xmm4
    3869:	c4 c1 79 db c7       	vpand  %xmm15,%xmm0,%xmm0
    386e:	83 e2 0f             	and    $0xf,%edx
    3871:	40 0f b6 ff          	movzbl %dil,%edi
    3875:	c4 e2 11 00 c0       	vpshufb %xmm0,%xmm13,%xmm0
    387a:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    387f:	8d 54 12 01          	lea    0x1(%rdx,%rdx,1),%edx
    3883:	c4 e2 71 04 cc       	vpmaddubsw %xmm4,%xmm1,%xmm1
    3888:	c4 e2 61 08 d8       	vpsignb %xmm0,%xmm3,%xmm3
    388d:	c4 e2 79 1c c0       	vpabsb %xmm0,%xmm0
    3892:	c5 f9 6e d2          	vmovd  %edx,%xmm2
    3896:	c4 e2 79 04 c3       	vpmaddubsw %xmm3,%xmm0,%xmm0
    389b:	c5 7a 6f 50 a0       	vmovdqu -0x60(%rax),%xmm10
    38a0:	8d 54 3f 01          	lea    0x1(%rdi,%rdi,1),%edx
    38a4:	c4 c2 69 00 d6       	vpshufb %xmm14,%xmm2,%xmm2
    38a9:	c5 f1 f5 f2          	vpmaddwd %xmm2,%xmm1,%xmm6
    38ad:	c5 fa 6f 49 80       	vmovdqu -0x80(%rcx),%xmm1
    38b2:	c5 f9 f5 ea          	vpmaddwd %xmm2,%xmm0,%xmm5
    38b6:	c5 f9 6e c2          	vmovd  %edx,%xmm0
    38ba:	0f b6 51 f5          	movzbl -0xb(%rcx),%edx
    38be:	c5 f9 7f 74 24 d8    	vmovdqa %xmm6,-0x28(%rsp)
    38c4:	c4 c1 71 db d7       	vpand  %xmm15,%xmm1,%xmm2
    38c9:	c5 f1 71 d1 04       	vpsrlw $0x4,%xmm1,%xmm1
    38ce:	c4 c2 79 00 c6       	vpshufb %xmm14,%xmm0,%xmm0
    38d3:	c5 fa 6f 98 10 ff ff 	vmovdqu -0xf0(%rax),%xmm3
    38da:	ff 
    38db:	c4 c1 71 db cf       	vpand  %xmm15,%xmm1,%xmm1
    38e0:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    38e5:	c5 fa 6f a0 00 ff ff 	vmovdqu -0x100(%rax),%xmm4
    38ec:	ff 
    38ed:	c4 e2 11 00 d2       	vpshufb %xmm2,%xmm13,%xmm2
    38f2:	c4 e2 61 08 d9       	vpsignb %xmm1,%xmm3,%xmm3
    38f7:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    38fc:	c4 e2 71 04 cb       	vpmaddubsw %xmm3,%xmm1,%xmm1
    3901:	c5 fa 6f 98 30 ff ff 	vmovdqu -0xd0(%rax),%xmm3
    3908:	ff 
    3909:	c4 e2 59 08 e2       	vpsignb %xmm2,%xmm4,%xmm4
    390e:	89 d7                	mov    %edx,%edi
    3910:	c4 e2 79 1c d2       	vpabsb %xmm2,%xmm2
    3915:	c4 e2 69 04 d4       	vpmaddubsw %xmm4,%xmm2,%xmm2
    391a:	c5 f1 f5 e0          	vpmaddwd %xmm0,%xmm1,%xmm4
    391e:	c5 fa 6f 49 90       	vmovdqu -0x70(%rcx),%xmm1
    3923:	83 e7 0f             	and    $0xf,%edi
    3926:	c0 ea 04             	shr    $0x4,%dl
    3929:	8d 7c 3f 01          	lea    0x1(%rdi,%rdi,1),%edi
    392d:	c5 69 f5 d8          	vpmaddwd %xmm0,%xmm2,%xmm11
    3931:	0f b6 d2             	movzbl %dl,%edx
    3934:	c5 fa 6f 78 b0       	vmovdqu -0x50(%rax),%xmm7
    3939:	c5 f9 7f 64 24 a8    	vmovdqa %xmm4,-0x58(%rsp)
    393f:	c4 c1 71 db c7       	vpand  %xmm15,%xmm1,%xmm0
    3944:	8d 54 12 01          	lea    0x1(%rdx,%rdx,1),%edx
    3948:	c5 f9 6e d7          	vmovd  %edi,%xmm2
    394c:	c5 f1 71 d1 04       	vpsrlw $0x4,%xmm1,%xmm1
    3951:	c4 e2 11 00 c0       	vpshufb %xmm0,%xmm13,%xmm0
    3956:	c4 c2 69 00 d6       	vpshufb %xmm14,%xmm2,%xmm2
    395b:	c5 fa 6f a0 20 ff ff 	vmovdqu -0xe0(%rax),%xmm4
    3962:	ff 
    3963:	c4 c1 71 db cf       	vpand  %xmm15,%xmm1,%xmm1
    3968:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    396d:	c4 e2 61 08 d9       	vpsignb %xmm1,%xmm3,%xmm3
    3972:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    3977:	c4 e2 59 08 e0       	vpsignb %xmm0,%xmm4,%xmm4
    397c:	c4 e2 71 04 cb       	vpmaddubsw %xmm3,%xmm1,%xmm1
    3981:	c4 e2 79 1c c0       	vpabsb %xmm0,%xmm0
    3986:	c4 e2 79 04 c4       	vpmaddubsw %xmm4,%xmm0,%xmm0
    398b:	c5 f1 f5 da          	vpmaddwd %xmm2,%xmm1,%xmm3
    398f:	c5 f9 f5 c2          	vpmaddwd %xmm2,%xmm0,%xmm0
    3993:	c5 f9 6e d2          	vmovd  %edx,%xmm2
    3997:	c5 fa 6f 49 a0       	vmovdqu -0x60(%rcx),%xmm1
    399c:	0f b6 51 f6          	movzbl -0xa(%rcx),%edx
    39a0:	c5 f9 7f 5c 24 98    	vmovdqa %xmm3,-0x68(%rsp)
    39a6:	c4 c2 69 00 d6       	vpshufb %xmm14,%xmm2,%xmm2
    39ab:	c5 fa 6f a0 40 ff ff 	vmovdqu -0xc0(%rax),%xmm4
    39b2:	ff 
    39b3:	c4 41 71 db cf       	vpand  %xmm15,%xmm1,%xmm9
    39b8:	c5 f1 71 d1 04       	vpsrlw $0x4,%xmm1,%xmm1
    39bd:	c4 42 11 00 c9       	vpshufb %xmm9,%xmm13,%xmm9
    39c2:	c5 fa 6f 98 50 ff ff 	vmovdqu -0xb0(%rax),%xmm3
    39c9:	ff 
    39ca:	c4 c1 71 db cf       	vpand  %xmm15,%xmm1,%xmm1
    39cf:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    39d4:	c5 f9 7f 6c 24 c8    	vmovdqa %xmm5,-0x38(%rsp)
    39da:	c4 c2 59 08 e1       	vpsignb %xmm9,%xmm4,%xmm4
    39df:	c4 42 79 1c c9       	vpabsb %xmm9,%xmm9
    39e4:	c4 62 31 04 cc       	vpmaddubsw %xmm4,%xmm9,%xmm9
    39e9:	c5 fa 6f 61 b0       	vmovdqu -0x50(%rcx),%xmm4
    39ee:	c4 e2 61 08 d9       	vpsignb %xmm1,%xmm3,%xmm3
    39f3:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    39f8:	c5 31 f5 ca          	vpmaddwd %xmm2,%xmm9,%xmm9
    39fc:	c4 e2 71 04 cb       	vpmaddubsw %xmm3,%xmm1,%xmm1
    3a01:	c5 f1 f5 d2          	vpmaddwd %xmm2,%xmm1,%xmm2
    3a05:	89 d7                	mov    %edx,%edi
    3a07:	c0 ea 04             	shr    $0x4,%dl
    3a0a:	c5 fa 6f 98 60 ff ff 	vmovdqu -0xa0(%rax),%xmm3
    3a11:	ff 
    3a12:	c5 f9 7f 54 24 88    	vmovdqa %xmm2,-0x78(%rsp)
    3a18:	83 e7 0f             	and    $0xf,%edi
    3a1b:	c4 c1 59 db f7       	vpand  %xmm15,%xmm4,%xmm6
    3a20:	0f b6 d2             	movzbl %dl,%edx
    3a23:	c5 fa 6f 90 70 ff ff 	vmovdqu -0x90(%rax),%xmm2
    3a2a:	ff 
    3a2b:	8d 7c 3f 01          	lea    0x1(%rdi,%rdi,1),%edi
    3a2f:	c5 d9 71 d4 04       	vpsrlw $0x4,%xmm4,%xmm4
    3a34:	c4 e2 11 00 f6       	vpshufb %xmm6,%xmm13,%xmm6
    3a39:	8d 54 12 01          	lea    0x1(%rdx,%rdx,1),%edx
    3a3d:	c4 e2 61 08 de       	vpsignb %xmm6,%xmm3,%xmm3
    3a42:	c5 f9 6e cf          	vmovd  %edi,%xmm1
    3a46:	c4 c1 59 db e7       	vpand  %xmm15,%xmm4,%xmm4
    3a4b:	c4 e2 79 1c f6       	vpabsb %xmm6,%xmm6
    3a50:	c4 e2 11 00 e4       	vpshufb %xmm4,%xmm13,%xmm4
    3a55:	c4 e2 49 04 f3       	vpmaddubsw %xmm3,%xmm6,%xmm6
    3a5a:	c4 e2 69 08 d4       	vpsignb %xmm4,%xmm2,%xmm2
    3a5f:	c4 e2 79 1c e4       	vpabsb %xmm4,%xmm4
    3a64:	c4 e2 59 04 e2       	vpmaddubsw %xmm2,%xmm4,%xmm4
    3a69:	c4 c2 71 00 ce       	vpshufb %xmm14,%xmm1,%xmm1
    3a6e:	c5 c9 f5 f1          	vpmaddwd %xmm1,%xmm6,%xmm6
    3a72:	c5 d9 f5 e1          	vpmaddwd %xmm1,%xmm4,%xmm4
    3a76:	c5 f9 6e ca          	vmovd  %edx,%xmm1
    3a7a:	0f b6 51 f7          	movzbl -0x9(%rcx),%edx
    3a7e:	c5 79 7f 5c 24 b8    	vmovdqa %xmm11,-0x48(%rsp)
    3a84:	c5 fa 6f 51 c0       	vmovdqu -0x40(%rcx),%xmm2
    3a89:	c4 c2 71 00 ce       	vpshufb %xmm14,%xmm1,%xmm1
    3a8e:	c4 c1 79 fe c1       	vpaddd %xmm9,%xmm0,%xmm0
    3a93:	c5 fa 6f 58 90       	vmovdqu -0x70(%rax),%xmm3
    3a98:	c4 41 69 db c7       	vpand  %xmm15,%xmm2,%xmm8
    3a9d:	c5 e9 71 d2 04       	vpsrlw $0x4,%xmm2,%xmm2
    3aa2:	c5 fa 6f 68 80       	vmovdqu -0x80(%rax),%xmm5
    3aa7:	c4 42 11 00 c0       	vpshufb %xmm8,%xmm13,%xmm8
    3aac:	89 d7                	mov    %edx,%edi
    3aae:	c4 c1 69 db d7       	vpand  %xmm15,%xmm2,%xmm2
    3ab3:	c4 e2 11 00 d2       	vpshufb %xmm2,%xmm13,%xmm2
    3ab8:	c4 e2 61 08 da       	vpsignb %xmm2,%xmm3,%xmm3
    3abd:	c4 e2 79 1c d2       	vpabsb %xmm2,%xmm2
    3ac2:	c4 e2 69 04 d3       	vpmaddubsw %xmm3,%xmm2,%xmm2
    3ac7:	c5 fa 6f 59 d0       	vmovdqu -0x30(%rcx),%xmm3
    3acc:	83 e7 0f             	and    $0xf,%edi
    3acf:	c4 c2 51 08 e8       	vpsignb %xmm8,%xmm5,%xmm5
    3ad4:	8d 7c 3f 01          	lea    0x1(%rdi,%rdi,1),%edi
    3ad8:	c4 42 79 1c c0       	vpabsb %xmm8,%xmm8
    3add:	c4 62 39 04 c5       	vpmaddubsw %xmm5,%xmm8,%xmm8
    3ae2:	c4 c1 61 db ef       	vpand  %xmm15,%xmm3,%xmm5
    3ae7:	c5 e1 71 d3 04       	vpsrlw $0x4,%xmm3,%xmm3
    3aec:	c5 39 f5 c1          	vpmaddwd %xmm1,%xmm8,%xmm8
    3af0:	c5 e9 f5 d1          	vpmaddwd %xmm1,%xmm2,%xmm2
    3af4:	c4 e2 11 00 ed       	vpshufb %xmm5,%xmm13,%xmm5
    3af9:	c5 f9 6e cf          	vmovd  %edi,%xmm1
    3afd:	c0 ea 04             	shr    $0x4,%dl
    3b00:	c4 62 29 08 d5       	vpsignb %xmm5,%xmm10,%xmm10
    3b05:	c4 c1 61 db df       	vpand  %xmm15,%xmm3,%xmm3
    3b0a:	c4 e2 79 1c ed       	vpabsb %xmm5,%xmm5
    3b0f:	c4 e2 11 00 db       	vpshufb %xmm3,%xmm13,%xmm3
    3b14:	0f b6 d2             	movzbl %dl,%edx
    3b17:	c4 e2 41 08 fb       	vpsignb %xmm3,%xmm7,%xmm7
    3b1c:	c4 c2 51 04 ea       	vpmaddubsw %xmm10,%xmm5,%xmm5
    3b21:	c4 e2 79 1c db       	vpabsb %xmm3,%xmm3
    3b26:	c4 c1 79 fe c0       	vpaddd %xmm8,%xmm0,%xmm0
    3b2b:	c4 c2 71 00 ce       	vpshufb %xmm14,%xmm1,%xmm1
    3b30:	c4 e2 61 04 df       	vpmaddubsw %xmm7,%xmm3,%xmm3
    3b35:	c5 d1 f5 e9          	vpmaddwd %xmm1,%xmm5,%xmm5
    3b39:	8d 54 12 01          	lea    0x1(%rdx,%rdx,1),%edx
    3b3d:	c5 e1 f5 d9          	vpmaddwd %xmm1,%xmm3,%xmm3
    3b41:	c5 fa 6f 49 e0       	vmovdqu -0x20(%rcx),%xmm1
    3b46:	c5 7a 6f 60 c0       	vmovdqu -0x40(%rax),%xmm12
    3b4b:	c5 79 6e d2          	vmovd  %edx,%xmm10
    3b4f:	c4 c1 71 db ff       	vpand  %xmm15,%xmm1,%xmm7
    3b54:	c5 7a 6f 58 d0       	vmovdqu -0x30(%rax),%xmm11
    3b59:	c4 e2 11 00 ff       	vpshufb %xmm7,%xmm13,%xmm7
    3b5e:	c4 42 29 00 d6       	vpshufb %xmm14,%xmm10,%xmm10
    3b63:	c4 62 19 08 e7       	vpsignb %xmm7,%xmm12,%xmm12
    3b68:	c4 e2 79 1c ff       	vpabsb %xmm7,%xmm7
    3b6d:	c4 c2 41 04 fc       	vpmaddubsw %xmm12,%xmm7,%xmm7
    3b72:	c5 f1 71 d1 04       	vpsrlw $0x4,%xmm1,%xmm1
    3b77:	c4 c1 41 f5 fa       	vpmaddwd %xmm10,%xmm7,%xmm7
    3b7c:	c4 c1 71 db cf       	vpand  %xmm15,%xmm1,%xmm1
    3b81:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    3b86:	c4 62 21 08 d9       	vpsignb %xmm1,%xmm11,%xmm11
    3b8b:	c5 f9 fe c7          	vpaddd %xmm7,%xmm0,%xmm0
    3b8f:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    3b94:	c4 c2 71 04 cb       	vpmaddubsw %xmm11,%xmm1,%xmm1
    3b99:	c5 79 6f 5c 24 b8    	vmovdqa -0x48(%rsp),%xmm11
    3b9f:	c5 a1 fe 7c 24 d8    	vpaddd -0x28(%rsp),%xmm11,%xmm7
    3ba5:	c4 c1 71 f5 ca       	vpmaddwd %xmm10,%xmm1,%xmm1
    3baa:	c5 c1 fe f6          	vpaddd %xmm6,%xmm7,%xmm6
    3bae:	c5 f9 6f 7c 24 88    	vmovdqa -0x78(%rsp),%xmm7
    3bb4:	c5 c9 fe ed          	vpaddd %xmm5,%xmm6,%xmm5
    3bb8:	c5 f9 6f 74 24 a8    	vmovdqa -0x58(%rsp),%xmm6
    3bbe:	c5 f9 fe c5          	vpaddd %xmm5,%xmm0,%xmm0
    3bc2:	c5 c9 fe 6c 24 c8    	vpaddd -0x38(%rsp),%xmm6,%xmm5
    3bc8:	c5 d1 fe e4          	vpaddd %xmm4,%xmm5,%xmm4
    3bcc:	c5 d9 fe db          	vpaddd %xmm3,%xmm4,%xmm3
    3bd0:	c5 c1 fe 64 24 98    	vpaddd -0x68(%rsp),%xmm7,%xmm4
    3bd6:	c5 d9 fe d2          	vpaddd %xmm2,%xmm4,%xmm2
    3bda:	c5 e9 fe c9          	vpaddd %xmm1,%xmm2,%xmm1
    3bde:	c5 e1 fe c9          	vpaddd %xmm1,%xmm3,%xmm1
    3be2:	c4 e3 7d 18 c1 01    	vinsertf128 $0x1,%xmm1,%ymm0,%ymm0
    3be8:	c5 fc 5b c8          	vcvtdq2ps %ymm0,%ymm1
    3bec:	c5 fa 10 41 f0       	vmovss -0x10(%rcx),%xmm0
    3bf1:	c5 fa 59 80 dc fe ff 	vmulss -0x124(%rax),%xmm0,%xmm0
    3bf8:	ff 
    3bf9:	c5 f8 c6 c0 00       	vshufps $0x0,%xmm0,%xmm0,%xmm0
    3bfe:	c4 e3 7d 18 c0 01    	vinsertf128 $0x1,%xmm0,%ymm0,%ymm0
    3c04:	c5 fc 59 c1          	vmulps %ymm1,%ymm0,%ymm0
    3c08:	c5 fc 58 74 24 e8    	vaddps -0x18(%rsp),%ymm0,%ymm6
    3c0e:	c5 fc 29 74 24 e8    	vmovaps %ymm6,-0x18(%rsp)
    3c14:	44 39 c6             	cmp    %r8d,%esi
    3c17:	0f 8c 03 fc ff ff    	jl     3820 <offset4_compat_dot(int, offset4_iq3s_block const*, block_q8_K const*)+0x60>
    3c1d:	c5 fc 28 fe          	vmovaps %ymm6,%ymm7
    3c21:	c4 e3 7d 19 f8 01    	vextractf128 $0x1,%ymm7,%xmm0
    3c27:	c5 c0 58 c0          	vaddps %xmm0,%xmm7,%xmm0
    3c2b:	c5 f8 12 c8          	vmovhlps %xmm0,%xmm0,%xmm1
    3c2f:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    3c33:	c5 fa 16 c8          	vmovshdup %xmm0,%xmm1
    3c37:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    3c3b:	c5 f8 77             	vzeroupper
    3c3e:	c9                   	leave
    3c3f:	c3                   	ret
    3c40:	c5 c0 57 ff          	vxorps %xmm7,%xmm7,%xmm7
    3c44:	c5 fc 29 7c 24 e8    	vmovaps %ymm7,-0x18(%rsp)
    3c4a:	eb d5                	jmp    3c21 <offset4_compat_dot(int, offset4_iq3s_block const*, block_q8_K const*)+0x461>
    3c4c:	0f 1f 40 00          	nopl   0x0(%rax)

