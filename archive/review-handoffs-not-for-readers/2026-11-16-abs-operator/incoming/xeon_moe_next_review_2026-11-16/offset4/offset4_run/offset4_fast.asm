00000000000033c0 <offset4_fast_dot(int, offset4_iq3s_block const*, block_q8_K const*, offset4_rhs_block const*)>:
    33c0:	55                   	push   %rbp
    33c1:	44 8d 87 ff 00 00 00 	lea    0xff(%rdi),%r8d
    33c8:	48 89 e5             	mov    %rsp,%rbp
    33cb:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    33cf:	85 ff                	test   %edi,%edi
    33d1:	44 0f 49 c7          	cmovns %edi,%r8d
    33d5:	41 c1 f8 08          	sar    $0x8,%r8d
    33d9:	81 ff ff 00 00 00    	cmp    $0xff,%edi
    33df:	0f 8e c1 03 00 00    	jle    37a6 <offset4_fast_dot(int, offset4_iq3s_block const*, block_q8_K const*, offset4_rhs_block const*)+0x3e6>
    33e5:	48 89 d0             	mov    %rdx,%rax
    33e8:	48 89 ca             	mov    %rcx,%rdx
    33eb:	b9 0f 0f 0f 0f       	mov    $0xf0f0f0f,%ecx
    33f0:	c5 c8 57 f6          	vxorps %xmm6,%xmm6,%xmm6
    33f4:	31 ff                	xor    %edi,%edi
    33f6:	c5 fc 29 74 24 e0    	vmovaps %ymm6,-0x20(%rsp)
    33fc:	c5 f9 6e f1          	vmovd  %ecx,%xmm6
    3400:	c5 79 70 fe 00       	vpshufd $0x0,%xmm6,%xmm15
    3405:	0f 1f 00             	nopl   (%rax)
    3408:	c5 fa 6f 16          	vmovdqu (%rsi),%xmm2
    340c:	83 c7 01             	add    $0x1,%edi
    340f:	48 81 c6 90 00 00 00 	add    $0x90,%rsi
    3416:	48 81 c2 00 01 00 00 	add    $0x100,%rdx
    341d:	44 0f b6 4e f4       	movzbl -0xc(%rsi),%r9d
    3422:	c5 e1 71 d2 04       	vpsrlw $0x4,%xmm2,%xmm3
    3427:	c4 c1 69 db d7       	vpand  %xmm15,%xmm2,%xmm2
    342c:	c4 e2 69 04 50 04    	vpmaddubsw 0x4(%rax),%xmm2,%xmm2
    3432:	c4 c1 61 db df       	vpand  %xmm15,%xmm3,%xmm3
    3437:	c5 fa 6f 76 80       	vmovdqu -0x80(%rsi),%xmm6
    343c:	c4 e2 61 04 58 14    	vpmaddubsw 0x14(%rax),%xmm3,%xmm3
    3442:	c5 e9 fd d2          	vpaddw %xmm2,%xmm2,%xmm2
    3446:	c5 e9 f9 92 00 ff ff 	vpsubw -0x100(%rdx),%xmm2,%xmm2
    344d:	ff 
    344e:	44 89 c9             	mov    %r9d,%ecx
    3451:	c5 e1 fd db          	vpaddw %xmm3,%xmm3,%xmm3
    3455:	c5 7a 6f 4e a0       	vmovdqu -0x60(%rsi),%xmm9
    345a:	c5 e1 f9 9a 10 ff ff 	vpsubw -0xf0(%rdx),%xmm3,%xmm3
    3461:	ff 
    3462:	83 e1 0f             	and    $0xf,%ecx
    3465:	41 c0 e9 04          	shr    $0x4,%r9b
    3469:	8d 4c 09 01          	lea    0x1(%rcx,%rcx,1),%ecx
    346d:	c4 c1 11 71 d1 04    	vpsrlw $0x4,%xmm9,%xmm13
    3473:	45 0f b6 c9          	movzbl %r9b,%r9d
    3477:	c4 41 31 db cf       	vpand  %xmm15,%xmm9,%xmm9
    347c:	c5 f9 6e c1          	vmovd  %ecx,%xmm0
    3480:	c4 62 31 04 48 64    	vpmaddubsw 0x64(%rax),%xmm9,%xmm9
    3486:	c4 41 11 db ef       	vpand  %xmm15,%xmm13,%xmm13
    348b:	c4 e2 79 00 05 7c 2e 	vpshufb 0x2e7c(%rip),%xmm0,%xmm0        # 6310 <iq3s_grid+0x810>
    3492:	00 00 
    3494:	c5 e9 f5 d0          	vpmaddwd %xmm0,%xmm2,%xmm2
    3498:	c5 e1 f5 d8          	vpmaddwd %xmm0,%xmm3,%xmm3
    349c:	c5 f9 71 d6 04       	vpsrlw $0x4,%xmm6,%xmm0
    34a1:	c4 c1 79 db c7       	vpand  %xmm15,%xmm0,%xmm0
    34a6:	c4 e2 79 04 40 34    	vpmaddubsw 0x34(%rax),%xmm0,%xmm0
    34ac:	c4 c1 49 db f7       	vpand  %xmm15,%xmm6,%xmm6
    34b1:	c4 e2 49 04 70 24    	vpmaddubsw 0x24(%rax),%xmm6,%xmm6
    34b7:	43 8d 4c 09 01       	lea    0x1(%r9,%r9,1),%ecx
    34bc:	44 0f b6 4e f5       	movzbl -0xb(%rsi),%r9d
    34c1:	c4 62 11 04 68 74    	vpmaddubsw 0x74(%rax),%xmm13,%xmm13
    34c7:	c5 f9 fd c0          	vpaddw %xmm0,%xmm0,%xmm0
    34cb:	c5 f9 f9 82 30 ff ff 	vpsubw -0xd0(%rdx),%xmm0,%xmm0
    34d2:	ff 
    34d3:	c5 f9 6e c9          	vmovd  %ecx,%xmm1
    34d7:	c4 e2 71 00 0d 30 2e 	vpshufb 0x2e30(%rip),%xmm1,%xmm1        # 6310 <iq3s_grid+0x810>
    34de:	00 00 
    34e0:	c5 c9 fd f6          	vpaddw %xmm6,%xmm6,%xmm6
    34e4:	c5 c9 f9 b2 20 ff ff 	vpsubw -0xe0(%rdx),%xmm6,%xmm6
    34eb:	ff 
    34ec:	c5 f9 f5 e9          	vpmaddwd %xmm1,%xmm0,%xmm5
    34f0:	c5 fa 6f 46 90       	vmovdqu -0x70(%rsi),%xmm0
    34f5:	c4 41 31 fd c9       	vpaddw %xmm9,%xmm9,%xmm9
    34fa:	c5 c9 f5 f1          	vpmaddwd %xmm1,%xmm6,%xmm6
    34fe:	44 89 c9             	mov    %r9d,%ecx
    3501:	41 c0 e9 04          	shr    $0x4,%r9b
    3505:	c5 f9 7f 6c 24 d0    	vmovdqa %xmm5,-0x30(%rsp)
    350b:	c5 f1 71 d0 04       	vpsrlw $0x4,%xmm0,%xmm1
    3510:	c4 c1 79 db c7       	vpand  %xmm15,%xmm0,%xmm0
    3515:	83 e1 0f             	and    $0xf,%ecx
    3518:	45 0f b6 c9          	movzbl %r9b,%r9d
    351c:	c4 e2 79 04 40 44    	vpmaddubsw 0x44(%rax),%xmm0,%xmm0
    3522:	c4 c1 71 db cf       	vpand  %xmm15,%xmm1,%xmm1
    3527:	c4 e2 71 04 48 54    	vpmaddubsw 0x54(%rax),%xmm1,%xmm1
    352d:	8d 4c 09 01          	lea    0x1(%rcx,%rcx,1),%ecx
    3531:	c4 41 11 fd ed       	vpaddw %xmm13,%xmm13,%xmm13
    3536:	c5 f9 fd c0          	vpaddw %xmm0,%xmm0,%xmm0
    353a:	c5 f1 fd c9          	vpaddw %xmm1,%xmm1,%xmm1
    353e:	c5 f9 6e e1          	vmovd  %ecx,%xmm4
    3542:	c5 f9 f9 82 40 ff ff 	vpsubw -0xc0(%rdx),%xmm0,%xmm0
    3549:	ff 
    354a:	c5 e9 fe d6          	vpaddd %xmm6,%xmm2,%xmm2
    354e:	c4 e2 59 00 25 b9 2d 	vpshufb 0x2db9(%rip),%xmm4,%xmm4        # 6310 <iq3s_grid+0x810>
    3555:	00 00 
    3557:	43 8d 4c 09 01       	lea    0x1(%r9,%r9,1),%ecx
    355c:	c5 f1 f9 8a 50 ff ff 	vpsubw -0xb0(%rdx),%xmm1,%xmm1
    3563:	ff 
    3564:	c5 f9 f5 c4          	vpmaddwd %xmm4,%xmm0,%xmm0
    3568:	c5 f1 f5 cc          	vpmaddwd %xmm4,%xmm1,%xmm1
    356c:	c5 f9 6e e1          	vmovd  %ecx,%xmm4
    3570:	c4 e2 59 00 25 97 2d 	vpshufb 0x2d97(%rip),%xmm4,%xmm4        # 6310 <iq3s_grid+0x810>
    3577:	00 00 
    3579:	0f b6 4e f6          	movzbl -0xa(%rsi),%ecx
    357d:	c5 31 f9 8a 60 ff ff 	vpsubw -0xa0(%rdx),%xmm9,%xmm9
    3584:	ff 
    3585:	c5 11 f9 aa 70 ff ff 	vpsubw -0x90(%rdx),%xmm13,%xmm13
    358c:	ff 
    358d:	c5 fa 6f 6e b0       	vmovdqu -0x50(%rsi),%xmm5
    3592:	c5 7a 6f 46 c0       	vmovdqu -0x40(%rsi),%xmm8
    3597:	c5 31 f5 cc          	vpmaddwd %xmm4,%xmm9,%xmm9
    359b:	c5 11 f5 ec          	vpmaddwd %xmm4,%xmm13,%xmm13
    359f:	c5 d9 71 d5 04       	vpsrlw $0x4,%xmm5,%xmm4
    35a4:	c4 c1 51 db ef       	vpand  %xmm15,%xmm5,%xmm5
    35a9:	c4 c1 79 fe c1       	vpaddd %xmm9,%xmm0,%xmm0
    35ae:	c4 e2 51 04 a8 84 00 	vpmaddubsw 0x84(%rax),%xmm5,%xmm5
    35b5:	00 00 
    35b7:	41 89 c9             	mov    %ecx,%r9d
    35ba:	c4 c1 19 71 d0 04    	vpsrlw $0x4,%xmm8,%xmm12
    35c0:	c4 c1 59 db e7       	vpand  %xmm15,%xmm4,%xmm4
    35c5:	c0 e9 04             	shr    $0x4,%cl
    35c8:	41 83 e1 0f          	and    $0xf,%r9d
    35cc:	c4 41 39 db c7       	vpand  %xmm15,%xmm8,%xmm8
    35d1:	c4 41 19 db e7       	vpand  %xmm15,%xmm12,%xmm12
    35d6:	c5 d1 fd ed          	vpaddw %xmm5,%xmm5,%xmm5
    35da:	c4 e2 59 04 a0 94 00 	vpmaddubsw 0x94(%rax),%xmm4,%xmm4
    35e1:	00 00 
    35e3:	c5 d1 f9 6a 80       	vpsubw -0x80(%rdx),%xmm5,%xmm5
    35e8:	0f b6 c9             	movzbl %cl,%ecx
    35eb:	c4 62 39 04 80 a4 00 	vpmaddubsw 0xa4(%rax),%xmm8,%xmm8
    35f2:	00 00 
    35f4:	c4 c1 71 fe cd       	vpaddd %xmm13,%xmm1,%xmm1
    35f9:	c4 62 19 04 a0 b4 00 	vpmaddubsw 0xb4(%rax),%xmm12,%xmm12
    3600:	00 00 
    3602:	c5 d9 fd e4          	vpaddw %xmm4,%xmm4,%xmm4
    3606:	47 8d 4c 09 01       	lea    0x1(%r9,%r9,1),%r9d
    360b:	c5 d9 f9 62 90       	vpsubw -0x70(%rdx),%xmm4,%xmm4
    3610:	8d 4c 09 01          	lea    0x1(%rcx,%rcx,1),%ecx
    3614:	c4 c1 79 6e f9       	vmovd  %r9d,%xmm7
    3619:	c4 41 39 fd c0       	vpaddw %xmm8,%xmm8,%xmm8
    361e:	c4 41 19 fd e4       	vpaddw %xmm12,%xmm12,%xmm12
    3623:	c5 39 f9 42 a0       	vpsubw -0x60(%rdx),%xmm8,%xmm8
    3628:	c4 e2 41 00 3d df 2c 	vpshufb 0x2cdf(%rip),%xmm7,%xmm7        # 6310 <iq3s_grid+0x810>
    362f:	00 00 
    3631:	c5 19 f9 62 b0       	vpsubw -0x50(%rdx),%xmm12,%xmm12
    3636:	c5 d1 f5 ef          	vpmaddwd %xmm7,%xmm5,%xmm5
    363a:	c5 d9 f5 ff          	vpmaddwd %xmm7,%xmm4,%xmm7
    363e:	c5 f9 6e e1          	vmovd  %ecx,%xmm4
    3642:	0f b6 4e f7          	movzbl -0x9(%rsi),%ecx
    3646:	c5 e9 fe d5          	vpaddd %xmm5,%xmm2,%xmm2
    364a:	c4 e2 59 00 25 bd 2c 	vpshufb 0x2cbd(%rip),%xmm4,%xmm4        # 6310 <iq3s_grid+0x810>
    3651:	00 00 
    3653:	c5 39 f5 c4          	vpmaddwd %xmm4,%xmm8,%xmm8
    3657:	c5 19 f5 e4          	vpmaddwd %xmm4,%xmm12,%xmm12
    365b:	c5 fa 6f 66 d0       	vmovdqu -0x30(%rsi),%xmm4
    3660:	c5 f9 7f 7c 24 c0    	vmovdqa %xmm7,-0x40(%rsp)
    3666:	c4 c1 79 fe c0       	vpaddd %xmm8,%xmm0,%xmm0
    366b:	c4 c1 71 fe cc       	vpaddd %xmm12,%xmm1,%xmm1
    3670:	c5 a9 71 d4 04       	vpsrlw $0x4,%xmm4,%xmm10
    3675:	c4 c1 59 db e7       	vpand  %xmm15,%xmm4,%xmm4
    367a:	c4 e2 59 04 a0 c4 00 	vpmaddubsw 0xc4(%rax),%xmm4,%xmm4
    3681:	00 00 
    3683:	41 89 c9             	mov    %ecx,%r9d
    3686:	c4 41 29 db d7       	vpand  %xmm15,%xmm10,%xmm10
    368b:	c0 e9 04             	shr    $0x4,%cl
    368e:	c4 62 29 04 90 d4 00 	vpmaddubsw 0xd4(%rax),%xmm10,%xmm10
    3695:	00 00 
    3697:	41 83 e1 0f          	and    $0xf,%r9d
    369b:	c5 d9 fd e4          	vpaddw %xmm4,%xmm4,%xmm4
    369f:	47 8d 4c 09 01       	lea    0x1(%r9,%r9,1),%r9d
    36a4:	c5 d9 f9 62 c0       	vpsubw -0x40(%rdx),%xmm4,%xmm4
    36a9:	0f b6 c9             	movzbl %cl,%ecx
    36ac:	8d 4c 09 01          	lea    0x1(%rcx,%rcx,1),%ecx
    36b0:	c4 41 29 fd d2       	vpaddw %xmm10,%xmm10,%xmm10
    36b5:	c4 c1 79 6e f9       	vmovd  %r9d,%xmm7
    36ba:	c5 29 f9 52 d0       	vpsubw -0x30(%rdx),%xmm10,%xmm10
    36bf:	c5 79 6e f1          	vmovd  %ecx,%xmm14
    36c3:	c4 e2 41 00 3d 44 2c 	vpshufb 0x2c44(%rip),%xmm7,%xmm7        # 6310 <iq3s_grid+0x810>
    36ca:	00 00 
    36cc:	c5 d9 f5 e7          	vpmaddwd %xmm7,%xmm4,%xmm4
    36d0:	c5 e9 fe d4          	vpaddd %xmm4,%xmm2,%xmm2
    36d4:	c5 29 f5 d7          	vpmaddwd %xmm7,%xmm10,%xmm10
    36d8:	c5 fa 6f 7e e0       	vmovdqu -0x20(%rsi),%xmm7
    36dd:	c5 a1 71 d7 04       	vpsrlw $0x4,%xmm7,%xmm11
    36e2:	c4 c1 41 db ff       	vpand  %xmm15,%xmm7,%xmm7
    36e7:	c4 e2 41 04 b8 e4 00 	vpmaddubsw 0xe4(%rax),%xmm7,%xmm7
    36ee:	00 00 
    36f0:	c5 e1 fe 5c 24 d0    	vpaddd -0x30(%rsp),%xmm3,%xmm3
    36f6:	c4 41 21 db df       	vpand  %xmm15,%xmm11,%xmm11
    36fb:	c5 e1 fe 5c 24 c0    	vpaddd -0x40(%rsp),%xmm3,%xmm3
    3701:	c4 62 21 04 98 f4 00 	vpmaddubsw 0xf4(%rax),%xmm11,%xmm11
    3708:	00 00 
    370a:	c4 62 09 00 35 fd 2b 	vpshufb 0x2bfd(%rip),%xmm14,%xmm14        # 6310 <iq3s_grid+0x810>
    3711:	00 00 
    3713:	c5 c1 fd ff          	vpaddw %xmm7,%xmm7,%xmm7
    3717:	c5 c1 f9 7a e0       	vpsubw -0x20(%rdx),%xmm7,%xmm7
    371c:	48 05 24 01 00 00    	add    $0x124,%rax
    3722:	c4 41 21 fd db       	vpaddw %xmm11,%xmm11,%xmm11
    3727:	c5 21 f9 5a f0       	vpsubw -0x10(%rdx),%xmm11,%xmm11
    372c:	c4 c1 41 f5 fe       	vpmaddwd %xmm14,%xmm7,%xmm7
    3731:	c4 c1 61 fe da       	vpaddd %xmm10,%xmm3,%xmm3
    3736:	c4 41 21 f5 de       	vpmaddwd %xmm14,%xmm11,%xmm11
    373b:	c5 f9 fe c7          	vpaddd %xmm7,%xmm0,%xmm0
    373f:	c4 c1 71 fe cb       	vpaddd %xmm11,%xmm1,%xmm1
    3744:	c5 f9 fe c2          	vpaddd %xmm2,%xmm0,%xmm0
    3748:	c5 f1 fe cb          	vpaddd %xmm3,%xmm1,%xmm1
    374c:	c4 e3 7d 18 c1 01    	vinsertf128 $0x1,%xmm1,%ymm0,%ymm0
    3752:	c5 fc 5b c8          	vcvtdq2ps %ymm0,%ymm1
    3756:	c5 fa 10 46 f0       	vmovss -0x10(%rsi),%xmm0
    375b:	c5 fa 59 80 dc fe ff 	vmulss -0x124(%rax),%xmm0,%xmm0
    3762:	ff 
    3763:	c5 f8 c6 c0 00       	vshufps $0x0,%xmm0,%xmm0,%xmm0
    3768:	c4 e3 7d 18 c0 01    	vinsertf128 $0x1,%xmm0,%ymm0,%ymm0
    376e:	c5 fc 59 c1          	vmulps %ymm1,%ymm0,%ymm0
    3772:	c5 fc 58 74 24 e0    	vaddps -0x20(%rsp),%ymm0,%ymm6
    3778:	c5 fc 29 74 24 e0    	vmovaps %ymm6,-0x20(%rsp)
    377e:	44 39 c7             	cmp    %r8d,%edi
    3781:	0f 8c 81 fc ff ff    	jl     3408 <offset4_fast_dot(int, offset4_iq3s_block const*, block_q8_K const*, offset4_rhs_block const*)+0x48>
    3787:	c4 e3 7d 19 f0 01    	vextractf128 $0x1,%ymm6,%xmm0
    378d:	c5 c8 58 c0          	vaddps %xmm0,%xmm6,%xmm0
    3791:	c5 f8 12 c8          	vmovhlps %xmm0,%xmm0,%xmm1
    3795:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    3799:	c5 fa 16 c8          	vmovshdup %xmm0,%xmm1
    379d:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    37a1:	c5 f8 77             	vzeroupper
    37a4:	c9                   	leave
    37a5:	c3                   	ret
    37a6:	c5 c8 57 f6          	vxorps %xmm6,%xmm6,%xmm6
    37aa:	c5 fc 29 74 24 e0    	vmovaps %ymm6,-0x20(%rsp)
    37b0:	eb d5                	jmp    3787 <offset4_fast_dot(int, offset4_iq3s_block const*, block_q8_K const*, offset4_rhs_block const*)+0x3c7>
    37b2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    37b9:	00 00 00 
    37bc:	0f 1f 40 00          	nopl   0x0(%rax)

