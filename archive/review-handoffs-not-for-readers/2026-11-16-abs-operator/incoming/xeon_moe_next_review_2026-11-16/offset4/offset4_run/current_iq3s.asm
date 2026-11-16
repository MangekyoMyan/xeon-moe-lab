0000000000002f30 <float iq_compact_dot<false>(int, iq_compact_block const*, block_q8_K const*)>:
    2f30:	55                   	push   %rbp
    2f31:	44 8d 87 ff 00 00 00 	lea    0xff(%rdi),%r8d
    2f38:	48 89 e5             	mov    %rsp,%rbp
    2f3b:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    2f3f:	48 83 ec 08          	sub    $0x8,%rsp
    2f43:	85 ff                	test   %edi,%edi
    2f45:	44 0f 49 c7          	cmovns %edi,%r8d
    2f49:	41 c1 f8 08          	sar    $0x8,%r8d
    2f4d:	81 ff ff 00 00 00    	cmp    $0xff,%edi
    2f53:	0f 8e 57 04 00 00    	jle    33b0 <float iq_compact_dot<false>(int, iq_compact_block const*, block_q8_K const*)+0x480>
    2f59:	48 89 d0             	mov    %rdx,%rax
    2f5c:	ba 0f 0f 0f 0f       	mov    $0xf0f0f0f,%edx
    2f61:	c5 c0 57 ff          	vxorps %xmm7,%xmm7,%xmm7
    2f65:	48 89 f1             	mov    %rsi,%rcx
    2f68:	c5 79 6f 2d d0 33 00 	vmovdqa 0x33d0(%rip),%xmm13        # 6340 <iq3s_grid+0x840>
    2f6f:	00 
    2f70:	c5 fc 29 7c 24 e8    	vmovaps %ymm7,-0x18(%rsp)
    2f76:	31 f6                	xor    %esi,%esi
    2f78:	c5 f9 6e fa          	vmovd  %edx,%xmm7
    2f7c:	c5 79 6f 35 8c 33 00 	vmovdqa 0x338c(%rip),%xmm14        # 6310 <iq3s_grid+0x810>
    2f83:	00 
    2f84:	c5 79 70 ff 00       	vpshufd $0x0,%xmm7,%xmm15
    2f89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2f90:	0f b6 b9 84 00 00 00 	movzbl 0x84(%rcx),%edi
    2f97:	83 c6 01             	add    $0x1,%esi
    2f9a:	48 81 c1 90 00 00 00 	add    $0x90,%rcx
    2fa1:	48 05 24 01 00 00    	add    $0x124,%rax
    2fa7:	c5 fa 6f 81 70 ff ff 	vmovdqu -0x90(%rcx),%xmm0
    2fae:	ff 
    2faf:	c5 fa 6f a0 e0 fe ff 	vmovdqu -0x120(%rax),%xmm4
    2fb6:	ff 
    2fb7:	c5 fa 6f 98 f0 fe ff 	vmovdqu -0x110(%rax),%xmm3
    2fbe:	ff 
    2fbf:	c4 c1 79 db cf       	vpand  %xmm15,%xmm0,%xmm1
    2fc4:	c5 f9 71 d0 04       	vpsrlw $0x4,%xmm0,%xmm0
    2fc9:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    2fce:	89 fa                	mov    %edi,%edx
    2fd0:	40 c0 ef 04          	shr    $0x4,%dil
    2fd4:	c4 e2 59 08 e1       	vpsignb %xmm1,%xmm4,%xmm4
    2fd9:	c4 c1 79 db c7       	vpand  %xmm15,%xmm0,%xmm0
    2fde:	83 e2 0f             	and    $0xf,%edx
    2fe1:	40 0f b6 ff          	movzbl %dil,%edi
    2fe5:	c4 e2 11 00 c0       	vpshufb %xmm0,%xmm13,%xmm0
    2fea:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    2fef:	8d 54 12 01          	lea    0x1(%rdx,%rdx,1),%edx
    2ff3:	c4 e2 71 04 cc       	vpmaddubsw %xmm4,%xmm1,%xmm1
    2ff8:	c4 e2 61 08 d8       	vpsignb %xmm0,%xmm3,%xmm3
    2ffd:	c4 e2 79 1c c0       	vpabsb %xmm0,%xmm0
    3002:	c5 f9 6e d2          	vmovd  %edx,%xmm2
    3006:	c4 e2 79 04 c3       	vpmaddubsw %xmm3,%xmm0,%xmm0
    300b:	c5 7a 6f 50 a0       	vmovdqu -0x60(%rax),%xmm10
    3010:	8d 54 3f 01          	lea    0x1(%rdi,%rdi,1),%edx
    3014:	c4 c2 69 00 d6       	vpshufb %xmm14,%xmm2,%xmm2
    3019:	c5 f1 f5 f2          	vpmaddwd %xmm2,%xmm1,%xmm6
    301d:	c5 fa 6f 49 80       	vmovdqu -0x80(%rcx),%xmm1
    3022:	c5 f9 f5 ea          	vpmaddwd %xmm2,%xmm0,%xmm5
    3026:	c5 f9 6e c2          	vmovd  %edx,%xmm0
    302a:	0f b6 51 f5          	movzbl -0xb(%rcx),%edx
    302e:	c5 f9 7f 74 24 d8    	vmovdqa %xmm6,-0x28(%rsp)
    3034:	c4 c1 71 db d7       	vpand  %xmm15,%xmm1,%xmm2
    3039:	c5 f1 71 d1 04       	vpsrlw $0x4,%xmm1,%xmm1
    303e:	c4 c2 79 00 c6       	vpshufb %xmm14,%xmm0,%xmm0
    3043:	c5 fa 6f 98 10 ff ff 	vmovdqu -0xf0(%rax),%xmm3
    304a:	ff 
    304b:	c4 c1 71 db cf       	vpand  %xmm15,%xmm1,%xmm1
    3050:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    3055:	c5 fa 6f a0 00 ff ff 	vmovdqu -0x100(%rax),%xmm4
    305c:	ff 
    305d:	c4 e2 11 00 d2       	vpshufb %xmm2,%xmm13,%xmm2
    3062:	c4 e2 61 08 d9       	vpsignb %xmm1,%xmm3,%xmm3
    3067:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    306c:	c4 e2 71 04 cb       	vpmaddubsw %xmm3,%xmm1,%xmm1
    3071:	c5 fa 6f 98 30 ff ff 	vmovdqu -0xd0(%rax),%xmm3
    3078:	ff 
    3079:	c4 e2 59 08 e2       	vpsignb %xmm2,%xmm4,%xmm4
    307e:	89 d7                	mov    %edx,%edi
    3080:	c4 e2 79 1c d2       	vpabsb %xmm2,%xmm2
    3085:	c4 e2 69 04 d4       	vpmaddubsw %xmm4,%xmm2,%xmm2
    308a:	c5 f1 f5 e0          	vpmaddwd %xmm0,%xmm1,%xmm4
    308e:	c5 fa 6f 49 90       	vmovdqu -0x70(%rcx),%xmm1
    3093:	83 e7 0f             	and    $0xf,%edi
    3096:	c0 ea 04             	shr    $0x4,%dl
    3099:	8d 7c 3f 01          	lea    0x1(%rdi,%rdi,1),%edi
    309d:	c5 69 f5 d8          	vpmaddwd %xmm0,%xmm2,%xmm11
    30a1:	0f b6 d2             	movzbl %dl,%edx
    30a4:	c5 fa 6f 78 b0       	vmovdqu -0x50(%rax),%xmm7
    30a9:	c5 f9 7f 64 24 a8    	vmovdqa %xmm4,-0x58(%rsp)
    30af:	c4 c1 71 db c7       	vpand  %xmm15,%xmm1,%xmm0
    30b4:	8d 54 12 01          	lea    0x1(%rdx,%rdx,1),%edx
    30b8:	c5 f9 6e d7          	vmovd  %edi,%xmm2
    30bc:	c5 f1 71 d1 04       	vpsrlw $0x4,%xmm1,%xmm1
    30c1:	c4 e2 11 00 c0       	vpshufb %xmm0,%xmm13,%xmm0
    30c6:	c4 c2 69 00 d6       	vpshufb %xmm14,%xmm2,%xmm2
    30cb:	c5 fa 6f a0 20 ff ff 	vmovdqu -0xe0(%rax),%xmm4
    30d2:	ff 
    30d3:	c4 c1 71 db cf       	vpand  %xmm15,%xmm1,%xmm1
    30d8:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    30dd:	c4 e2 61 08 d9       	vpsignb %xmm1,%xmm3,%xmm3
    30e2:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    30e7:	c4 e2 59 08 e0       	vpsignb %xmm0,%xmm4,%xmm4
    30ec:	c4 e2 71 04 cb       	vpmaddubsw %xmm3,%xmm1,%xmm1
    30f1:	c4 e2 79 1c c0       	vpabsb %xmm0,%xmm0
    30f6:	c4 e2 79 04 c4       	vpmaddubsw %xmm4,%xmm0,%xmm0
    30fb:	c5 f1 f5 da          	vpmaddwd %xmm2,%xmm1,%xmm3
    30ff:	c5 f9 f5 c2          	vpmaddwd %xmm2,%xmm0,%xmm0
    3103:	c5 f9 6e d2          	vmovd  %edx,%xmm2
    3107:	c5 fa 6f 49 a0       	vmovdqu -0x60(%rcx),%xmm1
    310c:	0f b6 51 f6          	movzbl -0xa(%rcx),%edx
    3110:	c5 f9 7f 5c 24 98    	vmovdqa %xmm3,-0x68(%rsp)
    3116:	c4 c2 69 00 d6       	vpshufb %xmm14,%xmm2,%xmm2
    311b:	c5 fa 6f a0 40 ff ff 	vmovdqu -0xc0(%rax),%xmm4
    3122:	ff 
    3123:	c4 41 71 db cf       	vpand  %xmm15,%xmm1,%xmm9
    3128:	c5 f1 71 d1 04       	vpsrlw $0x4,%xmm1,%xmm1
    312d:	c4 42 11 00 c9       	vpshufb %xmm9,%xmm13,%xmm9
    3132:	c5 fa 6f 98 50 ff ff 	vmovdqu -0xb0(%rax),%xmm3
    3139:	ff 
    313a:	c4 c1 71 db cf       	vpand  %xmm15,%xmm1,%xmm1
    313f:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    3144:	c5 f9 7f 6c 24 c8    	vmovdqa %xmm5,-0x38(%rsp)
    314a:	c4 c2 59 08 e1       	vpsignb %xmm9,%xmm4,%xmm4
    314f:	c4 42 79 1c c9       	vpabsb %xmm9,%xmm9
    3154:	c4 62 31 04 cc       	vpmaddubsw %xmm4,%xmm9,%xmm9
    3159:	c5 fa 6f 61 b0       	vmovdqu -0x50(%rcx),%xmm4
    315e:	c4 e2 61 08 d9       	vpsignb %xmm1,%xmm3,%xmm3
    3163:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    3168:	c5 31 f5 ca          	vpmaddwd %xmm2,%xmm9,%xmm9
    316c:	c4 e2 71 04 cb       	vpmaddubsw %xmm3,%xmm1,%xmm1
    3171:	c5 f1 f5 d2          	vpmaddwd %xmm2,%xmm1,%xmm2
    3175:	89 d7                	mov    %edx,%edi
    3177:	c0 ea 04             	shr    $0x4,%dl
    317a:	c5 fa 6f 98 60 ff ff 	vmovdqu -0xa0(%rax),%xmm3
    3181:	ff 
    3182:	c5 f9 7f 54 24 88    	vmovdqa %xmm2,-0x78(%rsp)
    3188:	83 e7 0f             	and    $0xf,%edi
    318b:	c4 c1 59 db f7       	vpand  %xmm15,%xmm4,%xmm6
    3190:	0f b6 d2             	movzbl %dl,%edx
    3193:	c5 fa 6f 90 70 ff ff 	vmovdqu -0x90(%rax),%xmm2
    319a:	ff 
    319b:	8d 7c 3f 01          	lea    0x1(%rdi,%rdi,1),%edi
    319f:	c5 d9 71 d4 04       	vpsrlw $0x4,%xmm4,%xmm4
    31a4:	c4 e2 11 00 f6       	vpshufb %xmm6,%xmm13,%xmm6
    31a9:	8d 54 12 01          	lea    0x1(%rdx,%rdx,1),%edx
    31ad:	c4 e2 61 08 de       	vpsignb %xmm6,%xmm3,%xmm3
    31b2:	c5 f9 6e cf          	vmovd  %edi,%xmm1
    31b6:	c4 c1 59 db e7       	vpand  %xmm15,%xmm4,%xmm4
    31bb:	c4 e2 79 1c f6       	vpabsb %xmm6,%xmm6
    31c0:	c4 e2 11 00 e4       	vpshufb %xmm4,%xmm13,%xmm4
    31c5:	c4 e2 49 04 f3       	vpmaddubsw %xmm3,%xmm6,%xmm6
    31ca:	c4 e2 69 08 d4       	vpsignb %xmm4,%xmm2,%xmm2
    31cf:	c4 e2 79 1c e4       	vpabsb %xmm4,%xmm4
    31d4:	c4 e2 59 04 e2       	vpmaddubsw %xmm2,%xmm4,%xmm4
    31d9:	c4 c2 71 00 ce       	vpshufb %xmm14,%xmm1,%xmm1
    31de:	c5 c9 f5 f1          	vpmaddwd %xmm1,%xmm6,%xmm6
    31e2:	c5 d9 f5 e1          	vpmaddwd %xmm1,%xmm4,%xmm4
    31e6:	c5 f9 6e ca          	vmovd  %edx,%xmm1
    31ea:	0f b6 51 f7          	movzbl -0x9(%rcx),%edx
    31ee:	c5 79 7f 5c 24 b8    	vmovdqa %xmm11,-0x48(%rsp)
    31f4:	c5 fa 6f 51 c0       	vmovdqu -0x40(%rcx),%xmm2
    31f9:	c4 c2 71 00 ce       	vpshufb %xmm14,%xmm1,%xmm1
    31fe:	c4 c1 79 fe c1       	vpaddd %xmm9,%xmm0,%xmm0
    3203:	c5 fa 6f 58 90       	vmovdqu -0x70(%rax),%xmm3
    3208:	c4 41 69 db c7       	vpand  %xmm15,%xmm2,%xmm8
    320d:	c5 e9 71 d2 04       	vpsrlw $0x4,%xmm2,%xmm2
    3212:	c5 fa 6f 68 80       	vmovdqu -0x80(%rax),%xmm5
    3217:	c4 42 11 00 c0       	vpshufb %xmm8,%xmm13,%xmm8
    321c:	89 d7                	mov    %edx,%edi
    321e:	c4 c1 69 db d7       	vpand  %xmm15,%xmm2,%xmm2
    3223:	c4 e2 11 00 d2       	vpshufb %xmm2,%xmm13,%xmm2
    3228:	c4 e2 61 08 da       	vpsignb %xmm2,%xmm3,%xmm3
    322d:	c4 e2 79 1c d2       	vpabsb %xmm2,%xmm2
    3232:	c4 e2 69 04 d3       	vpmaddubsw %xmm3,%xmm2,%xmm2
    3237:	c5 fa 6f 59 d0       	vmovdqu -0x30(%rcx),%xmm3
    323c:	83 e7 0f             	and    $0xf,%edi
    323f:	c4 c2 51 08 e8       	vpsignb %xmm8,%xmm5,%xmm5
    3244:	8d 7c 3f 01          	lea    0x1(%rdi,%rdi,1),%edi
    3248:	c4 42 79 1c c0       	vpabsb %xmm8,%xmm8
    324d:	c4 62 39 04 c5       	vpmaddubsw %xmm5,%xmm8,%xmm8
    3252:	c4 c1 61 db ef       	vpand  %xmm15,%xmm3,%xmm5
    3257:	c5 e1 71 d3 04       	vpsrlw $0x4,%xmm3,%xmm3
    325c:	c5 39 f5 c1          	vpmaddwd %xmm1,%xmm8,%xmm8
    3260:	c5 e9 f5 d1          	vpmaddwd %xmm1,%xmm2,%xmm2
    3264:	c4 e2 11 00 ed       	vpshufb %xmm5,%xmm13,%xmm5
    3269:	c5 f9 6e cf          	vmovd  %edi,%xmm1
    326d:	c0 ea 04             	shr    $0x4,%dl
    3270:	c4 62 29 08 d5       	vpsignb %xmm5,%xmm10,%xmm10
    3275:	c4 c1 61 db df       	vpand  %xmm15,%xmm3,%xmm3
    327a:	c4 e2 79 1c ed       	vpabsb %xmm5,%xmm5
    327f:	c4 e2 11 00 db       	vpshufb %xmm3,%xmm13,%xmm3
    3284:	0f b6 d2             	movzbl %dl,%edx
    3287:	c4 e2 41 08 fb       	vpsignb %xmm3,%xmm7,%xmm7
    328c:	c4 c2 51 04 ea       	vpmaddubsw %xmm10,%xmm5,%xmm5
    3291:	c4 e2 79 1c db       	vpabsb %xmm3,%xmm3
    3296:	c4 c1 79 fe c0       	vpaddd %xmm8,%xmm0,%xmm0
    329b:	c4 c2 71 00 ce       	vpshufb %xmm14,%xmm1,%xmm1
    32a0:	c4 e2 61 04 df       	vpmaddubsw %xmm7,%xmm3,%xmm3
    32a5:	c5 d1 f5 e9          	vpmaddwd %xmm1,%xmm5,%xmm5
    32a9:	8d 54 12 01          	lea    0x1(%rdx,%rdx,1),%edx
    32ad:	c5 e1 f5 d9          	vpmaddwd %xmm1,%xmm3,%xmm3
    32b1:	c5 fa 6f 49 e0       	vmovdqu -0x20(%rcx),%xmm1
    32b6:	c5 7a 6f 60 c0       	vmovdqu -0x40(%rax),%xmm12
    32bb:	c5 79 6e d2          	vmovd  %edx,%xmm10
    32bf:	c4 c1 71 db ff       	vpand  %xmm15,%xmm1,%xmm7
    32c4:	c5 7a 6f 58 d0       	vmovdqu -0x30(%rax),%xmm11
    32c9:	c4 e2 11 00 ff       	vpshufb %xmm7,%xmm13,%xmm7
    32ce:	c4 42 29 00 d6       	vpshufb %xmm14,%xmm10,%xmm10
    32d3:	c4 62 19 08 e7       	vpsignb %xmm7,%xmm12,%xmm12
    32d8:	c4 e2 79 1c ff       	vpabsb %xmm7,%xmm7
    32dd:	c4 c2 41 04 fc       	vpmaddubsw %xmm12,%xmm7,%xmm7
    32e2:	c5 f1 71 d1 04       	vpsrlw $0x4,%xmm1,%xmm1
    32e7:	c4 c1 41 f5 fa       	vpmaddwd %xmm10,%xmm7,%xmm7
    32ec:	c4 c1 71 db cf       	vpand  %xmm15,%xmm1,%xmm1
    32f1:	c4 e2 11 00 c9       	vpshufb %xmm1,%xmm13,%xmm1
    32f6:	c4 62 21 08 d9       	vpsignb %xmm1,%xmm11,%xmm11
    32fb:	c5 f9 fe c7          	vpaddd %xmm7,%xmm0,%xmm0
    32ff:	c4 e2 79 1c c9       	vpabsb %xmm1,%xmm1
    3304:	c4 c2 71 04 cb       	vpmaddubsw %xmm11,%xmm1,%xmm1
    3309:	c5 79 6f 5c 24 b8    	vmovdqa -0x48(%rsp),%xmm11
    330f:	c5 a1 fe 7c 24 d8    	vpaddd -0x28(%rsp),%xmm11,%xmm7
    3315:	c4 c1 71 f5 ca       	vpmaddwd %xmm10,%xmm1,%xmm1
    331a:	c5 c1 fe f6          	vpaddd %xmm6,%xmm7,%xmm6
    331e:	c5 f9 6f 7c 24 88    	vmovdqa -0x78(%rsp),%xmm7
    3324:	c5 c9 fe ed          	vpaddd %xmm5,%xmm6,%xmm5
    3328:	c5 f9 6f 74 24 a8    	vmovdqa -0x58(%rsp),%xmm6
    332e:	c5 f9 fe c5          	vpaddd %xmm5,%xmm0,%xmm0
    3332:	c5 c9 fe 6c 24 c8    	vpaddd -0x38(%rsp),%xmm6,%xmm5
    3338:	c5 d1 fe e4          	vpaddd %xmm4,%xmm5,%xmm4
    333c:	c5 d9 fe db          	vpaddd %xmm3,%xmm4,%xmm3
    3340:	c5 c1 fe 64 24 98    	vpaddd -0x68(%rsp),%xmm7,%xmm4
    3346:	c5 d9 fe d2          	vpaddd %xmm2,%xmm4,%xmm2
    334a:	c5 e9 fe c9          	vpaddd %xmm1,%xmm2,%xmm1
    334e:	c5 e1 fe c9          	vpaddd %xmm1,%xmm3,%xmm1
    3352:	c4 e3 7d 18 c1 01    	vinsertf128 $0x1,%xmm1,%ymm0,%ymm0
    3358:	c5 fc 5b c8          	vcvtdq2ps %ymm0,%ymm1
    335c:	c5 fa 10 41 f0       	vmovss -0x10(%rcx),%xmm0
    3361:	c5 fa 59 80 dc fe ff 	vmulss -0x124(%rax),%xmm0,%xmm0
    3368:	ff 
    3369:	c5 f8 c6 c0 00       	vshufps $0x0,%xmm0,%xmm0,%xmm0
    336e:	c4 e3 7d 18 c0 01    	vinsertf128 $0x1,%xmm0,%ymm0,%ymm0
    3374:	c5 fc 59 c1          	vmulps %ymm1,%ymm0,%ymm0
    3378:	c5 fc 58 74 24 e8    	vaddps -0x18(%rsp),%ymm0,%ymm6
    337e:	c5 fc 29 74 24 e8    	vmovaps %ymm6,-0x18(%rsp)
    3384:	44 39 c6             	cmp    %r8d,%esi
    3387:	0f 8c 03 fc ff ff    	jl     2f90 <float iq_compact_dot<false>(int, iq_compact_block const*, block_q8_K const*)+0x60>
    338d:	c5 fc 28 fe          	vmovaps %ymm6,%ymm7
    3391:	c4 e3 7d 19 f8 01    	vextractf128 $0x1,%ymm7,%xmm0
    3397:	c5 c0 58 c0          	vaddps %xmm0,%xmm7,%xmm0
    339b:	c5 f8 12 c8          	vmovhlps %xmm0,%xmm0,%xmm1
    339f:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    33a3:	c5 fa 16 c8          	vmovshdup %xmm0,%xmm1
    33a7:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    33ab:	c5 f8 77             	vzeroupper
    33ae:	c9                   	leave
    33af:	c3                   	ret
    33b0:	c5 c0 57 ff          	vxorps %xmm7,%xmm7,%xmm7
    33b4:	c5 fc 29 7c 24 e8    	vmovaps %ymm7,-0x18(%rsp)
    33ba:	eb d5                	jmp    3391 <float iq_compact_dot<false>(int, iq_compact_block const*, block_q8_K const*)+0x461>
    33bc:	0f 1f 40 00          	nopl   0x0(%rax)

