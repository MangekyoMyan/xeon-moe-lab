
/workspace/scratch/1f624f9ac929/audit/next_optimization_20261116/root/offset4_run/probe:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	48 83 ec 08          	sub    $0x8,%rsp
    1004:	48 8b 05 cd 6f 00 00 	mov    0x6fcd(%rip),%rax        # 7fd8 <__gmon_start__@Base>
    100b:	48 85 c0             	test   %rax,%rax
    100e:	74 02                	je     1012 <_init+0x12>
    1010:	ff d0                	call   *%rax
    1012:	48 83 c4 08          	add    $0x8,%rsp
    1016:	c3                   	ret

Disassembly of section .plt:

0000000000001020 <printf@plt-0x10>:
    1020:	ff 35 ca 6f 00 00    	push   0x6fca(%rip)        # 7ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	ff 25 cc 6f 00 00    	jmp    *0x6fcc(%rip)        # 7ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000001030 <printf@plt>:
    1030:	ff 25 ca 6f 00 00    	jmp    *0x6fca(%rip)        # 8000 <printf@GLIBC_2.2.5>
    1036:	68 00 00 00 00       	push   $0x0
    103b:	e9 e0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001040 <std::runtime_error::runtime_error(char const*)@plt>:
    1040:	ff 25 c2 6f 00 00    	jmp    *0x6fc2(%rip)        # 8008 <std::runtime_error::runtime_error(char const*)@GLIBCXX_3.4.21>
    1046:	68 01 00 00 00       	push   $0x1
    104b:	e9 d0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001050 <__cxa_begin_catch@plt>:
    1050:	ff 25 ba 6f 00 00    	jmp    *0x6fba(%rip)        # 8010 <__cxa_begin_catch@CXXABI_1.3>
    1056:	68 02 00 00 00       	push   $0x2
    105b:	e9 c0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001060 <__cxa_allocate_exception@plt>:
    1060:	ff 25 b2 6f 00 00    	jmp    *0x6fb2(%rip)        # 8018 <__cxa_allocate_exception@CXXABI_1.3>
    1066:	68 03 00 00 00       	push   $0x3
    106b:	e9 b0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001070 <std::__throw_length_error(char const*)@plt>:
    1070:	ff 25 aa 6f 00 00    	jmp    *0x6faa(%rip)        # 8020 <std::__throw_length_error(char const*)@GLIBCXX_3.4>
    1076:	68 04 00 00 00       	push   $0x4
    107b:	e9 a0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001080 <__cxa_free_exception@plt>:
    1080:	ff 25 a2 6f 00 00    	jmp    *0x6fa2(%rip)        # 8028 <__cxa_free_exception@CXXABI_1.3>
    1086:	68 05 00 00 00       	push   $0x5
    108b:	e9 90 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001090 <operator new(unsigned long)@plt>:
    1090:	ff 25 9a 6f 00 00    	jmp    *0x6f9a(%rip)        # 8030 <operator new(unsigned long)@GLIBCXX_3.4>
    1096:	68 06 00 00 00       	push   $0x6
    109b:	e9 80 ff ff ff       	jmp    1020 <_init+0x20>

00000000000010a0 <operator delete(void*, unsigned long)@plt>:
    10a0:	ff 25 92 6f 00 00    	jmp    *0x6f92(%rip)        # 8038 <operator delete(void*, unsigned long)@CXXABI_1.3.9>
    10a6:	68 07 00 00 00       	push   $0x7
    10ab:	e9 70 ff ff ff       	jmp    1020 <_init+0x20>

00000000000010b0 <ldexpf@plt>:
    10b0:	ff 25 8a 6f 00 00    	jmp    *0x6f8a(%rip)        # 8040 <ldexpf@GLIBC_2.2.5>
    10b6:	68 08 00 00 00       	push   $0x8
    10bb:	e9 60 ff ff ff       	jmp    1020 <_init+0x20>

00000000000010c0 <fprintf@plt>:
    10c0:	ff 25 82 6f 00 00    	jmp    *0x6f82(%rip)        # 8048 <fprintf@GLIBC_2.2.5>
    10c6:	68 09 00 00 00       	push   $0x9
    10cb:	e9 50 ff ff ff       	jmp    1020 <_init+0x20>

00000000000010d0 <__cxa_end_catch@plt>:
    10d0:	ff 25 7a 6f 00 00    	jmp    *0x6f7a(%rip)        # 8050 <__cxa_end_catch@CXXABI_1.3>
    10d6:	68 0a 00 00 00       	push   $0xa
    10db:	e9 40 ff ff ff       	jmp    1020 <_init+0x20>

00000000000010e0 <__cxa_throw@plt>:
    10e0:	ff 25 72 6f 00 00    	jmp    *0x6f72(%rip)        # 8058 <__cxa_throw@CXXABI_1.3>
    10e6:	68 0b 00 00 00       	push   $0xb
    10eb:	e9 30 ff ff ff       	jmp    1020 <_init+0x20>

00000000000010f0 <_Unwind_Resume@plt>:
    10f0:	ff 25 6a 6f 00 00    	jmp    *0x6f6a(%rip)        # 8060 <_Unwind_Resume@GCC_3.0>
    10f6:	68 0c 00 00 00       	push   $0xc
    10fb:	e9 20 ff ff ff       	jmp    1020 <_init+0x20>

Disassembly of section .plt.got:

0000000000001100 <__cxa_finalize@plt>:
    1100:	ff 25 b2 6e 00 00    	jmp    *0x6eb2(%rip)        # 7fb8 <__cxa_finalize@GLIBC_2.2.5>
    1106:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000001110 <main.cold>:
    1110:	bf 10 00 00 00       	mov    $0x10,%edi
    1115:	c5 f8 77             	vzeroupper
    1118:	e8 43 ff ff ff       	call   1060 <__cxa_allocate_exception@plt>
    111d:	48 8d 35 51 3f 00 00 	lea    0x3f51(%rip),%rsi        # 5075 <_fini+0xffd>
    1124:	48 89 c7             	mov    %rax,%rdi
    1127:	48 89 c3             	mov    %rax,%rbx
    112a:	e8 11 ff ff ff       	call   1040 <std::runtime_error::runtime_error(char const*)@plt>
    112f:	48 8b 15 92 6e 00 00 	mov    0x6e92(%rip),%rdx        # 7fc8 <std::runtime_error::~runtime_error()@GLIBCXX_3.4>
    1136:	48 8d 35 53 6c 00 00 	lea    0x6c53(%rip),%rsi        # 7d90 <typeinfo for std::runtime_error@GLIBCXX_3.4>
    113d:	48 89 df             	mov    %rbx,%rdi
    1140:	e8 9b ff ff ff       	call   10e0 <__cxa_throw@plt>
    1145:	bf 10 00 00 00       	mov    $0x10,%edi
    114a:	e8 11 ff ff ff       	call   1060 <__cxa_allocate_exception@plt>
    114f:	48 8d 35 0b 3f 00 00 	lea    0x3f0b(%rip),%rsi        # 5061 <_fini+0xfe9>
    1156:	48 89 c7             	mov    %rax,%rdi
    1159:	48 89 c3             	mov    %rax,%rbx
    115c:	e8 df fe ff ff       	call   1040 <std::runtime_error::runtime_error(char const*)@plt>
    1161:	48 8b 15 60 6e 00 00 	mov    0x6e60(%rip),%rdx        # 7fc8 <std::runtime_error::~runtime_error()@GLIBCXX_3.4>
    1168:	48 8d 35 21 6c 00 00 	lea    0x6c21(%rip),%rsi        # 7d90 <typeinfo for std::runtime_error@GLIBCXX_3.4>
    116f:	48 89 df             	mov    %rbx,%rdi
    1172:	e8 69 ff ff ff       	call   10e0 <__cxa_throw@plt>
    1177:	48 89 c3             	mov    %rax,%rbx
    117a:	49 89 d5             	mov    %rdx,%r13
    117d:	c5 f8 77             	vzeroupper
    1180:	48 83 bd e0 ea ff ff 	cmpq   $0x0,-0x1520(%rbp)
    1187:	00 
    1188:	74 13                	je     119d <main.cold+0x8d>
    118a:	48 8b b5 80 ea ff ff 	mov    -0x1580(%rbp),%rsi
    1191:	48 8b bd e0 ea ff ff 	mov    -0x1520(%rbp),%rdi
    1198:	e8 03 ff ff ff       	call   10a0 <operator delete(void*, unsigned long)@plt>
    119d:	4d 85 f6             	test   %r14,%r14
    11a0:	0f 84 df 00 00 00    	je     1285 <main.cold+0x175>
    11a6:	48 8b 85 78 ea ff ff 	mov    -0x1588(%rbp),%rax
    11ad:	48 89 85 90 ea ff ff 	mov    %rax,-0x1570(%rbp)
    11b4:	48 8b 85 b8 ea ff ff 	mov    -0x1548(%rbp),%rax
    11bb:	48 89 85 a0 ea ff ff 	mov    %rax,-0x1560(%rbp)
    11c2:	e9 a1 00 00 00       	jmp    1268 <main.cold+0x158>
    11c7:	bf 10 00 00 00       	mov    $0x10,%edi
    11cc:	e8 8f fe ff ff       	call   1060 <__cxa_allocate_exception@plt>
    11d1:	48 8d 35 28 3e 00 00 	lea    0x3e28(%rip),%rsi        # 5000 <_fini+0xf88>
    11d8:	48 89 c7             	mov    %rax,%rdi
    11db:	49 89 c5             	mov    %rax,%r13
    11de:	e8 5d fe ff ff       	call   1040 <std::runtime_error::runtime_error(char const*)@plt>
    11e3:	48 8b 15 de 6d 00 00 	mov    0x6dde(%rip),%rdx        # 7fc8 <std::runtime_error::~runtime_error()@GLIBCXX_3.4>
    11ea:	48 8d 35 9f 6b 00 00 	lea    0x6b9f(%rip),%rsi        # 7d90 <typeinfo for std::runtime_error@GLIBCXX_3.4>
    11f1:	4c 89 ef             	mov    %r13,%rdi
    11f4:	e8 e7 fe ff ff       	call   10e0 <__cxa_throw@plt>
    11f9:	48 89 c3             	mov    %rax,%rbx
    11fc:	48 89 d0             	mov    %rdx,%rax
    11ff:	c5 f8 77             	vzeroupper
    1202:	48 83 e8 01          	sub    $0x1,%rax
    1206:	48 89 df             	mov    %rbx,%rdi
    1209:	0f 85 cc 00 00 00    	jne    12db <main.cold+0x1cb>
    120f:	e8 3c fe ff ff       	call   1050 <__cxa_begin_catch@plt>
    1214:	48 89 c7             	mov    %rax,%rdi
    1217:	48 8b 00             	mov    (%rax),%rax
    121a:	ff 50 10             	call   *0x10(%rax)
    121d:	48 8b 3d 5c 6e 00 00 	mov    0x6e5c(%rip),%rdi        # 8080 <stderr@GLIBC_2.2.5>
    1224:	48 8d 35 8c 3e 00 00 	lea    0x3e8c(%rip),%rsi        # 50b7 <_fini+0x103f>
    122b:	48 89 c2             	mov    %rax,%rdx
    122e:	31 c0                	xor    %eax,%eax
    1230:	e8 8b fe ff ff       	call   10c0 <fprintf@plt>
    1235:	e8 96 fe ff ff       	call   10d0 <__cxa_end_catch@plt>
    123a:	b8 01 00 00 00       	mov    $0x1,%eax
    123f:	e9 56 15 00 00       	jmp    279a <main+0x135a>
    1244:	49 89 d4             	mov    %rdx,%r12
    1247:	4c 89 ef             	mov    %r13,%rdi
    124a:	48 89 c3             	mov    %rax,%rbx
    124d:	c5 f8 77             	vzeroupper
    1250:	e8 2b fe ff ff       	call   1080 <__cxa_free_exception@plt>
    1255:	4c 89 e0             	mov    %r12,%rax
    1258:	eb a8                	jmp    1202 <main.cold+0xf2>
    125a:	48 c7 85 90 ea ff ff 	movq   $0x124,-0x1570(%rbp)
    1261:	24 01 00 00 
    1265:	c5 f8 77             	vzeroupper
    1268:	48 8b b5 90 ea ff ff 	mov    -0x1570(%rbp),%rsi
    126f:	4c 89 f7             	mov    %r14,%rdi
    1272:	e8 29 fe ff ff       	call   10a0 <operator delete(void*, unsigned long)@plt>
    1277:	48 8b 85 a0 ea ff ff 	mov    -0x1560(%rbp),%rax
    127e:	48 89 85 b8 ea ff ff 	mov    %rax,-0x1548(%rbp)
    1285:	4d 85 e4             	test   %r12,%r12
    1288:	74 27                	je     12b1 <main.cold+0x1a1>
    128a:	48 8b 85 b8 ea ff ff 	mov    -0x1548(%rbp),%rax
    1291:	48 89 85 a0 ea ff ff 	mov    %rax,-0x1560(%rbp)
    1298:	4c 8b b5 a0 ea ff ff 	mov    -0x1560(%rbp),%r14
    129f:	4c 89 e7             	mov    %r12,%rdi
    12a2:	4c 89 f6             	mov    %r14,%rsi
    12a5:	e8 f6 fd ff ff       	call   10a0 <operator delete(void*, unsigned long)@plt>
    12aa:	4c 89 b5 b8 ea ff ff 	mov    %r14,-0x1548(%rbp)
    12b1:	4d 85 ff             	test   %r15,%r15
    12b4:	74 1d                	je     12d3 <main.cold+0x1c3>
    12b6:	48 8b 85 b8 ea ff ff 	mov    -0x1548(%rbp),%rax
    12bd:	48 89 85 a0 ea ff ff 	mov    %rax,-0x1560(%rbp)
    12c4:	48 8b b5 a0 ea ff ff 	mov    -0x1560(%rbp),%rsi
    12cb:	4c 89 ff             	mov    %r15,%rdi
    12ce:	e8 cd fd ff ff       	call   10a0 <operator delete(void*, unsigned long)@plt>
    12d3:	4c 89 e8             	mov    %r13,%rax
    12d6:	e9 27 ff ff ff       	jmp    1202 <main.cold+0xf2>
    12db:	e8 10 fe ff ff       	call   10f0 <_Unwind_Resume@plt>
    12e0:	48 89 c3             	mov    %rax,%rbx
    12e3:	c5 f8 77             	vzeroupper
    12e6:	e8 e5 fd ff ff       	call   10d0 <__cxa_end_catch@plt>
    12eb:	48 89 df             	mov    %rbx,%rdi
    12ee:	e8 fd fd ff ff       	call   10f0 <_Unwind_Resume@plt>
    12f3:	bf 10 00 00 00       	mov    $0x10,%edi
    12f8:	e8 63 fd ff ff       	call   1060 <__cxa_allocate_exception@plt>
    12fd:	48 8d 35 21 3d 00 00 	lea    0x3d21(%rip),%rsi        # 5025 <_fini+0xfad>
    1304:	48 89 c7             	mov    %rax,%rdi
    1307:	49 89 c5             	mov    %rax,%r13
    130a:	e8 31 fd ff ff       	call   1040 <std::runtime_error::runtime_error(char const*)@plt>
    130f:	48 8b 15 b2 6c 00 00 	mov    0x6cb2(%rip),%rdx        # 7fc8 <std::runtime_error::~runtime_error()@GLIBCXX_3.4>
    1316:	48 8d 35 73 6a 00 00 	lea    0x6a73(%rip),%rsi        # 7d90 <typeinfo for std::runtime_error@GLIBCXX_3.4>
    131d:	4c 89 ef             	mov    %r13,%rdi
    1320:	e8 bb fd ff ff       	call   10e0 <__cxa_throw@plt>
    1325:	e9 1a ff ff ff       	jmp    1244 <main.cold+0x134>
    132a:	c5 79 7e d9          	vmovd  %xmm11,%ecx
    132e:	c5 f9 7e c8          	vmovd  %xmm1,%eax
    1332:	c4 61 f9 7e d2       	vmovq  %xmm10,%rdx
    1337:	e9 de 03 00 00       	jmp    171a <main+0x2da>
    133c:	bf 10 00 00 00       	mov    $0x10,%edi
    1341:	e8 1a fd ff ff       	call   1060 <__cxa_allocate_exception@plt>
    1346:	48 8d 35 c5 3c 00 00 	lea    0x3cc5(%rip),%rsi        # 5012 <_fini+0xf9a>
    134d:	48 89 c7             	mov    %rax,%rdi
    1350:	49 89 c5             	mov    %rax,%r13
    1353:	e8 e8 fc ff ff       	call   1040 <std::runtime_error::runtime_error(char const*)@plt>
    1358:	48 8b 15 69 6c 00 00 	mov    0x6c69(%rip),%rdx        # 7fc8 <std::runtime_error::~runtime_error()@GLIBCXX_3.4>
    135f:	48 8d 35 2a 6a 00 00 	lea    0x6a2a(%rip),%rsi        # 7d90 <typeinfo for std::runtime_error@GLIBCXX_3.4>
    1366:	4c 89 ef             	mov    %r13,%rdi
    1369:	e8 72 fd ff ff       	call   10e0 <__cxa_throw@plt>
    136e:	e9 d1 fe ff ff       	jmp    1244 <main.cold+0x134>
    1373:	bf 10 00 00 00       	mov    $0x10,%edi
    1378:	e8 e3 fc ff ff       	call   1060 <__cxa_allocate_exception@plt>
    137d:	48 8d 35 cb 3c 00 00 	lea    0x3ccb(%rip),%rsi        # 504f <_fini+0xfd7>
    1384:	48 89 c7             	mov    %rax,%rdi
    1387:	48 89 c3             	mov    %rax,%rbx
    138a:	e8 b1 fc ff ff       	call   1040 <std::runtime_error::runtime_error(char const*)@plt>
    138f:	48 8b 15 32 6c 00 00 	mov    0x6c32(%rip),%rdx        # 7fc8 <std::runtime_error::~runtime_error()@GLIBCXX_3.4>
    1396:	48 8d 35 f3 69 00 00 	lea    0x69f3(%rip),%rsi        # 7d90 <typeinfo for std::runtime_error@GLIBCXX_3.4>
    139d:	48 89 df             	mov    %rbx,%rdi
    13a0:	e8 3b fd ff ff       	call   10e0 <__cxa_throw@plt>
    13a5:	bf 10 00 00 00       	mov    $0x10,%edi
    13aa:	e8 b1 fc ff ff       	call   1060 <__cxa_allocate_exception@plt>
    13af:	48 8d 35 85 3c 00 00 	lea    0x3c85(%rip),%rsi        # 503b <_fini+0xfc3>
    13b6:	48 89 c7             	mov    %rax,%rdi
    13b9:	48 89 c3             	mov    %rax,%rbx
    13bc:	e8 7f fc ff ff       	call   1040 <std::runtime_error::runtime_error(char const*)@plt>
    13c1:	48 8b 15 00 6c 00 00 	mov    0x6c00(%rip),%rdx        # 7fc8 <std::runtime_error::~runtime_error()@GLIBCXX_3.4>
    13c8:	48 8d 35 c1 69 00 00 	lea    0x69c1(%rip),%rsi        # 7d90 <typeinfo for std::runtime_error@GLIBCXX_3.4>
    13cf:	48 89 df             	mov    %rbx,%rdi
    13d2:	e8 09 fd ff ff       	call   10e0 <__cxa_throw@plt>
    13d7:	48 89 85 d0 ea ff ff 	mov    %rax,-0x1530(%rbp)
    13de:	48 89 df             	mov    %rbx,%rdi
    13e1:	49 89 d5             	mov    %rdx,%r13
    13e4:	c5 f8 77             	vzeroupper
    13e7:	e8 94 fc ff ff       	call   1080 <__cxa_free_exception@plt>
    13ec:	48 8b 9d d0 ea ff ff 	mov    -0x1530(%rbp),%rbx
    13f3:	e9 88 fd ff ff       	jmp    1180 <main.cold+0x70>
    13f8:	eb dd                	jmp    13d7 <main.cold+0x2c7>
    13fa:	eb db                	jmp    13d7 <main.cold+0x2c7>
    13fc:	eb d9                	jmp    13d7 <main.cold+0x2c7>
    13fe:	48 8d 3d c3 3c 00 00 	lea    0x3cc3(%rip),%rdi        # 50c8 <_fini+0x1050>
    1405:	e8 66 fc ff ff       	call   1070 <std::__throw_length_error(char const*)@plt>
    140a:	bf 10 00 00 00       	mov    $0x10,%edi
    140f:	e8 4c fc ff ff       	call   1060 <__cxa_allocate_exception@plt>
    1414:	48 8d 35 6a 3c 00 00 	lea    0x3c6a(%rip),%rsi        # 5085 <_fini+0x100d>
    141b:	48 89 c7             	mov    %rax,%rdi
    141e:	48 89 c3             	mov    %rax,%rbx
    1421:	e8 1a fc ff ff       	call   1040 <std::runtime_error::runtime_error(char const*)@plt>
    1426:	48 8b 15 9b 6b 00 00 	mov    0x6b9b(%rip),%rdx        # 7fc8 <std::runtime_error::~runtime_error()@GLIBCXX_3.4>
    142d:	48 8d 35 5c 69 00 00 	lea    0x695c(%rip),%rsi        # 7d90 <typeinfo for std::runtime_error@GLIBCXX_3.4>
    1434:	48 89 df             	mov    %rbx,%rdi
    1437:	e8 a4 fc ff ff       	call   10e0 <__cxa_throw@plt>
    143c:	eb 99                	jmp    13d7 <main.cold+0x2c7>
    143e:	66 90                	xchg   %ax,%ax

0000000000001440 <main>:
    1440:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    1445:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    1449:	31 d2                	xor    %edx,%edx
    144b:	41 ff 72 f8          	push   -0x8(%r10)
    144f:	55                   	push   %rbp
    1450:	48 8d 05 49 6c 00 00 	lea    0x6c49(%rip),%rax        # 80a0 <ggml_table_f32_f16>
    1457:	31 f6                	xor    %esi,%esi
    1459:	4c 8d 80 00 00 04 00 	lea    0x40000(%rax),%r8
    1460:	48 89 e5             	mov    %rsp,%rbp
    1463:	41 57                	push   %r15
    1465:	41 56                	push   %r14
    1467:	41 55                	push   %r13
    1469:	41 54                	push   %r12
    146b:	41 52                	push   %r10
    146d:	53                   	push   %rbx
    146e:	48 81 ec 00 16 00 00 	sub    $0x1600,%rsp
    1475:	c5 fa 10 0d 5b 4f 00 	vmovss 0x4f5b(%rip),%xmm1        # 63d8 <iq3s_grid+0x8d8>
    147c:	00 
    147d:	c5 fa 10 05 77 4f 00 	vmovss 0x4f77(%rip),%xmm0        # 63fc <iq3s_grid+0x8fc>
    1484:	00 
    1485:	eb 2f                	jmp    14b6 <main+0x76>
    1487:	c1 ef 11             	shr    $0x11,%edi
    148a:	81 cf 00 00 00 3f    	or     $0x3f000000,%edi
    1490:	c5 f9 6e ff          	vmovd  %edi,%xmm7
    1494:	c5 c2 5c f8          	vsubss %xmm0,%xmm7,%xmm7
    1498:	c5 f9 7e ff          	vmovd  %xmm7,%edi
    149c:	48 83 c0 04          	add    $0x4,%rax
    14a0:	81 c6 00 00 01 00    	add    $0x10000,%esi
    14a6:	09 f9                	or     %edi,%ecx
    14a8:	81 c2 00 00 02 00    	add    $0x20000,%edx
    14ae:	89 48 fc             	mov    %ecx,-0x4(%rax)
    14b1:	49 39 c0             	cmp    %rax,%r8
    14b4:	74 25                	je     14db <main+0x9b>
    14b6:	89 f1                	mov    %esi,%ecx
    14b8:	89 d7                	mov    %edx,%edi
    14ba:	81 e1 00 00 00 80    	and    $0x80000000,%ecx
    14c0:	81 fa ff ff ff 07    	cmp    $0x7ffffff,%edx
    14c6:	76 bf                	jbe    1487 <main+0x47>
    14c8:	c1 ef 04             	shr    $0x4,%edi
    14cb:	81 c7 00 00 00 70    	add    $0x70000000,%edi
    14d1:	c5 f9 6e ff          	vmovd  %edi,%xmm7
    14d5:	c5 c2 59 f9          	vmulss %xmm1,%xmm7,%xmm7
    14d9:	eb bd                	jmp    1498 <main+0x58>
    14db:	b8 81 02 c4 78       	mov    $0x78c40281,%eax
    14e0:	b9 01 00 00 00       	mov    $0x1,%ecx
    14e5:	48 c7 85 30 ec ff ff 	movq   $0x78c40281,-0x13d0(%rbp)
    14ec:	81 02 c4 78 
    14f0:	48 8d b5 30 ec ff ff 	lea    -0x13d0(%rbp),%rsi
    14f7:	48 89 c2             	mov    %rax,%rdx
    14fa:	48 89 b5 b0 ea ff ff 	mov    %rsi,-0x1550(%rbp)
    1501:	48 89 d0             	mov    %rdx,%rax
    1504:	48 c1 e8 1e          	shr    $0x1e,%rax
    1508:	48 31 d0             	xor    %rdx,%rax
    150b:	48 69 c0 65 89 07 6c 	imul   $0x6c078965,%rax,%rax
    1512:	8d 14 08             	lea    (%rax,%rcx,1),%edx
    1515:	48 8b 85 b0 ea ff ff 	mov    -0x1550(%rbp),%rax
    151c:	48 89 14 c8          	mov    %rdx,(%rax,%rcx,8)
    1520:	48 83 c1 01          	add    $0x1,%rcx
    1524:	48 81 f9 70 02 00 00 	cmp    $0x270,%rcx
    152b:	75 d4                	jne    1501 <main+0xc1>
    152d:	48 c7 45 b0 70 02 00 	movq   $0x270,-0x50(%rbp)
    1534:	00 
    1535:	c5 f8 ae 9d ec ea ff 	vstmxcsr -0x1514(%rbp)
    153c:	ff 
    153d:	8b 85 ec ea ff ff    	mov    -0x1514(%rbp),%eax
    1543:	ba 08 00 00 00       	mov    $0x8,%edx
    1548:	48 c7 85 18 ea ff ff 	movq   $0x0,-0x15e8(%rbp)
    154f:	00 00 00 00 
    1553:	48 8d 3d 46 3d 00 00 	lea    0x3d46(%rip),%rdi        # 52a0 <main::signed_values>
    155a:	31 c9                	xor    %ecx,%ecx
    155c:	48 8d 35 7d 3d 00 00 	lea    0x3d7d(%rip),%rsi        # 52e0 <offset4_pack(iq_compact_block const&, offset4_iq3s_block&)::map>
    1563:	89 85 24 ea ff ff    	mov    %eax,-0x15dc(%rbp)
    1569:	b8 08 00 00 00       	mov    $0x8,%eax
    156e:	eb 15                	jmp    1585 <main+0x145>
    1570:	89 c8                	mov    %ecx,%eax
    1572:	83 e0 0f             	and    $0xf,%eax
    1575:	0f b6 14 06          	movzbl (%rsi,%rax,1),%edx
    1579:	89 c8                	mov    %ecx,%eax
    157b:	c0 e8 04             	shr    $0x4,%al
    157e:	83 e0 0f             	and    $0xf,%eax
    1581:	0f b6 04 06          	movzbl (%rsi,%rax,1),%eax
    1585:	c1 e0 04             	shl    $0x4,%eax
    1588:	09 d0                	or     %edx,%eax
    158a:	83 e2 0f             	and    $0xf,%edx
    158d:	44 8d 44 12 f1       	lea    -0xf(%rdx,%rdx,1),%r8d
    1592:	89 ca                	mov    %ecx,%edx
    1594:	83 e2 0f             	and    $0xf,%edx
    1597:	44 3b 04 97          	cmp    (%rdi,%rdx,4),%r8d
    159b:	0f 85 26 fc ff ff    	jne    11c7 <main.cold+0xb7>
    15a1:	c0 e8 04             	shr    $0x4,%al
    15a4:	0f b6 c0             	movzbl %al,%eax
    15a7:	8d 54 00 f1          	lea    -0xf(%rax,%rax,1),%edx
    15ab:	89 c8                	mov    %ecx,%eax
    15ad:	c1 f8 04             	sar    $0x4,%eax
    15b0:	48 98                	cltq
    15b2:	3b 14 87             	cmp    (%rdi,%rax,4),%edx
    15b5:	0f 85 81 fd ff ff    	jne    133c <main.cold+0x22c>
    15bb:	48 83 85 18 ea ff ff 	addq   $0x2,-0x15e8(%rbp)
    15c2:	02 
    15c3:	83 c1 01             	add    $0x1,%ecx
    15c6:	81 f9 00 01 00 00    	cmp    $0x100,%ecx
    15cc:	75 a2                	jne    1570 <main+0x130>
    15ce:	be fc ff ff ff       	mov    $0xfffffffc,%esi
    15d3:	ba 04 00 00 00       	mov    $0x4,%edx
    15d8:	31 db                	xor    %ebx,%ebx
    15da:	c7 85 e0 ea ff ff f1 	movl   $0xfffffff1,-0x1520(%rbp)
    15e1:	ff ff ff 
    15e4:	c5 79 6e ee          	vmovd  %esi,%xmm13
    15e8:	c4 e1 f9 6e fa       	vmovq  %rdx,%xmm7
    15ed:	be 04 00 00 00       	mov    $0x4,%esi
    15f2:	c7 85 c0 ea ff ff 00 	movl   $0x0,-0x1540(%rbp)
    15f9:	00 00 00 
    15fc:	c5 79 6e e6          	vmovd  %esi,%xmm12
    1600:	c5 c1 6c ff          	vpunpcklqdq %xmm7,%xmm7,%xmm7
    1604:	b8 71 07 00 00       	mov    $0x771,%eax
    1609:	45 31 ed             	xor    %r13d,%r13d
    160c:	c5 f9 7f bd d0 ea ff 	vmovdqa %xmm7,-0x1530(%rbp)
    1613:	ff 
    1614:	48 89 da             	mov    %rbx,%rdx
    1617:	c4 41 79 70 ed 00    	vpshufd $0x0,%xmm13,%xmm13
    161d:	c4 41 79 70 e4 00    	vpshufd $0x0,%xmm12,%xmm12
    1623:	41 bc f1 ff ff ff    	mov    $0xfffffff1,%r12d
    1629:	45 31 db             	xor    %r11d,%r11d
    162c:	c4 c1 79 6e fb       	vmovd  %r11d,%xmm7
    1631:	41 89 c2             	mov    %eax,%r10d
    1634:	45 89 e9             	mov    %r13d,%r9d
    1637:	89 85 b8 ea ff ff    	mov    %eax,-0x1548(%rbp)
    163d:	c5 79 70 cf 00       	vpshufd $0x0,%xmm7,%xmm9
    1642:	41 b8 81 ff ff ff    	mov    $0xffffff81,%r8d
    1648:	c4 c1 79 6e fc       	vmovd  %r12d,%xmm7
    164d:	44 89 ad e8 ea ff ff 	mov    %r13d,-0x1518(%rbp)
    1654:	c5 79 70 c7 00       	vpshufd $0x0,%xmm7,%xmm8
    1659:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1660:	48 8d 42 01          	lea    0x1(%rdx),%rax
    1664:	c4 c1 79 6e f9       	vmovd  %r9d,%xmm7
    1669:	c4 c1 79 6e ea       	vmovd  %r10d,%xmm5
    166e:	c5 f9 6f 25 fa 4c 00 	vmovdqa 0x4cfa(%rip),%xmm4        # 6370 <iq3s_grid+0x870>
    1675:	00 
    1676:	c4 e1 f9 6e f2       	vmovq  %rdx,%xmm6
    167b:	c5 f9 70 ff 00       	vpshufd $0x0,%xmm7,%xmm7
    1680:	c5 f9 70 ed 00       	vpshufd $0x0,%xmm5,%xmm5
    1685:	c5 f9 6f 1d f3 4c 00 	vmovdqa 0x4cf3(%rip),%xmm3        # 6380 <iq3s_grid+0x880>
    168c:	00 
    168d:	c4 e3 c9 22 d0 01    	vpinsrq $0x1,%rax,%xmm6,%xmm2
    1693:	c4 c1 79 6e f0       	vmovd  %r8d,%xmm6
    1698:	31 c0                	xor    %eax,%eax
    169a:	c5 f9 70 f6 00       	vpshufd $0x0,%xmm6,%xmm6
    169f:	90                   	nop
    16a0:	c5 f9 6f cb          	vmovdqa %xmm3,%xmm1
    16a4:	c5 79 6f dc          	vmovdqa %xmm4,%xmm11
    16a8:	c5 79 6f d2          	vmovdqa %xmm2,%xmm10
    16ac:	c4 e2 31 40 c1       	vpmulld %xmm1,%xmm9,%xmm0
    16b1:	c5 49 fe f9          	vpaddd %xmm1,%xmm6,%xmm15
    16b5:	c5 e9 d4 95 d0 ea ff 	vpaddq -0x1530(%rbp),%xmm2,%xmm2
    16bc:	ff 
    16bd:	c4 c1 09 72 f7 04    	vpslld $0x4,%xmm15,%xmm14
    16c3:	c4 c1 59 fe e5       	vpaddd %xmm13,%xmm4,%xmm4
    16c8:	c4 c1 61 fe dc       	vpaddd %xmm12,%xmm3,%xmm3
    16cd:	c4 41 09 fa f7       	vpsubd %xmm15,%xmm14,%xmm14
    16d2:	c5 f9 fe c7          	vpaddd %xmm7,%xmm0,%xmm0
    16d6:	c5 f9 72 f0 01       	vpslld $0x1,%xmm0,%xmm0
    16db:	c4 c1 79 fa c6       	vpsubd %xmm14,%xmm0,%xmm0
    16e0:	c4 62 39 40 f1       	vpmulld %xmm1,%xmm8,%xmm14
    16e5:	c5 09 fe f5          	vpaddd %xmm5,%xmm14,%xmm14
    16e9:	c4 c1 79 76 c6       	vpcmpeqd %xmm14,%xmm0,%xmm0
    16ee:	c5 f9 ef 05 aa 4c 00 	vpxor  0x4caa(%rip),%xmm0,%xmm0        # 63a0 <iq3s_grid+0x8a0>
    16f5:	00 
    16f6:	c4 e2 79 17 c0       	vptest %xmm0,%xmm0
    16fb:	0f 85 29 fc ff ff    	jne    132a <main.cold+0x21a>
    1701:	83 c0 01             	add    $0x1,%eax
    1704:	83 f8 3f             	cmp    $0x3f,%eax
    1707:	75 97                	jne    16a0 <main+0x260>
    1709:	48 81 c2 fc 00 00 00 	add    $0xfc,%rdx
    1710:	b9 03 00 00 00       	mov    $0x3,%ecx
    1715:	b8 7d 00 00 00       	mov    $0x7d,%eax
    171a:	41 8d 1c 00          	lea    (%r8,%rax,1),%ebx
    171e:	44 89 df             	mov    %r11d,%edi
    1721:	44 89 e6             	mov    %r12d,%esi
    1724:	0f af f8             	imul   %eax,%edi
    1727:	41 89 de             	mov    %ebx,%r14d
    172a:	0f af f0             	imul   %eax,%esi
    172d:	41 c1 e6 04          	shl    $0x4,%r14d
    1731:	45 8d 3c 39          	lea    (%r9,%rdi,1),%r15d
    1735:	42 8d 1c 7b          	lea    (%rbx,%r15,2),%ebx
    1739:	45 8d 2c 32          	lea    (%r10,%rsi,1),%r13d
    173d:	44 29 f3             	sub    %r14d,%ebx
    1740:	41 39 dd             	cmp    %ebx,%r13d
    1743:	0f 85 aa fb ff ff    	jne    12f3 <main.cold+0x1e3>
    1749:	8d 58 01             	lea    0x1(%rax),%ebx
    174c:	83 f9 01             	cmp    $0x1,%ecx
    174f:	0f 84 be 03 00 00    	je     1b13 <main+0x6d3>
    1755:	44 01 c3             	add    %r8d,%ebx
    1758:	44 01 df             	add    %r11d,%edi
    175b:	44 01 e6             	add    %r12d,%esi
    175e:	45 8d 2c 39          	lea    (%r9,%rdi,1),%r13d
    1762:	41 89 de             	mov    %ebx,%r14d
    1765:	42 8d 1c 6b          	lea    (%rbx,%r13,2),%ebx
    1769:	41 c1 e6 04          	shl    $0x4,%r14d
    176d:	45 8d 2c 32          	lea    (%r10,%rsi,1),%r13d
    1771:	44 29 f3             	sub    %r14d,%ebx
    1774:	44 39 eb             	cmp    %r13d,%ebx
    1777:	0f 85 76 fb ff ff    	jne    12f3 <main.cold+0x1e3>
    177d:	8d 58 02             	lea    0x2(%rax),%ebx
    1780:	83 f9 02             	cmp    $0x2,%ecx
    1783:	0f 84 93 03 00 00    	je     1b1c <main+0x6dc>
    1789:	44 01 c3             	add    %r8d,%ebx
    178c:	44 01 df             	add    %r11d,%edi
    178f:	44 01 e6             	add    %r12d,%esi
    1792:	45 8d 2c 39          	lea    (%r9,%rdi,1),%r13d
    1796:	41 89 de             	mov    %ebx,%r14d
    1799:	42 8d 1c 6b          	lea    (%rbx,%r13,2),%ebx
    179d:	41 c1 e6 04          	shl    $0x4,%r14d
    17a1:	45 8d 2c 32          	lea    (%r10,%rsi,1),%r13d
    17a5:	44 29 f3             	sub    %r14d,%ebx
    17a8:	44 39 eb             	cmp    %r13d,%ebx
    17ab:	0f 85 42 fb ff ff    	jne    12f3 <main.cold+0x1e3>
    17b1:	83 f9 03             	cmp    $0x3,%ecx
    17b4:	0f 84 6b 03 00 00    	je     1b25 <main+0x6e5>
    17ba:	41 8d 44 00 03       	lea    0x3(%r8,%rax,1),%eax
    17bf:	44 01 e6             	add    %r12d,%esi
    17c2:	44 01 df             	add    %r11d,%edi
    17c5:	42 8d 0c 16          	lea    (%rsi,%r10,1),%ecx
    17c9:	42 8d 34 0f          	lea    (%rdi,%r9,1),%esi
    17cd:	89 c7                	mov    %eax,%edi
    17cf:	8d 04 70             	lea    (%rax,%rsi,2),%eax
    17d2:	c1 e7 04             	shl    $0x4,%edi
    17d5:	29 f8                	sub    %edi,%eax
    17d7:	39 c1                	cmp    %eax,%ecx
    17d9:	0f 85 14 fb ff ff    	jne    12f3 <main.cold+0x1e3>
    17df:	48 83 c2 04          	add    $0x4,%rdx
    17e3:	8b 85 c0 ea ff ff    	mov    -0x1540(%rbp),%eax
    17e9:	41 83 c0 01          	add    $0x1,%r8d
    17ed:	41 01 c1             	add    %eax,%r9d
    17f0:	8b 85 e0 ea ff ff    	mov    -0x1520(%rbp),%eax
    17f6:	41 01 c2             	add    %eax,%r10d
    17f9:	41 81 f8 80 00 00 00 	cmp    $0x80,%r8d
    1800:	0f 85 5a fe ff ff    	jne    1660 <main+0x220>
    1806:	41 83 c3 01          	add    $0x1,%r11d
    180a:	8b 85 b8 ea ff ff    	mov    -0x1548(%rbp),%eax
    1810:	41 83 c4 02          	add    $0x2,%r12d
    1814:	44 8b ad e8 ea ff ff 	mov    -0x1518(%rbp),%r13d
    181b:	41 83 fb 10          	cmp    $0x10,%r11d
    181f:	0f 85 07 fe ff ff    	jne    162c <main+0x1ec>
    1825:	83 85 c0 ea ff ff 01 	addl   $0x1,-0x1540(%rbp)
    182c:	41 83 ed 7f          	sub    $0x7f,%r13d
    1830:	2d fe 00 00 00       	sub    $0xfe,%eax
    1835:	8b 8d c0 ea ff ff    	mov    -0x1540(%rbp),%ecx
    183b:	83 85 e0 ea ff ff 02 	addl   $0x2,-0x1520(%rbp)
    1842:	83 f9 10             	cmp    $0x10,%ecx
    1845:	0f 85 d8 fd ff ff    	jne    1623 <main+0x1e3>
    184b:	48 8b 05 96 4b 00 00 	mov    0x4b96(%rip),%rax        # 63e8 <iq3s_grid+0x8e8>
    1852:	48 89 d3             	mov    %rdx,%rbx
    1855:	48 c7 85 58 ea ff ff 	movq   $0x0,-0x15a8(%rbp)
    185c:	00 00 00 00 
    1860:	48 c7 85 98 ea ff ff 	movq   $0x0,-0x1568(%rbp)
    1867:	00 00 00 00 
    186b:	48 c7 85 28 ea ff ff 	movq   $0x0,-0x15d8(%rbp)
    1872:	00 00 00 00 
    1876:	48 89 85 f8 ea ff ff 	mov    %rax,-0x1508(%rbp)
    187d:	48 8d 85 f8 ea ff ff 	lea    -0x1508(%rbp),%rax
    1884:	48 89 85 70 ea ff ff 	mov    %rax,-0x1590(%rbp)
    188b:	48 8d 85 00 eb ff ff 	lea    -0x1500(%rbp),%rax
    1892:	48 89 85 38 ea ff ff 	mov    %rax,-0x15c8(%rbp)
    1899:	48 c7 c0 00 00 00 80 	mov    $0xffffffff80000000,%rax
    18a0:	c4 e1 f9 6e f8       	vmovq  %rax,%xmm7
    18a5:	c5 c1 6c ff          	vpunpcklqdq %xmm7,%xmm7,%xmm7
    18a9:	c5 f9 7f bd c0 ea ff 	vmovdqa %xmm7,-0x1540(%rbp)
    18b0:	ff 
    18b1:	48 8d 05 78 39 00 00 	lea    0x3978(%rip),%rax        # 5230 <C.168.6>
    18b8:	48 8b b5 70 ea ff ff 	mov    -0x1590(%rbp),%rsi
    18bf:	48 89 85 68 ea ff ff 	mov    %rax,-0x1598(%rbp)
    18c6:	8b 85 24 ea ff ff    	mov    -0x15dc(%rbp),%eax
    18cc:	25 bf 1f ff ff       	and    $0xffff1fbf,%eax
    18d1:	0b 06                	or     (%rsi),%eax
    18d3:	89 85 20 ea ff ff    	mov    %eax,-0x15e0(%rbp)
    18d9:	48 8b b5 68 ea ff ff 	mov    -0x1598(%rbp),%rsi
    18e0:	8b 85 20 ea ff ff    	mov    -0x15e0(%rbp),%eax
    18e6:	0b 06                	or     (%rsi),%eax
    18e8:	89 85 ec ea ff ff    	mov    %eax,-0x1514(%rbp)
    18ee:	c5 f8 ae 95 ec ea ff 	vldmxcsr -0x1514(%rbp)
    18f5:	ff 
    18f6:	48 8d 05 13 39 00 00 	lea    0x3913(%rip),%rax        # 5210 <C.169.7>
    18fd:	48 89 85 60 ea ff ff 	mov    %rax,-0x15a0(%rbp)
    1904:	8b 38                	mov    (%rax),%edi
    1906:	8d 87 ff 00 00 00    	lea    0xff(%rdi),%eax
    190c:	85 ff                	test   %edi,%edi
    190e:	89 bd ac ea ff ff    	mov    %edi,-0x1554(%rbp)
    1914:	0f 49 c7             	cmovns %edi,%eax
    1917:	c1 f8 08             	sar    $0x8,%eax
    191a:	89 85 a8 ea ff ff    	mov    %eax,-0x1558(%rbp)
    1920:	48 98                	cltq
    1922:	48 89 c6             	mov    %rax,%rsi
    1925:	48 89 85 88 ea ff ff 	mov    %rax,-0x1578(%rbp)
    192c:	48 b8 e3 38 8e e3 38 	movabs $0xe38e38e38e38e3,%rax
    1933:	8e e3 00 
    1936:	48 39 f0             	cmp    %rsi,%rax
    1939:	0f 82 bf fa ff ff    	jb     13fe <main.cold+0x2ee>
    193f:	c7 85 e8 ea ff ff 00 	movl   $0x0,-0x1518(%rbp)
    1946:	00 00 00 
    1949:	48 8d 04 f6          	lea    (%rsi,%rsi,8),%rax
    194d:	48 c1 e0 04          	shl    $0x4,%rax
    1951:	48 89 85 40 ea ff ff 	mov    %rax,-0x15c0(%rbp)
    1958:	48 89 85 a0 ea ff ff 	mov    %rax,-0x1560(%rbp)
    195f:	48 69 c6 24 01 00 00 	imul   $0x124,%rsi,%rax
    1966:	48 c1 e6 08          	shl    $0x8,%rsi
    196a:	48 89 b5 48 ea ff ff 	mov    %rsi,-0x15b8(%rbp)
    1971:	48 89 85 90 ea ff ff 	mov    %rax,-0x1570(%rbp)
    1978:	48 8b 85 98 ea ff ff 	mov    -0x1568(%rbp),%rax
    197f:	48 89 85 30 ea ff ff 	mov    %rax,-0x15d0(%rbp)
    1986:	b8 ff ff ff 7f       	mov    $0x7fffffff,%eax
    198b:	c4 e1 f9 6e f8       	vmovq  %rax,%xmm7
    1990:	c5 c1 6c ff          	vpunpcklqdq %xmm7,%xmm7,%xmm7
    1994:	c5 f9 7f bd d0 ea ff 	vmovdqa %xmm7,-0x1530(%rbp)
    199b:	ff 
    199c:	48 83 bd 88 ea ff ff 	cmpq   $0x0,-0x1578(%rbp)
    19a3:	00 
    19a4:	0f 84 84 01 00 00    	je     1b2e <main+0x6ee>
    19aa:	4c 8b b5 40 ea ff ff 	mov    -0x15c0(%rbp),%r14
    19b1:	4c 89 f7             	mov    %r14,%rdi
    19b4:	e8 d7 f6 ff ff       	call   1090 <operator new(unsigned long)@plt>
    19b9:	48 83 bd 88 ea ff ff 	cmpq   $0x1,-0x1578(%rbp)
    19c0:	01 
    19c1:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    19c5:	49 89 c7             	mov    %rax,%r15
    19c8:	c5 f9 7f 00          	vmovdqa %xmm0,(%rax)
    19cc:	c4 e3 7d 19 40 10 01 	vextractf128 $0x1,%ymm0,0x10(%rax)
    19d3:	c4 e3 7d 19 40 20 01 	vextractf128 $0x1,%ymm0,0x20(%rax)
    19da:	c4 e3 7d 19 40 30 01 	vextractf128 $0x1,%ymm0,0x30(%rax)
    19e1:	c4 e3 7d 19 40 40 01 	vextractf128 $0x1,%ymm0,0x40(%rax)
    19e8:	4a 8d 14 30          	lea    (%rax,%r14,1),%rdx
    19ec:	c4 e3 7d 19 40 50 01 	vextractf128 $0x1,%ymm0,0x50(%rax)
    19f3:	c4 e3 7d 19 40 60 01 	vextractf128 $0x1,%ymm0,0x60(%rax)
    19fa:	c4 e3 7d 19 40 70 01 	vextractf128 $0x1,%ymm0,0x70(%rax)
    1a01:	c4 e3 7d 19 80 80 00 	vextractf128 $0x1,%ymm0,0x80(%rax)
    1a08:	00 00 01 
    1a0b:	48 8d 80 90 00 00 00 	lea    0x90(%rax),%rax
    1a12:	0f 84 5a 05 00 00    	je     1f72 <main+0xb32>
    1a18:	c4 c1 7e 6f 07       	vmovdqu (%r15),%ymm0
    1a1d:	48 05 90 00 00 00    	add    $0x90,%rax
    1a23:	c5 fe 7f 80 70 ff ff 	vmovdqu %ymm0,-0x90(%rax)
    1a2a:	ff 
    1a2b:	c4 c1 7e 6f 47 20    	vmovdqu 0x20(%r15),%ymm0
    1a31:	c5 fe 7f 40 90       	vmovdqu %ymm0,-0x70(%rax)
    1a36:	c4 c1 7e 6f 47 40    	vmovdqu 0x40(%r15),%ymm0
    1a3c:	c5 fe 7f 40 b0       	vmovdqu %ymm0,-0x50(%rax)
    1a41:	c4 c1 7e 6f 47 60    	vmovdqu 0x60(%r15),%ymm0
    1a47:	c5 fe 7f 40 d0       	vmovdqu %ymm0,-0x30(%rax)
    1a4c:	c4 c1 79 6f 87 80 00 	vmovdqa 0x80(%r15),%xmm0
    1a53:	00 00 
    1a55:	c5 f9 7f 40 f0       	vmovdqa %xmm0,-0x10(%rax)
    1a5a:	48 39 d0             	cmp    %rdx,%rax
    1a5d:	75 b9                	jne    1a18 <main+0x5d8>
    1a5f:	4c 8b b5 40 ea ff ff 	mov    -0x15c0(%rbp),%r14
    1a66:	4c 89 f7             	mov    %r14,%rdi
    1a69:	c5 f8 77             	vzeroupper
    1a6c:	e8 1f f6 ff ff       	call   1090 <operator new(unsigned long)@plt>
    1a71:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    1a75:	4a 8d 14 30          	lea    (%rax,%r14,1),%rdx
    1a79:	49 89 c4             	mov    %rax,%r12
    1a7c:	c5 f9 7f 00          	vmovdqa %xmm0,(%rax)
    1a80:	c4 e3 7d 19 40 10 01 	vextractf128 $0x1,%ymm0,0x10(%rax)
    1a87:	c4 e3 7d 19 40 20 01 	vextractf128 $0x1,%ymm0,0x20(%rax)
    1a8e:	c4 e3 7d 19 40 30 01 	vextractf128 $0x1,%ymm0,0x30(%rax)
    1a95:	c4 e3 7d 19 40 40 01 	vextractf128 $0x1,%ymm0,0x40(%rax)
    1a9c:	c4 e3 7d 19 40 50 01 	vextractf128 $0x1,%ymm0,0x50(%rax)
    1aa3:	c4 e3 7d 19 40 60 01 	vextractf128 $0x1,%ymm0,0x60(%rax)
    1aaa:	c4 e3 7d 19 40 70 01 	vextractf128 $0x1,%ymm0,0x70(%rax)
    1ab1:	c4 e3 7d 19 80 80 00 	vextractf128 $0x1,%ymm0,0x80(%rax)
    1ab8:	00 00 01 
    1abb:	48 8d 80 90 00 00 00 	lea    0x90(%rax),%rax
    1ac2:	c4 c1 7e 6f 04 24    	vmovdqu (%r12),%ymm0
    1ac8:	48 05 90 00 00 00    	add    $0x90,%rax
    1ace:	c5 fe 7f 80 70 ff ff 	vmovdqu %ymm0,-0x90(%rax)
    1ad5:	ff 
    1ad6:	c4 c1 7e 6f 44 24 20 	vmovdqu 0x20(%r12),%ymm0
    1add:	c5 fe 7f 40 90       	vmovdqu %ymm0,-0x70(%rax)
    1ae2:	c4 c1 7e 6f 44 24 40 	vmovdqu 0x40(%r12),%ymm0
    1ae9:	c5 fe 7f 40 b0       	vmovdqu %ymm0,-0x50(%rax)
    1aee:	c4 c1 7e 6f 44 24 60 	vmovdqu 0x60(%r12),%ymm0
    1af5:	c5 fe 7f 40 d0       	vmovdqu %ymm0,-0x30(%rax)
    1afa:	c4 c1 79 6f 84 24 80 	vmovdqa 0x80(%r12),%xmm0
    1b01:	00 00 00 
    1b04:	c5 f9 7f 40 f0       	vmovdqa %xmm0,-0x10(%rax)
    1b09:	48 39 d0             	cmp    %rdx,%rax
    1b0c:	75 b4                	jne    1ac2 <main+0x682>
    1b0e:	e9 b2 04 00 00       	jmp    1fc5 <main+0xb85>
    1b13:	48 83 c2 01          	add    $0x1,%rdx
    1b17:	e9 c7 fc ff ff       	jmp    17e3 <main+0x3a3>
    1b1c:	48 83 c2 02          	add    $0x2,%rdx
    1b20:	e9 be fc ff ff       	jmp    17e3 <main+0x3a3>
    1b25:	48 83 c2 03          	add    $0x3,%rdx
    1b29:	e9 b5 fc ff ff       	jmp    17e3 <main+0x3a3>
    1b2e:	48 c7 85 78 ea ff ff 	movq   $0x0,-0x1588(%rbp)
    1b35:	00 00 00 00 
    1b39:	45 31 f6             	xor    %r14d,%r14d
    1b3c:	45 31 ed             	xor    %r13d,%r13d
    1b3f:	45 31 ff             	xor    %r15d,%r15d
    1b42:	48 c7 85 80 ea ff ff 	movq   $0x0,-0x1580(%rbp)
    1b49:	00 00 00 00 
    1b4d:	45 31 e4             	xor    %r12d,%r12d
    1b50:	48 c7 85 b8 ea ff ff 	movq   $0x0,-0x1548(%rbp)
    1b57:	00 00 00 00 
    1b5b:	48 c7 85 e0 ea ff ff 	movq   $0x0,-0x1520(%rbp)
    1b62:	00 00 00 00 
    1b66:	81 bd ac ea ff ff ff 	cmpl   $0xff,-0x1554(%rbp)
    1b6d:	00 00 00 
    1b70:	0f 8e 39 09 00 00    	jle    24af <main+0x106f>
    1b76:	4c 8b 5d b0          	mov    -0x50(%rbp),%r11
    1b7a:	49 8d 46 04          	lea    0x4(%r14),%rax
    1b7e:	48 89 9d 00 ea ff ff 	mov    %rbx,-0x1600(%rbp)
    1b85:	4c 89 fb             	mov    %r15,%rbx
    1b88:	48 89 85 10 ea ff ff 	mov    %rax,-0x15f0(%rbp)
    1b8f:	c7 85 50 ea ff ff 00 	movl   $0x0,-0x15b0(%rbp)
    1b96:	00 00 00 
    1b99:	4c 89 a5 f8 e9 ff ff 	mov    %r12,-0x1608(%rbp)
    1ba0:	31 d2                	xor    %edx,%edx
    1ba2:	0f b6 0c 13          	movzbl (%rbx,%rdx,1),%ecx
    1ba6:	48 8d 3d 33 37 00 00 	lea    0x3733(%rip),%rdi        # 52e0 <offset4_pack(iq_compact_block const&, offset4_iq3s_block&)::map>
    1bad:	89 c8                	mov    %ecx,%eax
    1baf:	83 e1 0f             	and    $0xf,%ecx
    1bb2:	c0 e8 04             	shr    $0x4,%al
    1bb5:	83 e0 0f             	and    $0xf,%eax
    1bb8:	0f b6 04 07          	movzbl (%rdi,%rax,1),%eax
    1bbc:	c1 e0 04             	shl    $0x4,%eax
    1bbf:	0a 04 0f             	or     (%rdi,%rcx,1),%al
    1bc2:	41 88 04 14          	mov    %al,(%r12,%rdx,1)
    1bc6:	48 83 c2 01          	add    $0x1,%rdx
    1bca:	48 81 fa 80 00 00 00 	cmp    $0x80,%rdx
    1bd1:	75 cf                	jne    1ba2 <main+0x762>
    1bd3:	48 8b 83 84 00 00 00 	mov    0x84(%rbx),%rax
    1bda:	c5 fa 10 83 80 00 00 	vmovss 0x80(%rbx),%xmm0
    1be1:	00 
    1be2:	c4 c1 7a 11 84 24 80 	vmovss %xmm0,0x80(%r12)
    1be9:	00 00 00 
    1bec:	49 89 84 24 84 00 00 	mov    %rax,0x84(%r12)
    1bf3:	00 
    1bf4:	49 81 fb 6f 02 00 00 	cmp    $0x26f,%r11
    1bfb:	0f 87 bd 09 00 00    	ja     25be <main+0x117e>
    1c01:	4a 8b 8c dd 30 ec ff 	mov    -0x13d0(%rbp,%r11,8),%rcx
    1c08:	ff 
    1c09:	49 8d 73 01          	lea    0x1(%r11),%rsi
    1c0d:	48 89 75 b0          	mov    %rsi,-0x50(%rbp)
    1c11:	48 89 c8             	mov    %rcx,%rax
    1c14:	48 c1 e8 0b          	shr    $0xb,%rax
    1c18:	89 c0                	mov    %eax,%eax
    1c1a:	48 31 c1             	xor    %rax,%rcx
    1c1d:	48 89 c8             	mov    %rcx,%rax
    1c20:	48 c1 e0 07          	shl    $0x7,%rax
    1c24:	25 80 56 2c 9d       	and    $0x9d2c5680,%eax
    1c29:	48 31 c1             	xor    %rax,%rcx
    1c2c:	48 89 c8             	mov    %rcx,%rax
    1c2f:	48 c1 e0 0f          	shl    $0xf,%rax
    1c33:	25 00 00 c6 ef       	and    $0xefc60000,%eax
    1c38:	48 31 c1             	xor    %rax,%rcx
    1c3b:	48 89 c8             	mov    %rcx,%rax
    1c3e:	48 c1 e8 12          	shr    $0x12,%rax
    1c42:	48 31 c1             	xor    %rax,%rcx
    1c45:	48 b8 8f e3 38 8e e3 	movabs $0xe38e38e38e38e38f,%rax
    1c4c:	38 8e e3 
    1c4f:	48 f7 e1             	mul    %rcx
    1c52:	48 89 d7             	mov    %rdx,%rdi
    1c55:	48 89 d0             	mov    %rdx,%rax
    1c58:	48 c1 ef 03          	shr    $0x3,%rdi
    1c5c:	48 83 e0 f8          	and    $0xfffffffffffffff8,%rax
    1c60:	48 01 f8             	add    %rdi,%rax
    1c63:	48 29 c1             	sub    %rax,%rcx
    1c66:	48 8d 05 f3 35 00 00 	lea    0x35f3(%rip),%rax        # 5260 <main::exponents>
    1c6d:	8b 04 88             	mov    (%rax,%rcx,4),%eax
    1c70:	89 85 08 ea ff ff    	mov    %eax,-0x15f8(%rbp)
    1c76:	48 81 fe 6f 02 00 00 	cmp    $0x26f,%rsi
    1c7d:	0f 87 26 09 00 00    	ja     25a9 <main+0x1169>
    1c83:	48 8b 8c f5 30 ec ff 	mov    -0x13d0(%rbp,%rsi,8),%rcx
    1c8a:	ff 
    1c8b:	c5 c0 57 ff          	vxorps %xmm7,%xmm7,%xmm7
    1c8f:	4c 8d 5e 01          	lea    0x1(%rsi),%r11
    1c93:	8b bd 08 ea ff ff    	mov    -0x15f8(%rbp),%edi
    1c99:	4c 89 5d b0          	mov    %r11,-0x50(%rbp)
    1c9d:	4c 89 9d f0 e9 ff ff 	mov    %r11,-0x1610(%rbp)
    1ca4:	48 89 c8             	mov    %rcx,%rax
    1ca7:	48 c1 e8 0b          	shr    $0xb,%rax
    1cab:	89 c0                	mov    %eax,%eax
    1cad:	48 31 c1             	xor    %rax,%rcx
    1cb0:	48 89 c8             	mov    %rcx,%rax
    1cb3:	48 c1 e0 07          	shl    $0x7,%rax
    1cb7:	25 80 56 2c 9d       	and    $0x9d2c5680,%eax
    1cbc:	48 31 c1             	xor    %rax,%rcx
    1cbf:	48 89 c8             	mov    %rcx,%rax
    1cc2:	48 c1 e0 0f          	shl    $0xf,%rax
    1cc6:	25 00 00 c6 ef       	and    $0xefc60000,%eax
    1ccb:	48 31 c1             	xor    %rax,%rcx
    1cce:	48 89 c8             	mov    %rcx,%rax
    1cd1:	48 c1 e8 12          	shr    $0x12,%rax
    1cd5:	48 31 c1             	xor    %rax,%rcx
    1cd8:	48 b8 01 10 00 02 40 	movabs $0x8004002001001,%rax
    1cdf:	00 08 00 
    1ce2:	48 f7 e1             	mul    %rcx
    1ce5:	48 89 c8             	mov    %rcx,%rax
    1ce8:	48 29 d0             	sub    %rdx,%rax
    1ceb:	48 d1 e8             	shr    $1,%rax
    1cee:	48 01 d0             	add    %rdx,%rax
    1cf1:	48 c1 e8 0c          	shr    $0xc,%rax
    1cf5:	48 89 c2             	mov    %rax,%rdx
    1cf8:	48 c1 e2 0d          	shl    $0xd,%rdx
    1cfc:	48 29 c2             	sub    %rax,%rdx
    1cff:	48 89 c8             	mov    %rcx,%rax
    1d02:	48 29 d0             	sub    %rdx,%rax
    1d05:	2d ff 0f 00 00       	sub    $0xfff,%eax
    1d0a:	c5 c2 2a c0          	vcvtsi2ss %eax,%xmm7,%xmm0
    1d0e:	c5 fa 59 05 c6 46 00 	vmulss 0x46c6(%rip),%xmm0,%xmm0        # 63dc <iq3s_grid+0x8dc>
    1d15:	00 
    1d16:	e8 95 f3 ff ff       	call   10b0 <ldexpf@plt>
    1d1b:	48 8b 85 10 ea ff ff 	mov    -0x15f0(%rbp),%rax
    1d22:	48 8d b5 40 f3 ff ff 	lea    -0xcc0(%rbp),%rsi
    1d29:	49 b8 81 80 80 80 80 	movabs $0x8080808080808081,%r8
    1d30:	80 80 80 
    1d33:	4c 8b 9d f0 e9 ff ff 	mov    -0x1610(%rbp),%r11
    1d3a:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
    1d3f:	4c 8d 88 00 01 00 00 	lea    0x100(%rax),%r9
    1d46:	48 89 c7             	mov    %rax,%rdi
    1d49:	b8 01 00 00 00       	mov    $0x1,%eax
    1d4e:	c4 e1 f9 6e c8       	vmovq  %rax,%xmm1
    1d53:	48 8b 85 b0 ea ff ff 	mov    -0x1550(%rbp),%rax
    1d5a:	c5 f1 6c c9          	vpunpcklqdq %xmm1,%xmm1,%xmm1
    1d5e:	48 8d 88 78 13 00 00 	lea    0x1378(%rax),%rcx
    1d65:	eb 61                	jmp    1dc8 <main+0x988>
    1d67:	4a 8b 84 dd 30 ec ff 	mov    -0x13d0(%rbp,%r11,8),%rax
    1d6e:	ff 
    1d6f:	49 83 c3 01          	add    $0x1,%r11
    1d73:	49 89 c2             	mov    %rax,%r10
    1d76:	48 83 c7 01          	add    $0x1,%rdi
    1d7a:	49 c1 ea 0b          	shr    $0xb,%r10
    1d7e:	45 89 d2             	mov    %r10d,%r10d
    1d81:	49 31 c2             	xor    %rax,%r10
    1d84:	4c 89 d0             	mov    %r10,%rax
    1d87:	48 c1 e0 07          	shl    $0x7,%rax
    1d8b:	25 80 56 2c 9d       	and    $0x9d2c5680,%eax
    1d90:	49 31 c2             	xor    %rax,%r10
    1d93:	4c 89 d0             	mov    %r10,%rax
    1d96:	48 c1 e0 0f          	shl    $0xf,%rax
    1d9a:	25 00 00 c6 ef       	and    $0xefc60000,%eax
    1d9f:	49 31 c2             	xor    %rax,%r10
    1da2:	4c 89 d0             	mov    %r10,%rax
    1da5:	48 c1 e8 12          	shr    $0x12,%rax
    1da9:	49 31 c2             	xor    %rax,%r10
    1dac:	4c 89 d0             	mov    %r10,%rax
    1daf:	49 f7 e0             	mul    %r8
    1db2:	48 c1 ea 07          	shr    $0x7,%rdx
    1db6:	45 8d 54 12 81       	lea    -0x7f(%r10,%rdx,1),%r10d
    1dbb:	44 88 57 ff          	mov    %r10b,-0x1(%rdi)
    1dbf:	49 39 f9             	cmp    %rdi,%r9
    1dc2:	0f 84 b9 03 00 00    	je     2181 <main+0xd41>
    1dc8:	49 81 fb 6f 02 00 00 	cmp    $0x26f,%r11
    1dcf:	76 96                	jbe    1d67 <main+0x927>
    1dd1:	48 8b 85 b0 ea ff ff 	mov    -0x1550(%rbp),%rax
    1dd8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    1ddf:	00 
    1de0:	c5 f9 6f ad d0 ea ff 	vmovdqa -0x1530(%rbp),%xmm5
    1de7:	ff 
    1de8:	c5 d1 db 50 08       	vpand  0x8(%rax),%xmm5,%xmm2
    1ded:	48 83 c0 10          	add    $0x10,%rax
    1df1:	c5 f9 6f b5 c0 ea ff 	vmovdqa -0x1540(%rbp),%xmm6
    1df8:	ff 
    1df9:	c5 c9 db 40 f0       	vpand  -0x10(%rax),%xmm6,%xmm0
    1dfe:	c5 e9 eb d0          	vpor   %xmm0,%xmm2,%xmm2
    1e02:	c5 e1 73 d2 01       	vpsrlq $0x1,%xmm2,%xmm3
    1e07:	c5 e9 db d1          	vpand  %xmm1,%xmm2,%xmm2
    1e0b:	c5 f9 73 f2 03       	vpsllq $0x3,%xmm2,%xmm0
    1e10:	c5 e1 ef 98 58 0c 00 	vpxor  0xc58(%rax),%xmm3,%xmm3
    1e17:	00 
    1e18:	c5 f9 d4 c2          	vpaddq %xmm2,%xmm0,%xmm0
    1e1c:	c5 f9 73 f0 09       	vpsllq $0x9,%xmm0,%xmm0
    1e21:	c5 f9 d4 c2          	vpaddq %xmm2,%xmm0,%xmm0
    1e25:	c5 f9 73 f0 05       	vpsllq $0x5,%xmm0,%xmm0
    1e2a:	c5 f9 d4 c2          	vpaddq %xmm2,%xmm0,%xmm0
    1e2e:	c5 f9 73 f0 02       	vpsllq $0x2,%xmm0,%xmm0
    1e33:	c5 f9 fb c2          	vpsubq %xmm2,%xmm0,%xmm0
    1e37:	c5 f9 73 f0 03       	vpsllq $0x3,%xmm0,%xmm0
    1e3c:	c5 f9 fb c2          	vpsubq %xmm2,%xmm0,%xmm0
    1e40:	c5 d9 73 f0 04       	vpsllq $0x4,%xmm0,%xmm4
    1e45:	c5 f9 d4 c4          	vpaddq %xmm4,%xmm0,%xmm0
    1e49:	c5 f9 73 f0 05       	vpsllq $0x5,%xmm0,%xmm0
    1e4e:	c5 f9 fb c2          	vpsubq %xmm2,%xmm0,%xmm0
    1e52:	c5 e1 ef d8          	vpxor  %xmm0,%xmm3,%xmm3
    1e56:	c5 f9 7f 58 f0       	vmovdqa %xmm3,-0x10(%rax)
    1e5b:	48 39 c6             	cmp    %rax,%rsi
    1e5e:	75 80                	jne    1de0 <main+0x9a0>
    1e60:	48 8b 95 40 f3 ff ff 	mov    -0xcc0(%rbp),%rdx
    1e67:	48 8b 85 48 f3 ff ff 	mov    -0xcb8(%rbp),%rax
    1e6e:	48 81 e2 00 00 00 80 	and    $0xffffffff80000000,%rdx
    1e75:	25 ff ff ff 7f       	and    $0x7fffffff,%eax
    1e7a:	48 09 d0             	or     %rdx,%rax
    1e7d:	48 89 c2             	mov    %rax,%rdx
    1e80:	83 e0 01             	and    $0x1,%eax
    1e83:	48 d1 ea             	shr    $1,%rdx
    1e86:	48 33 55 a8          	xor    -0x58(%rbp),%rdx
    1e8a:	48 f7 d8             	neg    %rax
    1e8d:	25 df b0 08 99       	and    $0x9908b0df,%eax
    1e92:	48 31 d0             	xor    %rdx,%rax
    1e95:	48 89 85 40 f3 ff ff 	mov    %rax,-0xcc0(%rbp)
    1e9c:	48 8d 85 48 f3 ff ff 	lea    -0xcb8(%rbp),%rax
    1ea3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1ea8:	c5 f9 6f bd d0 ea ff 	vmovdqa -0x1530(%rbp),%xmm7
    1eaf:	ff 
    1eb0:	c5 c1 db 50 08       	vpand  0x8(%rax),%xmm7,%xmm2
    1eb5:	48 83 c0 10          	add    $0x10,%rax
    1eb9:	c5 f9 6f ad c0 ea ff 	vmovdqa -0x1540(%rbp),%xmm5
    1ec0:	ff 
    1ec1:	c5 d1 db 40 f0       	vpand  -0x10(%rax),%xmm5,%xmm0
    1ec6:	c5 e9 eb d0          	vpor   %xmm0,%xmm2,%xmm2
    1eca:	c5 e1 73 d2 01       	vpsrlq $0x1,%xmm2,%xmm3
    1ecf:	c5 e9 db d1          	vpand  %xmm1,%xmm2,%xmm2
    1ed3:	c5 f9 73 f2 03       	vpsllq $0x3,%xmm2,%xmm0
    1ed8:	c5 e1 ef 98 d8 f8 ff 	vpxor  -0x728(%rax),%xmm3,%xmm3
    1edf:	ff 
    1ee0:	c5 f9 d4 c2          	vpaddq %xmm2,%xmm0,%xmm0
    1ee4:	c5 f9 73 f0 09       	vpsllq $0x9,%xmm0,%xmm0
    1ee9:	c5 f9 d4 c2          	vpaddq %xmm2,%xmm0,%xmm0
    1eed:	c5 f9 73 f0 05       	vpsllq $0x5,%xmm0,%xmm0
    1ef2:	c5 f9 d4 c2          	vpaddq %xmm2,%xmm0,%xmm0
    1ef6:	c5 f9 73 f0 02       	vpsllq $0x2,%xmm0,%xmm0
    1efb:	c5 f9 fb c2          	vpsubq %xmm2,%xmm0,%xmm0
    1eff:	c5 f9 73 f0 03       	vpsllq $0x3,%xmm0,%xmm0
    1f04:	c5 f9 fb c2          	vpsubq %xmm2,%xmm0,%xmm0
    1f08:	c5 d9 73 f0 04       	vpsllq $0x4,%xmm0,%xmm4
    1f0d:	c5 f9 d4 c4          	vpaddq %xmm4,%xmm0,%xmm0
    1f11:	c5 f9 73 f0 05       	vpsllq $0x5,%xmm0,%xmm0
    1f16:	c5 f9 fb c2          	vpsubq %xmm2,%xmm0,%xmm0
    1f1a:	c5 e1 ef d8          	vpxor  %xmm0,%xmm3,%xmm3
    1f1e:	c5 fa 7f 58 f0       	vmovdqu %xmm3,-0x10(%rax)
    1f23:	48 39 c8             	cmp    %rcx,%rax
    1f26:	75 80                	jne    1ea8 <main+0xa68>
    1f28:	48 8b 85 30 ec ff ff 	mov    -0x13d0(%rbp),%rax
    1f2f:	41 bb 01 00 00 00    	mov    $0x1,%r11d
    1f35:	48 8b 55 a8          	mov    -0x58(%rbp),%rdx
    1f39:	49 89 c2             	mov    %rax,%r10
    1f3c:	41 81 e2 ff ff ff 7f 	and    $0x7fffffff,%r10d
    1f43:	48 81 e2 00 00 00 80 	and    $0xffffffff80000000,%rdx
    1f4a:	4c 09 d2             	or     %r10,%rdx
    1f4d:	49 89 d2             	mov    %rdx,%r10
    1f50:	83 e2 01             	and    $0x1,%edx
    1f53:	49 d1 ea             	shr    $1,%r10
    1f56:	4c 33 95 90 f8 ff ff 	xor    -0x770(%rbp),%r10
    1f5d:	48 f7 da             	neg    %rdx
    1f60:	81 e2 df b0 08 99    	and    $0x9908b0df,%edx
    1f66:	49 31 d2             	xor    %rdx,%r10
    1f69:	4c 89 55 a8          	mov    %r10,-0x58(%rbp)
    1f6d:	e9 01 fe ff ff       	jmp    1d73 <main+0x933>
    1f72:	bf 90 00 00 00       	mov    $0x90,%edi
    1f77:	c5 f8 77             	vzeroupper
    1f7a:	e8 11 f1 ff ff       	call   1090 <operator new(unsigned long)@plt>
    1f7f:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    1f83:	49 89 c4             	mov    %rax,%r12
    1f86:	c5 f9 7f 00          	vmovdqa %xmm0,(%rax)
    1f8a:	c4 e3 7d 19 40 10 01 	vextractf128 $0x1,%ymm0,0x10(%rax)
    1f91:	c4 e3 7d 19 40 20 01 	vextractf128 $0x1,%ymm0,0x20(%rax)
    1f98:	c4 e3 7d 19 40 30 01 	vextractf128 $0x1,%ymm0,0x30(%rax)
    1f9f:	c4 e3 7d 19 40 40 01 	vextractf128 $0x1,%ymm0,0x40(%rax)
    1fa6:	c4 e3 7d 19 40 50 01 	vextractf128 $0x1,%ymm0,0x50(%rax)
    1fad:	c4 e3 7d 19 40 60 01 	vextractf128 $0x1,%ymm0,0x60(%rax)
    1fb4:	c4 e3 7d 19 40 70 01 	vextractf128 $0x1,%ymm0,0x70(%rax)
    1fbb:	c4 e3 7d 19 80 80 00 	vextractf128 $0x1,%ymm0,0x80(%rax)
    1fc2:	00 00 01 
    1fc5:	4c 8b ad 90 ea ff ff 	mov    -0x1570(%rbp),%r13
    1fcc:	4c 89 ef             	mov    %r13,%rdi
    1fcf:	c5 f8 77             	vzeroupper
    1fd2:	e8 b9 f0 ff ff       	call   1090 <operator new(unsigned long)@plt>
    1fd7:	49 89 c6             	mov    %rax,%r14
    1fda:	b9 24 00 00 00       	mov    $0x24,%ecx
    1fdf:	31 c0                	xor    %eax,%eax
    1fe1:	49 8d 96 24 01 00 00 	lea    0x124(%r14),%rdx
    1fe8:	4c 89 f7             	mov    %r14,%rdi
    1feb:	4d 01 f5             	add    %r14,%r13
    1fee:	48 83 bd 88 ea ff ff 	cmpq   $0x1,-0x1578(%rbp)
    1ff5:	01 
    1ff6:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    1ff9:	c7 07 00 00 00 00    	movl   $0x0,(%rdi)
    1fff:	0f 84 ce 05 00 00    	je     25d3 <main+0x1193>
    2005:	c4 c1 7e 6f 06       	vmovdqu (%r14),%ymm0
    200a:	48 81 c2 24 01 00 00 	add    $0x124,%rdx
    2011:	c5 fe 7f 82 dc fe ff 	vmovdqu %ymm0,-0x124(%rdx)
    2018:	ff 
    2019:	c4 c1 7e 6f 46 20    	vmovdqu 0x20(%r14),%ymm0
    201f:	c5 fe 7f 82 fc fe ff 	vmovdqu %ymm0,-0x104(%rdx)
    2026:	ff 
    2027:	c4 c1 7e 6f 46 40    	vmovdqu 0x40(%r14),%ymm0
    202d:	c5 fe 7f 82 1c ff ff 	vmovdqu %ymm0,-0xe4(%rdx)
    2034:	ff 
    2035:	c4 c1 7e 6f 46 60    	vmovdqu 0x60(%r14),%ymm0
    203b:	c5 fe 7f 82 3c ff ff 	vmovdqu %ymm0,-0xc4(%rdx)
    2042:	ff 
    2043:	c4 c1 7e 6f 86 80 00 	vmovdqu 0x80(%r14),%ymm0
    204a:	00 00 
    204c:	c5 fe 7f 82 5c ff ff 	vmovdqu %ymm0,-0xa4(%rdx)
    2053:	ff 
    2054:	c4 c1 7e 6f 86 a0 00 	vmovdqu 0xa0(%r14),%ymm0
    205b:	00 00 
    205d:	c5 fe 7f 82 7c ff ff 	vmovdqu %ymm0,-0x84(%rdx)
    2064:	ff 
    2065:	c4 c1 7e 6f 86 c0 00 	vmovdqu 0xc0(%r14),%ymm0
    206c:	00 00 
    206e:	c5 fe 7f 42 9c       	vmovdqu %ymm0,-0x64(%rdx)
    2073:	c4 c1 7e 6f 86 e0 00 	vmovdqu 0xe0(%r14),%ymm0
    207a:	00 00 
    207c:	c5 fe 7f 42 bc       	vmovdqu %ymm0,-0x44(%rdx)
    2081:	c4 c1 7e 6f 86 00 01 	vmovdqu 0x100(%r14),%ymm0
    2088:	00 00 
    208a:	c5 fe 7f 42 dc       	vmovdqu %ymm0,-0x24(%rdx)
    208f:	41 8b 86 20 01 00 00 	mov    0x120(%r14),%eax
    2096:	89 42 fc             	mov    %eax,-0x4(%rdx)
    2099:	4c 39 ea             	cmp    %r13,%rdx
    209c:	0f 85 63 ff ff ff    	jne    2005 <main+0xbc5>
    20a2:	48 8b bd 48 ea ff ff 	mov    -0x15b8(%rbp),%rdi
    20a9:	c5 f8 77             	vzeroupper
    20ac:	e8 df ef ff ff       	call   1090 <operator new(unsigned long)@plt>
    20b1:	48 89 c6             	mov    %rax,%rsi
    20b4:	b9 20 00 00 00       	mov    $0x20,%ecx
    20b9:	31 c0                	xor    %eax,%eax
    20bb:	48 89 f7             	mov    %rsi,%rdi
    20be:	48 89 b5 e0 ea ff ff 	mov    %rsi,-0x1520(%rbp)
    20c5:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    20c8:	48 8b 85 48 ea ff ff 	mov    -0x15b8(%rbp),%rax
    20cf:	48 89 f1             	mov    %rsi,%rcx
    20d2:	48 8d 14 06          	lea    (%rsi,%rax,1),%rdx
    20d6:	48 8d 86 00 01 00 00 	lea    0x100(%rsi),%rax
    20dd:	c5 fe 6f 01          	vmovdqu (%rcx),%ymm0
    20e1:	48 05 00 01 00 00    	add    $0x100,%rax
    20e7:	c5 fe 7f 80 00 ff ff 	vmovdqu %ymm0,-0x100(%rax)
    20ee:	ff 
    20ef:	c5 fe 6f 41 20       	vmovdqu 0x20(%rcx),%ymm0
    20f4:	c5 fe 7f 80 20 ff ff 	vmovdqu %ymm0,-0xe0(%rax)
    20fb:	ff 
    20fc:	c5 fe 6f 41 40       	vmovdqu 0x40(%rcx),%ymm0
    2101:	c5 fe 7f 80 40 ff ff 	vmovdqu %ymm0,-0xc0(%rax)
    2108:	ff 
    2109:	c5 fe 6f 41 60       	vmovdqu 0x60(%rcx),%ymm0
    210e:	c5 fe 7f 80 60 ff ff 	vmovdqu %ymm0,-0xa0(%rax)
    2115:	ff 
    2116:	c5 fe 6f 81 80 00 00 	vmovdqu 0x80(%rcx),%ymm0
    211d:	00 
    211e:	c5 fe 7f 40 80       	vmovdqu %ymm0,-0x80(%rax)
    2123:	c5 fe 6f 81 a0 00 00 	vmovdqu 0xa0(%rcx),%ymm0
    212a:	00 
    212b:	c5 fe 7f 40 a0       	vmovdqu %ymm0,-0x60(%rax)
    2130:	c5 fe 6f 81 c0 00 00 	vmovdqu 0xc0(%rcx),%ymm0
    2137:	00 
    2138:	c5 fe 7f 40 c0       	vmovdqu %ymm0,-0x40(%rax)
    213d:	c5 fe 6f 81 e0 00 00 	vmovdqu 0xe0(%rcx),%ymm0
    2144:	00 
    2145:	c5 fe 7f 40 e0       	vmovdqu %ymm0,-0x20(%rax)
    214a:	48 39 d0             	cmp    %rdx,%rax
    214d:	75 8e                	jne    20dd <main+0xc9d>
    214f:	48 8b 85 90 ea ff ff 	mov    -0x1570(%rbp),%rax
    2156:	48 89 85 78 ea ff ff 	mov    %rax,-0x1588(%rbp)
    215d:	48 8b 85 48 ea ff ff 	mov    -0x15b8(%rbp),%rax
    2164:	48 89 85 80 ea ff ff 	mov    %rax,-0x1580(%rbp)
    216b:	48 8b 85 40 ea ff ff 	mov    -0x15c0(%rbp),%rax
    2172:	48 89 85 b8 ea ff ff 	mov    %rax,-0x1548(%rbp)
    2179:	c5 f8 77             	vzeroupper
    217c:	e9 9f 04 00 00       	jmp    2620 <main+0x11e0>
    2181:	83 85 50 ea ff ff 01 	addl   $0x1,-0x15b0(%rbp)
    2188:	4c 89 5d b0          	mov    %r11,-0x50(%rbp)
    218c:	48 81 c3 90 00 00 00 	add    $0x90,%rbx
    2193:	49 81 c4 90 00 00 00 	add    $0x90,%r12
    219a:	8b 85 50 ea ff ff    	mov    -0x15b0(%rbp),%eax
    21a0:	8b bd a8 ea ff ff    	mov    -0x1558(%rbp),%edi
    21a6:	48 81 85 10 ea ff ff 	addq   $0x124,-0x15f0(%rbp)
    21ad:	24 01 00 00 
    21b1:	39 f8                	cmp    %edi,%eax
    21b3:	0f 8c e7 f9 ff ff    	jl     1ba0 <main+0x760>
    21b9:	48 8b 9d 00 ea ff ff 	mov    -0x1600(%rbp),%rbx
    21c0:	b8 80 80 80 80       	mov    $0x80808080,%eax
    21c5:	4d 89 f0             	mov    %r14,%r8
    21c8:	31 f6                	xor    %esi,%esi
    21ca:	4c 8b a5 f8 e9 ff ff 	mov    -0x1608(%rbp),%r12
    21d1:	c5 f9 6e c8          	vmovd  %eax,%xmm1
    21d5:	b8 0f 0f 0f 0f       	mov    $0xf0f0f0f,%eax
    21da:	b9 01 00 00 00       	mov    $0x1,%ecx
    21df:	48 8b bd e0 ea ff ff 	mov    -0x1520(%rbp),%rdi
    21e6:	c5 f9 6e c0          	vmovd  %eax,%xmm0
    21ea:	c5 f9 70 c9 00       	vpshufd $0x0,%xmm1,%xmm1
    21ef:	c5 f9 70 c0 00       	vpshufd $0x0,%xmm0,%xmm0
    21f4:	c4 c1 7a 6f 50 04    	vmovdqu 0x4(%r8),%xmm2
    21fa:	41 8b 00             	mov    (%r8),%eax
    21fd:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    2201:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    2206:	c5 f9 7f 17          	vmovdqa %xmm2,(%rdi)
    220a:	c4 c1 7a 6f 50 14    	vmovdqu 0x14(%r8),%xmm2
    2210:	f7 d0                	not    %eax
    2212:	a9 00 00 80 7f       	test   $0x7f800000,%eax
    2217:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    221b:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    221f:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    2224:	89 85 50 ea ff ff    	mov    %eax,-0x15b0(%rbp)
    222a:	0f 95 c2             	setne  %dl
    222d:	c5 f9 7f 57 10       	vmovdqa %xmm2,0x10(%rdi)
    2232:	c4 c1 7a 6f 50 24    	vmovdqu 0x24(%r8),%xmm2
    2238:	c5 79 d7 db          	vpmovmskb %xmm3,%r11d
    223c:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    2240:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    2245:	c5 f9 7f 57 20       	vmovdqa %xmm2,0x20(%rdi)
    224a:	c4 c1 7a 6f 50 34    	vmovdqu 0x34(%r8),%xmm2
    2250:	c5 79 d7 d3          	vpmovmskb %xmm3,%r10d
    2254:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    2258:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    225d:	c5 f9 7f 57 30       	vmovdqa %xmm2,0x30(%rdi)
    2262:	c4 c1 7a 6f 50 44    	vmovdqu 0x44(%r8),%xmm2
    2268:	c5 79 d7 cb          	vpmovmskb %xmm3,%r9d
    226c:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    2270:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    2275:	c5 f9 7f 57 40       	vmovdqa %xmm2,0x40(%rdi)
    227a:	c4 c1 7a 6f 50 54    	vmovdqu 0x54(%r8),%xmm2
    2280:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2284:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    2288:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    228d:	c5 f9 7f 57 50       	vmovdqa %xmm2,0x50(%rdi)
    2292:	c4 c1 7a 6f 50 64    	vmovdqu 0x64(%r8),%xmm2
    2298:	89 85 10 ea ff ff    	mov    %eax,-0x15f0(%rbp)
    229e:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    22a2:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    22a6:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    22ab:	c5 f9 7f 57 60       	vmovdqa %xmm2,0x60(%rdi)
    22b0:	c4 c1 7a 6f 50 74    	vmovdqu 0x74(%r8),%xmm2
    22b6:	89 85 08 ea ff ff    	mov    %eax,-0x15f8(%rbp)
    22bc:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    22c0:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    22c4:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    22c9:	c5 f9 7f 57 70       	vmovdqa %xmm2,0x70(%rdi)
    22ce:	c4 c1 7a 6f 90 84 00 	vmovdqu 0x84(%r8),%xmm2
    22d5:	00 00 
    22d7:	89 85 00 ea ff ff    	mov    %eax,-0x1600(%rbp)
    22dd:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    22e1:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    22e5:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    22ea:	c5 f9 7f 97 80 00 00 	vmovdqa %xmm2,0x80(%rdi)
    22f1:	00 
    22f2:	c4 c1 7a 6f 90 94 00 	vmovdqu 0x94(%r8),%xmm2
    22f9:	00 00 
    22fb:	89 85 f8 e9 ff ff    	mov    %eax,-0x1608(%rbp)
    2301:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2305:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    2309:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    230e:	c5 f9 7f 97 90 00 00 	vmovdqa %xmm2,0x90(%rdi)
    2315:	00 
    2316:	c4 c1 7a 6f 90 a4 00 	vmovdqu 0xa4(%r8),%xmm2
    231d:	00 00 
    231f:	89 85 f0 e9 ff ff    	mov    %eax,-0x1610(%rbp)
    2325:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2329:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    232d:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    2332:	c5 f9 7f 97 a0 00 00 	vmovdqa %xmm2,0xa0(%rdi)
    2339:	00 
    233a:	c4 c1 7a 6f 90 b4 00 	vmovdqu 0xb4(%r8),%xmm2
    2341:	00 00 
    2343:	89 85 ec e9 ff ff    	mov    %eax,-0x1614(%rbp)
    2349:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    234d:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    2351:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    2356:	89 85 e8 e9 ff ff    	mov    %eax,-0x1618(%rbp)
    235c:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2360:	89 85 e4 e9 ff ff    	mov    %eax,-0x161c(%rbp)
    2366:	c5 f9 7f 97 b0 00 00 	vmovdqa %xmm2,0xb0(%rdi)
    236d:	00 
    236e:	c4 c1 7a 6f 90 c4 00 	vmovdqu 0xc4(%r8),%xmm2
    2375:	00 00 
    2377:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    237b:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    2380:	c5 f9 7f 97 c0 00 00 	vmovdqa %xmm2,0xc0(%rdi)
    2387:	00 
    2388:	c4 c1 7a 6f 90 d4 00 	vmovdqu 0xd4(%r8),%xmm2
    238f:	00 00 
    2391:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2395:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    2399:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    239e:	c5 f9 7f 97 d0 00 00 	vmovdqa %xmm2,0xd0(%rdi)
    23a5:	00 
    23a6:	c4 c1 7a 6f 90 e4 00 	vmovdqu 0xe4(%r8),%xmm2
    23ad:	00 00 
    23af:	89 85 e0 e9 ff ff    	mov    %eax,-0x1620(%rbp)
    23b5:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    23b9:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    23bd:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    23c2:	c5 f9 7f 97 e0 00 00 	vmovdqa %xmm2,0xe0(%rdi)
    23c9:	00 
    23ca:	c4 c1 7a 6f 90 f4 00 	vmovdqu 0xf4(%r8),%xmm2
    23d1:	00 00 
    23d3:	89 85 dc e9 ff ff    	mov    %eax,-0x1624(%rbp)
    23d9:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    23dd:	c5 e9 74 d9          	vpcmpeqb %xmm1,%xmm2,%xmm3
    23e1:	c4 e2 79 04 d2       	vpmaddubsw %xmm2,%xmm0,%xmm2
    23e6:	89 85 d8 e9 ff ff    	mov    %eax,-0x1628(%rbp)
    23ec:	c5 f9 7f 97 f0 00 00 	vmovdqa %xmm2,0xf0(%rdi)
    23f3:	00 
    23f4:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    23f8:	89 85 d4 e9 ff ff    	mov    %eax,-0x162c(%rbp)
    23fe:	8b 85 50 ea ff ff    	mov    -0x15b0(%rbp),%eax
    2404:	44 09 d8             	or     %r11d,%eax
    2407:	44 8b 9d 10 ea ff ff 	mov    -0x15f0(%rbp),%r11d
    240e:	44 09 d0             	or     %r10d,%eax
    2411:	44 09 c8             	or     %r9d,%eax
    2414:	44 09 d8             	or     %r11d,%eax
    2417:	44 8b 9d 08 ea ff ff 	mov    -0x15f8(%rbp),%r11d
    241e:	44 09 d8             	or     %r11d,%eax
    2421:	44 8b 9d 00 ea ff ff 	mov    -0x1600(%rbp),%r11d
    2428:	44 09 d8             	or     %r11d,%eax
    242b:	44 8b 9d f8 e9 ff ff 	mov    -0x1608(%rbp),%r11d
    2432:	44 09 d8             	or     %r11d,%eax
    2435:	44 8b 9d f0 e9 ff ff 	mov    -0x1610(%rbp),%r11d
    243c:	44 09 d8             	or     %r11d,%eax
    243f:	44 8b 9d ec e9 ff ff 	mov    -0x1614(%rbp),%r11d
    2446:	44 09 d8             	or     %r11d,%eax
    2449:	44 8b 9d e8 e9 ff ff 	mov    -0x1618(%rbp),%r11d
    2450:	44 09 d8             	or     %r11d,%eax
    2453:	44 8b 9d e4 e9 ff ff 	mov    -0x161c(%rbp),%r11d
    245a:	44 09 d8             	or     %r11d,%eax
    245d:	44 8b 9d e0 e9 ff ff 	mov    -0x1620(%rbp),%r11d
    2464:	44 09 d8             	or     %r11d,%eax
    2467:	44 8b 9d dc e9 ff ff 	mov    -0x1624(%rbp),%r11d
    246e:	44 09 d8             	or     %r11d,%eax
    2471:	44 8b 9d d8 e9 ff ff 	mov    -0x1628(%rbp),%r11d
    2478:	44 09 d8             	or     %r11d,%eax
    247b:	0b 85 d4 e9 ff ff    	or     -0x162c(%rbp),%eax
    2481:	0f 94 c0             	sete   %al
    2484:	83 c6 01             	add    $0x1,%esi
    2487:	49 81 c0 24 01 00 00 	add    $0x124,%r8
    248e:	21 d0                	and    %edx,%eax
    2490:	48 81 c7 00 01 00 00 	add    $0x100,%rdi
    2497:	21 c1                	and    %eax,%ecx
    2499:	8b 85 a8 ea ff ff    	mov    -0x1558(%rbp),%eax
    249f:	39 c6                	cmp    %eax,%esi
    24a1:	0f 8c 4d fd ff ff    	jl     21f4 <main+0xdb4>
    24a7:	84 c9                	test   %cl,%cl
    24a9:	0f 84 f6 ee ff ff    	je     13a5 <main.cold+0x295>
    24af:	44 8b 9d ac ea ff ff 	mov    -0x1554(%rbp),%r11d
    24b6:	4c 89 f2             	mov    %r14,%rdx
    24b9:	4c 89 fe             	mov    %r15,%rsi
    24bc:	44 89 df             	mov    %r11d,%edi
    24bf:	e8 6c 0a 00 00       	call   2f30 <float iq_compact_dot<false>(int, iq_compact_block const*, block_q8_K const*)>
    24c4:	4c 89 f2             	mov    %r14,%rdx
    24c7:	4c 89 e6             	mov    %r12,%rsi
    24ca:	44 89 df             	mov    %r11d,%edi
    24cd:	48 8b 8d e0 ea ff ff 	mov    -0x1520(%rbp),%rcx
    24d4:	c4 c1 79 7e c2       	vmovd  %xmm0,%r10d
    24d9:	e8 e2 0e 00 00       	call   33c0 <offset4_fast_dot(int, offset4_iq3s_block const*, block_q8_K const*, offset4_rhs_block const*)>
    24de:	c5 f9 7e c0          	vmovd  %xmm0,%eax
    24e2:	41 39 c2             	cmp    %eax,%r10d
    24e5:	0f 85 88 ee ff ff    	jne    1373 <main.cold+0x263>
    24eb:	44 89 df             	mov    %r11d,%edi
    24ee:	4c 89 f2             	mov    %r14,%rdx
    24f1:	4c 89 e6             	mov    %r12,%rsi
    24f4:	e8 c7 12 00 00       	call   37c0 <offset4_compat_dot(int, offset4_iq3s_block const*, block_q8_K const*)>
    24f9:	c5 f9 7e c0          	vmovd  %xmm0,%eax
    24fd:	41 39 c2             	cmp    %eax,%r10d
    2500:	0f 85 3f ec ff ff    	jne    1145 <main.cold+0x35>
    2506:	48 83 85 28 ea ff ff 	addq   $0x1,-0x15d8(%rbp)
    250d:	01 
    250e:	48 83 85 98 ea ff ff 	addq   $0x1,-0x1568(%rbp)
    2515:	01 
    2516:	83 bd e8 ea ff ff 0f 	cmpl   $0xf,-0x1518(%rbp)
    251d:	0f 8e ec 03 00 00    	jle    290f <main+0x14cf>
    2523:	48 8b 85 e0 ea ff ff 	mov    -0x1520(%rbp),%rax
    252a:	48 85 c0             	test   %rax,%rax
    252d:	74 0f                	je     253e <main+0x10fe>
    252f:	48 8b b5 80 ea ff ff 	mov    -0x1580(%rbp),%rsi
    2536:	48 89 c7             	mov    %rax,%rdi
    2539:	e8 62 eb ff ff       	call   10a0 <operator delete(void*, unsigned long)@plt>
    253e:	4d 85 f6             	test   %r14,%r14
    2541:	74 0f                	je     2552 <main+0x1112>
    2543:	48 8b b5 78 ea ff ff 	mov    -0x1588(%rbp),%rsi
    254a:	4c 89 f7             	mov    %r14,%rdi
    254d:	e8 4e eb ff ff       	call   10a0 <operator delete(void*, unsigned long)@plt>
    2552:	4d 85 e4             	test   %r12,%r12
    2555:	74 0f                	je     2566 <main+0x1126>
    2557:	48 8b b5 b8 ea ff ff 	mov    -0x1548(%rbp),%rsi
    255e:	4c 89 e7             	mov    %r12,%rdi
    2561:	e8 3a eb ff ff       	call   10a0 <operator delete(void*, unsigned long)@plt>
    2566:	4d 85 ff             	test   %r15,%r15
    2569:	0f 84 63 01 00 00    	je     26d2 <main+0x1292>
    256f:	48 8b b5 b8 ea ff ff 	mov    -0x1548(%rbp),%rsi
    2576:	4c 89 ff             	mov    %r15,%rdi
    2579:	e8 22 eb ff ff       	call   10a0 <operator delete(void*, unsigned long)@plt>
    257e:	83 85 e8 ea ff ff 01 	addl   $0x1,-0x1518(%rbp)
    2585:	8b 85 e8 ea ff ff    	mov    -0x1518(%rbp),%eax
    258b:	3d 00 01 00 00       	cmp    $0x100,%eax
    2590:	0f 84 54 01 00 00    	je     26ea <main+0x12aa>
    2596:	48 8b 85 98 ea ff ff 	mov    -0x1568(%rbp),%rax
    259d:	48 89 85 30 ea ff ff 	mov    %rax,-0x15d0(%rbp)
    25a4:	e9 f3 f3 ff ff       	jmp    199c <main+0x55c>
    25a9:	48 8b bd b0 ea ff ff 	mov    -0x1550(%rbp),%rdi
    25b0:	e8 db 18 00 00       	call   3e90 <std::mersenne_twister_engine<unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>::_M_gen_rand()>
    25b5:	48 8b 75 b0          	mov    -0x50(%rbp),%rsi
    25b9:	e9 c5 f6 ff ff       	jmp    1c83 <main+0x843>
    25be:	48 8b bd b0 ea ff ff 	mov    -0x1550(%rbp),%rdi
    25c5:	e8 c6 18 00 00       	call   3e90 <std::mersenne_twister_engine<unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>::_M_gen_rand()>
    25ca:	4c 8b 5d b0          	mov    -0x50(%rbp),%r11
    25ce:	e9 2e f6 ff ff       	jmp    1c01 <main+0x7c1>
    25d3:	bf 00 01 00 00       	mov    $0x100,%edi
    25d8:	48 89 95 b8 ea ff ff 	mov    %rdx,-0x1548(%rbp)
    25df:	e8 ac ea ff ff       	call   1090 <operator new(unsigned long)@plt>
    25e4:	48 89 c7             	mov    %rax,%rdi
    25e7:	48 89 85 e0 ea ff ff 	mov    %rax,-0x1520(%rbp)
    25ee:	31 c0                	xor    %eax,%eax
    25f0:	b9 20 00 00 00       	mov    $0x20,%ecx
    25f5:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    25f8:	4c 8b ad b8 ea ff ff 	mov    -0x1548(%rbp),%r13
    25ff:	48 c7 85 78 ea ff ff 	movq   $0x124,-0x1588(%rbp)
    2606:	24 01 00 00 
    260a:	48 c7 85 80 ea ff ff 	movq   $0x100,-0x1580(%rbp)
    2611:	00 01 00 00 
    2615:	48 c7 85 b8 ea ff ff 	movq   $0x90,-0x1548(%rbp)
    261c:	90 00 00 00 
    2620:	48 8b 55 b0          	mov    -0x50(%rbp),%rdx
    2624:	48 8d 85 6e eb ff ff 	lea    -0x1492(%rbp),%rax
    262b:	4c 89 bd 10 ea ff ff 	mov    %r15,-0x15f0(%rbp)
    2632:	48 c7 85 50 ea ff ff 	movq   $0x0,-0x15b0(%rbp)
    2639:	00 00 00 00 
    263d:	48 89 85 08 ea ff ff 	mov    %rax,-0x15f8(%rbp)
    2644:	48 8b 8d 38 ea ff ff 	mov    -0x15c8(%rbp),%rcx
    264b:	4c 89 a5 00 ea ff ff 	mov    %r12,-0x1600(%rbp)
    2652:	49 89 dc             	mov    %rbx,%r12
    2655:	48 89 cb             	mov    %rcx,%rbx
    2658:	eb 5a                	jmp    26b4 <main+0x1274>
    265a:	48 8d 50 01          	lea    0x1(%rax),%rdx
    265e:	48 8b 84 c5 30 ec ff 	mov    -0x13d0(%rbp,%rax,8),%rax
    2665:	ff 
    2666:	48 83 c3 01          	add    $0x1,%rbx
    266a:	48 89 55 b0          	mov    %rdx,-0x50(%rbp)
    266e:	48 89 c1             	mov    %rax,%rcx
    2671:	48 c1 e9 0b          	shr    $0xb,%rcx
    2675:	89 c9                	mov    %ecx,%ecx
    2677:	48 31 c8             	xor    %rcx,%rax
    267a:	48 89 c1             	mov    %rax,%rcx
    267d:	48 c1 e1 07          	shl    $0x7,%rcx
    2681:	81 e1 80 56 2c 9d    	and    $0x9d2c5680,%ecx
    2687:	48 31 c8             	xor    %rcx,%rax
    268a:	48 89 c1             	mov    %rax,%rcx
    268d:	48 c1 e1 0f          	shl    $0xf,%rcx
    2691:	81 e1 00 00 c6 ef    	and    $0xefc60000,%ecx
    2697:	48 31 c8             	xor    %rcx,%rax
    269a:	48 89 c1             	mov    %rax,%rcx
    269d:	48 c1 e9 12          	shr    $0x12,%rcx
    26a1:	48 31 c8             	xor    %rcx,%rax
    26a4:	88 43 ff             	mov    %al,-0x1(%rbx)
    26a7:	48 39 9d 08 ea ff ff 	cmp    %rbx,-0x15f8(%rbp)
    26ae:	0f 84 fc 00 00 00    	je     27b0 <main+0x1370>
    26b4:	48 89 d0             	mov    %rdx,%rax
    26b7:	48 81 fa 6f 02 00 00 	cmp    $0x26f,%rdx
    26be:	76 9a                	jbe    265a <main+0x121a>
    26c0:	48 8b bd b0 ea ff ff 	mov    -0x1550(%rbp),%rdi
    26c7:	e8 c4 17 00 00       	call   3e90 <std::mersenne_twister_engine<unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>::_M_gen_rand()>
    26cc:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
    26d0:	eb 88                	jmp    265a <main+0x121a>
    26d2:	83 85 e8 ea ff ff 01 	addl   $0x1,-0x1518(%rbp)
    26d9:	8b 85 e8 ea ff ff    	mov    -0x1518(%rbp),%eax
    26df:	3d 00 01 00 00       	cmp    $0x100,%eax
    26e4:	0f 85 ac fe ff ff    	jne    2596 <main+0x1156>
    26ea:	48 83 85 60 ea ff ff 	addq   $0x4,-0x15a0(%rbp)
    26f1:	04 
    26f2:	48 8d 0d 2b 2b 00 00 	lea    0x2b2b(%rip),%rcx        # 5224 <C.169.7+0x14>
    26f9:	48 8b 85 60 ea ff ff 	mov    -0x15a0(%rbp),%rax
    2700:	48 39 c1             	cmp    %rax,%rcx
    2703:	0f 85 fb f1 ff ff    	jne    1904 <main+0x4c4>
    2709:	48 83 85 68 ea ff ff 	addq   $0x4,-0x1598(%rbp)
    2710:	04 
    2711:	48 8d 35 28 2b 00 00 	lea    0x2b28(%rip),%rsi        # 5240 <main::bad>
    2718:	48 8b 85 68 ea ff ff 	mov    -0x1598(%rbp),%rax
    271f:	48 39 c6             	cmp    %rax,%rsi
    2722:	0f 85 b1 f1 ff ff    	jne    18d9 <main+0x499>
    2728:	48 83 85 70 ea ff ff 	addq   $0x4,-0x1590(%rbp)
    272f:	04 
    2730:	48 8b 85 70 ea ff ff 	mov    -0x1590(%rbp),%rax
    2737:	48 39 85 38 ea ff ff 	cmp    %rax,-0x15c8(%rbp)
    273e:	0f 85 6d f1 ff ff    	jne    18b1 <main+0x471>
    2744:	8b 85 24 ea ff ff    	mov    -0x15dc(%rbp),%eax
    274a:	89 85 ec ea ff ff    	mov    %eax,-0x1514(%rbp)
    2750:	c5 f8 ae 95 ec ea ff 	vldmxcsr -0x1514(%rbp)
    2757:	ff 
    2758:	49 89 d9             	mov    %rbx,%r9
    275b:	b9 00 01 00 00       	mov    $0x100,%ecx
    2760:	50                   	push   %rax
    2761:	4c 8b 85 18 ea ff ff 	mov    -0x15e8(%rbp),%r8
    2768:	31 c0                	xor    %eax,%eax
    276a:	ba 90 00 00 00       	mov    $0x90,%edx
    276f:	ff b5 58 ea ff ff    	push   -0x15a8(%rbp)
    2775:	48 8d 35 27 29 00 00 	lea    0x2927(%rip),%rsi        # 50a3 <_fini+0x102b>
    277c:	ff b5 98 ea ff ff    	push   -0x1568(%rbp)
    2782:	48 8d 3d 77 29 00 00 	lea    0x2977(%rip),%rdi        # 5100 <_fini+0x1088>
    2789:	ff b5 28 ea ff ff    	push   -0x15d8(%rbp)
    278f:	e8 9c e8 ff ff       	call   1030 <printf@plt>
    2794:	48 83 c4 20          	add    $0x20,%rsp
    2798:	31 c0                	xor    %eax,%eax
    279a:	48 8d 65 d0          	lea    -0x30(%rbp),%rsp
    279e:	5b                   	pop    %rbx
    279f:	41 5a                	pop    %r10
    27a1:	41 5c                	pop    %r12
    27a3:	41 5d                	pop    %r13
    27a5:	41 5e                	pop    %r14
    27a7:	41 5f                	pop    %r15
    27a9:	5d                   	pop    %rbp
    27aa:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    27ae:	c3                   	ret
    27af:	90                   	nop
    27b0:	4c 89 e3             	mov    %r12,%rbx
    27b3:	4c 8b a5 00 ea ff ff 	mov    -0x1600(%rbp),%r12
    27ba:	48 81 fa 6f 02 00 00 	cmp    $0x26f,%rdx
    27c1:	0f 87 33 01 00 00    	ja     28fa <main+0x14ba>
    27c7:	48 8b 8c d5 30 ec ff 	mov    -0x13d0(%rbp,%rdx,8),%rcx
    27ce:	ff 
    27cf:	48 8d 72 01          	lea    0x1(%rdx),%rsi
    27d3:	48 89 75 b0          	mov    %rsi,-0x50(%rbp)
    27d7:	48 89 c8             	mov    %rcx,%rax
    27da:	48 c1 e8 0b          	shr    $0xb,%rax
    27de:	89 c0                	mov    %eax,%eax
    27e0:	48 31 c1             	xor    %rax,%rcx
    27e3:	48 89 c8             	mov    %rcx,%rax
    27e6:	48 c1 e0 07          	shl    $0x7,%rax
    27ea:	25 80 56 2c 9d       	and    $0x9d2c5680,%eax
    27ef:	48 31 c1             	xor    %rax,%rcx
    27f2:	48 89 c8             	mov    %rcx,%rax
    27f5:	48 c1 e0 0f          	shl    $0xf,%rax
    27f9:	25 00 00 c6 ef       	and    $0xefc60000,%eax
    27fe:	48 31 c1             	xor    %rax,%rcx
    2801:	48 89 c8             	mov    %rcx,%rax
    2804:	48 c1 e8 12          	shr    $0x12,%rax
    2808:	48 31 c1             	xor    %rax,%rcx
    280b:	48 b8 31 42 08 21 84 	movabs $0x842108421084231,%rax
    2812:	10 42 08 
    2815:	48 89 ca             	mov    %rcx,%rdx
    2818:	48 c1 ea 0a          	shr    $0xa,%rdx
    281c:	48 f7 e2             	mul    %rdx
    281f:	48 69 d2 00 7c 00 00 	imul   $0x7c00,%rdx,%rdx
    2826:	48 29 d1             	sub    %rdx,%rcx
    2829:	48 89 8d 00 ea ff ff 	mov    %rcx,-0x1600(%rbp)
    2830:	48 81 fe 6f 02 00 00 	cmp    $0x26f,%rsi
    2837:	0f 87 a8 00 00 00    	ja     28e5 <main+0x14a5>
    283d:	48 8b 84 f5 30 ec ff 	mov    -0x13d0(%rbp,%rsi,8),%rax
    2844:	ff 
    2845:	48 8d 56 01          	lea    0x1(%rsi),%rdx
    2849:	48 8b bd 38 ea ff ff 	mov    -0x15c8(%rbp),%rdi
    2850:	48 89 55 b0          	mov    %rdx,-0x50(%rbp)
    2854:	48 8b b5 10 ea ff ff 	mov    -0x15f0(%rbp),%rsi
    285b:	48 89 95 f8 e9 ff ff 	mov    %rdx,-0x1608(%rbp)
    2862:	48 89 c1             	mov    %rax,%rcx
    2865:	48 c1 e9 0b          	shr    $0xb,%rcx
    2869:	89 c9                	mov    %ecx,%ecx
    286b:	48 31 c8             	xor    %rcx,%rax
    286e:	48 89 c1             	mov    %rax,%rcx
    2871:	48 c1 e1 07          	shl    $0x7,%rcx
    2875:	81 e1 80 56 2c 9d    	and    $0x9d2c5680,%ecx
    287b:	48 31 c8             	xor    %rcx,%rax
    287e:	48 89 c1             	mov    %rax,%rcx
    2881:	48 c1 e1 0f          	shl    $0xf,%rcx
    2885:	81 e1 00 00 c6 ef    	and    $0xefc60000,%ecx
    288b:	48 31 c8             	xor    %rcx,%rax
    288e:	48 89 c1             	mov    %rax,%rcx
    2891:	48 c1 e9 12          	shr    $0x12,%rcx
    2895:	48 31 c8             	xor    %rcx,%rax
    2898:	66 25 00 80          	and    $0x8000,%ax
    289c:	66 0b 85 00 ea ff ff 	or     -0x1600(%rbp),%ax
    28a3:	66 89 85 00 eb ff ff 	mov    %ax,-0x1500(%rbp)
    28aa:	e8 a1 13 00 00       	call   3c50 <void iq_pack<false>(std::conditional<false, block_iq2_s, block_iq3_s>::type const&, iq_compact_block&)>
    28af:	48 8b b5 88 ea ff ff 	mov    -0x1578(%rbp),%rsi
    28b6:	48 83 85 50 ea ff ff 	addq   $0x1,-0x15b0(%rbp)
    28bd:	01 
    28be:	48 8b 85 50 ea ff ff 	mov    -0x15b0(%rbp),%rax
    28c5:	48 81 85 10 ea ff ff 	addq   $0x90,-0x15f0(%rbp)
    28cc:	90 00 00 00 
    28d0:	48 8b 95 f8 e9 ff ff 	mov    -0x1608(%rbp),%rdx
    28d7:	48 39 f0             	cmp    %rsi,%rax
    28da:	0f 82 64 fd ff ff    	jb     2644 <main+0x1204>
    28e0:	e9 81 f2 ff ff       	jmp    1b66 <main+0x726>
    28e5:	48 8b bd b0 ea ff ff 	mov    -0x1550(%rbp),%rdi
    28ec:	e8 9f 15 00 00       	call   3e90 <std::mersenne_twister_engine<unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>::_M_gen_rand()>
    28f1:	48 8b 75 b0          	mov    -0x50(%rbp),%rsi
    28f5:	e9 43 ff ff ff       	jmp    283d <main+0x13fd>
    28fa:	48 8b bd b0 ea ff ff 	mov    -0x1550(%rbp),%rdi
    2901:	e8 8a 15 00 00       	call   3e90 <std::mersenne_twister_engine<unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>::_M_gen_rand()>
    2906:	48 8b 55 b0          	mov    -0x50(%rbp),%rdx
    290a:	e9 b8 fe ff ff       	jmp    27c7 <main+0x1387>
    290f:	c4 e2 79 18 15 78 3a 	vbroadcastss 0x3a78(%rip),%xmm2        # 6390 <iq3s_grid+0x890>
    2916:	00 00 
    2918:	45 31 c9             	xor    %r9d,%r9d
    291b:	c4 e2 79 18 0d fc 39 	vbroadcastss 0x39fc(%rip),%xmm1        # 6320 <iq3s_grid+0x820>
    2922:	00 00 
    2924:	4c 8d 1d 15 29 00 00 	lea    0x2915(%rip),%r11        # 5240 <main::bad>
    292b:	c4 c1 7e 6f 85 dc fe 	vmovdqu -0x124(%r13),%ymm0
    2932:	ff ff 
    2934:	41 8b 45 fc          	mov    -0x4(%r13),%eax
    2938:	c5 fe 7f 85 00 eb ff 	vmovdqu %ymm0,-0x1500(%rbp)
    293f:	ff 
    2940:	c4 c1 7e 6f 85 fc fe 	vmovdqu -0x104(%r13),%ymm0
    2947:	ff ff 
    2949:	c5 fe 7f 85 20 eb ff 	vmovdqu %ymm0,-0x14e0(%rbp)
    2950:	ff 
    2951:	c4 c1 7e 6f 85 1c ff 	vmovdqu -0xe4(%r13),%ymm0
    2958:	ff ff 
    295a:	89 85 20 ec ff ff    	mov    %eax,-0x13e0(%rbp)
    2960:	c5 fe 7f 85 40 eb ff 	vmovdqu %ymm0,-0x14c0(%rbp)
    2967:	ff 
    2968:	c4 c1 7e 6f 85 3c ff 	vmovdqu -0xc4(%r13),%ymm0
    296f:	ff ff 
    2971:	c5 fe 7f 85 60 eb ff 	vmovdqu %ymm0,-0x14a0(%rbp)
    2978:	ff 
    2979:	c4 c1 7e 6f 85 5c ff 	vmovdqu -0xa4(%r13),%ymm0
    2980:	ff ff 
    2982:	c5 fe 7f 85 80 eb ff 	vmovdqu %ymm0,-0x1480(%rbp)
    2989:	ff 
    298a:	c4 c1 7e 6f 85 7c ff 	vmovdqu -0x84(%r13),%ymm0
    2991:	ff ff 
    2993:	c5 fe 7f 85 a0 eb ff 	vmovdqu %ymm0,-0x1460(%rbp)
    299a:	ff 
    299b:	c4 c1 7e 6f 45 9c    	vmovdqu -0x64(%r13),%ymm0
    29a1:	c5 fe 7f 85 c0 eb ff 	vmovdqu %ymm0,-0x1440(%rbp)
    29a8:	ff 
    29a9:	c4 c1 7e 6f 45 bc    	vmovdqu -0x44(%r13),%ymm0
    29af:	c5 fe 7f 85 e0 eb ff 	vmovdqu %ymm0,-0x1420(%rbp)
    29b6:	ff 
    29b7:	c4 c1 7e 6f 45 dc    	vmovdqu -0x24(%r13),%ymm0
    29bd:	c5 fe 7f 85 00 ec ff 	vmovdqu %ymm0,-0x1400(%rbp)
    29c4:	ff 
    29c5:	4d 85 c9             	test   %r9,%r9
    29c8:	0f 84 f4 03 00 00    	je     2dc2 <main+0x1982>
    29ce:	43 8b 04 8b          	mov    (%r11,%r9,4),%eax
    29d2:	41 89 85 dc fe ff ff 	mov    %eax,-0x124(%r13)
    29d9:	81 bd ac ea ff ff ff 	cmpl   $0xff,-0x1554(%rbp)
    29e0:	00 00 00 
    29e3:	0f 8e 11 04 00 00    	jle    2dfa <main+0x19ba>
    29e9:	48 8b bd e0 ea ff ff 	mov    -0x1520(%rbp),%rdi
    29f0:	4d 89 f0             	mov    %r14,%r8
    29f3:	b9 01 00 00 00       	mov    $0x1,%ecx
    29f8:	31 f6                	xor    %esi,%esi
    29fa:	c4 c1 7a 6f 40 04    	vmovdqu 0x4(%r8),%xmm0
    2a00:	41 8b 00             	mov    (%r8),%eax
    2a03:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2a07:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2a0c:	c5 f9 7f 07          	vmovdqa %xmm0,(%rdi)
    2a10:	c4 c1 7a 6f 40 14    	vmovdqu 0x14(%r8),%xmm0
    2a16:	f7 d0                	not    %eax
    2a18:	a9 00 00 80 7f       	test   $0x7f800000,%eax
    2a1d:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2a21:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2a25:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2a2a:	89 85 98 ea ff ff    	mov    %eax,-0x1568(%rbp)
    2a30:	0f 95 c2             	setne  %dl
    2a33:	c5 f9 7f 47 10       	vmovdqa %xmm0,0x10(%rdi)
    2a38:	c4 c1 7a 6f 40 24    	vmovdqu 0x24(%r8),%xmm0
    2a3e:	c5 79 d7 d3          	vpmovmskb %xmm3,%r10d
    2a42:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2a46:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2a4b:	c5 f9 7f 47 20       	vmovdqa %xmm0,0x20(%rdi)
    2a50:	c4 c1 7a 6f 40 34    	vmovdqu 0x34(%r8),%xmm0
    2a56:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2a5a:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2a5e:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2a63:	c5 f9 7f 47 30       	vmovdqa %xmm0,0x30(%rdi)
    2a68:	c4 c1 7a 6f 40 44    	vmovdqu 0x44(%r8),%xmm0
    2a6e:	89 85 50 ea ff ff    	mov    %eax,-0x15b0(%rbp)
    2a74:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2a78:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2a7c:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2a81:	c5 f9 7f 47 40       	vmovdqa %xmm0,0x40(%rdi)
    2a86:	c4 c1 7a 6f 40 54    	vmovdqu 0x54(%r8),%xmm0
    2a8c:	89 85 10 ea ff ff    	mov    %eax,-0x15f0(%rbp)
    2a92:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2a96:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2a9a:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2a9f:	c5 f9 7f 47 50       	vmovdqa %xmm0,0x50(%rdi)
    2aa4:	c4 c1 7a 6f 40 64    	vmovdqu 0x64(%r8),%xmm0
    2aaa:	89 85 08 ea ff ff    	mov    %eax,-0x15f8(%rbp)
    2ab0:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2ab4:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2ab8:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2abd:	c5 f9 7f 47 60       	vmovdqa %xmm0,0x60(%rdi)
    2ac2:	c4 c1 7a 6f 40 74    	vmovdqu 0x74(%r8),%xmm0
    2ac8:	89 85 00 ea ff ff    	mov    %eax,-0x1600(%rbp)
    2ace:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2ad2:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2ad6:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2adb:	c5 f9 7f 47 70       	vmovdqa %xmm0,0x70(%rdi)
    2ae0:	c4 c1 7a 6f 80 84 00 	vmovdqu 0x84(%r8),%xmm0
    2ae7:	00 00 
    2ae9:	89 85 f8 e9 ff ff    	mov    %eax,-0x1608(%rbp)
    2aef:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2af3:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2af7:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2afc:	c5 f9 7f 87 80 00 00 	vmovdqa %xmm0,0x80(%rdi)
    2b03:	00 
    2b04:	c4 c1 7a 6f 80 94 00 	vmovdqu 0x94(%r8),%xmm0
    2b0b:	00 00 
    2b0d:	89 85 f0 e9 ff ff    	mov    %eax,-0x1610(%rbp)
    2b13:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2b17:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2b1b:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2b20:	c5 f9 7f 87 90 00 00 	vmovdqa %xmm0,0x90(%rdi)
    2b27:	00 
    2b28:	c4 c1 7a 6f 80 a4 00 	vmovdqu 0xa4(%r8),%xmm0
    2b2f:	00 00 
    2b31:	89 85 ec e9 ff ff    	mov    %eax,-0x1614(%rbp)
    2b37:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2b3b:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2b3f:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2b44:	89 85 e8 e9 ff ff    	mov    %eax,-0x1618(%rbp)
    2b4a:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2b4e:	89 85 e4 e9 ff ff    	mov    %eax,-0x161c(%rbp)
    2b54:	c5 f9 7f 87 a0 00 00 	vmovdqa %xmm0,0xa0(%rdi)
    2b5b:	00 
    2b5c:	c4 c1 7a 6f 80 b4 00 	vmovdqu 0xb4(%r8),%xmm0
    2b63:	00 00 
    2b65:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2b69:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2b6e:	c5 f9 7f 87 b0 00 00 	vmovdqa %xmm0,0xb0(%rdi)
    2b75:	00 
    2b76:	c4 c1 7a 6f 80 c4 00 	vmovdqu 0xc4(%r8),%xmm0
    2b7d:	00 00 
    2b7f:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2b83:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2b87:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2b8c:	c5 f9 7f 87 c0 00 00 	vmovdqa %xmm0,0xc0(%rdi)
    2b93:	00 
    2b94:	c4 c1 7a 6f 80 d4 00 	vmovdqu 0xd4(%r8),%xmm0
    2b9b:	00 00 
    2b9d:	89 85 e0 e9 ff ff    	mov    %eax,-0x1620(%rbp)
    2ba3:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2ba7:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2bab:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2bb0:	c5 f9 7f 87 d0 00 00 	vmovdqa %xmm0,0xd0(%rdi)
    2bb7:	00 
    2bb8:	c4 c1 7a 6f 80 e4 00 	vmovdqu 0xe4(%r8),%xmm0
    2bbf:	00 00 
    2bc1:	89 85 dc e9 ff ff    	mov    %eax,-0x1624(%rbp)
    2bc7:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2bcb:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2bcf:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2bd4:	c5 f9 7f 87 e0 00 00 	vmovdqa %xmm0,0xe0(%rdi)
    2bdb:	00 
    2bdc:	c4 c1 7a 6f 80 f4 00 	vmovdqu 0xf4(%r8),%xmm0
    2be3:	00 00 
    2be5:	89 85 d8 e9 ff ff    	mov    %eax,-0x1628(%rbp)
    2beb:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2bef:	c5 f9 74 da          	vpcmpeqb %xmm2,%xmm0,%xmm3
    2bf3:	c4 e2 71 04 c0       	vpmaddubsw %xmm0,%xmm1,%xmm0
    2bf8:	89 85 d4 e9 ff ff    	mov    %eax,-0x162c(%rbp)
    2bfe:	c5 f9 7f 87 f0 00 00 	vmovdqa %xmm0,0xf0(%rdi)
    2c05:	00 
    2c06:	c5 f9 d7 c3          	vpmovmskb %xmm3,%eax
    2c0a:	89 85 d0 e9 ff ff    	mov    %eax,-0x1630(%rbp)
    2c10:	8b 85 98 ea ff ff    	mov    -0x1568(%rbp),%eax
    2c16:	44 09 d0             	or     %r10d,%eax
    2c19:	44 8b 95 50 ea ff ff 	mov    -0x15b0(%rbp),%r10d
    2c20:	44 09 d0             	or     %r10d,%eax
    2c23:	44 8b 95 10 ea ff ff 	mov    -0x15f0(%rbp),%r10d
    2c2a:	44 09 d0             	or     %r10d,%eax
    2c2d:	44 8b 95 08 ea ff ff 	mov    -0x15f8(%rbp),%r10d
    2c34:	44 09 d0             	or     %r10d,%eax
    2c37:	44 8b 95 00 ea ff ff 	mov    -0x1600(%rbp),%r10d
    2c3e:	44 09 d0             	or     %r10d,%eax
    2c41:	44 8b 95 f8 e9 ff ff 	mov    -0x1608(%rbp),%r10d
    2c48:	44 09 d0             	or     %r10d,%eax
    2c4b:	44 8b 95 f0 e9 ff ff 	mov    -0x1610(%rbp),%r10d
    2c52:	44 09 d0             	or     %r10d,%eax
    2c55:	44 8b 95 ec e9 ff ff 	mov    -0x1614(%rbp),%r10d
    2c5c:	44 09 d0             	or     %r10d,%eax
    2c5f:	44 8b 95 e8 e9 ff ff 	mov    -0x1618(%rbp),%r10d
    2c66:	44 09 d0             	or     %r10d,%eax
    2c69:	44 8b 95 e4 e9 ff ff 	mov    -0x161c(%rbp),%r10d
    2c70:	44 09 d0             	or     %r10d,%eax
    2c73:	44 8b 95 e0 e9 ff ff 	mov    -0x1620(%rbp),%r10d
    2c7a:	44 09 d0             	or     %r10d,%eax
    2c7d:	44 8b 95 dc e9 ff ff 	mov    -0x1624(%rbp),%r10d
    2c84:	44 09 d0             	or     %r10d,%eax
    2c87:	44 8b 95 d8 e9 ff ff 	mov    -0x1628(%rbp),%r10d
    2c8e:	44 09 d0             	or     %r10d,%eax
    2c91:	44 8b 95 d4 e9 ff ff 	mov    -0x162c(%rbp),%r10d
    2c98:	44 09 d0             	or     %r10d,%eax
    2c9b:	0b 85 d0 e9 ff ff    	or     -0x1630(%rbp),%eax
    2ca1:	0f 94 c0             	sete   %al
    2ca4:	83 c6 01             	add    $0x1,%esi
    2ca7:	49 81 c0 24 01 00 00 	add    $0x124,%r8
    2cae:	21 d0                	and    %edx,%eax
    2cb0:	48 81 c7 00 01 00 00 	add    $0x100,%rdi
    2cb7:	21 c1                	and    %eax,%ecx
    2cb9:	8b 85 a8 ea ff ff    	mov    -0x1558(%rbp),%eax
    2cbf:	39 c6                	cmp    %eax,%esi
    2cc1:	0f 8c 33 fd ff ff    	jl     29fa <main+0x15ba>
    2cc7:	84 c9                	test   %cl,%cl
    2cc9:	0f 85 57 01 00 00    	jne    2e26 <main+0x19e6>
    2ccf:	8b bd ac ea ff ff    	mov    -0x1554(%rbp),%edi
    2cd5:	4c 89 f2             	mov    %r14,%rdx
    2cd8:	4c 89 fe             	mov    %r15,%rsi
    2cdb:	c5 f8 77             	vzeroupper
    2cde:	e8 4d 02 00 00       	call   2f30 <float iq_compact_dot<false>(int, iq_compact_block const*, block_q8_K const*)>
    2ce3:	8b bd ac ea ff ff    	mov    -0x1554(%rbp),%edi
    2ce9:	4c 89 f2             	mov    %r14,%rdx
    2cec:	4c 89 e6             	mov    %r12,%rsi
    2cef:	c4 c1 79 7e c2       	vmovd  %xmm0,%r10d
    2cf4:	e8 c7 0a 00 00       	call   37c0 <offset4_compat_dot(int, offset4_iq3s_block const*, block_q8_K const*)>
    2cf9:	c5 f9 7e c0          	vmovd  %xmm0,%eax
    2cfd:	44 39 d0             	cmp    %r10d,%eax
    2d00:	0f 85 04 e7 ff ff    	jne    140a <main.cold+0x2fa>
    2d06:	c5 fe 6f 85 00 eb ff 	vmovdqu -0x1500(%rbp),%ymm0
    2d0d:	ff 
    2d0e:	8b 85 20 ec ff ff    	mov    -0x13e0(%rbp),%eax
    2d14:	c4 c1 7e 7f 85 dc fe 	vmovdqu %ymm0,-0x124(%r13)
    2d1b:	ff ff 
    2d1d:	c5 fe 6f 85 20 eb ff 	vmovdqu -0x14e0(%rbp),%ymm0
    2d24:	ff 
    2d25:	c5 f9 6f 15 63 36 00 	vmovdqa 0x3663(%rip),%xmm2        # 6390 <iq3s_grid+0x890>
    2d2c:	00 
    2d2d:	c4 c1 7e 7f 85 fc fe 	vmovdqu %ymm0,-0x104(%r13)
    2d34:	ff ff 
    2d36:	c5 fe 6f 85 40 eb ff 	vmovdqu -0x14c0(%rbp),%ymm0
    2d3d:	ff 
    2d3e:	41 89 45 fc          	mov    %eax,-0x4(%r13)
    2d42:	c5 f9 6f 0d d6 35 00 	vmovdqa 0x35d6(%rip),%xmm1        # 6320 <iq3s_grid+0x820>
    2d49:	00 
    2d4a:	49 8d 41 01          	lea    0x1(%r9),%rax
    2d4e:	c4 c1 7e 7f 85 1c ff 	vmovdqu %ymm0,-0xe4(%r13)
    2d55:	ff ff 
    2d57:	c5 fe 6f 85 60 eb ff 	vmovdqu -0x14a0(%rbp),%ymm0
    2d5e:	ff 
    2d5f:	c4 c1 7e 7f 85 3c ff 	vmovdqu %ymm0,-0xc4(%r13)
    2d66:	ff ff 
    2d68:	c5 fe 6f 85 80 eb ff 	vmovdqu -0x1480(%rbp),%ymm0
    2d6f:	ff 
    2d70:	c4 c1 7e 7f 85 5c ff 	vmovdqu %ymm0,-0xa4(%r13)
    2d77:	ff ff 
    2d79:	c5 fe 6f 85 a0 eb ff 	vmovdqu -0x1460(%rbp),%ymm0
    2d80:	ff 
    2d81:	c4 c1 7e 7f 85 7c ff 	vmovdqu %ymm0,-0x84(%r13)
    2d88:	ff ff 
    2d8a:	c5 fe 6f 85 c0 eb ff 	vmovdqu -0x1440(%rbp),%ymm0
    2d91:	ff 
    2d92:	c4 c1 7e 7f 45 9c    	vmovdqu %ymm0,-0x64(%r13)
    2d98:	c5 fe 6f 85 e0 eb ff 	vmovdqu -0x1420(%rbp),%ymm0
    2d9f:	ff 
    2da0:	c4 c1 7e 7f 45 bc    	vmovdqu %ymm0,-0x44(%r13)
    2da6:	c5 fe 6f 85 00 ec ff 	vmovdqu -0x1400(%rbp),%ymm0
    2dad:	ff 
    2dae:	c4 c1 7e 7f 45 dc    	vmovdqu %ymm0,-0x24(%r13)
    2db4:	49 83 f9 04          	cmp    $0x4,%r9
    2db8:	74 12                	je     2dcc <main+0x198c>
    2dba:	49 89 c1             	mov    %rax,%r9
    2dbd:	e9 69 fb ff ff       	jmp    292b <main+0x14eb>
    2dc2:	41 c6 45 df 80       	movb   $0x80,-0x21(%r13)
    2dc7:	e9 0d fc ff ff       	jmp    29d9 <main+0x1599>
    2dcc:	48 8b 85 58 ea ff ff 	mov    -0x15a8(%rbp),%rax
    2dd3:	4a 8d 44 08 01       	lea    0x1(%rax,%r9,1),%rax
    2dd8:	48 89 85 58 ea ff ff 	mov    %rax,-0x15a8(%rbp)
    2ddf:	48 8b 85 30 ea ff ff 	mov    -0x15d0(%rbp),%rax
    2de6:	4a 8d 44 08 02       	lea    0x2(%rax,%r9,1),%rax
    2deb:	48 89 85 98 ea ff ff 	mov    %rax,-0x1568(%rbp)
    2df2:	c5 f8 77             	vzeroupper
    2df5:	e9 29 f7 ff ff       	jmp    2523 <main+0x10e3>
    2dfa:	e9 11 e3 ff ff       	jmp    1110 <main.cold>
    2dff:	48 89 c3             	mov    %rax,%rbx
    2e02:	49 89 d5             	mov    %rdx,%r13
    2e05:	c5 f8 77             	vzeroupper
    2e08:	e9 8b e4 ff ff       	jmp    1298 <main.cold+0x188>
    2e0d:	48 89 c3             	mov    %rax,%rbx
    2e10:	49 89 d5             	mov    %rdx,%r13
    2e13:	e9 42 e4 ff ff       	jmp    125a <main.cold+0x14a>
    2e18:	48 89 c3             	mov    %rax,%rbx
    2e1b:	49 89 d5             	mov    %rdx,%r13
    2e1e:	c5 f8 77             	vzeroupper
    2e21:	e9 42 e4 ff ff       	jmp    1268 <main.cold+0x158>
    2e26:	e9 e5 e2 ff ff       	jmp    1110 <main.cold>
    2e2b:	48 89 c3             	mov    %rax,%rbx
    2e2e:	49 89 d5             	mov    %rdx,%r13
    2e31:	c5 f8 77             	vzeroupper
    2e34:	e9 8b e4 ff ff       	jmp    12c4 <main.cold+0x1b4>
    2e39:	e9 bb e3 ff ff       	jmp    11f9 <main.cold+0xe9>
    2e3e:	66 90                	xchg   %ax,%ax

0000000000002e40 <_start>:
    2e40:	31 ed                	xor    %ebp,%ebp
    2e42:	49 89 d1             	mov    %rdx,%r9
    2e45:	5e                   	pop    %rsi
    2e46:	48 89 e2             	mov    %rsp,%rdx
    2e49:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    2e4d:	50                   	push   %rax
    2e4e:	54                   	push   %rsp
    2e4f:	45 31 c0             	xor    %r8d,%r8d
    2e52:	31 c9                	xor    %ecx,%ecx
    2e54:	48 8d 3d e5 e5 ff ff 	lea    -0x1a1b(%rip),%rdi        # 1440 <main>
    2e5b:	ff 15 5f 51 00 00    	call   *0x515f(%rip)        # 7fc0 <__libc_start_main@GLIBC_2.34>
    2e61:	f4                   	hlt
    2e62:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2e69:	00 00 00 
    2e6c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002e70 <deregister_tm_clones>:
    2e70:	48 8d 3d 09 52 00 00 	lea    0x5209(%rip),%rdi        # 8080 <stderr@GLIBC_2.2.5>
    2e77:	48 8d 05 02 52 00 00 	lea    0x5202(%rip),%rax        # 8080 <stderr@GLIBC_2.2.5>
    2e7e:	48 39 f8             	cmp    %rdi,%rax
    2e81:	74 15                	je     2e98 <deregister_tm_clones+0x28>
    2e83:	48 8b 05 46 51 00 00 	mov    0x5146(%rip),%rax        # 7fd0 <_ITM_deregisterTMCloneTable@Base>
    2e8a:	48 85 c0             	test   %rax,%rax
    2e8d:	74 09                	je     2e98 <deregister_tm_clones+0x28>
    2e8f:	ff e0                	jmp    *%rax
    2e91:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2e98:	c3                   	ret
    2e99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002ea0 <register_tm_clones>:
    2ea0:	48 8d 3d d9 51 00 00 	lea    0x51d9(%rip),%rdi        # 8080 <stderr@GLIBC_2.2.5>
    2ea7:	48 8d 35 d2 51 00 00 	lea    0x51d2(%rip),%rsi        # 8080 <stderr@GLIBC_2.2.5>
    2eae:	48 29 fe             	sub    %rdi,%rsi
    2eb1:	48 89 f0             	mov    %rsi,%rax
    2eb4:	48 c1 ee 3f          	shr    $0x3f,%rsi
    2eb8:	48 c1 f8 03          	sar    $0x3,%rax
    2ebc:	48 01 c6             	add    %rax,%rsi
    2ebf:	48 d1 fe             	sar    $1,%rsi
    2ec2:	74 14                	je     2ed8 <register_tm_clones+0x38>
    2ec4:	48 8b 05 15 51 00 00 	mov    0x5115(%rip),%rax        # 7fe0 <_ITM_registerTMCloneTable@Base>
    2ecb:	48 85 c0             	test   %rax,%rax
    2ece:	74 08                	je     2ed8 <register_tm_clones+0x38>
    2ed0:	ff e0                	jmp    *%rax
    2ed2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2ed8:	c3                   	ret
    2ed9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002ee0 <__do_global_dtors_aux>:
    2ee0:	f3 0f 1e fa          	endbr64
    2ee4:	80 3d 9d 51 00 00 00 	cmpb   $0x0,0x519d(%rip)        # 8088 <completed.0>
    2eeb:	75 2b                	jne    2f18 <__do_global_dtors_aux+0x38>
    2eed:	55                   	push   %rbp
    2eee:	48 83 3d c2 50 00 00 	cmpq   $0x0,0x50c2(%rip)        # 7fb8 <__cxa_finalize@GLIBC_2.2.5>
    2ef5:	00 
    2ef6:	48 89 e5             	mov    %rsp,%rbp
    2ef9:	74 0c                	je     2f07 <__do_global_dtors_aux+0x27>
    2efb:	48 8b 3d 66 51 00 00 	mov    0x5166(%rip),%rdi        # 8068 <__dso_handle>
    2f02:	e8 f9 e1 ff ff       	call   1100 <__cxa_finalize@plt>
    2f07:	e8 64 ff ff ff       	call   2e70 <deregister_tm_clones>
    2f0c:	c6 05 75 51 00 00 01 	movb   $0x1,0x5175(%rip)        # 8088 <completed.0>
    2f13:	5d                   	pop    %rbp
    2f14:	c3                   	ret
    2f15:	0f 1f 00             	nopl   (%rax)
    2f18:	c3                   	ret
    2f19:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002f20 <frame_dummy>:
    2f20:	f3 0f 1e fa          	endbr64
    2f24:	e9 77 ff ff ff       	jmp    2ea0 <register_tm_clones>
    2f29:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

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

0000000000003c50 <void iq_pack<false>(std::conditional<false, block_iq2_s, block_iq3_s>::type const&, iq_compact_block&)>:
    3c50:	0f b7 07             	movzwl (%rdi),%eax
    3c53:	41 54                	push   %r12
    3c55:	49 89 f1             	mov    %rsi,%r9
    3c58:	55                   	push   %rbp
    3c59:	53                   	push   %rbx
    3c5a:	89 c2                	mov    %eax,%edx
    3c5c:	c1 e0 11             	shl    $0x11,%eax
    3c5f:	c1 e2 10             	shl    $0x10,%edx
    3c62:	81 e2 00 00 00 80    	and    $0x80000000,%edx
    3c68:	3d ff ff ff 07       	cmp    $0x7ffffff,%eax
    3c6d:	0f 87 fd 01 00 00    	ja     3e70 <void iq_pack<false>(std::conditional<false, block_iq2_s, block_iq3_s>::type const&, iq_compact_block&)+0x220>
    3c73:	c1 e8 11             	shr    $0x11,%eax
    3c76:	0d 00 00 00 3f       	or     $0x3f000000,%eax
    3c7b:	c5 f9 6e f8          	vmovd  %eax,%xmm7
    3c7f:	c5 c2 5c 3d 75 27 00 	vsubss 0x2775(%rip),%xmm7,%xmm7        # 63fc <iq3s_grid+0x8fc>
    3c86:	00 
    3c87:	c5 f9 7e f8          	vmovd  %xmm7,%eax
    3c8b:	09 d0                	or     %edx,%eax
    3c8d:	31 c9                	xor    %ecx,%ecx
    3c8f:	c5 c9 76 f6          	vpcmpeqd %xmm6,%xmm6,%xmm6
    3c93:	49 c7 81 84 00 00 00 	movq   $0x0,0x84(%r9)
    3c9a:	00 00 00 00 
    3c9e:	41 89 81 80 00 00 00 	mov    %eax,0x80(%r9)
    3ca5:	8b 47 6a             	mov    0x6a(%rdi),%eax
    3ca8:	48 8d 57 02          	lea    0x2(%rdi),%rdx
    3cac:	c5 d1 ef ed          	vpxor  %xmm5,%xmm5,%xmm5
    3cb0:	48 8d 35 49 1e 00 00 	lea    0x1e49(%rip),%rsi        # 5b00 <iq3s_grid>
    3cb7:	4c 8d 05 42 16 00 00 	lea    0x1642(%rip),%r8        # 5300 <iq_signs>
    3cbe:	41 89 81 84 00 00 00 	mov    %eax,0x84(%r9)
    3cc5:	b8 08 08 08 08       	mov    $0x8080808,%eax
    3cca:	c5 f9 6e e0          	vmovd  %eax,%xmm4
    3cce:	b8 07 07 07 07       	mov    $0x7070707,%eax
    3cd3:	c5 f9 6e d8          	vmovd  %eax,%xmm3
    3cd7:	c5 f9 70 e4 00       	vpshufd $0x0,%xmm4,%xmm4
    3cdc:	c5 f9 70 db 00       	vpshufd $0x0,%xmm3,%xmm3
    3ce1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3ce8:	0f b6 44 0f 42       	movzbl 0x42(%rdi,%rcx,1),%eax
    3ced:	48 83 c2 08          	add    $0x8,%rdx
    3cf1:	44 0f b6 52 f8       	movzbl -0x8(%rdx),%r10d
    3cf6:	0f b6 5a f9          	movzbl -0x7(%rdx),%ebx
    3cfa:	44 0f b6 62 fb       	movzbl -0x5(%rdx),%r12d
    3cff:	41 89 c3             	mov    %eax,%r11d
    3d02:	89 c5                	mov    %eax,%ebp
    3d04:	41 c1 e3 08          	shl    $0x8,%r11d
    3d08:	c1 e5 06             	shl    $0x6,%ebp
    3d0b:	41 81 e3 00 01 00 00 	and    $0x100,%r11d
    3d12:	81 e5 00 01 00 00    	and    $0x100,%ebp
    3d18:	45 09 d3             	or     %r10d,%r11d
    3d1b:	41 89 c2             	mov    %eax,%r10d
    3d1e:	41 c1 e2 07          	shl    $0x7,%r10d
    3d22:	4d 63 db             	movslq %r11d,%r11
    3d25:	41 81 e2 00 01 00 00 	and    $0x100,%r10d
    3d2c:	c4 a1 79 6e 04 9e    	vmovd  (%rsi,%r11,4),%xmm0
    3d32:	41 89 c3             	mov    %eax,%r11d
    3d35:	41 09 da             	or     %ebx,%r10d
    3d38:	0f b6 5a fa          	movzbl -0x6(%rdx),%ebx
    3d3c:	41 c1 e3 04          	shl    $0x4,%r11d
    3d40:	4d 63 d2             	movslq %r10d,%r10
    3d43:	41 81 e3 00 01 00 00 	and    $0x100,%r11d
    3d4a:	c4 a3 79 22 04 96 01 	vpinsrd $0x1,(%rsi,%r10,4),%xmm0,%xmm0
    3d51:	44 0f b6 52 fc       	movzbl -0x4(%rdx),%r10d
    3d56:	09 dd                	or     %ebx,%ebp
    3d58:	89 c3                	mov    %eax,%ebx
    3d5a:	c1 e3 05             	shl    $0x5,%ebx
    3d5d:	48 63 ed             	movslq %ebp,%rbp
    3d60:	81 e3 00 01 00 00    	and    $0x100,%ebx
    3d66:	c5 f9 6e 0c ae       	vmovd  (%rsi,%rbp,4),%xmm1
    3d6b:	45 09 d3             	or     %r10d,%r11d
    3d6e:	44 09 e3             	or     %r12d,%ebx
    3d71:	0f b6 6a fe          	movzbl -0x2(%rdx),%ebp
    3d75:	4d 63 db             	movslq %r11d,%r11
    3d78:	48 63 db             	movslq %ebx,%rbx
    3d7b:	c4 a1 79 6e 14 9e    	vmovd  (%rsi,%r11,4),%xmm2
    3d81:	c4 e3 71 22 0c 9e 01 	vpinsrd $0x1,(%rsi,%rbx,4),%xmm1,%xmm1
    3d88:	0f b6 5a fd          	movzbl -0x3(%rdx),%ebx
    3d8c:	c5 f9 6c c1          	vpunpcklqdq %xmm1,%xmm0,%xmm0
    3d90:	44 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%r10d
    3d97:	00 
    3d98:	41 81 e2 00 01 00 00 	and    $0x100,%r10d
    3d9f:	c5 f9 fc c6          	vpaddb %xmm6,%xmm0,%xmm0
    3da3:	c5 f9 71 d0 01       	vpsrlw $0x1,%xmm0,%xmm0
    3da8:	c5 f9 db c3          	vpand  %xmm3,%xmm0,%xmm0
    3dac:	41 09 da             	or     %ebx,%r10d
    3daf:	8d 1c 85 00 00 00 00 	lea    0x0(,%rax,4),%ebx
    3db6:	c1 f8 07             	sar    $0x7,%eax
    3db9:	4d 63 d2             	movslq %r10d,%r10
    3dbc:	81 e3 00 01 00 00    	and    $0x100,%ebx
    3dc2:	c1 e0 08             	shl    $0x8,%eax
    3dc5:	c4 a3 69 22 14 96 01 	vpinsrd $0x1,(%rsi,%r10,4),%xmm2,%xmm2
    3dcc:	44 0f b6 54 8f 4c    	movzbl 0x4c(%rdi,%rcx,4),%r10d
    3dd2:	09 eb                	or     %ebp,%ebx
    3dd4:	0f b6 6a ff          	movzbl -0x1(%rdx),%ebp
    3dd8:	48 63 db             	movslq %ebx,%rbx
    3ddb:	c5 f9 6e 0c 9e       	vmovd  (%rsi,%rbx,4),%xmm1
    3de0:	09 e8                	or     %ebp,%eax
    3de2:	48 98                	cltq
    3de4:	c4 e3 71 22 0c 86 01 	vpinsrd $0x1,(%rsi,%rax,4),%xmm1,%xmm1
    3deb:	0f b6 44 8f 4d       	movzbl 0x4d(%rdi,%rcx,4),%eax
    3df0:	c5 e9 6c d1          	vpunpcklqdq %xmm1,%xmm2,%xmm2
    3df4:	c5 e9 fc d6          	vpaddb %xmm6,%xmm2,%xmm2
    3df8:	c4 81 7a 7e 0c d0    	vmovq  (%r8,%r10,8),%xmm1
    3dfe:	c5 e9 71 d2 01       	vpsrlw $0x1,%xmm2,%xmm2
    3e03:	44 0f b6 54 8f 4a    	movzbl 0x4a(%rdi,%rcx,4),%r10d
    3e09:	c5 e9 db d3          	vpand  %xmm3,%xmm2,%xmm2
    3e0d:	c4 c3 f1 22 0c c0 01 	vpinsrq $0x1,(%r8,%rax,8),%xmm1,%xmm1
    3e14:	0f b6 44 8f 4b       	movzbl 0x4b(%rdi,%rcx,4),%eax
    3e19:	c5 d1 64 c9          	vpcmpgtb %xmm1,%xmm5,%xmm1
    3e1d:	c5 f1 db cc          	vpand  %xmm4,%xmm1,%xmm1
    3e21:	c5 f1 eb ca          	vpor   %xmm2,%xmm1,%xmm1
    3e25:	c4 81 7a 7e 14 d0    	vmovq  (%r8,%r10,8),%xmm2
    3e2b:	c5 f1 71 f1 04       	vpsllw $0x4,%xmm1,%xmm1
    3e30:	c4 c3 e9 22 14 c0 01 	vpinsrq $0x1,(%r8,%rax,8),%xmm2,%xmm2
    3e37:	c5 d1 64 d2          	vpcmpgtb %xmm2,%xmm5,%xmm2
    3e3b:	48 89 c8             	mov    %rcx,%rax
    3e3e:	48 83 c1 01          	add    $0x1,%rcx
    3e42:	48 c1 e0 04          	shl    $0x4,%rax
    3e46:	c5 f9 eb c1          	vpor   %xmm1,%xmm0,%xmm0
    3e4a:	c5 e9 db d4          	vpand  %xmm4,%xmm2,%xmm2
    3e4e:	c5 e9 eb c0          	vpor   %xmm0,%xmm2,%xmm0
    3e52:	c4 c1 7a 7f 04 01    	vmovdqu %xmm0,(%r9,%rax,1)
    3e58:	48 83 f9 08          	cmp    $0x8,%rcx
    3e5c:	0f 85 86 fe ff ff    	jne    3ce8 <void iq_pack<false>(std::conditional<false, block_iq2_s, block_iq3_s>::type const&, iq_compact_block&)+0x98>
    3e62:	5b                   	pop    %rbx
    3e63:	5d                   	pop    %rbp
    3e64:	41 5c                	pop    %r12
    3e66:	c3                   	ret
    3e67:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    3e6e:	00 00 
    3e70:	c1 e8 04             	shr    $0x4,%eax
    3e73:	05 00 00 00 70       	add    $0x70000000,%eax
    3e78:	c5 f9 6e f8          	vmovd  %eax,%xmm7
    3e7c:	c5 c2 59 3d 54 25 00 	vmulss 0x2554(%rip),%xmm7,%xmm7        # 63d8 <iq3s_grid+0x8d8>
    3e83:	00 
    3e84:	c5 f9 7e f8          	vmovd  %xmm7,%eax
    3e88:	e9 fe fd ff ff       	jmp    3c8b <void iq_pack<false>(std::conditional<false, block_iq2_s, block_iq3_s>::type const&, iq_compact_block&)+0x3b>
    3e8d:	0f 1f 00             	nopl   (%rax)

0000000000003e90 <std::mersenne_twister_engine<unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>::_M_gen_rand()>:
    3e90:	48 c7 c6 00 00 00 80 	mov    $0xffffffff80000000,%rsi
    3e97:	48 89 f9             	mov    %rdi,%rcx
    3e9a:	48 89 f8             	mov    %rdi,%rax
    3e9d:	48 8d 97 10 07 00 00 	lea    0x710(%rdi),%rdx
    3ea4:	c4 e1 f9 6e e6       	vmovq  %rsi,%xmm4
    3ea9:	bf 01 00 00 00       	mov    $0x1,%edi
    3eae:	be ff ff ff 7f       	mov    $0x7fffffff,%esi
    3eb3:	c4 e1 f9 6e d7       	vmovq  %rdi,%xmm2
    3eb8:	c5 d9 6c e4          	vpunpcklqdq %xmm4,%xmm4,%xmm4
    3ebc:	c4 e1 f9 6e de       	vmovq  %rsi,%xmm3
    3ec1:	c5 e9 6c d2          	vpunpcklqdq %xmm2,%xmm2,%xmm2
    3ec5:	c5 e1 6c db          	vpunpcklqdq %xmm3,%xmm3,%xmm3
    3ec9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3ed0:	c5 e1 db 40 08       	vpand  0x8(%rax),%xmm3,%xmm0
    3ed5:	c5 d9 db 28          	vpand  (%rax),%xmm4,%xmm5
    3ed9:	c5 d1 eb e8          	vpor   %xmm0,%xmm5,%xmm5
    3edd:	c5 f1 73 d5 01       	vpsrlq $0x1,%xmm5,%xmm1
    3ee2:	c5 d1 db ea          	vpand  %xmm2,%xmm5,%xmm5
    3ee6:	c5 f9 73 f5 03       	vpsllq $0x3,%xmm5,%xmm0
    3eeb:	c5 f1 ef 88 68 0c 00 	vpxor  0xc68(%rax),%xmm1,%xmm1
    3ef2:	00 
    3ef3:	48 83 c0 10          	add    $0x10,%rax
    3ef7:	c5 f9 d4 c5          	vpaddq %xmm5,%xmm0,%xmm0
    3efb:	c5 f9 73 f0 09       	vpsllq $0x9,%xmm0,%xmm0
    3f00:	c5 f9 d4 c5          	vpaddq %xmm5,%xmm0,%xmm0
    3f04:	c5 f9 73 f0 05       	vpsllq $0x5,%xmm0,%xmm0
    3f09:	c5 f9 d4 c5          	vpaddq %xmm5,%xmm0,%xmm0
    3f0d:	c5 f9 73 f0 02       	vpsllq $0x2,%xmm0,%xmm0
    3f12:	c5 f9 fb c5          	vpsubq %xmm5,%xmm0,%xmm0
    3f16:	c5 f9 73 f0 03       	vpsllq $0x3,%xmm0,%xmm0
    3f1b:	c5 f9 fb c5          	vpsubq %xmm5,%xmm0,%xmm0
    3f1f:	c5 c9 73 f0 04       	vpsllq $0x4,%xmm0,%xmm6
    3f24:	c5 f9 d4 c6          	vpaddq %xmm6,%xmm0,%xmm0
    3f28:	c5 f9 73 f0 05       	vpsllq $0x5,%xmm0,%xmm0
    3f2d:	c5 f9 fb c5          	vpsubq %xmm5,%xmm0,%xmm0
    3f31:	c5 f1 ef c8          	vpxor  %xmm0,%xmm1,%xmm1
    3f35:	c5 fa 7f 48 f0       	vmovdqu %xmm1,-0x10(%rax)
    3f3a:	48 39 d0             	cmp    %rdx,%rax
    3f3d:	75 91                	jne    3ed0 <std::mersenne_twister_engine<unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>::_M_gen_rand()+0x40>
    3f3f:	48 8b 81 10 07 00 00 	mov    0x710(%rcx),%rax
    3f46:	48 c7 c6 00 00 00 80 	mov    $0xffffffff80000000,%rsi
    3f4d:	bf ff ff ff 7f       	mov    $0x7fffffff,%edi
    3f52:	48 8b 91 18 07 00 00 	mov    0x718(%rcx),%rdx
    3f59:	c4 e1 f9 6e e6       	vmovq  %rsi,%xmm4
    3f5e:	be 01 00 00 00       	mov    $0x1,%esi
    3f63:	c4 e1 f9 6e df       	vmovq  %rdi,%xmm3
    3f68:	c4 e1 f9 6e d6       	vmovq  %rsi,%xmm2
    3f6d:	c5 d9 6c e4          	vpunpcklqdq %xmm4,%xmm4,%xmm4
    3f71:	c5 e1 6c db          	vpunpcklqdq %xmm3,%xmm3,%xmm3
    3f75:	c5 e9 6c d2          	vpunpcklqdq %xmm2,%xmm2,%xmm2
    3f79:	48 25 00 00 00 80    	and    $0xffffffff80000000,%rax
    3f7f:	81 e2 ff ff ff 7f    	and    $0x7fffffff,%edx
    3f85:	48 09 d0             	or     %rdx,%rax
    3f88:	48 89 c2             	mov    %rax,%rdx
    3f8b:	83 e0 01             	and    $0x1,%eax
    3f8e:	48 d1 ea             	shr    $1,%rdx
    3f91:	48 33 91 78 13 00 00 	xor    0x1378(%rcx),%rdx
    3f98:	48 f7 d8             	neg    %rax
    3f9b:	25 df b0 08 99       	and    $0x9908b0df,%eax
    3fa0:	48 31 d0             	xor    %rdx,%rax
    3fa3:	48 89 81 10 07 00 00 	mov    %rax,0x710(%rcx)
    3faa:	48 8d 91 78 13 00 00 	lea    0x1378(%rcx),%rdx
    3fb1:	48 8d 81 18 07 00 00 	lea    0x718(%rcx),%rax
    3fb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    3fbf:	00 
    3fc0:	c5 e1 db 40 08       	vpand  0x8(%rax),%xmm3,%xmm0
    3fc5:	c5 d9 db 28          	vpand  (%rax),%xmm4,%xmm5
    3fc9:	c5 d1 eb e8          	vpor   %xmm0,%xmm5,%xmm5
    3fcd:	c5 f1 73 d5 01       	vpsrlq $0x1,%xmm5,%xmm1
    3fd2:	c5 d1 db ea          	vpand  %xmm2,%xmm5,%xmm5
    3fd6:	c5 f9 73 f5 03       	vpsllq $0x3,%xmm5,%xmm0
    3fdb:	c5 f1 ef 88 e8 f8 ff 	vpxor  -0x718(%rax),%xmm1,%xmm1
    3fe2:	ff 
    3fe3:	48 83 c0 10          	add    $0x10,%rax
    3fe7:	c5 f9 d4 c5          	vpaddq %xmm5,%xmm0,%xmm0
    3feb:	c5 f9 73 f0 09       	vpsllq $0x9,%xmm0,%xmm0
    3ff0:	c5 f9 d4 c5          	vpaddq %xmm5,%xmm0,%xmm0
    3ff4:	c5 f9 73 f0 05       	vpsllq $0x5,%xmm0,%xmm0
    3ff9:	c5 f9 d4 c5          	vpaddq %xmm5,%xmm0,%xmm0
    3ffd:	c5 f9 73 f0 02       	vpsllq $0x2,%xmm0,%xmm0
    4002:	c5 f9 fb c5          	vpsubq %xmm5,%xmm0,%xmm0
    4006:	c5 f9 73 f0 03       	vpsllq $0x3,%xmm0,%xmm0
    400b:	c5 f9 fb c5          	vpsubq %xmm5,%xmm0,%xmm0
    400f:	c5 c9 73 f0 04       	vpsllq $0x4,%xmm0,%xmm6
    4014:	c5 f9 d4 c6          	vpaddq %xmm6,%xmm0,%xmm0
    4018:	c5 f9 73 f0 05       	vpsllq $0x5,%xmm0,%xmm0
    401d:	c5 f9 fb c5          	vpsubq %xmm5,%xmm0,%xmm0
    4021:	c5 f1 ef c8          	vpxor  %xmm0,%xmm1,%xmm1
    4025:	c5 fa 7f 48 f0       	vmovdqu %xmm1,-0x10(%rax)
    402a:	48 39 c2             	cmp    %rax,%rdx
    402d:	75 91                	jne    3fc0 <std::mersenne_twister_engine<unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>::_M_gen_rand()+0x130>
    402f:	48 c7 81 80 13 00 00 	movq   $0x0,0x1380(%rcx)
    4036:	00 00 00 00 
    403a:	48 8b 11             	mov    (%rcx),%rdx
    403d:	48 8b 81 78 13 00 00 	mov    0x1378(%rcx),%rax
    4044:	81 e2 ff ff ff 7f    	and    $0x7fffffff,%edx
    404a:	48 25 00 00 00 80    	and    $0xffffffff80000000,%rax
    4050:	48 09 d0             	or     %rdx,%rax
    4053:	48 89 c2             	mov    %rax,%rdx
    4056:	83 e0 01             	and    $0x1,%eax
    4059:	48 d1 ea             	shr    $1,%rdx
    405c:	48 33 91 60 0c 00 00 	xor    0xc60(%rcx),%rdx
    4063:	48 f7 d8             	neg    %rax
    4066:	25 df b0 08 99       	and    $0x9908b0df,%eax
    406b:	48 31 d0             	xor    %rdx,%rax
    406e:	48 89 81 78 13 00 00 	mov    %rax,0x1378(%rcx)
    4075:	c3                   	ret

Disassembly of section .fini:

0000000000004078 <_fini>:
    4078:	48 83 ec 08          	sub    $0x8,%rsp
    407c:	48 83 c4 08          	add    $0x8,%rsp
    4080:	c3                   	ret
