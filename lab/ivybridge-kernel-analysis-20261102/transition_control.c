#include <stdio.h>
#include <string.h>

int main(int argc,char **argv) {
    if(argc!=2 || (strcmp(argv[1],"legacy") && strcmp(argv[1],"vex")))return 2;
    const float values[8]={1,1,1,1,1,1,1,1};
    unsigned long n=200000;
    if(!strcmp(argv[1],"legacy")) {
        __asm__ volatile(
            "vmovups (%1), %%ymm0\n\t"
            "1: vaddps %%ymm0, %%ymm0, %%ymm0\n\t"
            "addps %%xmm0, %%xmm0\n\t"
            "sub $1, %0\n\tjnz 1b\n\tvzeroupper\n\t"
            : "+r"(n) : "r"(values) : "ymm0","cc","memory");
    } else {
        __asm__ volatile(
            "vmovups (%1), %%ymm0\n\t"
            "1: vaddps %%ymm0, %%ymm0, %%ymm0\n\t"
            "vaddps %%xmm0, %%xmm0, %%xmm0\n\t"
            "sub $1, %0\n\tjnz 1b\n\tvzeroupper\n\t"
            : "+r"(n) : "r"(values) : "ymm0","cc","memory");
    }
    printf("{\"mode\":\"%s\",\"iterations\":200000}\n",argv[1]);
    return 0;
}
