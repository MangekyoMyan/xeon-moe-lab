#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
int main(int argc,char**argv) {
    if(argc!=3)return 2;
    std::ifstream a(argv[1],std::ios::binary),b(argv[2],std::ios::binary);
    if(!a||!b)return 2;
    std::vector<float>x(16384),y(16384);
    double se=0,power=0,ma=0;uint64_t total=0,diff=0;
    for(;;){
        a.read((char*)x.data(),x.size()*4);b.read((char*)y.data(),y.size()*4);
        auto n=a.gcount();if(n!=b.gcount()||n%4)return 3;if(!n)break;
        for(int i=0;i<n/4;i++){if(!std::isfinite(x[i])||!std::isfinite(y[i]))return 4;
            ++total;diff+=std::memcmp(&x[i],&y[i],4)!=0;double e=double(x[i])-y[i];se+=e*e;power+=double(x[i])*x[i];ma=std::max(ma,std::abs(e));}
    }
    if(!total||!power)return 5;
    std::cout<<std::setprecision(13)<<"{\"elements\":"<<total<<",\"different_bits\":"<<diff<<",\"RMSE\":"<<std::sqrt(se/total)
             <<",\"relative_L2\":"<<std::sqrt(se/power)<<",\"max_abs\":"<<ma<<"}\n";
}
