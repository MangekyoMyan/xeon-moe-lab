// Compare every vocabulary score in matched histories; probabilities use untruncated temperature-one softmax.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool v){if(!v)throw std::runtime_error("invalid matched full-vocabulary data");}
static constexpr int V=129280;

struct Distribution {
    std::vector<float> x=std::vector<float>(V);
    std::vector<double> p=std::vector<double>(V);
    std::array<int,5> top{};
    double log_normalizer=0;
    void calculate() {
        top.fill(-1);
        for(int i=0;i<V;i++){
            check(std::isfinite(x[i]));
            if(top.back()>=0 && x[i]<=x[top.back()])continue;
            for(int j=0;j<5;j++)if(top[j]<0 || x[i]>x[top[j]]){
                std::move_backward(top.begin()+j,top.end()-1,top.end());
                top[j]=i;break;
            }
        }
        const double m=x[top[0]];
        double z=0;
        for(int i=0;i<V;i++){p[i]=std::exp(double(x[i])-m);z+=p[i];}
        for(double &v:p)v/=z;
        log_normalizer=m+std::log(z);
    }
    int rank(int k)const {
        int count=1;
        for(int i=0;i<V;i++)if(x[i]>x[k] || (x[i]==x[k] && i<k))++count;
        return count;
    }
};
int main(int argc,char **argv){
    try{
        check(argc==5);
        std::ifstream a(argv[1],std::ios::binary),b(argv[2],std::ios::binary),targets(argv[3]);
        std::ofstream o(argv[4]);
        check(bool(a)&&bool(b)&&bool(targets)&&bool(o));
        o<<"row\ttarget\texact\tA_top\tB_top\tA_second\tB_second\tA_margin\tB_margin\tA_gap_to_Btop\tB_gap_to_Atop\tBtop_rank_in_A\tAtop_rank_in_B\tA_p_Atop\tB_p_Atop\tA_p_Btop\tB_p_Btop\tA_target_nll\tB_target_nll\tTV\tJS_nats\tKL_A_B\tKL_B_A\ttop5_intersection\tmax_abs\n";
        o<<std::setprecision(13);
        Distribution u,v;int k,index=0;
        while(targets>>k) {
            check(k>=0&&k<V);
            a.read((char*)u.x.data(),V*4);b.read((char*)v.x.data(),V*4);check(bool(a)&&bool(b));
            u.calculate();bool exact=std::memcmp(u.x.data(),v.x.data(),V*4)==0;
            if(exact){v.top=u.top;v.p=u.p;v.log_normalizer=u.log_normalizer;}else v.calculate();
            const int ax=u.top[0],by=v.top[0];
            double tv=0.,js=0.,ab=0.,ba=0.,maxdiff=0.;
            if(!exact){
                for(int i=0;i<V;i++){
                    const double x=u.p[i],y=v.p[i],mix=(x+y)/2;
                    tv+=std::abs(x-y);
                    if(x>0)js+=.5*x*std::log(x/mix);
                    if(y>0)js+=.5*y*std::log(y/mix);
                    const double logratio=(double(u.x[i])-u.log_normalizer)-(double(v.x[i])-v.log_normalizer);
                    ab+=x*logratio;ba-=y*logratio;
                    maxdiff=std::max(maxdiff,std::abs(double(u.x[i])-v.x[i]));
                }
            }
            int overlap=0;for(int x:u.top)overlap+=std::find(v.top.begin(),v.top.end(),x)!=v.top.end();
            o<<index++<<'\t'<<k<<'\t'<<exact<<'\t'<<ax<<'\t'<<by<<'\t'<<u.top[1]<<'\t'<<v.top[1]
             <<'\t'<<double(u.x[ax])-u.x[u.top[1]]<<'\t'<<double(v.x[by])-v.x[v.top[1]]
             <<'\t'<<double(u.x[ax])-u.x[by]<<'\t'<<double(v.x[by])-v.x[ax]
             <<'\t'<<u.rank(by)<<'\t'<<v.rank(ax)<<'\t'<<u.p[ax]<<'\t'<<v.p[ax]<<'\t'<<u.p[by]<<'\t'<<v.p[by]
             <<'\t'<<u.log_normalizer-u.x[k]<<'\t'<<v.log_normalizer-v.x[k]<<'\t'<<tv/2<<'\t'<<js<<'\t'<<ab<<'\t'<<ba
             <<'\t'<<overlap<<'\t'<<maxdiff<<'\n';
        }
        check(index>0 && a.peek()==std::char_traits<char>::eof()&&b.peek()==std::char_traits<char>::eof()&&o.good());
    }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
