#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <fcntl.h>
#define GGML_ASSERT(x) do { if (!(x)) std::abort(); } while(0)
#include "sort-ids.inc"
static void need(bool x){if(!x)throw std::runtime_error("index equivalence");}
static void old_sort(const char* ids,size_t nb0,size_t nb1,int64_t e,int64_t t,int64_t used,int64_t cols,std::vector<int32_t>&to,std::vector<int32_t>&from,std::vector<int32_t>&counts){
 for(int64_t expert=0;expert<e;++expert)for(int64_t token=0;token<t;++token)for(int64_t slot=0;slot<used;++slot){
  int32_t id;std::memcpy(&id,ids+token*nb1+slot*nb0,4);assert(id>=0&&id<e);
  if(id==expert){from[token*used+slot]=to.size();to.push_back(token*cols+slot%cols);++counts[expert];break;}
 }
 GGML_ASSERT(to.size()==size_t(t*used));
}
static void death(bool candidate,int kind){
 const pid_t child=fork();need(child>=0);
 if(!child){int fd=open("/dev/null",O_WRONLY);dup2(fd,2);close(fd);std::vector<int32_t>ids={0,1,2,3,4,5},a,b(6),c(8);
  ids[5]=kind==0?0:kind==1?-1:8;
  if(candidate)xeon_sort_moe_ids(reinterpret_cast<const char*>(ids.data()),4,24,8,1,6,1,a,b,c);else old_sort(reinterpret_cast<const char*>(ids.data()),4,24,8,1,6,1,a,b,c);
  _exit(0);
 }
 int status=0;need(waitpid(child,&status,0)==child);need(WIFSIGNALED(status)&&WTERMSIG(status)==SIGABRT);
}
int main()try{
 rlimit limit{0,0};need(setrlimit(RLIMIT_CORE,&limit)==0);
 std::mt19937 rng(884913);size_t cases=0,values=0;
 for(int e:{1,6,24,96,384})for(int used:{1,6})if(used<=e)
 for(int t:{0,1,32,63,64,65,128,1024,2048})for(int cols:{1,2,3,6})if(used%cols==0)
 for(int layout=0;layout<4;++layout)for(int pattern=0;pattern<4;++pattern){
  const size_t nb0=layout==3?12:4,nb1=std::max(size_t(e)*4,size_t(used)*nb0)+(layout?16:0),offset=layout>=2?8:0;
  std::vector<char>storage(offset+std::max(1,t)*nb1, char(0xff));std::vector<int>permutation(e);std::iota(permutation.begin(),permutation.end(),0);
  for(int token=0;token<t;++token){if(pattern==0)std::shuffle(permutation.begin(),permutation.end(),rng);
   for(int slot=0;slot<used;++slot){int32_t id=pattern==0?permutation[slot]:pattern==1?slot:pattern==2?(token*used+slot)%e:e-1-slot;std::memcpy(storage.data()+offset+token*nb1+slot*nb0,&id,4);}}
  std::vector<int32_t>a,b(size_t(t)*used),c(e),d,f(size_t(t)*used),g(e);
  old_sort(storage.data()+offset,nb0,nb1,e,t,used,cols,a,b,c);
  xeon_sort_moe_ids(storage.data()+offset,nb0,nb1,e,t,used,cols,d,f,g);
  need(a==d&&b==f&&c==g);++cases;values+=a.size()+b.size()+c.size();
 }
 for(bool candidate:{false,true})for(int kind:{0,1,2})death(candidate,kind);
 std::printf("{\"pass\":true,\"cases\":%zu,\"compared_index_values\":%zu,\"invalid_rejections\":6}\n",cases,values);
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
