#include "backend.h"
#include "graph.h"
#include "fp8_small_m.cuh"
#include "fp8_cutlass.cuh"
#include <functional>
#include <iostream>
#include <cmath>
using namespace tts;
using bf16=__nv_bfloat16;
float measure(Backend& g,const std::function<void(int)>& f,int copies) {
    Graph graph;graph.capture(g.stream,[&]{for(int j=0;j<copies;++j)f(j);});
    for(int i=0;i<5;++i)graph.run(g.stream);g.sync();
    cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));std::vector<float> times;
    for(int i=0;i<15;++i){CUDA(cudaEventRecord(a,g.stream));graph.run(g.stream);CUDA(cudaEventRecord(b,g.stream));CUDA(cudaEventSynchronize(b));float ms;CUDA(cudaEventElapsedTime(&ms,a,b));times.push_back(ms*1000/copies);}
    cudaEventDestroy(a);cudaEventDestroy(b);std::sort(times.begin(),times.end());return times[times.size()/2];
}
int main(int argc,char**argv) {
    int m=argc>1?atoi(argv[1]):1,n=argc>2?atoi(argv[2]):1024,k=argc>3?atoi(argv[3]):8192;
    Backend g(0,Quantization::FP8);auto& q=g.quantized();constexpr int copies=32;
    Buffer x(size_t(m)*k*2, &g.memory),qx(size_t(m)*k, &g.memory),sx(m*4, &g.memory),sw(n*4, &g.memory),y(size_t(m)*n*2, &g.memory);
    RowScaledView ax{qx.view(m,k,Type::FP8),sx.view(m,1,Type::F32)};
    std::vector<Buffer> w,qw;for(int i=0;i<copies;++i){w.emplace_back(size_t(n)*k*2, &g.memory);qw.emplace_back(size_t(n)*k, &g.memory);}
    std::vector<bf16> hx(m*k),hw(size_t(n)*k);
    for(size_t i=0;i<hx.size();++i)hx[i]=bf16(std::sin(i*.137)*(1+(i/k)%7));
    for(size_t i=0;i<hw.size();++i)hw[i]=bf16((std::sin(i*.013)+std::cos(i*.071))/std::sqrt(float(k)));
    g.upload(x.data(),hx.data(),x.size());g.upload(w[0].data(),hw.data(),w[0].size());
    q.quantize(x.view(m,k),ax);q.quantize(w[0].view(n,k),{qw[0].view(n,k,Type::FP8),sw.view(n,1,Type::F32)});
    for(int i=1;i<copies;++i){CUDA(cudaMemcpyAsync(w[i].data(),w[0].data(),w[0].size(),cudaMemcpyDeviceToDevice,g.stream));CUDA(cudaMemcpyAsync(qw[i].data(),qw[0].data(),qw[0].size(),cudaMemcpyDeviceToDevice,g.stream));}
    q.prepare(m,n,k);g.sync();
    std::vector<std::pair<std::string,std::function<void(int)>>> paths;
    paths.push_back({"bf16",[&](int i){g.linear(x.view(m,k),w[i].view(n,k),y.view(m,n));}});
    paths.push_back({"fp8_lt",[&](int i){q.linear(ax,{qw[i].view(n,k,Type::FP8),sw.view(n,1,Type::F32)},y.view(m,n));}});
#define GEMV(T,S) paths.push_back({"gemv" #T "split" #S,[&](int i){fp8_gemv_shared<T,S><<<dim3((n+(8/S)-1)/(8/S),(m+T-1)/T),256,T*k,g.stream>>>(qx.ptr<unsigned char>(),sx.ptr<float>(),qw[i].ptr<unsigned char>(),sw.ptr<float>(),y.ptr<bf16>(),m,n,k);}})
    GEMV(1,1);GEMV(1,2);GEMV(1,4);GEMV(2,2);GEMV(4,4);
#define MMA(S) paths.push_back({"mma" #S,[&](int i){fp8_mma_small<S><<<dim3((n+15)/16,(m+7)/8),S*32,8*k,g.stream>>>(qx.ptr<unsigned char>(),sx.ptr<float>(),qw[i].ptr<unsigned char>(),sw.ptr<float>(),y.ptr<bf16>(),m,n,k);}})
    if(8*k<=48000){MMA(4);MMA(8);}
    Buffer workspace(8<<20, &g.memory);
    paths.push_back({"cutlass64x16",[&](int i){fp8_bench::run(qx.data(),sx.ptr<float>(),qw[i].data(),sw.ptr<float>(),y.data(),m,n,k,workspace.data(),workspace.size(),g.stream);}});
    std::vector<bf16> ref(m*n),actual(m*n);paths[1].second(0);g.download(ref.data(),y.data(),y.size());
    for(auto&[name,f]:paths){f(0);g.download(actual.data(),y.data(),y.size());double err=0,norm=0;for(size_t i=0;i<ref.size();++i){double d=float(actual[i])-float(ref[i]);err+=d*d;norm+=double(float(ref[i]))*float(ref[i]);}double rel=sqrt(err/norm);if(name!="bf16" && rel>.01)throw std::runtime_error(name+" invalid "+std::to_string(rel));std::cout<<"{\"path\":\""<<name<<"\",\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k<<",\"us\":"<<measure(g,f,copies)<<",\"l2_vs_lt\":"<<rel<<"}\n"<<std::flush;}
}
