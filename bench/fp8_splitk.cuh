#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/thread/linear_combination.h>
namespace fp8_split {
using E=cutlass::float_e4m3_t;
using Gemm=cutlass::gemm::device::GemmUniversal<E,cutlass::layout::RowMajor,E,
    cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<16,64,128>,cutlass::gemm::GemmShape<16,32,64>,
    cutlass::gemm::GemmShape<16,8,32>,
    cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,16,16>;
__global__ void finish(const float *partial,const float *sx,const float *sw,
                       __nv_bfloat16 *y,int m,int n,int splits) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<m*n){float v=0;for(int s=0;s<splits;++s)v+=partial[size_t(s)*m*n+i];
        y[i]=__nv_bfloat16(v*sx[i/n]*sw[i%n]);}
}
inline void run(void*x,const float*sx,void*w,const float*sw,void*y,
                int m,int n,int k,void*workspace,size_t bytes,cudaStream_t stream,int splits) {
    size_t partial_bytes=size_t(splits)*m*n*4;
    if(partial_bytes>bytes)throw std::runtime_error("splitK partial workspace overflow");
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
        {m,n,k},splits,{1.f,0.f},x,w,nullptr,workspace,
        int64_t(m)*k,int64_t(n)*k,int64_t(m)*n,int64_t(m)*n,k,k,n,n};
    size_t extra=Gemm::get_workspace_size(args);
    if(extra+partial_bytes>bytes)throw std::runtime_error("splitK kernel workspace overflow");
    auto check=[](cutlass::Status s){if(s!=cutlass::Status::kSuccess)throw std::runtime_error(cutlassGetStatusString(s));};
    Gemm gemm;check(gemm.can_implement(args));
    check(gemm.initialize(args,static_cast<char*>(workspace)+partial_bytes,stream));check(gemm.run(stream));
    finish<<<(m*n+255)/256,256,0,stream>>>(static_cast<float*>(workspace),sx,sw,static_cast<__nv_bfloat16*>(y),m,n,splits);
}
}
