#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cuda_bf16.h>
namespace fp8_native {
using namespace cute;
__global__ void finish(const float *partial,const float *sx,const float *sw,
                       __nv_bfloat16 *y,int m,int n,int splits) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<m*n){float v=0;for(int s=0;s<splits;++s)v+=partial[size_t(s)*m*n+i];
        y[i]=__nv_bfloat16(v*sx[i/n]*sw[i%n]);}
}
template<int BM=64,int BN=16>struct Native {
using E=cutlass::float_e4m3_t;
using Tile=Shape<Int<BM>,Int<BN>,_128>;using Cluster=Shape<_1,_1,_1>;
using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,float,float,void,
    cutlass::layout::ColumnMajor,4,float,cutlass::layout::ColumnMajor,4,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,E,cutlass::layout::RowMajor,16,
    E,cutlass::layout::ColumnMajor,16,float,Tile,Cluster,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
static void run(void*x,const float*sx,void*w,const float*sw,void*y,
                int m,int n,int k,void*workspace,size_t bytes,cudaStream_t stream,int splits) {
    if(k%splits || (k/splits)%16)throw std::runtime_error("unaligned native K partition");
    if(size_t(m)*n*splits*4>bytes)throw std::runtime_error("native partial workspace overflow");
    typename Mainloop::Arguments mainloop{};
    mainloop.mma_promotion_interval=4;
    mainloop.ptr_A=static_cast<E*>(w);mainloop.dA={k,Int<1>{},k/splits};
    mainloop.ptr_B=static_cast<E*>(x);mainloop.dB={k,Int<1>{},k/splits};
    typename Epilogue::Arguments epilogue{};
    epilogue.thread={1.f,0.f};
    epilogue.ptr_D=static_cast<float*>(workspace);epilogue.dD={Int<1>{},n,n*m};
    epilogue.dC={Int<1>{},n,n*m};
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched,{n,m,k/splits,splits},mainloop,epilogue};
    if(Gemm::get_workspace_size(args))throw std::runtime_error("unexpected native workspace");
    Gemm gemm;
    auto check=[](cutlass::Status s){if(s!=cutlass::Status::kSuccess)throw std::runtime_error(cutlassGetStatusString(s));};
    check(gemm.can_implement(args));check(gemm.initialize(args,nullptr,stream));check(gemm.run(stream));
    finish<<<(m*n+255)/256,256,0,stream>>>(static_cast<float*>(workspace),sx,sw,static_cast<__nv_bfloat16*>(y),m,n,splits);
}
};
}
