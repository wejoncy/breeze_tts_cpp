#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
namespace fp8_stream {
using namespace cute;
using E=cutlass::float_e4m3_t;using D=cutlass::bfloat16_t;
using Tile=Shape<_128,_16,_128>;using Cluster=Shape<_1,_1,_1>;
namespace fusion=cutlass::epilogue::fusion;
using ScaleW=fusion::Sm90ColBroadcast<0,Tile,float,float,Stride<_1,_0,_0>>;
using ScaleX=fusion::Sm90RowBroadcast<0,Tile,float,float,Stride<_0,_1,_0>>;
using Mul=fusion::Sm90Compute<cutlass::multiplies,float,float,cutlass::FloatRoundStyle::round_to_nearest>;
using MulOut=fusion::Sm90Compute<cutlass::multiplies,D,float,cutlass::FloatRoundStyle::round_to_nearest>;
using EVT0=fusion::Sm90EVT<Mul,ScaleX,fusion::Sm90AccFetch>;
using EVT=fusion::Sm90EVT<MulOut,ScaleW,EVT0>;
using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,float,float,void,
    cutlass::layout::ColumnMajor,8,D,cutlass::layout::ColumnMajor,8,
    cutlass::epilogue::TmaWarpSpecializedCooperative,EVT>::CollectiveOp;
using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,E,cutlass::layout::RowMajor,16,
    E,cutlass::layout::ColumnMajor,16,float,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue,cutlass::gemm::StreamKScheduler>;
using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
inline void run(void* x,const float* sx,void* w,const float* sw,void* y,
                int m,int n,int k,void* workspace,size_t bytes,cudaStream_t stream,int splits) {
    typename Mainloop::Arguments mainloop{};
    mainloop.mma_promotion_interval=4;
    mainloop.ptr_A=static_cast<E*>(w);mainloop.dA={k,Int<1>{},n*k};
    mainloop.ptr_B=static_cast<E*>(x);mainloop.dB={k,Int<1>{},m*k};
    typename Epilogue::Arguments epilogue{};
    epilogue.thread={ {sw}, { {sx},{},{} }, {} };
    epilogue.ptr_D=static_cast<D*>(y);epilogue.dD={Int<1>{},n,n*m};
    epilogue.dC={Int<1>{},n,n*m};
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{n,m,k,1},mainloop,epilogue};
    static const cutlass::KernelHardwareInfo hw=[] { cutlass::KernelHardwareInfo result{};
        cudaGetDevice(&result.device_id);cudaDeviceProp p{};cudaGetDeviceProperties(&p,result.device_id);
        result.sm_count=p.multiProcessorCount;return result;}();
    args.hw_info=hw;args.scheduler.splits=splits;
    args.scheduler.decomposition_mode=splits==1?Kernel::TileScheduler::DecompositionMode::StreamK:Kernel::TileScheduler::DecompositionMode::SplitK;
    if(Gemm::get_workspace_size(args)>bytes)throw std::runtime_error("CUTLASS workspace overflow");
    Gemm gemm;
    auto check=[](cutlass::Status s){if(s!=cutlass::Status::kSuccess)throw std::runtime_error(cutlassGetStatusString(s));};
    check(gemm.can_implement(args));check(gemm.initialize(args,workspace,stream));check(gemm.run(stream));
}
}
