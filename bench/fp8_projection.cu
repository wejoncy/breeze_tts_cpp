// Experimental only: compare complete BF16-input projection paths before dispatch decisions.
#include "backends/cuda/backend.h"
#include "runtime/graph.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <functional>
#include <iostream>
#include <vector>
using bf16 = __nv_bfloat16;
using namespace tts;
__device__ float maximum(float v) {
    for (int d=16;d;d/=2) v=fmaxf(v,__shfl_down_sync(0xffffffff,v,d));
    return v;
}
__device__ float block_max(float v) {
    __shared__ float partial[8];
    v=maximum(v);
    if (!(threadIdx.x%32)) partial[threadIdx.x/32]=v;
    __syncthreads();
    float result=0;
    for(int i=0;i<blockDim.x/32;++i) result=fmaxf(result,partial[i]);
    return result;
}
__device__ float scale_from_max(float value) {
    return value == 0 ? 1.f : fmaxf(value / 448.f, FLT_MIN);
}
__global__ void benchmark_quantize(const bf16 *x, unsigned char *q, float *s, int width) {
    int row=blockIdx.x;
    float amax=0;
    for(int k=threadIdx.x;k<width;k+=blockDim.x) amax=fmaxf(amax,fabsf(float(x[size_t(row)*width+k])));
    float scale=scale_from_max(block_max(amax)), inv=1.f/scale;
    if(!threadIdx.x) s[row]=scale;
    for(int k=threadIdx.x;k<width;k+=blockDim.x)
        q[size_t(row)*width+k]=__nv_fp8_e4m3(float(x[size_t(row)*width+k])*inv).__x;
}
__device__ float4 fp8x4(unsigned int bits) {
    __nv_fp8x4_e4m3 v;v.__x=bits;return float4(v);
}
// One input token per CTA, shared quantized row, eight output channels per CTA.
__global__ void projection_fp8_fused_gemv(const bf16 *x,const unsigned char *w,const float *sw,
                                         bf16 *y,int rows,int n,int k) {
    extern __shared__ unsigned int aq[];
    __shared__ float sa;
    int row=blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32,col=blockIdx.x*8+warp;
    float amax=0;
    for(int c=threadIdx.x;c<k;c+=blockDim.x) amax=fmaxf(amax,fabsf(float(x[size_t(row)*k+c])));
    float scale=scale_from_max(block_max(amax)), inv=1.f/scale;
    if(!threadIdx.x)sa=scale;
    for(int c=threadIdx.x*4;c<k;c+=blockDim.x*4) {
        const bf16 *p=x+size_t(row)*k+c;
        aq[c/4]=__nv_fp8x4_e4m3(make_float4(float(p[0])*inv,float(p[1])*inv,float(p[2])*inv,float(p[3])*inv)).__x;
    }
    __syncthreads();
    float sum=0;
    if(col<n) for(int c=lane*4;c<k;c+=128) {
        float4 a=fp8x4(aq[c/4]);
        float4 b=fp8x4(*reinterpret_cast<const unsigned int*>(w+size_t(col)*k+c));
        sum=fmaf(a.x,b.x,sum);sum=fmaf(a.y,b.y,sum);sum=fmaf(a.z,b.z,sum);sum=fmaf(a.w,b.w,sum);
    }
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(!lane && col<n)y[size_t(row)*n+col]=bf16(sum*sw[col]*sa);
}
template<int K,int Tokens,bool Prequantized=false>
__global__ void projection_fp8_token_tile(const bf16 *x,const unsigned char *qx,const float *sx,
                                         const unsigned char *w,const float *sw,bf16 *y,int rows,int n) {
    __shared__ unsigned int aq[Tokens*K/4];
    __shared__ float partial[Tokens*8], scales[Tokens], inverse[Tokens];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.y*Tokens,col=blockIdx.x*8+warp;
    if constexpr (!Prequantized) {
        float values[Tokens][K/256];
#pragma unroll
        for(int r=0;r<Tokens;++r) {
            float amax=0;
#pragma unroll
            for(int j=0;j<K/256;++j) {
                int c=threadIdx.x+j*256;
                float value=row+r<rows?float(x[size_t(row+r)*K+c]):0;
                values[r][j]=value;amax=fmaxf(amax,fabsf(value));
            }
            amax=maximum(amax);
            if(!lane)partial[r*8+warp]=amax;
        }
        __syncthreads();
        if(threadIdx.x<Tokens) {
            float amax=0;
#pragma unroll
            for(int w=0;w<8;++w)amax=fmaxf(amax,partial[threadIdx.x*8+w]);
            scales[threadIdx.x]=scale_from_max(amax);inverse[threadIdx.x]=1.f/scales[threadIdx.x];
        }
        __syncthreads();
#pragma unroll
        for(int r=0;r<Tokens;++r) {
#pragma unroll
            for(int j=0;j<K/256;++j)
                reinterpret_cast<unsigned char*>(aq)[r*K+threadIdx.x+j*256]=__nv_fp8_e4m3(values[r][j]*inverse[r]).__x;
        }
        __syncthreads();
    }
    float sums[Tokens]={};
    if(col<n) {
#pragma unroll 4
        for(int c=lane*4;c<K;c+=128) {
            float4 b=fp8x4(*reinterpret_cast<const unsigned int*>(w+size_t(col)*K+c));
#pragma unroll
            for(int r=0;r<Tokens;++r) {
                unsigned int bits;
                if constexpr (Prequantized)
                    bits=row+r<rows?*reinterpret_cast<const unsigned int*>(qx+size_t(row+r)*K+c):0;
                else bits=aq[(r*K+c)/4];
                float4 a=fp8x4(bits);
                sums[r]=fmaf(a.x,b.x,sums[r]);sums[r]=fmaf(a.y,b.y,sums[r]);
                sums[r]=fmaf(a.z,b.z,sums[r]);sums[r]=fmaf(a.w,b.w,sums[r]);
            }
        }
    }
#pragma unroll
    for(int r=0;r<Tokens;++r) {
        for(int d=16;d;d/=2)sums[r]+=__shfl_down_sync(0xffffffff,sums[r],d);
        if(!lane && col<n && row+r<rows) {
            float scale=Prequantized?sx[row+r]:scales[r];
            y[size_t(row+r)*n+col]=bf16(sums[r]*sw[col]*scale);
        }
    }
}
// Full input row stays in shared memory; four warps split K inside each CTA.
// This reduces synchronization and increases output-tile parallelism at decode sizes.
template<int K>
__global__ void projection_fp8_fused_mma(const bf16 *x,const unsigned char *w,const float *sw,
                                        bf16 *y,int rows,int n) {
    constexpr int Stride=K+16;
    __shared__ __align__(16) unsigned char aq[16*Stride];
    __shared__ float scales[16], inverses[16], partial[4*16*16];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,m0=blockIdx.y*16,n0=blockIdx.x*16;
    for(int r=warp;r<16;r+=8) {
        float amax=0;
        if(m0+r<rows) {
#pragma unroll 4
            for(int c=lane*4;c<K;c+=128) {
                const bf16 *p=x+size_t(m0+r)*K+c;
                amax=fmaxf(amax,fabsf(float(p[0])));amax=fmaxf(amax,fabsf(float(p[1])));
                amax=fmaxf(amax,fabsf(float(p[2])));amax=fmaxf(amax,fabsf(float(p[3])));
            }
        }
        amax=maximum(amax);
        if(!lane){scales[r]=scale_from_max(amax);inverses[r]=1.f/scales[r];}
    }
    __syncthreads();
    for(int i=threadIdx.x;i<16*K/4;i+=blockDim.x) {
        int r=i/(K/4),c=(i%(K/4))*4;
        unsigned int bits=0;
        if(m0+r<rows) {
            const bf16 *p=x+size_t(m0+r)*K+c;float inv=inverses[r];
            bits=__nv_fp8x4_e4m3(make_float4(float(p[0])*inv,float(p[1])*inv,float(p[2])*inv,float(p[3])*inv)).__x;
        }
        *reinterpret_cast<unsigned int*>(aq+r*Stride+c)=bits;
    }
    __syncthreads();
    float d0=0,d1=0,d2=0,d3=0;
    int split=warp%4,group=warp/4,ar=lane/4,ac=(lane%4)*4,bn=n0+group*8+lane/4;
#pragma unroll 4
    for(int p=split*(K/4);p<(split+1)*(K/4);p+=32) {
        unsigned int a0=*reinterpret_cast<const unsigned int*>(aq+ar*Stride+ac+p);
        unsigned int a1=*reinterpret_cast<const unsigned int*>(aq+(ar+8)*Stride+ac+p);
        unsigned int a2=*reinterpret_cast<const unsigned int*>(aq+ar*Stride+ac+p+16);
        unsigned int a3=*reinterpret_cast<const unsigned int*>(aq+(ar+8)*Stride+ac+p+16);
        unsigned int b0=bn<n?*reinterpret_cast<const unsigned int*>(w+size_t(bn)*K+p+ac):0;
        unsigned int b1=bn<n?*reinterpret_cast<const unsigned int*>(w+size_t(bn)*K+p+ac+16):0;
#if __CUDA_ARCH__ >= 890
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
                     : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#endif
    }
    int col=group*8+2*(lane%4),offset=split*256+ar*16+col;
    partial[offset]=d0;partial[offset+1]=d1;partial[offset+8*16]=d2;partial[offset+8*16+1]=d3;
    __syncthreads();
    int r=threadIdx.x/16,c=threadIdx.x%16;
    if(m0+r<rows && n0+c<n) {
        float sum=partial[threadIdx.x]+partial[256+threadIdx.x]+partial[512+threadIdx.x]+partial[768+threadIdx.x];
        y[size_t(m0+r)*n+n0+c]=bf16(sum*sw[n0+c]*scales[r]);
    }
}

struct LtPlan {
    bool activation_first;
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a,b,d;
    cublasLtMatmulHeuristicResult_t algorithm;
    Buffer workspace;
    LtPlan(DeviceContext &memory,int rows,int n,int k,const float *sw,const float *sx,bool first=false) : activation_first(first), workspace(32<<20, &memory) {
        if(first)std::swap(sw,sx);
        BLAS(cublasLtCreate(&handle));
        BLAS(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        auto transpose=first?CUBLAS_OP_N:CUBLAS_OP_T;
        BLAS(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
        auto mode=CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        BLAS(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
        BLAS(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
        BLAS(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sw,sizeof(sw)));
        BLAS(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sx,sizeof(sx)));
        BLAS(cublasLtMatrixLayoutCreate(&a,CUDA_R_8F_E4M3,first?rows:k,first?k:n,k));
        BLAS(cublasLtMatrixLayoutCreate(&b,CUDA_R_8F_E4M3,k,first?n:rows,k));
        BLAS(cublasLtMatrixLayoutCreate(&d,CUDA_R_16BF,first?rows:n,first?n:rows,n));
        if(first){auto order=CUBLASLT_ORDER_ROW;BLAS(cublasLtMatrixLayoutSetAttribute(a,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order)));BLAS(cublasLtMatrixLayoutSetAttribute(d,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order)));}
        cublasLtMatmulPreference_t pref;BLAS(cublasLtMatmulPreferenceCreate(&pref));size_t size=workspace.size();
        BLAS(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&size,sizeof(size)));
        int count=0;
        BLAS(cublasLtMatmulAlgoGetHeuristic(handle,operation,a,b,d,d,pref,1,&algorithm,&count));
        cublasLtMatmulPreferenceDestroy(pref);
        if(!count)throw std::runtime_error("FP8 outer-scale GEMM unavailable");
    }
    ~LtPlan(){cublasLtMatmulDescDestroy(operation);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(d);cublasLtDestroy(handle);}
    void run(Backend &g,const Buffer &x,const Buffer &w,Buffer &y) {
        float alpha=1,beta=0;
        BLAS(cublasLtMatmul(handle,operation,&alpha,activation_first?x.data():w.data(),a,activation_first?w.data():x.data(),b,&beta,y.data(),d,y.data(),d,&algorithm.algo,workspace.data(),workspace.size(),g.stream));
    }
};
static float measure(Backend &g,const std::function<void(int)> &run,int copies,int repeats) {
    Graph graph;graph.capture(g.stream,[&]{for(int j=0;j<copies;++j)run(j);});
    for(int i=0;i<4;++i)graph.run(g.stream);g.sync();
    cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));std::vector<float> times;
    for(int i=0;i<repeats;++i){CUDA(cudaEventRecord(a,g.stream));graph.run(g.stream);CUDA(cudaEventRecord(b,g.stream));CUDA(cudaEventSynchronize(b));float ms;CUDA(cudaEventElapsedTime(&ms,a,b));times.push_back(ms*1000/copies);}
    cudaEventDestroy(a);cudaEventDestroy(b);std::sort(times.begin(),times.end());return times[times.size()/2];
}
int main(int argc,char **argv) {
    try {
        int rows=argc>1?std::atoi(argv[1]):1,n=argc>2?std::atoi(argv[2]):2048,k=argc>3?std::atoi(argv[3]):2048;
        int repeats=argc>4?std::atoi(argv[4]):20;
        if(rows<1||rows>512||n<1||k<32||k%32||repeats<1)throw std::invalid_argument("benchmark shape");
        constexpr int copies=64;
        Backend g(0, Quantization::FP8);Buffer x(size_t(rows)*k*2, &g.memory),qx(size_t(rows)*k, &g.memory),sx(rows*4, &g.memory),sw(n*4, &g.memory),y(size_t(rows)*n*2, &g.memory);
        std::vector<Buffer> weights,quantized;
        for(int i=0;i<copies;++i){weights.emplace_back(size_t(n)*k*2, &g.memory);quantized.emplace_back(size_t(n)*k, &g.memory);}
        std::vector<bf16> hx(rows*k),hw(size_t(n)*k);
        for(size_t i=0;i<hx.size();++i)hx[i]=bf16(std::sin(i*.017)*(1+(i/k)%7));
        for(size_t i=0;i<hw.size();++i)hw[i]=bf16((std::sin(i*.013)+std::cos(i*.071))/std::sqrt(float(k)));
        g.upload(x.data(),hx.data(),x.size());g.upload(weights[0].data(),hw.data(),weights[0].size());
        benchmark_quantize<<<n,256,0,g.stream>>>(weights[0].ptr<bf16>(),quantized[0].ptr<unsigned char>(),sw.ptr<float>(),k);
        benchmark_quantize<<<rows,256,0,g.stream>>>(x.ptr<bf16>(),qx.ptr<unsigned char>(),sx.ptr<float>(),k);
        for(int i=1;i<copies;++i){CUDA(cudaMemcpyAsync(weights[i].data(),weights[0].data(),weights[0].size(),cudaMemcpyDeviceToDevice,g.stream));CUDA(cudaMemcpyAsync(quantized[i].data(),quantized[0].data(),quantized[0].size(),cudaMemcpyDeviceToDevice,g.stream));}
        g.sync();LtPlan plan(g.memory,rows,n,k,sw.ptr<float>(),sx.ptr<float>()), row_plan(g.memory,rows,n,k,sw.ptr<float>(),sx.ptr<float>(),true);
        g.quantized().prepare(rows,n,k);
        std::vector<std::pair<std::string,std::function<void(int)>>> paths;
        paths.push_back({"bf16",[&](int i){g.linear(x.view(rows,k),weights[i].view(n,k),y.view(rows,n));}});
        paths.push_back({"fp8_current_separate",[&](int i){
            RowScaledView input{qx.view(rows,k,Type::FP8),sx.view(rows,1,Type::F32)};
            g.quantized().quantize(x.view(rows,k),input);
            g.quantized().linear(input,{quantized[i].view(n,k,Type::FP8),sw.view(n,1,Type::F32)},y.view(rows,n));
        }});
        paths.push_back({"fp8_prequantized",[&](int i){plan.run(g,qx,quantized[i],y);}});
        paths.push_back({"fp8_separate_quant_total",[&](int i){benchmark_quantize<<<rows,256,0,g.stream>>>(x.ptr<bf16>(),qx.ptr<unsigned char>(),sx.ptr<float>(),k);plan.run(g,qx,quantized[i],y);}});
        paths.push_back({"fp8_fused_gemv",[&](int i){projection_fp8_fused_gemv<<<dim3((n+7)/8,rows),256,k,g.stream>>>(x.ptr<bf16>(),quantized[i].ptr<unsigned char>(),sw.ptr<float>(),y.ptr<bf16>(),rows,n,k);}});
        paths.push_back({"fp8_fused_mma",[&](int i){if(k==1024)projection_fp8_fused_mma<1024><<<dim3((n+15)/16,(rows+15)/16),256,0,g.stream>>>(x.ptr<bf16>(),quantized[i].ptr<unsigned char>(),sw.ptr<float>(),y.ptr<bf16>(),rows,n);
            else if(k==2048)projection_fp8_fused_mma<2048><<<dim3((n+15)/16,(rows+15)/16),256,0,g.stream>>>(x.ptr<bf16>(),quantized[i].ptr<unsigned char>(),sw.ptr<float>(),y.ptr<bf16>(),rows,n);
            else throw std::invalid_argument("fused prototype supports the actual projection K=1024/2048");}});
        paths.push_back({"fp8_prequantized_row",[&](int i){row_plan.run(g,qx,quantized[i],y);}});
#define TILE_PATH(K,T,P) projection_fp8_token_tile<K,T,P><<<dim3((n+7)/8,(rows+T-1)/T),256,0,g.stream>>>(x.ptr<bf16>(),qx.ptr<unsigned char>(),sx.ptr<float>(),quantized[i].ptr<unsigned char>(),sw.ptr<float>(),y.ptr<bf16>(),rows,n)
        paths.push_back({"fp8_fused_tile4",[&](int i){if(k==1024){TILE_PATH(1024,4,false);}else{TILE_PATH(2048,4,false);}}});
        paths.push_back({"fp8_fused_tile8",[&](int i){if(k==1024){TILE_PATH(1024,8,false);}else{TILE_PATH(2048,8,false);}}});
        paths.push_back({"fp8_prequant_tile4",[&](int i){if(k==1024){TILE_PATH(1024,4,true);}else{TILE_PATH(2048,4,true);}}});
#undef TILE_PATH
        std::vector<bf16> reference(rows*n),actual(rows*n);
        paths[1].second(0);g.download(reference.data(),y.data(),y.size());
        for(auto &[name,run]:paths) {
            run(0);g.download(actual.data(),y.data(),y.size());double squared=0,error=0;float max_error=0;
            for(size_t i=0;i<actual.size();++i){float diff=float(actual[i])-float(reference[i]);error+=diff*diff;squared+=float(reference[i])*float(reference[i]);max_error=std::max(max_error,std::abs(diff));}
            double relative=std::sqrt(error/std::max(squared,1e-30));
            if(name!="bf16" && relative>.01)throw std::runtime_error(name+" failed FP8 reference comparison: "+std::to_string(relative));
            float us=measure(g,run,copies,repeats);
            std::cout<<"{\"path\":\""<<name<<"\",\"rows\":"<<rows<<",\"n\":"<<n<<",\"k\":"<<k<<",\"us\":"<<us<<",\"relative_l2_vs_fp8\":"<<relative<<",\"max_abs_vs_fp8\":"<<max_error<<"}\n"<<std::flush;
        }
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
