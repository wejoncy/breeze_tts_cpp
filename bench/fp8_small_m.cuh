#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
// Experimental small-M kernels, with no intermediate layout conversion.
template<int Tokens, int Splits=1>
__global__ void fp8_gemv_shared(const unsigned char *x, const float *sx,
                              const unsigned char *w, const float *sw,
                              __nv_bfloat16 *y, int rows, int n, int k) {
    extern __shared__ unsigned int shared_x[];
    int lane=threadIdx.x%32, warp=threadIdx.x/32, row=blockIdx.y*Tokens;
    int col=blockIdx.x*(8/Splits)+warp/Splits;
    int split=warp%Splits;
    __shared__ float partial[8][Tokens];
    for(int i=threadIdx.x;i<Tokens*k/4;i+=blockDim.x) {
        int r=i/(k/4), c=i%(k/4);
        shared_x[i]=row+r<rows?reinterpret_cast<const unsigned int*>(x)[(row+r)*(k/4)+c]:0;
    }
    __syncthreads();
    float sums[Tokens]={};
    if(col<n) {
#pragma unroll 4
        for(int c=lane*4+split*128;c<k;c+=128*Splits) {
            __nv_fp8x4_e4m3 wb;wb.__x=*reinterpret_cast<const unsigned int*>(w+size_t(col)*k+c);
            float4 b=float4(wb);
#pragma unroll
            for(int r=0;r<Tokens;++r) {
                __nv_fp8x4_e4m3 xa;xa.__x=shared_x[(r*k+c)/4];float4 a=float4(xa);
                sums[r]=fmaf(a.x,b.x,sums[r]);sums[r]=fmaf(a.y,b.y,sums[r]);
                sums[r]=fmaf(a.z,b.z,sums[r]);sums[r]=fmaf(a.w,b.w,sums[r]);
            }
        }
    }
#pragma unroll
    for(int r=0;r<Tokens;++r) {
        for(int d=16;d;d/=2)sums[r]+=__shfl_down_sync(0xffffffff,sums[r],d);
        if(!lane)partial[warp][r]=sums[r];
    }
    __syncthreads();
    if(!lane && !split && col<n) {
#pragma unroll
        for(int r=0;r<Tokens;++r) {
            float v=0;
#pragma unroll
            for(int s=0;s<Splits;++s)v+=partial[warp+s][r];
            if(row+r<rows)y[size_t(row+r)*n+col]=__nv_bfloat16(v*sw[col]*sx[row+r]);
        }
    }
}
// Swapped operands: rows of weights are M; up to eight tokens are N.
// Split K across the four warps in a CTA, reduce in shared, one final write.
template<int Splits=4>
__global__ void fp8_mma_small(const unsigned char *x, const float *sx,
                            const unsigned char *w, const float *sw,
                            __nv_bfloat16 *y, int rows, int n, int k) {
    extern __shared__ unsigned int shared_x[];
    __shared__ float partial[Splits][32][4];
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    int row=blockIdx.y*8, col=blockIdx.x*16, group=lane/4, t=lane%4;
    for(int i=threadIdx.x;i<8*k/4;i+=blockDim.x) {
        int r=i/(k/4), c=i%(k/4);
        shared_x[i]=row+r<rows?reinterpret_cast<const unsigned int*>(x)[(row+r)*(k/4)+c]:0;
    }
    __syncthreads();
    float sums[4]={};
#pragma unroll 4
    for(int kk=warp*32;kk<k;kk+=Splits*32) {
        unsigned int a0=0,a1=0,a2=0,a3=0;
        if(col+group<n) {
            a0=*reinterpret_cast<const unsigned int*>(w+size_t(col+group)*k+kk+t*4);
            a2=*reinterpret_cast<const unsigned int*>(w+size_t(col+group)*k+kk+t*4+16);
        }
        if(col+group+8<n) {
            a1=*reinterpret_cast<const unsigned int*>(w+size_t(col+group+8)*k+kk+t*4);
            a3=*reinterpret_cast<const unsigned int*>(w+size_t(col+group+8)*k+kk+t*4+16);
        }
        unsigned int b0=shared_x[(group*k+kk+t*4)/4],b1=shared_x[(group*k+kk+t*4+16)/4];
        float d0,d1,d2,d3;
#if __CUDA_ARCH__ >= 890
        // Promote each instruction result to full FP32 before the next K tile.
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {0f00000000,0f00000000,0f00000000,0f00000000};"
                     : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
                     : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#else
        d0=d1=d2=d3=0;
#endif
        sums[0]+=d0;sums[1]+=d1;sums[2]+=d2;sums[3]+=d3;
    }
#pragma unroll
    for(int j=0;j<4;++j)partial[warp][lane][j]=sums[j];
    __syncthreads();
    if(warp==0) {
#pragma unroll
        for(int j=0;j<4;++j) {
            float v=0;
#pragma unroll
            for(int split=0;split<Splits;++split)v+=partial[split][lane][j];
            int c=col+group+(j/2)*8, r=row+t*2+j%2;
            if(c<n && r<rows)y[size_t(r)*n+c]=__nv_bfloat16(v*sw[c]*sx[r]);
        }
    }
}
