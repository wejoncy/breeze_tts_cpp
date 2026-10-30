#include "backends/cuda/backend.h"
#include "runtime/graph.h"
#include "fp8_small_m.cuh"
#include <functional>
#include <iostream>
#include <cmath>
#include <cublasLt.h>
using namespace tts;
using bf16=__nv_bfloat16;
struct LtPlan {
    bool activation_first;
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a,b,d;
    cublasLtMatmulHeuristicResult_t algorithm;
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;
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
        int count=0;algorithms.resize(64);
        BLAS(cublasLtMatmulAlgoGetHeuristic(handle,operation,a,b,d,d,pref,64,algorithms.data(),&count));
        algorithms.resize(count);algorithm=algorithms[0];
        int support=0;uint32_t mask=0;size_t written=0;
        BLAS(cublasLtMatmulAlgoCapGetAttribute(&algorithm.algo,CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,&support,sizeof(support),&written));
        BLAS(cublasLtMatmulAlgoCapGetAttribute(&algorithm.algo,CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,&mask,sizeof(mask),&written));
        std::cerr<<"splitK support="<<support<<" reduction mask="<<mask<<"\n";
        if(support)for(uint32_t split:{2,4,8,16})for(uint32_t reduction:{1,2,4})if(mask&reduction) {
            auto candidate=algorithm;
            BLAS(cublasLtMatmulAlgoConfigSetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split)));
            BLAS(cublasLtMatmulAlgoConfigSetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction)));
            cublasLtMatmulHeuristicResult_t checked{};
            auto status=cublasLtMatmulAlgoCheck(handle,operation,a,b,d,d,&candidate.algo,&checked);
            if(status==CUBLAS_STATUS_SUCCESS && checked.state==CUBLAS_STATUS_SUCCESS && checked.workspaceSize<=size) {
                candidate.workspaceSize=checked.workspaceSize;algorithms.push_back(candidate);
                std::cerr<<"candidate "<<algorithms.size()-1<<" split="<<split<<" reduction="<<reduction<<" bytes="<<checked.workspaceSize<<"\n";
            }
        }
        cublasLtMatmulPreferenceDestroy(pref);
        if(!count)throw std::runtime_error("FP8 outer-scale GEMM unavailable");
    }
    ~LtPlan(){cublasLtMatmulDescDestroy(operation);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(d);cublasLtDestroy(handle);}
    void run(Backend &g,const Buffer &x,const Buffer &w,Buffer &y) {
        float alpha=1,beta=0;
        BLAS(cublasLtMatmul(handle,operation,&alpha,activation_first?x.data():w.data(),a,activation_first?w.data():x.data(),b,&beta,y.data(),d,y.data(),d,&algorithm.algo,workspace.data(),workspace.size(),g.stream));
    }
};
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
    LtPlan plan(g.memory,m,n,k,sw.ptr<float>(),sx.ptr<float>());
    for(size_t j=0;j<plan.algorithms.size();++j)paths.push_back({"candidate_"+std::to_string(j),[&,j](int i){plan.algorithm=plan.algorithms[j];plan.run(g,qx,qw[i],y);}});
    std::vector<bf16> ref(m*n),actual(m*n);paths[1].second(0);g.download(ref.data(),y.data(),y.size());
    for(auto&[name,f]:paths){f(0);g.download(actual.data(),y.data(),y.size());double err=0,norm=0;for(size_t i=0;i<ref.size();++i){double d=float(actual[i])-float(ref[i]);err+=d*d;norm+=double(float(ref[i]))*float(ref[i]);}double rel=sqrt(err/norm);if(name!="bf16" && rel>.01)throw std::runtime_error(name+" invalid "+std::to_string(rel));std::cout<<"{\"path\":\""<<name<<"\",\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k<<",\"us\":"<<measure(g,f,copies)<<",\"l2_vs_lt\":"<<rel<<"}\n"<<std::flush;}
}
