#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

void incisoA(float t0, float y0, float dt){ 
    float n = 10.0/dt;
    float *ys;
    ys = (float *)malloc(n * sizeof(float));
    float s = 0;
    for(int i = 1; i <= n; i++){
        s = s + (9*powf(i*dt, 2.0) - 4*(i*dt) + 5);
        ys[i-1] = y0 + dt*s;
    }
    free(ys);
}

__global__ void incisoB(float *ys, float t0, float y0, float dt, float n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float s = 0;
    if(tid < n){
        for(int i = 0; i <= tid; i++){
            s = s + (9*powf(i*dt, 2.0) - 4*i*dt+5);
        }
        ys[tid] = y0 + dt*s;
    }
}

__global__ void incisoCGPU(float *sums, float *ys, float t0, float y0, float dt, float n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n){
        ys[tid] = y0 + dt*sums[tid];
    }
}

void incisoC(float t0, float y0, float dt){ 
    float n = 10.0/dt;
    float *sums, *ys, *ida, *vuelta;
    float s = 0;
    sums = (float *)malloc(n * sizeof(float));
    for(int i = 0; i < n; i++){
        s = s + (9*powf(i*dt, 2.0) - 4*(i*dt) + 5);
        sums[i] = s;
    }
    ys = (float *)malloc(n*sizeof(float));
    int bs = 256;
    int gs = (int)ceil((float)n / bs);
    
    cudaMalloc(&ida, sizeof(float)*n);
    cudaMalloc(&vuelta, sizeof(float)*n);
    cudaMemcpy(ida, sums, sizeof(float)*n, cudaMemcpyHostToDevice);
    incisoCGPU<<<gs, bs>>>(ida, vuelta, 0, y0, dt, n);
    cudaMemcpy(ys, vuelta, sizeof(float)*n, cudaMemcpyDeviceToHost);
    cudaFree(vuelta);
    cudaFree(ida);
    free(ys);
    free(sums);
}


int main(){
    float dts[6] = {0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001};

    //inciso a
    clock_t t1, t2;
    double ms;
    for(int i = 0; i<6; i++){
        t1 = clock();
        incisoA(0.0, 4.0, dts[i]);
        t2 = clock();
        ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
        std::cout << "a) Tiempo CPU para el valor delta t igual a "<< dts[i] << " es: "<< ms << "[ms]" << std::endl;
    }
    std::cout << "----------------------" << std::endl;

    //inciso b

    float dt;
    cudaEvent_t ct1, ct2;
    for(int i = 0; i<4; i++){
        float n = 10.0/dts[i];
        float *p;
        float *ys = (float *)malloc(sizeof(float)*n);
        int bs = 256;
        int gs = (int)ceil((float)n / bs);
        
        cudaMalloc(&p, sizeof(float)*n);

        cudaEventCreate(&ct1);
	    cudaEventCreate(&ct2);
	    cudaEventRecord(ct1);
        incisoB<<<gs, bs>>>(p, 0, 4.0, dts[i], n);
        cudaEventRecord(ct2);
	    cudaEventSynchronize(ct2);
	    cudaEventElapsedTime(&dt, ct1, ct2);
	    std::cout << "b) Tiempo GPU para el valor delta t igual a "<< dts[i] << " es: "<< dt << "[ms]" << std::endl;

        cudaMemcpy(ys, p, sizeof(float)*n, cudaMemcpyDeviceToHost);
        cudaFree(p);
        free(ys);
    }
    std::cout << "----------------------" << std::endl;

    //inciso c 

    for(int i = 0; i<6; i++){
        t1 = clock();
        incisoC(0.0, 4.0, dts[i]);
        t2 = clock();
        ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
        std::cout << "c) Tiempo CPU-GPU para el valor delta t igual a "<< dts[i] << " es: "<< ms << "[ms]" << std::endl;
    }

    return 0;
}