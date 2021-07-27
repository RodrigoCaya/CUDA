#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

void incisoA(int m, float *yij, float dt, float ti){
    for(int j = 0; j<=m; j++){
        yij[j] = yij[j] + dt*(4*ti - yij[j] + 3 + j);
    }
}

__global__ void incisoB(int m, float *yij, float dt, float ti){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < m){
        yij[tid] = yij[tid] + dt*(4*ti - yij[tid] + 3 + tid);
    }
}

int main(){
    int ms[5] = {10000, 100000, 1000000, 10000000, 100000000};

    //inciso a

    clock_t t1, t2;
    double tms;
    int n = 1000;
    float dt = 0.001;
    for(int k = 0; k<5; k++){

        float *yj0 = (float *)malloc(sizeof(float)*ms[k]);
        for(int j = 0; j<=ms[k]; j++){
            yj0[j] = j;
        }
        float *ysGPU = (float *)malloc(sizeof(float)*ms[k]);
        memcpy(ysGPU, yj0, sizeof(float)*ms[k]);

        t1 = clock();
        for(int i = 0; i<=n; i++){
            incisoA(ms[k], yj0, dt, (0+i*dt));
        }
        t2 = clock();
        tms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
        std::cout << "a) Tiempo CPU para el valor de m igual a "<< ms[k] << " es: "<< tms << "[ms]" << std::endl;
        free(yj0);
        std::cout << "----------------------" << std::endl;

        //inciso b

        cudaEvent_t ct1, ct2;
        float tgpu, *ida;
        int bs = 256;
        int gs = (int)ceil((float)ms[k] / bs);
        
        cudaEventCreate(&ct1);
	      cudaEventCreate(&ct2);
	      cudaEventRecord(ct1);

        cudaMalloc(&ida, sizeof(float)*ms[k]);
        cudaMemcpy(ida, ysGPU, sizeof(float)*ms[k], cudaMemcpyHostToDevice);
        for(int i = 0; i<=n; i++){
            incisoB<<<gs, bs>>>(ms[k], ida, dt, (0+i*dt));
        }
        cudaMemcpy(ysGPU, ida, sizeof(float)*ms[k], cudaMemcpyDeviceToHost);

        cudaEventRecord(ct2);
	    cudaEventSynchronize(ct2);
	    cudaEventElapsedTime(&tgpu, ct1, ct2);
	    std::cout << "b) Tiempo GPU para el valor de m igual a "<< ms[k] << " es: "<< tgpu << "[ms]" << std::endl;
        cudaFree(ida);
        free(ysGPU);
        std::cout << "----------------------" << std::endl;  
    }

    //inciso c

    int m = 100000000;
    int bss[4] = {64, 128, 256, 512};
    for(int b = 0; b < 4; b++){
        float *ys = (float *)malloc(sizeof(float)*m);
        for(int j = 0; j<=m; j++){
            ys[j] = j;
        }
        cudaEvent_t ct1, ct2;
        float tgpu, *ida;
        int bs = bss[b];
        int gs = (int)ceil((float)m / bs);
        
        cudaEventCreate(&ct1);
	    cudaEventCreate(&ct2);
	    cudaEventRecord(ct1);

        cudaMalloc(&ida, sizeof(float)*m);
        cudaMemcpy(ida, ys, sizeof(float)*m, cudaMemcpyHostToDevice);
        for(int i = 0; i<=n; i++){
            incisoB<<<gs, bs>>>(m, ida, dt, (0+i*dt));
        }
        cudaMemcpy(ys, ida, sizeof(float)*m, cudaMemcpyDeviceToHost);
        cudaEventRecord(ct2);
	    cudaEventSynchronize(ct2);
	    cudaEventElapsedTime(&tgpu, ct1, ct2);
	    std::cout << "c) Tiempo GPU para el valor de m igual a "<< m << " y bloque igual a " << bss[b] <<" es: "<< tgpu << "[ms]" << std::endl;
        cudaFree(ida);
        free(ys);
    }

}