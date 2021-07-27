#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
/*
 *  Escritura Archivo
    Funcion extraida de actividad de curso
 */
void Write(float* R, float* G, float* B, 
	       int M, int N, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", M, N);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", R[i]);
    fprintf(fp, "%f\n", R[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", G[i]);
    fprintf(fp, "%f\n", G[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", B[i]);
    fprintf(fp, "%f\n", B[M*N-1]);
    fclose(fp);
}
/*
 *  Funcion Pregunta 1 inciso a
 */
void A(){

    FILE *pAr;
    float *R, *G, *B, Rv, Gv, Bv;
    int M, N, P;
    pAr = fopen("img.txt", "r");
    fscanf(pAr, "%d %d %d\n", &M, &N, &P);

    int *rs = (int *)malloc(sizeof(int)*P);
    int *gs = (int *)malloc(sizeof(int)*P);
    int *bs = (int *)malloc(sizeof(int)*P);
    for(int j = 0; j < P; j++)
        fscanf(pAr, "%d ", &rs[j]);
    for(int j = 0; j < P; j++)
        fscanf(pAr, "%d ", &gs[j]);
    for(int j = 0; j < P; j++)
        fscanf(pAr, "%d ", &bs[j]);

         
    R = (float *)malloc(M*N*sizeof(float));
    G = (float *)malloc(M*N*sizeof(float));
    B = (float *)malloc(M*N*sizeof(float));

    clock_t t1, t2;
    double ms;
    t1 = clock();
    for(int i = 0; i < M*N; i++){
        Rv = 0, Gv = 0, Bv = 0;
        for(int k = 0; k < P; k++){
            float pk = 0;
            fscanf(pAr, "%f ", &pk);
            Rv = Rv + pk*rs[k];
            Gv = Gv + pk*gs[k];
            Bv = Bv + pk*bs[k];
        }
        R[i] = Rv;
        G[i] = Gv;
        B[i] = Bv;
    }
    t2 = clock();
    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
    std::cout << "a) Tiempo empleado es: "<< ms << "[ms]" << std::endl;

    Write(R, G, B, M, N, "salidaA.txt");
    fclose(pAr);

    free(rs);
    free(gs);
    free(bs);
}
/*
 *  Kernel Pregunta 1 inciso b
 */
 __global__ void B(float *pixels, int *rs, int *gs, int *bs, float *Rout, float *Gout, float *Bout, int M, int N, int P){
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     if(tid < M*N){
         float Rv = 0, Gv = 0, Bv = 0;
         for(int i = 0; i < P; i++){
             Rv = Rv + pixels[tid*P + i]*rs[i];
             Gv = Gv + pixels[tid*P + i]*gs[i];
             Bv = Bv + pixels[tid*P + i]*bs[i];
         }
         Rout[tid] = Rv;
         Gout[tid] = Gv;
         Bout[tid] = Bv;
     }
 }
 /*
 *  Kernel Pregunta 1 inciso c
 */
 __global__ void C(float *pixels, int *rs, int *gs, int *bs, float *Rout, float *Gout, float *Bout, int M, int N, int P){
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
     if(tid < M*N){
         float Rv = 0, Gv = 0, Bv = 0;
         for(int i = 0; i < P; i++){
             Rv = Rv + pixels[M*N*i + tid]*rs[i];
             Gv = Gv + pixels[M*N*i + tid]*gs[i];
             Bv = Bv + pixels[M*N*i + tid]*bs[i];
         }
         Rout[tid] = Rv;
         Gout[tid] = Gv;
         Bout[tid] = Bv;
     }
 }

int main(){

    //inciso a

    A();

    //inciso b
    
    FILE *pAr;
    int M, N, P, cont = 0, *rsGPU, *gsGPU, *bsGPU;
    float dt, *Rout, *Gout, *Bout, *pxs;
    cudaEvent_t ct1, ct2;
    pAr = fopen("img.txt", "r");
    fscanf(pAr, "%d %d %d\n", &M, &N, &P);
    int *rs = (int *)malloc(P*sizeof(int));
    int *gs = (int *)malloc(P*sizeof(int));
    int *bs = (int *)malloc(P*sizeof(int));
    float *pixels = (float *)malloc(M*N*P*sizeof(float));

    for(int j = 0; j < P; j++)
        fscanf(pAr, "%d ", &rs[j]);
    for(int j = 0; j < P; j++)
        fscanf(pAr, "%d ", &gs[j]);
    for(int j = 0; j < P; j++)
        fscanf(pAr, "%d ", &bs[j]);
    for(int i = 0; i < M*N; i++){ //ordenamos en formato AoS
        for(int k = 0; k < P; k++){
            fscanf(pAr, "%f ", &pixels[cont]);
            cont++;
        }
    }

    int bsize = 256;
    int gsize = (int)ceil((float)M*N / bsize);
    cudaMalloc(&rsGPU, sizeof(int)*P);
    cudaMalloc(&gsGPU, sizeof(int)*P);
    cudaMalloc(&bsGPU, sizeof(int)*P);
    cudaMalloc(&pxs, sizeof(float)*P*M*N);
    cudaMemcpy(rsGPU, rs, P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gsGPU, gs, P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bsGPU, bs, P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pxs, pixels, P*M*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&Rout, M * N * sizeof(float));
	  cudaMalloc(&Gout, M * N * sizeof(float));
	  cudaMalloc(&Bout, M * N * sizeof(float));

    cudaEventCreate(&ct1);
	  cudaEventCreate(&ct2);
	  cudaEventRecord(ct1);
    B<<<gsize, bsize>>>(pxs, rsGPU, gsGPU, bsGPU, Rout, Gout, Bout, M, N, P);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    std::cout << "b) Tiempo GPU AoS: " << dt << "[ms]" << std::endl;
    
    float *Rhostout = new float[M*N];
    float *Ghostout = new float[M*N];
    float *Bhostout = new float[M*N];
    cudaMemcpy(Rhostout, Rout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ghostout, Gout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bhostout, Bout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    Write(Rhostout, Ghostout, Bhostout, M, N, "salidaB.txt");

    cudaFree(Rout); cudaFree(Gout); cudaFree(Bout);
    cudaFree(rsGPU); cudaFree(gsGPU); cudaFree(bsGPU); cudaFree(pxs);
    free(rs); free(gs); free(bs);
    fclose(pAr);
    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;


    //inciso C
    
    pAr = fopen("img.txt", "r");
    fscanf(pAr, "%d %d %d\n", &M, &N, &P);
    rs = (int *)malloc(P*sizeof(int));
    gs = (int *)malloc(P*sizeof(int));
    bs = (int *)malloc(P*sizeof(int));
    float *pixelsC = (float *)malloc(M*N*P*sizeof(float));

    for(int j = 0; j < P; j++)
        fscanf(pAr, "%d ", &rs[j]);
    for(int j = 0; j < P; j++)
        fscanf(pAr, "%d ", &gs[j]);
    for(int j = 0; j < P; j++)
        fscanf(pAr, "%d ", &bs[j]);
    //ordenamos en SoA
    cont = 0;
    for(int i = 0; i < P; i++){
        for(int j = 0; j < N*M; j++){
            pixelsC[cont] = pixels[P*j + i];
            cont++;
        }
    }

    bsize = 256;
    gsize = (int)ceil((float)M*N / bsize);
    cudaMalloc(&rsGPU, sizeof(int)*P);
    cudaMalloc(&gsGPU, sizeof(int)*P);
    cudaMalloc(&bsGPU, sizeof(int)*P);
    cudaMalloc(&pxs, sizeof(float)*P*M*N);
    cudaMemcpy(rsGPU, rs, P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gsGPU, gs, P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bsGPU, bs, P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pxs, pixelsC, P*M*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&Rout, M * N * sizeof(float));
	  cudaMalloc(&Gout, M * N * sizeof(float));
	  cudaMalloc(&Bout, M * N * sizeof(float));

    cudaEventCreate(&ct1);
	  cudaEventCreate(&ct2);
	  cudaEventRecord(ct1);
    C<<<gsize, bsize>>>(pxs, rsGPU, gsGPU, bsGPU, Rout, Gout, Bout, M, N, P);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    std::cout << "c) Tiempo GPU SoA: " << dt << "[ms]" << std::endl;
    
    Rhostout = new float[M*N];
    Ghostout = new float[M*N];
    Bhostout = new float[M*N];
    cudaMemcpy(Rhostout, Rout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ghostout, Gout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bhostout, Bout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    Write(Rhostout, Ghostout, Bhostout, M, N, "salidaC.txt");

    cudaFree(Rout); cudaFree(Gout); cudaFree(Bout);
    cudaFree(rsGPU); cudaFree(gsGPU); cudaFree(bsGPU); cudaFree(pxs);
    free(rs); free(gs); free(bs); free(pixelsC); free(pixels);
    fclose(pAr);
    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
    return 0;
}