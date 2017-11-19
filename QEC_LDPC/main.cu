#include <thrust/version.h>
#include <cusp/version.h>
#include <iostream>
#include <iomanip>
//#include "QC_LDPC_CSS.h"
#include <random>
#include <chrono>
#include <cuda_runtime_api.h>
#include "Quantum_LDPC_Code.h"
#include "DecoderCPU.h"
#include "ArrayOutput.h"
#include <omp.h>
#include "DecoderGPU.h"

int main(void)
{
	//	int cuda_major = CUDA_VERSION / 1000;
	//	int cuda_minor = (CUDA_VERSION % 1000) / 10;
	int thrust_major = THRUST_MAJOR_VERSION;
	int thrust_minor = THRUST_MINOR_VERSION;
	int cusp_major = CUSP_MAJOR_VERSION;
	int cusp_minor = CUSP_MINOR_VERSION;
	//	std::cout << "CUDA   v" << cuda_major << "." << cuda_minor << std::endl;
	std::cout << "Thrust v" << thrust_major << "." << thrust_minor << std::endl;
	std::cout << "Cusp   v" << cusp_major << "." << cusp_minor << std::endl;

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
        size_t mem_tot;
        size_t mem_free;
        cudaMemGetInfo(&mem_free, &mem_tot);
        std::cout << "Total Memory : " << mem_tot << std::endl;
        std::cout << "Free memory : " << mem_free << std::endl;
    }
    
    std::cout << std::endl;
	
	int J = 3; // rows of Hc 
	int K = 3; // rows of Hd
	int L = 6; // cols 
	int P = 7;
	int sigma = 2;
	int tau = 3;
    int j = 2;
    int k = 2;

    //QC_LDPC_CSS code(J, K, L, P, sigma, tau);
    Quantum_LDPC_Code code(J, K, L, P, sigma, tau, j, k);
    //DecoderGPU decoder(code);
    DecoderCPU decoder(code);

	// Given the code, generate N strings of weight W dephasing errors and attempt to decode.
	// Record success or failure.
	// generate a random string of Weight N errors, then decide whether it is x, y, or z
	int numVars = P*L;

    std::random_device rd; // random seed or mersene twister engine.  could use this exclusively, but mt is faster
    std::mt19937 mt(rd()); // engine to produce random number
    std::uniform_int_distribution<int> indexDist(0, numVars - 1); // distribution for rng of index where errror occurs
    std::uniform_int_distribution<int> errorDist(0, 2); // distribution for rng of error type. x=0, y=1, z=2

    IntArray1d_h xErrors(numVars, 0);
    IntArray1d_h zErrors(numVars, 0);

    IntArray1d_h xDecodedErrors(numVars, 0);
    IntArray1d_h zDecodedErrors(numVars, 0);

    int W = 1;
    int COUNT = 10000;
    int MAX_ITERATIONS = 1000;

    for(auto w = 1; w <= W; ++w)
    {
        std::stringstream fileName;
        fileName << "results/ResultsCPU_RELEASE_" << w << ".txt";
        std::cout << fileName.str() << std::endl;
        std::ofstream outFile;
        outFile.open(fileName.str());
        auto stats = decoder.GetStatistics(w, COUNT, 0.02f, MAX_ITERATIONS);
        outFile << stats;
        outFile.close();
    }
	return 0;
}

//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "QC_LDPC_CSS.h"
//#include <cusp\version.h>
//
//#include <stdio.h>
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
