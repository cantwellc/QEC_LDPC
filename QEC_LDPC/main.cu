#include <thrust/version.h>
#include <cusp/version.h>
#include <iostream>
#include <iomanip>
#include "QC_LDPC_CSS.h"
#include <random>
#include <chrono>

//// include the csr_matrix header file
//#include <cusp/csr_matrix.h>
//#include <cusp/print.h>
//int main()
//{
//	// allocate storage for (4,3) matrix with 4 nonzeros
//	cusp::csr_matrix<int, float, cusp::host_memory> A(4, 3, 6);
//	// initialize matrix entries on host
//	A.row_offsets[0] = 0;  // first offset is always zero
//	A.row_offsets[1] = 2;
//	A.row_offsets[2] = 2;
//	A.row_offsets[3] = 3;
//	A.row_offsets[4] = 6; // last offset is always num_entries
//	A.column_indices[0] = 0; A.values[0] = 10;
//	A.column_indices[1] = 2; A.values[1] = 20;
//	A.column_indices[2] = 2; A.values[2] = 30;
//	A.column_indices[3] = 0; A.values[3] = 40;
//	A.column_indices[4] = 1; A.values[4] = 50;
//	A.column_indices[5] = 2; A.values[5] = 60;
//	// A now represents the following matrix
//	//    [10  0 20]
//	//    [ 0  0  0]
//	//    [ 0  0 30]
//	//    [40 50 60]
//	// copy to the device
//	cusp::csr_matrix<int, float, cusp::device_memory> B(A);
//	cusp::print(B);
//}

void WriteVectorToFile(const std::vector<int>& vec, const char* str)
{
    std::ofstream file;
    file.open(str, std::ios::app);
    if (file.is_open()) {
        std::cout << "Writing to file " << str << std::endl;
        for (auto i = 0; i < vec.size(); ++i)
        {
            auto v = vec[i];
            file << v << ",";
        }
        file << "\n";
        file.close();
    }
    else
    {
        std::cout << "Failed to open file " << str << std::endl;
    }
}

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
	
	int J = 3; // rows of Hc 
	int K = 3; // rows of Hd
	int L = 6; // cols 
	int P = 7;
	int sigma = 2;
	int tau = 3;

    QC_LDPC_CSS code(J, K, L, P, sigma, tau);
    

	// Given the code, generate N strings of weight W dephasing errors and attempt to decode.
	// Record success or failure.
	// generate a random string of Weight N errors, then decide whether it is x, y, or z
	int numVars = P*L;

    std::random_device rd; // random seed or mersene twister engine.  could use this exclusively, but mt is faster
    std::mt19937 mt(rd()); // engine to produce random number
    std::uniform_int_distribution<int> indexDist(0, numVars - 1); // distribution for rng of index where errror occurs
    std::uniform_int_distribution<int> errorDist(0, 2); // distribution for rng of error type. x=0, y=1, z=2

    std::vector<int> xErrors(numVars, 0);
    std::vector<int> zErrors(numVars, 0);

    std::vector<int> xDecodedErrors(numVars, 0);
    std::vector<int> zDecodedErrors(numVars, 0);

    int cpuConvergenceErrorX = 0;
    int cpuConvergenceErrorZ = 0;
    int cpuSyndromeErrorX = 0;
    int cpuSyndromeErrorZ = 0;
    int cpuLogicalErrorX = 0;
    int cpuLogicalErrorZ = 0;
    int cpuCorrected = 0;
    
    int cudaConvergenceErrorX = 0;
    int cudaConvergenceErrorZ = 0;
    int cudaSyndromeErrorX = 0;
    int cudaSyndromeErrorZ = 0;
    int cudaLogicalErrorX = 0;
    int cudaLogicalErrorZ = 0;
    int cudaCorrected = 0;

    int W = 3;
    int COUNT = 1000;
    int MAX_ITERATIONS = 100;

    std::chrono::microseconds cpuDuration(0);
    std::chrono::microseconds cudaDuration(0);

    for (int c = 0; c < COUNT; ++c) {
        // clear all containers
        std::fill(xErrors.begin(), xErrors.end(), 0);
        std::fill(zErrors.begin(), zErrors.end(), 0);

        std::fill(xDecodedErrors.begin(), xDecodedErrors.end(), 0);
        std::fill(zDecodedErrors.begin(), zDecodedErrors.end(), 0);

        // construct random error string
        for (int i = 0; i<W; ++i)
        {
            // chose the index where an error will occur.
            int index = indexDist(mt);
            // determine whether the error is x, y, or z.
            int error = errorDist(mt);
            // set the correct error bits
            if (error == 0 || error == 1) xErrors[index] = 1;
            if (error == 2 || error == 1) zErrors[index] = 1;
        }

        auto sx = code.GetXSyndrome(xErrors);
        auto sz = code.GetZSyndrome(zErrors);

        //// decode error
        //auto start = std::chrono::high_resolution_clock::now();
        //QC_LDPC_CSS::ErrorCode errorCode = code.DecodeCPU2(sx, sz, 0.02, xDecodedErrors, zDecodedErrors,MAX_ITERATIONS);
        //auto finish = std::chrono::high_resolution_clock::now();
        //cpuDuration += std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

        //// increment error or corrected counters
        //if (errorCode == QC_LDPC_CSS::CONVERGENCE_FAIL_X) cpuConvergenceErrorX++;
        //else if (errorCode == QC_LDPC_CSS::CONVERGENCE_FAIL_Z) cpuConvergenceErrorZ++;
        //else if (errorCode == QC_LDPC_CSS::CONVERGENCE_FAIL_XZ) {
        //    cpuConvergenceErrorX++;
        //    cpuConvergenceErrorZ++;
        //}
        //else if (errorCode == QC_LDPC_CSS::SYNDROME_FAIL_X) cpuSyndromeErrorX++;
        //else if (errorCode == QC_LDPC_CSS::SYNDROME_FAIL_Z) cpuSyndromeErrorZ++;
        //else if (errorCode == QC_LDPC_CSS::SYNDROME_FAIL_XZ) {
        //    cpuSyndromeErrorX++;
        //    cpuSyndromeErrorZ++;
        //}
        //else { // the decoder thinks it correctly decoded the error
        //       // check for logical errors
        //       // What is e'-e?
        //    std::vector<int> xDiff(numVars, 0);
        //    std::vector<int> zDiff(numVars, 0);

        //    for (int i = 0; i < numVars; ++i)
        //    {
        //        if (xErrors[i] != xDecodedErrors[i]) xDiff[i] = 1;
        //        if (zErrors[i] != zDecodedErrors[i]) zDiff[i] = 1;
        //    }

        //    // the decoded error string can differ from the original by a stabilizer and still be decodeable.
        //    // however, if it falls outside the sstabilizer group it is a logical error. for stabilizer elements
        //    // H e = 0.
        //    auto xDiffSyndrome = code.GetXSyndrome(xDiff);
        //    auto zDiffSyndrome = code.GetZSyndrome(zDiff);

        //    bool xLogicalError = false;
        //    bool zLogicalError = false;
        //    for (int i = 0; i < xDiffSyndrome.size(); ++i)
        //    {
        //        if (xDiffSyndrome[i] != 0) xLogicalError = true;
        //        if (zDiffSyndrome[i] != 0) zLogicalError = true;
        //    }
        //    if (xLogicalError) cpuLogicalErrorX++;
        //    if (zLogicalError) cpuLogicalErrorZ++;
        //    if (!xLogicalError && !zLogicalError) cpuCorrected++;
        //}

        auto start = std::chrono::high_resolution_clock::now();
        auto errorCode = code.DecodeCUDA(sx, sz, 0.02, xDecodedErrors, zDecodedErrors,MAX_ITERATIONS);
        auto finish = std::chrono::high_resolution_clock::now();
        cudaDuration += std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

        // increment error or corrected counters
        if (errorCode == QC_LDPC_CSS::CONVERGENCE_FAIL_X) cudaConvergenceErrorX++;
        else if (errorCode == QC_LDPC_CSS::CONVERGENCE_FAIL_Z) cudaConvergenceErrorZ++;
        else if (errorCode == QC_LDPC_CSS::CONVERGENCE_FAIL_XZ) {
            cudaConvergenceErrorX++;
            cudaConvergenceErrorZ++;
        }
        else if (errorCode == QC_LDPC_CSS::SYNDROME_FAIL_X) cudaSyndromeErrorX++;
        else if (errorCode == QC_LDPC_CSS::SYNDROME_FAIL_Z) cudaSyndromeErrorZ++;
        else if (errorCode == QC_LDPC_CSS::SYNDROME_FAIL_XZ) {
            cudaSyndromeErrorX++;
            cudaSyndromeErrorZ++;
        }
        else { // the decoder thinks it correctly decoded the error
            // check for logical errors
            // What is e'-e?
            std::vector<int> xDiff(numVars, 0);
            std::vector<int> zDiff(numVars, 0);

            for (int i = 0; i < numVars; ++i)
            {
                if (xErrors[i] != xDecodedErrors[i]) xDiff[i] = 1;
                if (zErrors[i] != zDecodedErrors[i]) zDiff[i] = 1;
            }

            // the decoded error string can differ from the original by a stabilizer and still be decodeable.
            // however, if it falls outside the sstabilizer group it is a logical error. for stabilizer elements
            // H e = 0.
            auto xDiffSyndrome = code.GetXSyndrome(xDiff);
            auto zDiffSyndrome = code.GetZSyndrome(zDiff);

            bool xLogicalError = false;
            bool zLogicalError = false;
            for (int i = 0; i < xDiffSyndrome.size(); ++i)
            {
                if (xDiffSyndrome[i] != 0) xLogicalError = true;
                if (zDiffSyndrome[i] != 0) zLogicalError = true;
            }
            if (xLogicalError) cudaLogicalErrorX++;
            if (zLogicalError) cudaLogicalErrorZ++;
            if (!xLogicalError && !zLogicalError) cudaCorrected++;
        }
    }

    std::cout << "CPU: " << cpuDuration.count() / COUNT << " micro-seconds per error.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cpuCorrected / (float)COUNT * 100.0f) << " % corrected. \n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cpuConvergenceErrorX / (float)COUNT * 100.0f) << " % X convergence errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cpuConvergenceErrorZ / (float)COUNT * 100.0f) << " % Z convergence errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cpuSyndromeErrorX / (float)COUNT * 100.0f) << " % X syndrome errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cpuSyndromeErrorZ / (float)COUNT * 100.0f) << " % Z syndrome errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cpuLogicalErrorX / (float)COUNT * 100.0f) << " % X logical errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cpuLogicalErrorZ / (float)COUNT * 100.0f) << " % Z logical errors.\n";
    
    std::cout << "CUDA: " << cudaDuration.count() / COUNT << " micro-seconds per error.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cudaCorrected / (float)COUNT * 100.0f) << " % corrected. \n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cudaConvergenceErrorX / (float)COUNT * 100.0f) << " % X convergence errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cudaConvergenceErrorZ / (float)COUNT * 100.0f) << " % Z convergence errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cudaSyndromeErrorX / (float)COUNT * 100.0f) << " % X syndrome errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cudaSyndromeErrorZ / (float)COUNT * 100.0f) << " % Z syndrome errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cudaLogicalErrorX / (float)COUNT * 100.0f) << " % X logical errors.\n";
    std::cout << std::fixed << std::setprecision(3) << static_cast<float>((float)cudaLogicalErrorZ / (float)COUNT * 100.0f) << " % Z logical errors.\n";
    // at this point the decoder has either failed to decode the error and was able to tell
    // i.e. convergence or syndrome errors, or it thinks it succeeded.  it is still possible
    // for the error to have been a logical error and the decoder does not know it failed.

    //WriteVectorToFile(xDecodedErrors, "results/xDecodedErrors.txt");
    //WriteVectorToFile(xDecodedErrors, "results/zDecodedErrors.txt");
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
