#pragma once

#include "cuda_runtime.h"
//#include "assert.h"
//#include <curand.h>
//#include <curand_kernel.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <cuda.h>

__global__ void buildHC_kernel(int J, int L, int P, int sigma, int invSigma, int tau, int *C);

__global__ void beliefPropogation_kernel(float* eqNodes, float* varNodes, int* eqNodeVarIndices, int * varNodeEqIndices,
   int* syndrome, float errorProbability, int numVars, int numEqs, int numVarsPerEq, int numEqsPerVar, int maxIterations);

__global__ void eqNodeUpdate_kernel(int* eqNodeVarIndices, float* checkNodeBeliefs, float* varNodeEstimates, 
    int* syndrome, int numVars, int numEqs, int numVarsPerEq);

__global__ void varNodeUpdate_kernel(int* varNodeEqIndices, float* checkNodeBeliefs, float* varNodeEstimates, 
    float errorProbability, bool last, int numVars, int numEqs, int numEqsPerVar);

__global__ void varNodeUpdate2_kernel(int * varNodeEqIndices, float* eqNodes, float * varNodes,
    float errorProbability, bool last, int numVars, int numEqs, int numEqsPerVar);

__device__ bool checkConvergence(float* varNodes, int numVars, int numEqs, float high, float low);
//__device__ void InitUpdateArrays(float* eqNodes, float* varNodes, int n, int numEqs, int* pcm, float errorProbability);
//__device__ void GenerateError(int xErrors[], int zErrors[], int n, unsigned seed);

