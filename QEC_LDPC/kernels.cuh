#pragma once

#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <cuda.h>

__global__ void buildHC_kernel(int J, int L, int P, int sigma, int invSigma, int tau, int *C);
__global__ void beliefPropogation(int *parityCheckMatrix, float* checkNodeBeliefs, float* varNodeEstimates);

__global__ void eqNodeUpdate_kernel(int* eqNodeVarIndices, float* checkNodeBeliefs, float* varNodeEstimates, int* syndrome, int numVars, int numEqs, int numVarsPerEq);
__global__ void varNodeUpdate_kernel(int* varNodeEqIndices, float* checkNodeBeliefs, float* varNodeEstimates, float errorProbability, bool last, int numVars, int numEqs, int numEqsPerVar);