#pragma once
#include "Decoder.h"
#include "ArrayOutput.h"
#include <omp.h>
#include "CodeStatistics.h"
#include <random>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class DecoderGPU :
    public Decoder
{
private:
    typedef std::vector<int> IntArray2d;
    typedef std::vector<int> IntArray1d;
    typedef std::vector<float> FloatArray2d;
    typedef std::vector<float*> FloatPtrArray2d;

    const int _numVars;
    const int _numEqsX;
    const int _numEqsZ;
    const int _numVarsPerEqX;
    const int _numVarsPerEqZ;
    const int _numEqsPerVarX;
    const int _numEqsPerVarZ;

    thrust::host_vector<int> _eqNodeVarIndicesX;
    thrust::device_vector<int> _eqNodeVarIndicesX_d;
    thrust::host_vector<int> _eqNodeVarIndicesZ;
    thrust::device_vector<int> _eqNodeVarIndicesZ_d;
    thrust::host_vector<int> _varNodeEqIndicesX;
    thrust::device_vector<int> _varNodeEqIndicesX_d;
    thrust::host_vector<int> _varNodeEqIndicesZ;
    thrust::device_vector<int> _varNodeEqIndicesZ_d;

    static void InitIndexArrays(thrust::host_vector<int>& eqNodeVarIndices, thrust::host_vector<int>& varNodeEqIndices,
        const int* pcm, int numEqs, int numVars)
    {
        // set device index matrices for var node and check node updates
        // each equation will include L variables.
        // each variable will be involved in J equations
        // loop over all check node equations in the parity check matrix for X errors    
        std::vector<std::vector<int>> eqVarIndices(numEqs, std::vector<int>());
        std::vector<std::vector<int>> vnEqIndices(numVars, std::vector<int>());
        // loop over all equations
        for (auto eqIdx = 0; eqIdx < numEqs; ++eqIdx)
        {
            // loop over all variables
            for (auto varIdx = 0; varIdx < numVars; ++varIdx)
            {
                auto pcmIdx = eqIdx * numVars + varIdx;
                // if the entry in the pcm is 1, this check node involves this variable.  set the index entry
                if (pcm[pcmIdx])
                {
                    eqVarIndices[eqIdx].push_back(varIdx);
                    vnEqIndices[varIdx].push_back(eqIdx);
                }
            }
        }
        // copy data into provided array containers
        auto index = 0;
        for (auto i = 0; i<eqVarIndices.size(); ++i)
        {
            for (auto j = 0; j<eqVarIndices[0].size(); ++j)
            {
                eqNodeVarIndices[index] = eqVarIndices[i][j];
                ++index;
            }
        }
        index = 0;
        for (auto i = 0; i<vnEqIndices.size(); ++i)
        {
            for (auto j = 0; j<vnEqIndices[0].size(); ++j)
            {
                varNodeEqIndices[index] = vnEqIndices[i][j];
                ++index;
            }
        }
    }

    static void InitVarNodes(FloatArray2d_h& varNodes, const IntArray2d_h& eqNodesVarIndices, float probability)
    {
        int numVarsPerEq = eqNodesVarIndices.num_cols;
        int numEqs = varNodes.num_cols;
        for (int eqIdx = 0; eqIdx<numEqs; ++eqIdx)
        {
            for (int j = 0; j<numVarsPerEq; ++j)
            {
                int idx = eqIdx * numVarsPerEq + j;
                int varIdx = eqNodesVarIndices.values[idx];
                int varNodeIdx = varIdx * numEqs + eqIdx;
                varNodes.values[varNodeIdx] = probability;
            }
        }
    }

    static bool CheckConvergence(const float* estimates, float high, float low, int numVars, int numEqs)
    {
        // loop over all estimates
        for (auto i = 0; i < numVars; ++i) {
            for (auto j = 0; j < numEqs; ++j) {
                int index = i * numEqs + j;
                //if (estimates.values[index] != 0.0f) {
                if (estimates[index] != 0.0f) {
                    // if any estimate is between the bounds we have failed to converge
                    if (estimates[index] > low && estimates[index] < high) return false;
                    //if (estimates.values[index] > low && estimates.values[index] < high) return false;
                }
            }
        }
        return true;
    }

public:

    DecoderGPU(Quantum_LDPC_Code code) : Decoder(code),
        _numVars(code.n), _numEqsX(code.numEqsX), _numEqsZ(code.numEqsZ),
        _numVarsPerEqX(code.L), _numVarsPerEqZ(code.L),
        _numEqsPerVarX(code.J), _numEqsPerVarZ(code.K),
        _eqNodeVarIndicesX(IntArray2d(_numEqsX*_numVarsPerEqX)), _eqNodeVarIndicesZ(IntArray2d(_numEqsZ*_numVarsPerEqZ)),
        _varNodeEqIndicesX(IntArray2d(_numVars*_numEqsPerVarX)), _varNodeEqIndicesZ(_numVars*_numEqsPerVarZ)
    {
        InitIndexArrays(_eqNodeVarIndicesX, _varNodeEqIndicesX, &code.pcmX.values[0], _numEqsX, _numVars);
        InitIndexArrays(_eqNodeVarIndicesZ, _varNodeEqIndicesZ, &code.pcmZ.values[0], _numEqsZ, _numVars);
        _varNodeEqIndicesX_d = _varNodeEqIndicesX;
        _varNodeEqIndicesZ_d = _varNodeEqIndicesZ;
        _eqNodeVarIndicesX_d = _eqNodeVarIndicesX;
        _eqNodeVarIndicesZ_d = _eqNodeVarIndicesZ;
    }

    ~DecoderGPU()
    {
    }

    ErrorCode Decode(const IntArray1d& syndromeX, const IntArray1d& syndromeZ, float errorProbability, int maxIterations,
        IntArray1d& outErrorsX, IntArray1d& outErrorsZ)
    {
        // We will first decode xErrors and then zErrors
        // An NxM parity check matrix H can be viewed as a bipartite graph with
        // N symbol nodes and M parity check nodes.  Each symbol node is connected
        // to ds parity-check nodes, and each parity-check node is connected to dc
        // symbol nodes.
        float p = 2.0f / 3.0f * errorProbability; // a priori probability for x/z OR y error
        float high = 0.99f;
        float low = 0.01f;

        // accumulate the error estimates into a single vector
        std::vector<int> finalEstimatesX(_numVars, 0);
        std::vector<int> finalEstimatesZ(_numVars, 0);

        // check for correct error decoding
        ErrorCode code = SUCCESS;
        // check convergence errors
        for (auto varIdx = 0; varIdx < _numVars; ++varIdx) {
            for (auto eqIdx = 0; eqIdx < _numEqsX; ++eqIdx) {
                int index = varIdx * _numEqsX + eqIdx;
                //if (_varNodesX[index] >= 0.5f) // best guess of error
                //{
                //    finalEstimatesX[varIdx] = 1;
                //    break;
                //}
            }
        }
        for (auto varIdx = 0; varIdx < _numVars; ++varIdx) {
            for (auto eqIdx = 0; eqIdx < _numEqsZ; ++eqIdx) {
                int index = varIdx * _numEqsZ + eqIdx;
                //if (_varNodesZ[index] >= 0.5f) // best guess of error
                //{
                //    finalEstimatesZ[varIdx] = 1;
                //    break;
                //}
            }
        }
        // check for convergence failure
        /*if (!CheckConvergence(&_varNodesX[0], high, low, _numVars, _numEqsX)) {
            code = code | CONVERGENCE_FAIL_X;
        }
        if (!CheckConvergence(&_varNodesZ[0], high, low, _numVars, _numEqsZ)) code = code | CONVERGENCE_FAIL_Z;*/
        // check syndrome errors
        auto xS = _code.GetSyndromeX(finalEstimatesX);
        if (!std::equal(syndromeX.begin(), syndromeX.end(), xS.begin())) { code = code | SYNDROME_FAIL_X; }

        auto zS = _code.GetSyndromeZ(finalEstimatesZ);
        if (!std::equal(syndromeZ.begin(), syndromeZ.end(), zS.begin())) { code = code | SYNDROME_FAIL_Z; }

        outErrorsX = finalEstimatesX;
        outErrorsZ = finalEstimatesZ;

        return code;
    }

    CodeStatistics GetStats(int errorWeight, int numErrors, float errorProbability, int maxIterations, int seed, thrust::host_vector<int>& xErrors, thrust::host_vector<int>& zErrors)
    {
        int convergenceFailX = 0;
        int convergenceFailZ = 0;
        int syndromeErrorX = 0;
        int syndromeErrorZ = 0;
        int logicalError = 0;
        int corrected = 0;
        int xErrorsTested = 0;
        int zErrorsTested = 0;

        int W = errorWeight;
        int COUNT = numErrors;
        int MAX_ITERATIONS = maxIterations;

        auto start = std::chrono::high_resolution_clock::now();

        // transfer error arrays to device
        thrust::device_vector<int> xErrors_d(xErrors);
        thrust::device_vector<int> zErrors_d(zErrors);

        // set up calculation arrays on device
        thrust::device_vector<float> eqNodesX_d(_numEqsX*_numVars, 0.0f);
        thrust::device_vector<float> eqNodesZ_d(_numEqsZ*_numVars, 0.0f);
        thrust::device_vector<float> varNodesX_d(_numEqsX*_numVars, 0.0f);
        thrust::device_vector<float> varNodesZ_d(_numEqsZ*_numVars, 0.0f);

        // start kernel to collect statistics

        
        auto finish = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
        CodeStatistics stats = { _code,seed,COUNT, xErrorsTested, zErrorsTested, W,corrected,syndromeErrorX,syndromeErrorZ,logicalError,
            convergenceFailX,convergenceFailZ, duration };
        return stats;
    }

    CodeStatistics GetStatistics(int errorWeight, int numErrors, float errorProbability, int maxIterations, unsigned int seed) override
    {
        DecoderGPU decoder(_code);
        std::mt19937 mt(seed); // engine to produce random number
        std::uniform_int_distribution<int> indexDist(0, _code.n - 1); // distribution for rng of index where errror occurs
        std::uniform_int_distribution<int> errorDist(0, 2); // distribution for rng of error type. x=0, y=1, z=2

        int convergenceFailX = 0;
        int convergenceFailZ = 0;
        int syndromeErrorX = 0;
        int syndromeErrorZ = 0;
        int logicalError = 0;
        int corrected = 0;
        int xErrorsTested = 0;
        int zErrorsTested = 0;

        int W = errorWeight;
        int COUNT = numErrors;
        int MAX_ITERATIONS = maxIterations;

        auto start = std::chrono::high_resolution_clock::now();

        // create a host and device array to hold all errors generated a-priori.
        thrust::host_vector<int> xErrors_h(COUNT*_code.n);
        thrust::host_vector<int> zErrors_h(COUNT*_code.n);

        for (int c = 0; c < COUNT; ++c) {
            // clear all containers
            thrust::fill(xErrors_h.begin(), xErrors_h.end(), 0);
            thrust::fill(zErrors_h.begin(), zErrors_h.end(), 0);
            for (int i = 0; i < W; ++i) {
                // chose the index where an error will occur.
                int index = indexDist(mt);
                int index1d = c*_code.n + index;
                // determine whether the error is x, y, or z.
                int error = errorDist(mt);
                // set the correct error bits
                if (error == 0 || error == 1) xErrors_h[index] = 1;
                if (error == 2 || error == 1) zErrors_h[index] = 1;
            }
        }
        return decoder.GetStats(errorWeight, numErrors, errorProbability, maxIterations, seed, xErrors_h,zErrors_h);

    }

    CodeStatistics GetStatistics(int errorWeight, int numErrors, float errorProbability,
        int maxIterations) override {
        std::random_device rd; // random seed for mersene twister engine.  could use this exclusively, but mt is faster
        unsigned int seed = rd();
        return GetStatistics(errorWeight, numErrors, errorProbability, maxIterations, seed);
    }
};

//#pragma once
//#include "Decoder.h"
//#include "ArrayOutput.h"
//#include <omp.h>
//#include "CodeStatistics.h"
//#include <chrono>
//#include <random>
//#include "kernels.cuh"
//#include "cuda_runtime.h"
//#include "thrust/device_vector.h"
//#include <cusp/csr_matrix.h>
//#include <cusp/array2d.h>
//#include <cusp/array1d.h>
//#include <math.h>
//#include <cusp/print.h>
//#include <cuda.h>
//#include <fstream>
//#include <iostream>
//#include <iomanip>
//#include <cusp/iterator/random_iterator.h>
//#include <cusp/iterator/random_iterator.h>
//#include <cusp/iterator/random_iterator.h>
//#include "RandomErrorGenerator.h"
//#include <cusp/iterator/random_iterator.h>
//#include "HostDeviceArray.h"
//
//class DecoderGPU :
//    public Decoder
//{
//private:
//    IntArray2d_h _eqNodeVarIndicesX_h;
//    IntArray2d_h _eqNodeVarIndicesZ_h;
//    IntArray2d_h _varNodeEqIndicesX_h;
//    IntArray2d_h _varNodeEqIndicesZ_h;
//
//    FloatArray2d_h _eqNodesX_h;
//    FloatArray2d_h _eqNodesZ_h;
//    FloatArray2d_h _varNodesX_h;
//    FloatArray2d_h _varNodesZ_h;
//
//    IntArray2d_d _eqNodeVarIndicesX_d;
//    IntArray2d_d _eqNodeVarIndicesZ_d;
//    IntArray2d_d _varNodeEqIndicesX_d;
//    IntArray2d_d _varNodeEqIndicesZ_d;
//
//    FloatArray2d_d _eqNodesX_d;
//    FloatArray2d_d _eqNodesZ_d;
//    FloatArray2d_d _varNodesX_d;
//    FloatArray2d_d _varNodesZ_d;
//
//    int* _eqNodeVarIndicesX_d_ptr;
//    int* _eqNodeVarIndicesZ_d_ptr;
//    int* _varNodeEqIndicesX_d_ptr;
//    int* _varNodeEqIndicesZ_d_ptr;
//
//    float* _eqNodesX_d_ptr;
//    float* _eqNodesZ_d_ptr;
//    float* _varNodesX_d_ptr;
//    float* _varNodesZ_d_ptr;
//
//
//    static void InitIndexArrays(IntArray2d_h& eqNodeVarIndices, IntArray2d_h& varNodeEqIndices, const IntArray2d_h& pcm)
//    {
//        // set device index matrices for var node and check node updates
//        // each equation will include L variables.
//        // each variable will be involved in J equations
//        // loop over all check node equations in the parity check matrix for X errors    
//        int numEqs = pcm.num_rows;
//        int numVars = pcm.num_cols;
//        std::vector<std::vector<int>> cnVarIndices(numEqs, std::vector<int>());
//        std::vector<std::vector<int>> vnEqIndices(numVars, std::vector<int>());
//        // loop over all equations
//        for (auto eqIdx = 0; eqIdx < numEqs; ++eqIdx)
//        {
//            // loop over all variables
//            for (auto varIdx = 0; varIdx < numVars; ++varIdx)
//            {
//                auto pcmIdx = eqIdx * numVars + varIdx;
//                // if the entry in the pcm is 1, this check node involves this variable.  set the index entry
//                if (pcm.values[pcmIdx])
//                {
//                    cnVarIndices[eqIdx].push_back(varIdx);
//                    vnEqIndices[varIdx].push_back(eqIdx);
//                }
//            }
//        }
//        // copy data into provided array containers
//        auto index = 0;
//        for (auto i = 0; i<cnVarIndices.size(); ++i)
//        {
//            for (auto j = 0; j<cnVarIndices[0].size(); ++j)
//            {
//                eqNodeVarIndices.values[index] = cnVarIndices[i][j];
//                ++index;
//            }
//        }
//        index = 0;
//        for (auto i = 0; i<vnEqIndices.size(); ++i)
//        {
//            for (auto j = 0; j<vnEqIndices[0].size(); ++j)
//            {
//                varNodeEqIndices.values[index] = vnEqIndices[i][j];
//                ++index;
//            }
//        }
//    }
//

//
//    static void EqNodeUpdate(FloatArray2d_h &eqNodes, const FloatArray2d_h& varNodes, const IntArray2d_h& eqNodeVarIndices, IntArray1d_h syndrome)
//    {
//        // For a check node interested in variables a,b,c,d to estimate the updated probability for variable a
//        // syndrome = 0: even # of errors -> pa' = pb(1-pc)(1-pd) + pc(1-pb)(1-pd) + pd(1-pb)(1-pc) + pb*pc*pd
//        //                                       = 0.5 * (1 - (1-2pb)(1-2pc)(1-2pd))
//        // syndrome = 1: odd # of errors -> pa' = (1-pb)(1-pc)(1-pd) + pb*pc*(1-pd) + pb*(1-pc)*pd + (1-pb)*pc*pd
//        //                                      = 0.5 * (1 + (1-2pb)(1-2pc)(1-2pd))
//        int numEqs = eqNodes.num_rows;
//        int numVarsPerEq = eqNodeVarIndices.num_cols;
//        int numVars = varNodes.num_rows;
//#pragma omp parallel for
//        for (auto eqIdx = 0; eqIdx < numEqs; ++eqIdx) // loop over check nodes (parity equations)
//        {
//            auto firstVarIdx = eqIdx*numVarsPerEq;
//            // loop over variables to be updated for this check node
//            for (auto i = 0; i < numVarsPerEq; ++i)
//            {
//                auto index = firstVarIdx + i; // 1d array index to look up the variable index
//                auto varIdx = eqNodeVarIndices.values[index]; // variable index under investigation for this eq
//                auto product = 1.0f; // reset product
//                                     // loop over all other variables in the equation, accumulate (1-2p) terms
//                for (auto k = 0; k < numVarsPerEq; ++k)
//                {
//                    if (k == i) continue; // skip the variable being updated
//                    auto otherIndex = firstVarIdx + k; // 1d array index to look up the variable index
//                    auto otherVarIdx = eqNodeVarIndices.values[otherIndex];
//
//                    // the index holding the estimate beinng used for this eq
//                    auto varNodesIndex = otherVarIdx * numEqs + eqIdx;
//                    auto value = varNodes.values[varNodesIndex]; // belief value for this variable and this eq
//                    product *= (1.0f - 2.0f*value);
//                }
//                auto cnIdx = eqIdx * numVars + varIdx; // index for value within the check node array to update
//                if (syndrome[eqIdx]) {
//                    eqNodes.values[cnIdx] = 0.5 * (1.0f + product); // syndrome = 1 -> odd parity
//                }
//                else {
//                    eqNodes.values[cnIdx] = 0.5f * (1.0f - product); // syndrome = 0 -> even parity
//                }
//            }
//        }
//    }
//
//    static void VarNodeUpdate(const FloatArray2d_h& eqNodes, FloatArray2d_h& varNodes, const IntArray2d_h& varNodeEqIndices,
//        float errorProbability, bool last)
//    {
//        // For a variable node connected to check nodes 1,2,3,4 use the following formula to send an estimate to var node 1
//        // p1' = K*pch*p2*p3*p4   (pch is the channel error probability. ignore the estimate received from check node 1 unless last)
//        // where K = 1/[(1-pch)(1-p2)(1-p3)(1-p4)... + pch*p2*p3*p4...]
//        int numEqs = eqNodes.num_rows;
//        int numVars = varNodes.num_rows;
//        int numEqsPerVar = varNodeEqIndices.num_cols;
//
//#pragma omp parallel for
//        for (auto varIdx = 0; varIdx < numVars; ++varIdx) // loop over all variables
//        {
//            auto firstVarNode = varIdx * numEqs; // start of entries in VarNodes array for this variable
//            auto firstEqIndices = varIdx * numEqsPerVar; // starting point for first equation in the index list for this var.
//            for (auto j = 0; j < numEqsPerVar; ++j) // loop over all equations for this variable
//            {
//                // find the index of the equation estimate being updated
//                auto index = firstEqIndices + j;
//                auto eqIdx = varNodeEqIndices.values[index];
//
//                // 1d index for var nodes entry being updated
//                auto varNodesIdx = firstVarNode + eqIdx;
//
//                // start with a priori channel error probability
//                auto prodP = errorProbability;
//                auto prodOneMinusP = 1.0f - errorProbability;
//
//                // calculate the updated probability for this check node based on belief estimates of all OTHER check nodes
//                for (auto k = 0; k < numEqsPerVar; ++k)
//                {
//                    auto index2 = firstEqIndices + k; // 1d index for entry in the index array
//                    auto otherEQIdx = varNodeEqIndices.values[index2];
//
//                    if (otherEQIdx == eqIdx && !last) continue;
//                    // 1d index for check nodes belief being used
//                    auto checkNodesIdx = otherEQIdx * numVars + varIdx;
//                    auto p = eqNodes.values[checkNodesIdx];
//
//                    prodOneMinusP *= (1.0f - p);
//                    prodP *= p;
//                }
//                auto value = prodP / (prodOneMinusP + prodP);
//                varNodes.values[varNodesIdx] = value;
//            }
//        }
//    }
//
//    static bool CheckConvergence(const FloatArray2d_h& estimates, float high, float low)
//    {
//        // loop over all estimates
//        for (auto i = 0; i < estimates.num_rows; ++i) {
//            for (auto j = 0; j < estimates.num_cols; ++j) {
//                int index = i * estimates.num_cols + j;
//                if (estimates.values[index] != 0.0f) {
//                    // if any estimate is between the bounds we have failed to converge
//                    if (estimates.values[index] > low && estimates.values[index] < high) return false;
//                }
//            }
//        }
//        return true;
//    }
//
//    // helper function for decoding x or z errors individually
//    void BeliefPropogation(IntArray1d_h syndrome, float errorProbability, int maxIterations,
//        FloatArray2d_h& varNodes_h, FloatArray2d_h& eqNodes_h,
//        FloatArray2d_d& varNodes_d, FloatArray2d_d& eqNodes_d,
//        IntArray2d_h& eqNodeVarIndices_h, IntArray2d_h& varNodeEqIndices_h,
//        IntArray2d_d& eqNodeVarIndices_d, IntArray2d_d& varNodeEqIndices_d)
//    {
//        // We will first decode xErrors and then zErrors
//        // An NxM parity check matrix H can be viewed as a bipartite graph with
//        // N symbol nodes and M parity check nodes.  Each symbol node is connected
//        // to ds parity-check nodes, and each parity-check node is connected to dc
//        // symbol nodes.
//        float p = 2.0f / 3.0f * errorProbability; // a priori probability for x/z OR y error
//        float high = 0.99f;
//        float low = 0.01f;
//
//        thrust::fill(varNodes_h.values.begin(), varNodes_h.values.end(), 0.0f);
//        InitVarNodes(varNodes_h, eqNodeVarIndices_h, p);
//        thrust::copy(varNodes_h.values.begin(), varNodes_h.values.end(), varNodes_d.values.end());
//
//        IntArray1d_d syndrome_d(syndrome);
//        int* syndrome_ptr = thrust::raw_pointer_cast(&syndrome_d[0]);
//
//        float* eqNode_ptr = thrust::raw_pointer_cast(&eqNodes_d.values[0]);
//        float* varNode_ptr = thrust::raw_pointer_cast(&varNodes_d.values[0]);
//        int* eqNodeVarIndices_ptr = thrust::raw_pointer_cast(&eqNodeVarIndices_d.values[0]);
//        int* varNodeEqIndices_ptr = thrust::raw_pointer_cast(&varNodeEqIndices_d.values[0]);
//        auto numVars = varNodes_h.num_rows;
//        auto numEqs = eqNodes_h.num_rows;
//        auto numVarsPerEq = eqNodeVarIndices_h.num_cols;
//        auto numEqsPerVar = varNodeEqIndices_h.num_cols;
//
//        auto N = maxIterations; // maximum number of iterations
//        beliefPropogation_kernel <<<1, 1 >>> (eqNode_ptr, varNode_ptr, eqNodeVarIndices_ptr, varNodeEqIndices_ptr,
//                    syndrome_ptr, p, numVars, numEqs, numVarsPerEq, numEqsPerVar, N);
//        //bool converge = false;
//
//        //for (auto n = 0; n < N; n++)
//        //{
//        //    if (converge) break;
//        //    EqNodeUpdate(en1, varNodes_h, eqNodeVarIndices_h, syndrome);
//        //    VarNodeUpdate(en1, varNodes_h, varNodeEqIndices, p, n == N - 1);
//
//
//        //    if (n % 10 == 0)
//        //    {
//        //        converge = CheckConvergence(varNodes_h, high, low);
//        //    }
//        //}
//    }
//
//public:
//
//    DecoderGPU(Quantum_LDPC_Code code) : Decoder(code),
//        _eqNodeVarIndicesX_h(code.numEqsX, code.L), _eqNodeVarIndicesZ_h(code.numEqsZ, code.L),
//        _varNodeEqIndicesX_h(code.n, code.numEqsX/code.P), _varNodeEqIndicesZ_h(code.n, code.numEqsZ/code.P),
//        _eqNodesX_h(code.numEqsX, code.n, 0.0f), _eqNodesZ_h(code.numEqsZ, code.n, 0.0f),
//        _varNodesX_h(code.n, code.numEqsX, 0.0f), _varNodesZ_h(code.n, code.numEqsZ, 0.0f),
//        _eqNodeVarIndicesX_d(code.numEqsX, code.L), _eqNodeVarIndicesZ_d(code.numEqsZ, code.L),
//        _varNodeEqIndicesX_d(code.n, code.numEqsX / code.P), _varNodeEqIndicesZ_d(code.n, code.numEqsZ / code.P),
//        _eqNodesX_d(code.numEqsX, code.n, 0.0f), _eqNodesZ_d(code.numEqsZ, code.n, 0.0f),
//        _varNodesX_d(code.n, code.numEqsX, 0.0f), _varNodesZ_d(code.n, code.numEqsZ, 0.0f)
//    {
//        InitIndexArrays(_eqNodeVarIndicesX_h, _varNodeEqIndicesX_h, code.pcmX);
//        thrust::copy(_varNodeEqIndicesX_h.values.begin(), _varNodeEqIndicesX_h.values.end(), _varNodeEqIndicesX_d.values.begin());
//        thrust::copy(_eqNodeVarIndicesX_h.values.begin(), _eqNodeVarIndicesX_h.values.end(), _eqNodeVarIndicesX_d.values.begin());
//        
//        InitIndexArrays(_eqNodeVarIndicesZ_h, _varNodeEqIndicesZ_h, code.pcmZ);
//        thrust::copy(_varNodeEqIndicesZ_h.values.begin(), _varNodeEqIndicesZ_h.values.end(), _varNodeEqIndicesZ_d.values.begin());
//        thrust::copy(_eqNodeVarIndicesZ_h.values.begin(), _eqNodeVarIndicesZ_h.values.end(), _eqNodeVarIndicesZ_d.values.begin());
//
//        _eqNodeVarIndicesX_d_ptr = thrust::raw_pointer_cast(&_eqNodeVarIndicesX_d.values[0]);
//        _varNodeEqIndicesX_d_ptr = thrust::raw_pointer_cast(&_varNodeEqIndicesX_d.values[0]);
//        _eqNodeVarIndicesZ_d_ptr = thrust::raw_pointer_cast(&_eqNodeVarIndicesZ_d.values[0]);
//        _varNodeEqIndicesZ_d_ptr = thrust::raw_pointer_cast(&_varNodeEqIndicesZ_d.values[0]);
//
//        _eqNodesX_d_ptr = thrust::raw_pointer_cast(&_eqNodesX_d.values[0]);
//        _varNodesX_d_ptr = thrust::raw_pointer_cast(&_varNodesX_d.values[0]);
//        _eqNodesZ_d_ptr = thrust::raw_pointer_cast(&_eqNodesZ_d.values[0]);
//        _varNodesZ_d_ptr = thrust::raw_pointer_cast(&_varNodesZ_d.values[0]);
//    }
//
//    ~DecoderGPU()
//    {
//    }
//
//    ErrorCode Decode(const IntArray1d_h& syndromeX, const IntArray1d_h& syndromeZ, float errorProbability, int maxIterations,
//        IntArray1d_h& outErrorsX, IntArray1d_h& outErrorsZ) override
//    {
//        // We will first decode xErrors and then zErrors
//        // An NxM parity check matrix H can be viewed as a bipartite graph with
//        // N symbol nodes and M parity check nodes.  Each symbol node is connected
//        // to ds parity-check nodes, and each parity-check node is connected to dc
//        // symbol nodes.
//        float p = 2.0f / 3.0f * errorProbability; // a priori probability for x/z OR y error
//        float high = 0.99f;
//        float low = 0.01f;
//
//        IntArray1d_d syndromeX_d(syndromeX);
//        int* syndromeX_d_ptr = thrust::raw_pointer_cast(&syndromeX_d[0]);
//
//        IntArray1d_d syndromeZ_d(syndromeX);
//        int* syndromeZ_d_ptr = thrust::raw_pointer_cast(&syndromeX_d[0]);
//
//        thrust::fill(_varNodesX_h.values.begin(), _varNodesX_h.values.end(), 0.0f);
//        InitVarNodes(_varNodesX_h, _eqNodeVarIndicesX_h, p);
//        thrust::copy(_varNodesX_h.values.begin(), _varNodesX_h.values.end(), _varNodesX_d.values.begin());
//
//
//        thrust::fill(_varNodesZ_h.values.begin(), _varNodesZ_h.values.end(), 0.0f);
//        InitVarNodes(_varNodesZ_h, _eqNodeVarIndicesZ_h, p);
//        thrust::copy(_varNodesZ_h.values.begin(), _varNodesZ_h.values.end(), _varNodesZ_d.values.begin());
//
//        cudaDeviceSynchronize();
//        WriteToFile(_varNodesZ_h, "results/varNodesZ.txt");
//        WriteToFile(_varNodesX_h, "results/varNodesX.txt");
//
//        beliefPropogation_kernel <<<1, 1 >>> (_eqNodesX_d_ptr, _varNodesX_d_ptr, _eqNodeVarIndicesX_d_ptr, _varNodeEqIndicesX_d_ptr,
//            syndromeX_d_ptr, p, _code.n, _code.numEqsX, _eqNodeVarIndicesX_h.num_cols, _varNodeEqIndicesX_h.num_cols, maxIterations);
//
//        beliefPropogation_kernel <<<1, 1 >>> (_eqNodesZ_d_ptr, _varNodesZ_d_ptr, _eqNodeVarIndicesZ_d_ptr, _varNodeEqIndicesZ_d_ptr,
//            syndromeZ_d_ptr, p, _code.n, _code.numEqsZ, _eqNodeVarIndicesZ_h.num_cols, _varNodeEqIndicesZ_h.num_cols, maxIterations);
//
//        cudaDeviceSynchronize();
//
//        thrust::copy(_varNodesX_d.values.begin(), _varNodesX_d.values.end(), _varNodesX_h.values.begin());
//        thrust::copy(_varNodesZ_d.values.begin(), _varNodesZ_d.values.end(), _varNodesZ_h.values.begin());
//
//        WriteToFile(_varNodesZ_h, "results/varNodesZ.txt");
//        WriteToFile(_varNodesX_h, "results/varNodesX.txt");
//
//        thrust::copy(_eqNodesX_d.values.begin(), _eqNodesX_d.values.end(), _eqNodesX_h.values.begin());
//        thrust::copy(_eqNodesZ_d.values.begin(), _eqNodesZ_d.values.end(), _eqNodesZ_h.values.begin());
//        WriteToFile(_eqNodesZ_h, "results/eqNodesZ.txt");
//        WriteToFile(_eqNodesX_h, "results/eqNodesX.txt");
//
//        // accumulate the error estimates into a single vector
//        std::vector<int> finalEstimatesX(_varNodesX_h.num_rows, 0);
//        std::vector<int> finalEstimatesZ(_varNodesZ_h.num_rows, 0);
//
//        // check for correct error decoding
//        ErrorCode code = SUCCESS;
//        // check convergence errors
//        for (auto varIdx = 0; varIdx < _varNodesX_h.num_rows; ++varIdx) {
//            for (auto eqIdx = 0; eqIdx < _varNodesX_h.num_cols; ++eqIdx) {
//                int index = varIdx * _varNodesX_h.num_cols + eqIdx;
//                if (_varNodesX_h.values[index] >= 0.5f) // best guess of error
//                {
//                    finalEstimatesX[varIdx] = 1;
//                    break;
//                }
//            }
//        }
//        for (auto varIdx = 0; varIdx < _varNodesZ_h.num_rows; ++varIdx) {
//            for (auto eqIdx = 0; eqIdx < _varNodesZ_h.num_cols; ++eqIdx) {
//                int index = varIdx * _varNodesZ_h.num_cols + eqIdx;
//                if (_varNodesZ_h.values[index] >= 0.5f) // best guess of error
//                {
//                    finalEstimatesZ[varIdx] = 1;
//                    break;
//                }
//            }
//        }
//        // check for convergence failure
//        if (!CheckConvergence(_varNodesX_h, high, low)) {
//            code = code | CONVERGENCE_FAIL_X;
//        }
//        if (!CheckConvergence(_varNodesZ_h, high, low)) code = code | CONVERGENCE_FAIL_Z;
//        // check syndrome errors
//        auto xS = _code.GetSyndromeX(finalEstimatesX);
//        if (!std::equal(syndromeX.begin(), syndromeX.end(), xS.begin())) { code = code | SYNDROME_FAIL_X; }
//
//        auto zS = _code.GetSyndromeZ(finalEstimatesZ);
//        if (!std::equal(syndromeZ.begin(), syndromeZ.end(), zS.begin())) { code = code | SYNDROME_FAIL_Z; }
//
//        outErrorsX = finalEstimatesX;
//        outErrorsZ = finalEstimatesZ;
//
//        return code;
//    }
//
//    CodeStatistics GetStatistics(int errorWeight, int numErrors, float errorProbability, 
//        int maxIterations) override {
//        std::random_device rd; // random seed for mersene twister engine.  could use this exclusively, but mt is faster
//        unsigned int seed = rd();
//        std::mt19937 mt(seed); // engine to produce random number
//        std::uniform_int_distribution<int> indexDist(0, _code.n - 1); // distribution for rng of index where errror occurs
//        std::uniform_int_distribution<int> errorDist(0, 2); // distribution for rng of error type. x=0, y=1, z=2
//
//        int convergenceFailX = 0;
//        int convergenceFailZ = 0;
//        int syndromeErrorX = 0;
//        int syndromeErrorZ = 0;
//        int logicalError = 0;
//        int corrected = 0;
//
//        int W = errorWeight;
//        int COUNT = numErrors;
//        int MAX_ITERATIONS = maxIterations;
//
//        omp_lock_t randLock;
//        omp_init_lock(&randLock);
//
//        int nThreads;
//        int count;
//
//        auto start = std::chrono::high_resolution_clock::now();
//
////#pragma omp parallel 
//        {
//            int tID = omp_get_thread_num();
//            if (tID == 0)
//            {
//                nThreads = omp_get_num_threads();
//                std::cout << "Thread count: " << nThreads << std::endl;
//                count = COUNT / nThreads;
//            }
////#pragma omp barrier
//
//            DecoderGPU decoder(_code);
//            IntArray1d_h xErrors(_code.n, 0);
//            IntArray1d_h zErrors(_code.n, 0);
//
//            IntArray1d_h xDecodedErrors(_code.n, 0);
//            IntArray1d_h zDecodedErrors(_code.n, 0);
//
//            for (int c = 0; c < count; ++c) {
//
//                // clear all containers
//                thrust::fill(xErrors.begin(), xErrors.end(), 0);
//                thrust::fill(zErrors.begin(), zErrors.end(), 0);
//                thrust::fill(xDecodedErrors.begin(), xDecodedErrors.end(), 0);
//                thrust::fill(zDecodedErrors.begin(), zDecodedErrors.end(), 0);
//
//                // construct random error string
//                // lock the thread here to ensure reproducable errors given a seed
//                omp_set_lock(&randLock);
//                for (int i = 0; i < W; ++i)
//                {
//                    // chose the index where an error will occur.
//                    int index = indexDist(mt);
//                    // determine whether the error is x, y, or z.
//                    int error = errorDist(mt);
//                    // set the correct error bits
//                    if (error == 0 || error == 1) xErrors[index] = 1;
//                    if (error == 2 || error == 1) zErrors[index] = 1;
//                }
//                omp_unset_lock(&randLock);
//
//                auto sx = _code.GetSyndromeX(xErrors);
//                auto sz = _code.GetSyndromeZ(zErrors);
//
//                auto errorCode = decoder.Decode(sx, sz, errorProbability, MAX_ITERATIONS, xDecodedErrors, zDecodedErrors);
//
//                // increment error or corrected counters
//                if (errorCode == Decoder::CONVERGENCE_FAIL_X) {
//#pragma omp atomic
//                    convergenceFailX++;
//                }
//                if (errorCode == Decoder::CONVERGENCE_FAIL_Z) {
//#pragma omp atomic
//                    convergenceFailZ++;
//                }
//
//                if (errorCode == Decoder::SYNDROME_FAIL_X) {
//#pragma omp atomic
//                    syndromeErrorX++;
//                }
//                if (errorCode == Decoder::SYNDROME_FAIL_Z) {
//#pragma omp atomic
//                    syndromeErrorZ++;
//                }
//
//                if (errorCode != Decoder::SYNDROME_FAIL_X && errorCode != Decoder::SYNDROME_FAIL_Z)
//                { // the decoder thinks it correctly decoded the error
//                  // check for logical errors
//                  // What is e'-e?
//                    IntArray1d_h xDiff(_code.n, 0);
//                    IntArray1d_h zDiff(_code.n, 0);
//
//                    bool xIsDiff = false;
//                    bool zIsDiff = false;
//                    for (int i = 0; i < _code.n; ++i)
//                    {
//                        if (xErrors[i] != xDecodedErrors[i]) {
//                            xDiff[i] = 1;
//                            xIsDiff = true;
//                        }
//                        if (zErrors[i] != zDecodedErrors[i]) {
//                            zIsDiff = true;
//                            zDiff[i] = 1;
//                        }
//                    }
//                    if (xIsDiff || zIsDiff) {
//                        // the decoded error string can differ from the original by a stabilizer and still be decodeable.
//                        // however, if it falls outside the sstabilizer group it is a logical error. for stabilizer elements
//                        // H e = 0.
//                        auto xDiffSyndrome = _code.GetSyndromeX(xDiff);
//                        auto zDiffSyndrome = _code.GetSyndromeZ(zDiff);
//
//                        bool xLogicalError = false;
//                        bool zLogicalError = false;
//                        for (int i = 0; i < xDiffSyndrome.size(); ++i)
//                        {
//                            if (xDiffSyndrome[i] != 0) xLogicalError = true;
//                            if (zDiffSyndrome[i] != 0) zLogicalError = true;
//                        }
//                        if (xLogicalError) {
//#pragma omp atomic
//                            logicalError++;
//                        }
//                        if (zLogicalError) {
//#pragma omp atomic
//                            logicalError++;
//                        }
//                        if (!xLogicalError && !zLogicalError) {
//#pragma omp atomic
//                            corrected++;
//                        }
//                    }
//                    else
//                    {
//#pragma omp atomic
//                        corrected++;
//                    }
//                }
//            }
//        }
//        auto finish = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
//        CodeStatistics stats = { _code,seed,count * nThreads,W,corrected,syndromeErrorX,syndromeErrorZ,logicalError,
//            convergenceFailX,convergenceFailZ, duration };
//        return stats;
//    }
//};
//
