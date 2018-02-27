#pragma once
#include "Decoder.h"
#include "ArrayOutput.h"
#include <omp.h>
#include "CodeStatistics.h"
#include <random>
#include <chrono>

class DecoderCPU :
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

    IntArray2d _eqNodeVarIndicesX;
    IntArray2d _eqNodeVarIndicesZ;
    IntArray2d _varNodeEqIndicesX;
    IntArray2d _varNodeEqIndicesZ;

    FloatArray2d _eqNodesX;
    FloatArray2d _eqNodesZ;
    FloatArray2d _varNodesX;
    FloatArray2d _varNodesZ;

    FloatPtrArray2d _eqNodeVarPtrsX;
    FloatPtrArray2d _eqNodeVarPtrsZ;
    FloatPtrArray2d _varNodeEqPtrsX;
    FloatPtrArray2d _varNodeEqPtrsZ;
    
    static void InitIndexArrays(IntArray2d& eqNodeVarIndices, IntArray2d& varNodeEqIndices, 
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

    static void InitNodePtrs(FloatPtrArray2d& eqNodeVarPtrs, FloatPtrArray2d& varNodeEqPtrs, 
        FloatArray2d& eqNodes, FloatArray2d& varNodes, const int* pcm, const int numEqs, const int numVars)
    {
        // set device index matrices for var node and check node updates
        // each equation will include L variables.
        // each variable will be involved in J equations
        // loop over all check node equations in the parity check matrix for X errors    
        std::vector<std::vector<int>> eqNodeVarIndices(numEqs, std::vector<int>());
        std::vector<std::vector<int>> varNodeEqIndices(numVars, std::vector<int>());
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
                    eqNodeVarIndices[eqIdx].push_back(varIdx);
                    varNodeEqIndices[varIdx].push_back(eqIdx);
                }
            }
        }
        // set pointers to relevant entries
        auto index = 0;
        for (auto eqIdx = 0; eqIdx<eqNodeVarIndices.size(); ++eqIdx)
        {
            for (auto j = 0; j<eqNodeVarIndices[0].size(); ++j)
            {
                auto varIdx = eqNodeVarIndices[eqIdx][j];
                auto varNodesIdx = varIdx * numEqs + eqIdx;
                eqNodeVarPtrs[index] = &varNodes[varNodesIdx];
                ++index;
            }
        }
        index = 0;
        for (auto varIdx = 0; varIdx<varNodeEqIndices.size(); ++varIdx)
        {
            for (auto j = 0; j<varNodeEqIndices[0].size(); ++j)
            {
                auto eqIdx = varNodeEqIndices[varIdx][j];
                auto eqNodesIdx = eqIdx * numVars + varIdx;
                varNodeEqPtrs[index] = &eqNodes[eqNodesIdx];
                ++index;
            }
        }
    }

    static void InitVarNodes(FloatArray2d& varNodes, const IntArray2d& eqNodesVarIndices, 
        float probability, int numVarsPerEq, int numEqs)
    {
        for (int eqIdx = 0; eqIdx<numEqs; ++eqIdx)
        {
            for (int j = 0; j<numVarsPerEq; ++j)
            {
                int idx = eqIdx * numVarsPerEq + j;
                int varIdx = eqNodesVarIndices[idx];
                int varNodeIdx = varIdx * numEqs + eqIdx;
                varNodes[varNodeIdx] = probability;
            }
        }
    }

    static void EqNodeUpdate(float* eqNodesPtr, FloatPtrArray2d& eqNodeVarPtrs, const int* eqNodeVarIndicesPtr,
        const int* syndromePtr, int numEqs, int numVars, int numVarsPerEq)
    {
        // For a check node interested in variables a,b,c,d to estimate the updated probability for variable a
        // syndrome = 0: even # of errors -> pa' = pb(1-pc)(1-pd) + pc(1-pb)(1-pd) + pd(1-pb)(1-pc) + pb*pc*pd
        //                                       = 0.5 * (1 - (1-2pb)(1-2pc)(1-2pd))
        // syndrome = 1: odd # of errors -> pa' = (1-pb)(1-pc)(1-pd) + pb*pc*(1-pd) + pb*(1-pc)*pd + (1-pb)*pc*pd
        //                                      = 0.5 * (1 + (1-2pb)(1-2pc)(1-2pd))

#pragma omp parallel for
        for (auto eqIdx = 0; eqIdx < numEqs; ++eqIdx) // loop over check nodes (parity equations)
        {
            auto firstVarIdx = eqIdx*numVarsPerEq;
            // loop over variables to be updated for this check node
            for (auto i = 0; i < numVarsPerEq; ++i)
            {
                auto index = firstVarIdx + i; // 1d array index to look up the variable index
                auto varIdx = eqNodeVarIndicesPtr[index]; // variable index under investigation for this eq
                auto product = 1.0f; // reset product
                                     // loop over all other variables in the equation, accumulate (1-2p) terms
                for (auto k = 0; k < numVarsPerEq; ++k)
                {
                    if (k == i) continue; // skip the variable being updated
                    auto otherIndex = firstVarIdx + k; // 1d array index to look up the variable index
                    float value = *eqNodeVarPtrs[otherIndex]; // belief value for this variable and this eq
                    product *= (1.0f - 2.0f*value);
                }
                auto eqNodeIdx = eqIdx * numVars + varIdx; // index for value within the check node array to update
                if (syndromePtr[eqIdx]) {
                    eqNodesPtr[eqNodeIdx] = 0.5 * (1.0f + product); // syndrome = 1 -> odd parity
                }
                else {
                    eqNodesPtr[eqNodeIdx] = 0.5f * (1.0f - product); // syndrome = 0 -> even parity
                }
            }
        }
    }

    static void VarNodeUpdate(float* varNodesPtr, FloatPtrArray2d& varNodeEqPtrs, const int* varNodeEqIndicesPtr, 
        float errorProbability, bool last, int numEqs, int numVars, int numEqsPerVar)
    {
        // For a variable node connected to check nodes 1,2,3,4 use the following formula to send an estimate to var node 1
        // p1' = K*pch*p2*p3*p4   (pch is the channel error probability. ignore the estimate received from check node 1 unless last)
        // where K = 1/[(1-pch)(1-p2)(1-p3)(1-p4)... + pch*p2*p3*p4...]
#pragma omp parallel for
        for (auto varIdx = 0; varIdx < numVars; ++varIdx) // loop over all variables
        {
            auto firstVarNode = varIdx * numEqs; // start of entries in VarNodes array for this variable
            auto firstEqIndices = varIdx * numEqsPerVar; // starting point for first equation in the index list for this var.
            for (auto j = 0; j < numEqsPerVar; ++j) // loop over all equations for this variable
            {
                // find the index of the equation estimate being updated
                auto index = firstEqIndices + j;
                auto eqIdx = varNodeEqIndicesPtr[index];
                //auto eqIdx = varNodeEqIndicesPtr.values[index];

                // 1d index for var nodes entry being updated
                auto varNodesIdx = firstVarNode + eqIdx;

                // start with a priori channel error probability
                auto prodP = errorProbability;
                auto prodOneMinusP = 1.0f - errorProbability;

                // calculate the updated probability for this check node based on belief estimates of all OTHER check nodes
                for (auto k = 0; k < numEqsPerVar; ++k)
                {
                    if (j == k && !last) continue;
                    // 1d index for entry in value ptr array
                    auto index2 = firstEqIndices + k; 
                    float p = *varNodeEqPtrs[index2];

                    prodOneMinusP *= (1.0f - p);
                    prodP *= p;
                }
                auto value = prodP / (prodOneMinusP + prodP);
                varNodesPtr[varNodesIdx] = value;
                //varNodesPtr.values[varNodesIdx] = value;
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

    // helper function for decoding x or z errors individually
    void BeliefPropogation(const IntArray1d& syndrome, float errorProbability, int maxIterations,
        FloatArray2d& varNodes, FloatArray2d& eqNodes, const IntArray2d& eqNodeVarIndices, const IntArray2d& varNodeEqIndices,
        FloatPtrArray2d& eqNodeVarPtrs, FloatPtrArray2d& varNodeEqPtrs,
        const int numVars, const int numEqs, const int numVarsPerEq, const int numEqsPerVar)
    {
        // We will first decode xErrors and then zErrors
        // An NxM parity check matrix H can be viewed as a bipartite graph with
        // N symbol nodes and M parity check nodes.  Each symbol node is connected
        // to ds parity-check nodes, and each parity-check node is connected to dc
        // symbol nodes.
        float p = 2.0f / 3.0f * errorProbability; // a priori probability for x/z OR y error
        float high = 0.99f;
        float low = 0.01f;

        int numElements = numVars*numEqs;
        // reset var nodes
        std::fill(varNodes.begin(), varNodes.end(), 0.0f);
        //thrust::fill(varNodesPtr.values.begin(), varNodesPtr.values.end(), 0.0f);
        InitVarNodes(varNodes, eqNodeVarIndices, p, numVarsPerEq,numEqs);

        auto eqNodesPtr = &eqNodes[0];
        auto varNodesPtr = &varNodes[0];
        auto eqNodeVarIndicesPtr = &eqNodeVarIndices[0];
        auto varNodeEqIndicesPtr = &varNodeEqIndices[0];
        auto syndromePtr = &syndrome[0];

        auto N = maxIterations; // maximum number of iterations
        bool converge= false;
        //int tid = omp_get_thread_num();
        //std::string vnFile = "results/varNodesTest_" + std::to_string(tid) + ".txt";
        //std::string enFile = "results/eqNodesTest_" + std::to_string(tid) + ".txt";
        for (auto n = 0; n < N; n++)
        {
            if (converge) break;
            EqNodeUpdate(eqNodesPtr, eqNodeVarPtrs, eqNodeVarIndicesPtr, syndromePtr,  numEqs, numVars, numVarsPerEq);
            VarNodeUpdate(varNodesPtr, varNodeEqPtrs, varNodeEqIndicesPtr, p, n == N - 1,numEqs,numVars,numEqsPerVar);
            //WriteToFile(varNodes, vnFile.c_str(), numVars, numEqs);
            //WriteToFile(eqNodes, enFile.c_str(), numVars, numEqs);
            if (n % 10 == 0)
            {
                converge = CheckConvergence(varNodesPtr, high, low, numVars, numEqs);
            }
        }
    }

public:

    DecoderCPU(Quantum_LDPC_Code code) : Decoder(code),
        _numVars(code.n), _numEqsX(code.numEqsX), _numEqsZ(code.numEqsZ), 
        _numVarsPerEqX(code.L), _numVarsPerEqZ(code.L),
        _numEqsPerVarX(code.J), _numEqsPerVarZ(code.K),
        _eqNodeVarIndicesX(IntArray2d(_numEqsX*_numVarsPerEqX)), _eqNodeVarIndicesZ(IntArray2d(_numEqsZ*_numVarsPerEqZ)),
        _varNodeEqIndicesX(IntArray2d(_numVars*_numEqsPerVarX)), _varNodeEqIndicesZ(_numVars*_numEqsPerVarZ),
        _eqNodesX(_numEqsX*_numVars, 0.0f), _eqNodesZ(_numEqsZ*_numVars, 0.0f),
        _varNodesX(_numEqsX*_numVars, 0.0f), _varNodesZ(_numEqsZ*_numVars, 0.0f),
        _eqNodeVarPtrsX(FloatPtrArray2d(_numEqsX*_numVarsPerEqX)), _eqNodeVarPtrsZ(FloatPtrArray2d(_numEqsZ*_numVarsPerEqZ)),
        _varNodeEqPtrsX(FloatPtrArray2d(_numVars*_numEqsPerVarX)), _varNodeEqPtrsZ(FloatPtrArray2d(_numVars*_numEqsPerVarZ))
    {
        InitIndexArrays(_eqNodeVarIndicesX, _varNodeEqIndicesX, &code.pcmX.values[0], _numEqsX, _numVars);
        InitIndexArrays(_eqNodeVarIndicesZ, _varNodeEqIndicesZ, &code.pcmZ.values[0], _numEqsZ, _numVars);
        InitNodePtrs(_eqNodeVarPtrsX, _varNodeEqPtrsX, _eqNodesX, _varNodesX, &code.pcmX.values[0], _numEqsX, _numVars);
        InitNodePtrs(_eqNodeVarPtrsZ, _varNodeEqPtrsZ, _eqNodesZ, _varNodesZ, &code.pcmZ.values[0], _numEqsZ, _numVars);
    }

    ~DecoderCPU()
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

#pragma omp parallel sections
        {
#pragma omp section
            {
                BeliefPropogation(syndromeX, errorProbability, maxIterations, 
                    _varNodesX, _eqNodesX, _eqNodeVarIndicesX, _varNodeEqIndicesX,
                    _eqNodeVarPtrsX, _varNodeEqPtrsX,
                    _numVars, _numEqsX, _numVarsPerEqX, _numEqsPerVarX);
            }
#pragma omp section
            {
                BeliefPropogation(syndromeZ, errorProbability, maxIterations, 
                    _varNodesZ,_eqNodesZ, _eqNodeVarIndicesZ, _varNodeEqIndicesZ,
                    _eqNodeVarPtrsZ, _varNodeEqPtrsZ,
                    _numVars, _numEqsZ, _numVarsPerEqZ, _numEqsPerVarZ);
            }
        }

        // accumulate the error estimates into a single vector
        std::vector<int> finalEstimatesX(_numVars, 0);
        std::vector<int> finalEstimatesZ(_numVars, 0);

        // check for correct error decoding
        ErrorCode code = SUCCESS;
        // check convergence errors
        for (auto varIdx = 0; varIdx < _numVars; ++varIdx) {
            for (auto eqIdx = 0; eqIdx < _numEqsX; ++eqIdx) {
                int index = varIdx * _numEqsX + eqIdx;
                if (_varNodesX[index] >= 0.5f) // best guess of error
                {
                    finalEstimatesX[varIdx] = 1;
                    break;
                }
            }
        }
        for (auto varIdx = 0; varIdx < _numVars; ++varIdx) {
            for (auto eqIdx = 0; eqIdx < _numEqsZ; ++eqIdx) {
                int index = varIdx * _numEqsZ + eqIdx;
                if (_varNodesZ[index] >= 0.5f) // best guess of error
                {
                    finalEstimatesZ[varIdx] = 1;
                    break;
                }
            }
        }
        // check for convergence failure
        if (!CheckConvergence(&_varNodesX[0], high, low,_numVars, _numEqsX)) {
            code = code | CONVERGENCE_FAIL_X;
        }
        if (!CheckConvergence(&_varNodesZ[0], high, low, _numVars, _numEqsZ)) code = code | CONVERGENCE_FAIL_Z;
        // check syndrome errors
        auto xS = _code.GetSyndromeX(finalEstimatesX);
        if (!std::equal(syndromeX.begin(), syndromeX.end(), xS.begin())) { code = code | SYNDROME_FAIL_X; }

        auto zS = _code.GetSyndromeZ(finalEstimatesZ);
        if (!std::equal(syndromeZ.begin(), syndromeZ.end(), zS.begin())) { code = code | SYNDROME_FAIL_Z; }

        outErrorsX = finalEstimatesX;
        outErrorsZ = finalEstimatesZ;

        return code;
    }

    CodeStatistics GetStatistics(int errorWeight, int numErrors, float errorProbability,
        int maxIterations, unsigned int seed) override {
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

        omp_lock_t randLock;
        omp_init_lock(&randLock);

        int nThreads;
        int count;

        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel 
        {
            int tID = omp_get_thread_num();
            if (tID == 0)
            {
                nThreads = omp_get_num_threads();
                std::cout << "Thread count: " << nThreads << std::endl;
                count = COUNT / nThreads;
            }
#pragma omp barrier


            DecoderCPU decoder(_code);
            IntArray1d xErrors(_code.n, 0);
            IntArray1d zErrors(_code.n, 0);

            IntArray1d xDecodedErrors(_code.n, 0);
            IntArray1d zDecodedErrors(_code.n, 0);

            for (int c = 0; c < count; ++c) {

                // clear all containers
                thrust::fill(xErrors.begin(), xErrors.end(), 0);
                thrust::fill(zErrors.begin(), zErrors.end(), 0);
                thrust::fill(xDecodedErrors.begin(), xDecodedErrors.end(), 0);
                thrust::fill(zDecodedErrors.begin(), zDecodedErrors.end(), 0);

                // construct random error string
                // lock the thread here to ensure reproducable errors given a seed
                omp_set_lock(&randLock);
                for (int i = 0; i < W; ++i)
                {
                    // chose the index where an error will occur.
                    int index = indexDist(mt);
                    // determine whether the error is x, y, or z.
                    int error = errorDist(mt);
                    // set the correct error bits
                    if (error == 0 || error == 1) xErrors[index] = 1;
                    if (error == 2 || error == 1) zErrors[index] = 1;
                }
                omp_unset_lock(&randLock);

                auto sx = _code.GetSyndromeX(xErrors);
                auto sz = _code.GetSyndromeZ(zErrors);
                //
                bool nxE = std::all_of(xErrors.begin(), xErrors.end(), [](int i) { return i == 0; });
                bool nzE = std::all_of(zErrors.begin(), zErrors.end(), [](int i) { return i == 0; });
                if (!nxE) {
#pragma omp atomic
                    xErrorsTested++;
                }
                if (!nzE) {
#pragma omp atomic
                    zErrorsTested++;
                }

                IntArray1d sx1(sx.begin(), sx.end());
                IntArray1d sz1(sz.begin(), sz.end());
                auto errorCode = decoder.Decode(sx1, sz1, errorProbability, MAX_ITERATIONS, xDecodedErrors, zDecodedErrors);

                // check for detected syndrome errors
                bool dEX = errorCode & Decoder::SYNDROME_FAIL_X;
                if (dEX) {
#pragma omp atomic
                    syndromeErrorX++;
                }
                bool dEZ = errorCode & Decoder::SYNDROME_FAIL_Z;
                if (dEZ) {
#pragma omp atomic
                    syndromeErrorZ++;
                }

                // check the decoded errors for a logical error
                if (!(dEX || dEZ)) {
                    // this wasn't a detected error, could be logical error
                    // apply correction. mod 2 addition of error with decoded error
                    IntArray1d errors(xErrors.size() + zErrors.size());
                    for (int i = 0; i < xErrors.size(); ++i)
                        errors[i] = (xErrors[i] + xDecodedErrors[i]) % 2;
                    for (int i = 0; i < zErrors.size(); ++i)
                        errors[xErrors.size() + i] = (zErrors[i] + zDecodedErrors[i]) % 2;
                    bool lE = _code.CheckLogicalError(errors);
                    if (lE) {
#pragma omp atomic
                        logicalError++;
                    }
                    else
                    {// this wasn't a detected error or logical error. must be corrected
#pragma omp atomic
                        corrected++;
                    }
                }

                // increment convergence failure counters. it can fail to converge
                // and still potentially produce a correct output.
                if (errorCode & Decoder::CONVERGENCE_FAIL_X) {
#pragma omp atomic
                    convergenceFailX++;
                }
                if (errorCode & Decoder::CONVERGENCE_FAIL_Z) {
#pragma omp atomic
                    convergenceFailZ++;
                }
            }
        }

        auto finish = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
        CodeStatistics stats = { _code,seed,count * nThreads, xErrorsTested, zErrorsTested, W,corrected,syndromeErrorX,syndromeErrorZ,logicalError,
            convergenceFailX,convergenceFailZ, duration };
        return stats;
    }

    CodeStatistics GetStatistics(int errorWeight, int numErrors, float errorProbability, 
        int maxIterations) override{
        std::random_device rd; // random seed for mersene twister engine.  could use this exclusively, but mt is faster
        unsigned int seed = rd();
        return GetStatistics(errorWeight, numErrors, errorProbability, maxIterations, seed);
    }
};

