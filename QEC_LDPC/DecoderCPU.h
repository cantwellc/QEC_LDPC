#pragma once
#include "Decoder.h"
#include "ArrayOutput.h"
#include <omp.h>
#include "CodeStatistics.h"

class DecoderCPU :
    public Decoder
{
private:
    IntArray2d_h _eqNodeVarIndicesX_h;
    IntArray2d_h _eqNodeVarIndicesZ_h;
    IntArray2d_h _varNodeEqIndicesX_h;
    IntArray2d_h _varNodeEqIndicesZ_h;

    FloatArray2d_h _eqNodesX_h;
    FloatArray2d_h _eqNodesZ_h;
    FloatArray2d_h _varNodesX_h;
    FloatArray2d_h _varNodesZ_h;

    static void InitIndexArrays(IntArray2d_h& eqNodeVarIndices, IntArray2d_h& varNodeEqIndices, const IntArray2d_h& pcm)
    {
        // set device index matrices for var node and check node updates
        // each equation will include L variables.
        // each variable will be involved in J equations
        // loop over all check node equations in the parity check matrix for X errors    
        int numEqs = pcm.num_rows;
        int numVars = pcm.num_cols;
        std::vector<std::vector<int>> cnVarIndices(numEqs, std::vector<int>());
        std::vector<std::vector<int>> vnEqIndices(numVars, std::vector<int>());
        // loop over all equations
        for (auto eqIdx = 0; eqIdx < numEqs; ++eqIdx)
        {
            // loop over all variables
            for (auto varIdx = 0; varIdx < numVars; ++varIdx)
            {
                auto pcmIdx = eqIdx * numVars + varIdx;
                // if the entry in the pcm is 1, this check node involves this variable.  set the index entry
                if (pcm.values[pcmIdx])
                {
                    cnVarIndices[eqIdx].push_back(varIdx);
                    vnEqIndices[varIdx].push_back(eqIdx);
                }
            }
        }
        // copy data into provided array containers
        auto index = 0;
        for (auto i = 0; i<cnVarIndices.size(); ++i)
        {
            for (auto j = 0; j<cnVarIndices[0].size(); ++j)
            {
                eqNodeVarIndices.values[index] = cnVarIndices[i][j];
                ++index;
            }
        }
        index = 0;
        for (auto i = 0; i<vnEqIndices.size(); ++i)
        {
            for (auto j = 0; j<vnEqIndices[0].size(); ++j)
            {
                varNodeEqIndices.values[index] = vnEqIndices[i][j];
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

    static void EqNodeUpdate(FloatArray2d_h &eqNodes, const FloatArray2d_h& varNodes, const IntArray2d_h& eqNodeVarIndices, IntArray1d_h syndrome)
    {
        // For a check node interested in variables a,b,c,d to estimate the updated probability for variable a
        // syndrome = 0: even # of errors -> pa' = pb(1-pc)(1-pd) + pc(1-pb)(1-pd) + pd(1-pb)(1-pc) + pb*pc*pd
        //                                       = 0.5 * (1 - (1-2pb)(1-2pc)(1-2pd))
        // syndrome = 1: odd # of errors -> pa' = (1-pb)(1-pc)(1-pd) + pb*pc*(1-pd) + pb*(1-pc)*pd + (1-pb)*pc*pd
        //                                      = 0.5 * (1 + (1-2pb)(1-2pc)(1-2pd))
        int numEqs = eqNodes.num_rows;
        int numVarsPerEq = eqNodeVarIndices.num_cols;
        int numVars = varNodes.num_rows;
        auto eqNodePtr = &eqNodes.values[0];
        auto varNodePtr = &varNodes.values[0];
        auto eqNodeVarIndicesPtr = &eqNodeVarIndices.values[0];
        auto syndromePtr = &syndrome[0];
#pragma omp parallel for
        for (auto eqIdx = 0; eqIdx < numEqs; ++eqIdx) // loop over check nodes (parity equations)
        {
            auto firstVarIdx = eqIdx*numVarsPerEq;
            // loop over variables to be updated for this check node
            for (auto i = 0; i < numVarsPerEq; ++i)
            {
                auto index = firstVarIdx + i; // 1d array index to look up the variable index
                auto varIdx = eqNodeVarIndicesPtr[index]; // variable index under investigation for this eq
                //auto varIdx = eqNodeVarIndices.values[index]; // variable index under investigation for this eq
                auto product = 1.0f; // reset product
                // loop over all other variables in the equation, accumulate (1-2p) terms
                for (auto k = 0; k < numVarsPerEq; ++k)
                {
                    if (k == i) continue; // skip the variable being updated
                    auto otherIndex = firstVarIdx + k; // 1d array index to look up the variable index
                    auto otherVarIdx = eqNodeVarIndicesPtr[otherIndex];
                    //auto otherVarIdx = eqNodeVarIndices.values[otherIndex];

                    // the index holding the estimate beinng used for this eq
                    auto varNodesIndex = otherVarIdx * numEqs + eqIdx;
                    auto value = varNodePtr[varNodesIndex]; // belief value for this variable and this eq
                    //auto value = varNodes.values[varNodesIndex]; // belief value for this variable and this eq
                    product *= (1.0f - 2.0f*value);
                }
                auto cnIdx = eqIdx * numVars + varIdx; // index for value within the check node array to update
                //if (syndrome[eqIdx]) {
                if (syndromePtr[eqIdx]) {
                    eqNodePtr[cnIdx] = 0.5 * (1.0f + product); // syndrome = 1 -> odd parity
                    //eqNodes.values[cnIdx] = 0.5 * (1.0f + product); // syndrome = 1 -> odd parity
                }
                else {
                    eqNodePtr[cnIdx] = 0.5f * (1.0f - product); // syndrome = 0 -> even parity
                    //eqNodes.values[cnIdx] = 0.5f * (1.0f - product); // syndrome = 0 -> even parity
                }
            }
        }
    }

    static void VarNodeUpdate(const FloatArray2d_h& eqNodes, FloatArray2d_h& varNodes, const IntArray2d_h& varNodeEqIndices, 
        float errorProbability, bool last)
    {
        // For a variable node connected to check nodes 1,2,3,4 use the following formula to send an estimate to var node 1
        // p1' = K*pch*p2*p3*p4   (pch is the channel error probability. ignore the estimate received from check node 1 unless last)
        // where K = 1/[(1-pch)(1-p2)(1-p3)(1-p4)... + pch*p2*p3*p4...]
        int numEqs = eqNodes.num_rows;
        int numVars = varNodes.num_rows;
        int numEqsPerVar = varNodeEqIndices.num_cols;

        auto eqNodePtr = &eqNodes.values[0];
        auto varNodePtr = &varNodes.values[0];
        auto varNodeEqIndicesPtr = &varNodeEqIndices.values[0];

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
                //auto eqIdx = varNodeEqIndices.values[index];

                // 1d index for var nodes entry being updated
                auto varNodesIdx = firstVarNode + eqIdx;

                // start with a priori channel error probability
                auto prodP = errorProbability;
                auto prodOneMinusP = 1.0f - errorProbability;

                // calculate the updated probability for this check node based on belief estimates of all OTHER check nodes
                for (auto k = 0; k < numEqsPerVar; ++k)
                {
                    auto index2 = firstEqIndices + k; // 1d index for entry in the index array
                    auto otherEQIdx = varNodeEqIndicesPtr[index2];
                    //auto otherEQIdx = varNodeEqIndices.values[index2];

                    if (otherEQIdx == eqIdx && !last) continue;
                    // 1d index for check nodes belief being used
                    auto checkNodesIdx = otherEQIdx * numVars + varIdx;
                    auto p = eqNodePtr[checkNodesIdx];
                    //auto p = eqNodes.values[checkNodesIdx];

                    prodOneMinusP *= (1.0f - p);
                    prodP *= p;
                }
                auto value = prodP / (prodOneMinusP + prodP);
                varNodePtr[varNodesIdx] = value;
                //varNodes.values[varNodesIdx] = value;
            }
        }
    }

    static bool CheckConvergence(const FloatArray2d_h& estimates, float high, float low)
    {

        auto estimatesPtr = &estimates.values[0];
        auto numRows = estimates.num_rows;
        auto numCols = estimates.num_cols;
        // loop over all estimates
        for (auto i = 0; i < numRows; ++i) {
            for (auto j = 0; j < numCols; ++j) {
                int index = i * numCols + j;
                //if (estimates.values[index] != 0.0f) {
                if (estimatesPtr[index] != 0.0f) {
                    // if any estimate is between the bounds we have failed to converge
                    if (estimatesPtr[index] > low && estimatesPtr[index] < high) return false;
                    //if (estimates.values[index] > low && estimates.values[index] < high) return false;
                }
            }
        }
        return true;
    }

    // helper function for decoding x or z errors individually
    void BeliefPropogation(const IntArray1d_h& syndrome, float errorProbability, int maxIterations,
        FloatArray2d_h& vn1, FloatArray2d_h& en1, const IntArray2d_h& eqNodeVarIndices, const IntArray2d_h& varNodeEqIndices)
    {
        // We will first decode xErrors and then zErrors
        // An NxM parity check matrix H can be viewed as a bipartite graph with
        // N symbol nodes and M parity check nodes.  Each symbol node is connected
        // to ds parity-check nodes, and each parity-check node is connected to dc
        // symbol nodes.
        float p = 2.0f / 3.0f * errorProbability; // a priori probability for x/z OR y error
        float high = 0.99f;
        float low = 0.01f;

        thrust::fill(vn1.values.begin(), vn1.values.end(), 0.0f);
        InitVarNodes(vn1, eqNodeVarIndices, p);

        auto N = maxIterations; // maximum number of iterations
        bool converge= false;

        for (auto n = 0; n < N; n++)
        {
            if (converge) break;
            EqNodeUpdate(en1, vn1, eqNodeVarIndices, syndrome);
            VarNodeUpdate(en1, vn1, varNodeEqIndices, p, n == N - 1);
            

            if (n % 10 == 0)
            {
                converge = CheckConvergence(vn1, high, low);
            }
        }
    }

public:

    DecoderCPU(Quantum_LDPC_Code code) : Decoder(code),
        _eqNodeVarIndicesX_h(code.numCodeEqsX, code.L), _eqNodeVarIndicesZ_h(code.numCodeEqsZ, code.L),
        _varNodeEqIndicesX_h(code.numVars, code.numCodeEqsX/code.P), _varNodeEqIndicesZ_h(code.numVars, code.numCodeEqsX / code.P),
        _eqNodesX_h(code.numCodeEqsX, code.numVars, 0.0f), _eqNodesZ_h(code.numCodeEqsZ, code.numVars, 0.0f),
        _varNodesX_h(code.numVars, code.numCodeEqsX, 0.0f), _varNodesZ_h(code.numVars, code.numCodeEqsZ, 0.0f)
    {
        InitIndexArrays(_eqNodeVarIndicesX_h, _varNodeEqIndicesX_h, code.pcmX);
        InitIndexArrays(_eqNodeVarIndicesZ_h, _varNodeEqIndicesZ_h, code.pcmZ);
    }

    ~DecoderCPU()
    {
    }

    ErrorCode Decode(const IntArray1d_h& syndromeX, const IntArray1d_h& syndromeZ, float errorProbability, int maxIterations, 
        IntArray1d_h& outErrorsX, IntArray1d_h& outErrorsZ) override
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
                    _varNodesX_h, _eqNodesX_h, _eqNodeVarIndicesX_h, _varNodeEqIndicesX_h);
            }
#pragma omp section
            {
                BeliefPropogation(syndromeZ, errorProbability, maxIterations, 
                    _varNodesZ_h,_eqNodesZ_h, _eqNodeVarIndicesZ_h, _varNodeEqIndicesZ_h);
            }
        }

        // accumulate the error estimates into a single vector
        std::vector<int> finalEstimatesX(_varNodesX_h.num_rows, 0);
        std::vector<int> finalEstimatesZ(_varNodesZ_h.num_rows, 0);

        // check for correct error decoding
        ErrorCode code = SUCCESS;
        // check convergence errors
        for (auto varIdx = 0; varIdx < _varNodesX_h.num_rows; ++varIdx) {
            for (auto eqIdx = 0; eqIdx < _varNodesX_h.num_cols; ++eqIdx) {
                int index = varIdx * _varNodesX_h.num_cols + eqIdx;
                if (_varNodesX_h.values[index] >= 0.5f) // best guess of error
                {
                    finalEstimatesX[varIdx] = 1;
                    break;
                }
            }
        }
        for (auto varIdx = 0; varIdx < _varNodesZ_h.num_rows; ++varIdx) {
            for (auto eqIdx = 0; eqIdx < _varNodesZ_h.num_cols; ++eqIdx) {
                int index = varIdx * _varNodesZ_h.num_cols + eqIdx;
                if (_varNodesZ_h.values[index] >= 0.5f) // best guess of error
                {
                    finalEstimatesZ[varIdx] = 1;
                    break;
                }
            }
        }
        // check for convergence failure
        if (!CheckConvergence(_varNodesX_h, high, low)) {
            code = code | CONVERGENCE_FAIL_X;
        }
        if (!CheckConvergence(_varNodesZ_h, high, low)) code = code | CONVERGENCE_FAIL_Z;
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
        int maxIterations) override{
        std::random_device rd; // random seed for mersene twister engine.  could use this exclusively, but mt is faster
        unsigned int seed = rd();
        std::mt19937 mt(seed); // engine to produce random number
        std::uniform_int_distribution<int> indexDist(0, _code.numVars - 1); // distribution for rng of index where errror occurs
        std::uniform_int_distribution<int> errorDist(0, 2); // distribution for rng of error type. x=0, y=1, z=2

        int convergenceFailX = 0;
        int convergenceFailZ = 0;
        int syndromeErrorX = 0;
        int syndromeErrorZ = 0;
        int logicalErrorX = 0;
        int logicalErrorZ = 0;
        int corrected = 0;

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
            if(tID == 0)
            {
                nThreads = omp_get_num_threads();
                std::cout << "Thread count: " << nThreads << std::endl;
                count = COUNT / nThreads;
            }
#pragma omp barrier
            

            DecoderCPU decoder(_code);
            IntArray1d_h xErrors(_code.numVars, 0);
            IntArray1d_h zErrors(_code.numVars, 0);

            IntArray1d_h xDecodedErrors(_code.numVars, 0);
            IntArray1d_h zDecodedErrors(_code.numVars, 0);

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

                auto errorCode = decoder.Decode(sx, sz, errorProbability, MAX_ITERATIONS, xDecodedErrors, zDecodedErrors);


                // increment error or corrected counters
                if (errorCode & Decoder::CONVERGENCE_FAIL_X) {
#pragma omp atomic
                    convergenceFailX++;
                }
                if (errorCode & Decoder::CONVERGENCE_FAIL_Z) {
#pragma omp atomic
                    convergenceFailZ++;
                }

                if (errorCode & Decoder::SYNDROME_FAIL_X) {
#pragma omp atomic
                    syndromeErrorX++;
                }
                if (errorCode & Decoder::SYNDROME_FAIL_Z) {
#pragma omp atomic
                    syndromeErrorZ++;
                }

                if (!(errorCode & Decoder::SYNDROME_FAIL_XZ))
                { 
                    // the decoder thinks it correctly decoded the error
                    // check for logical errors
                    // If the diffference between the decoded error and actual error
                    // is in the row space of the code words, and is not zero, but it
                    // has the same syndrome as the actual error then it is a logical
                    // error.
                    IntArray1d_h xDiff(_code.numVars, 0);
                    IntArray1d_h zDiff(_code.numVars, 0);

                    bool xIsDiff = false;
                    bool zIsDiff = false;
                    for (auto i = 0; i < _code.numVars; ++i)
                    {
                        if (xErrors[i] != xDecodedErrors[i]) {
                            xDiff[i] = 1;
                            xIsDiff = true;
                        }
                        if (zErrors[i] != zDecodedErrors[i]) {
                            zIsDiff = true;
                            zDiff[i] = 1;
                        }
                    }
                    if (xIsDiff || zIsDiff) {
                        // the decoded error string can differ from the original by a stabilizer and still be decodeable.
                        // however, if it falls outside the stabilizer group it is a logical error. for stabilizer elements
                        // H e = 0.
                        auto xDiffSyndrome = _code.GetSyndromeX(xDiff);
                        auto zDiffSyndrome = _code.GetSyndromeZ(zDiff);

                        bool xLogicalError = false;
                        bool zLogicalError = false;
                        for (int i = 0; i < xDiffSyndrome.size(); ++i)
                        {
                            if (xDiffSyndrome[i] != 0) {
                                xLogicalError = true;
                            }
                            if (zDiffSyndrome[i] != 0) {
                                zLogicalError = true;
                            }
                        }
                        if (xLogicalError) {
#pragma omp atomic
                            logicalErrorX++;
                        }
                        if (zLogicalError) {
#pragma omp atomic
                            logicalErrorZ++;
                        }
                        if (!xLogicalError && !zLogicalError) {
#pragma omp atomic
                            corrected++;
                        }
                    }
                    else
                    {
#pragma omp atomic
                        corrected++;
                    }
                }
            }
        }
        auto finish = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
        CodeStatistics stats = {_code,seed,count * nThreads,W,corrected,syndromeErrorX,syndromeErrorZ,logicalErrorX,logicalErrorZ,
            convergenceFailX,convergenceFailZ, duration};
        return stats;
    }
};

