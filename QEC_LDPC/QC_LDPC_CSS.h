#pragma once
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "cusparse_v2.h"
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/array1d.h>
#include <math.h>
#include <cusp/print.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "kernels.cuh"
#include <cusp/iterator/random_iterator.h>
#include <cusp/iterator/random_iterator.h>
#include <cusp/iterator/random_iterator.h>


class QC_LDPC_CSS
{
private:
    int _numEqsX;
    int _numEqsZ;
    int _numVars;
    int _P;

    typedef cusp::array2d<int, cusp::host_memory, cusp::row_major> IntArray2d_h;
    typedef cusp::array2d<int, cusp::device_memory, cusp::row_major> IntArray2d_d;
    
    typedef cusp::array2d<float, cusp::host_memory, cusp::row_major> FloatArray2d_h;
    typedef cusp::array2d<float, cusp::device_memory, cusp::row_major> FloatArray2d_d;

    typedef cusp::array1d<int, cusp::host_memory> IntArray1d_h;
    typedef cusp::array1d<int, cusp::device_memory> IntArray1d_d;

    // Host memory for parity check matrices HC (X errors) and HD (Z errors)
    std::vector<std::vector<int>> _hHC_vec;
    std::vector<std::vector<int>> _hHD_vec;

    IntArray2d_h _pcmX_h;
    //IntArray2d_d _pcmX_d;
    //int* _pcmX_d_ptr;

    IntArray2d_h _pcmZ_h;
    //IntArray2d_d _pcmZ_d;
    //int* _pcmZ_d_ptr;

    IntArray1d_h _syndromeX_h;
    IntArray1d_d _syndromeX_d;
    int* _syndromeX_d_ptr;

    IntArray1d_h _syndromeZ_h;
    IntArray1d_d _syndromeZ_d;
    int* _syndromeZ_d_ptr;

    FloatArray2d_h _varNodesX_h;
    FloatArray2d_h _varNodesZ_h;
    FloatArray2d_d _varNodesX_d;
    FloatArray2d_d _varNodesZ_d;
    float* _varNodesX_d_ptr;
    float* _varNodesZ_d_ptr;

    FloatArray2d_h _eqNodesX_h;
    FloatArray2d_h _eqNodesZ_h;
    FloatArray2d_d _eqNodesX_d;
    FloatArray2d_d _checkNodesZ_d;
    float* _eqNodesX_d_ptr;
    float* _eqNodesZ_d_ptr;

    IntArray2d_h _eqNodeVarIndicesX_h;
    IntArray2d_h _eqNodeVarIndicesZ_h;
    IntArray2d_d _eqNodeVarIndicesX_d;
    IntArray2d_d _eqNodeVarIndicesZ_d;
    int* _eqNodeVarIndicesX_d_ptr;
    int* _eqNodeVarIndicesZ_d_ptr;

    IntArray2d_h _varNodeEqIndicesX_h;
    IntArray2d_h _varNodeEqIndicesZ_h;
    IntArray2d_d _varNodeEqIndicesX_d;
    IntArray2d_d _varNodeEqIndicesZ_d;
    int* _varNodeEqIndicesX_d_ptr;
    int* _varNodeEqIndicesZ_d_ptr;

    //std::vector<std::vector<float>> _qC; // array of error probability beliefs for each variable node
    //std::vector<std::vector<float>> _rC; // array of updated beliefs for each check node

    //	cusp::array2d<double, cusp::host_memory, cusp::row_major> _q1; // message from variable-node to check-node
    //	cusp::array2d<double, cusp::host_memory, cusp::row_major> _r1; // message from check-node to variable-node

    void VarNodeUpdate(std::vector<std::vector<float>>& varNodeEstimates,
                            std::vector<std::vector<float>>& checkNodeBeliefs, 
                            std::vector<std::vector<int>> parityCheckMatrix,
                            float errorProbability, bool last);
    void EqNodeUpdate(std::vector<std::vector<float>>& varNodeEstimates,
                         std::vector<std::vector<float>>& checkNodeBeliefs, 
                         std::vector<std::vector<int>> parityCheckMatrix, std::vector<int> syndrome);

    void VarNodeUpdate(FloatArray2d_h checkNodes, FloatArray2d_h &varNodes, IntArray2d_h varNodeEqIndices, float errorProbability, bool last);
    void EqNodeUpdate(FloatArray2d_h &checkNodes, FloatArray2d_h varNodes, IntArray2d_h checkNodeVarIndices, IntArray1d_h syndrome);

    static void SetIndexArrays(IntArray2d_h& checkNodeVarIndices, IntArray2d_h& varNodeEqIndices, IntArray2d_h& parityCheckMatrix);

    static bool CheckConvergence(const std::vector<std::vector<float>>& estimates, float high, float low);
    static bool CheckConvergence(const FloatArray2d_h& estimates, float high, float low);


public:
    static enum ErrorCode
    {
        SUCCESS = 0,
        SYNDROME_FAIL_X = 1<<0,
        SYNDROME_FAIL_Z = 1<<1,
        SYNDROME_FAIL_XZ = SYNDROME_FAIL_X | SYNDROME_FAIL_Z,
        CONVERGENCE_FAIL_X = 1<<2,
        CONVERGENCE_FAIL_Z = 1<<3,
        CONVERGENCE_FAIL_XZ = CONVERGENCE_FAIL_X | CONVERGENCE_FAIL_Z
    };
    friend inline ErrorCode operator|(const ErrorCode &a, const ErrorCode &b) {
        return static_cast<ErrorCode>(static_cast<int>(a) | static_cast<int>(b));
    }
    friend inline ErrorCode operator&(const ErrorCode &a, const ErrorCode &b) {
        return static_cast<ErrorCode>(static_cast<int>(a) & static_cast<int>(b));
    }

    
    ///	<summary>
    ///	Build the parity check matrix for code C (i.e. H_1), and dual code D (i.e. H_2) given parameters
    ///	J: rows of C (must be > 1)
    ///	K: rows of D (must be < L/2, may be same as J)
    ///	L: columns (must be even, > 0)
    ///	P: degree of circulant matrix ( I(1)^P )
    ///	sigma: an element of Z_P
    ///	tau: an element in (Z_P)* \ {1,sigma,sigma^2,...}
    ///	This kernel builds a single column of matrix C
    /// </summary>
    QC_LDPC_CSS(int J, int K, int L, int P, int sigma, int tau);

    ~QC_LDPC_CSS();

    static void WriteToFile(cusp::array2d<int,cusp::host_memory,cusp::row_major> matrix, const char* file);
    static void WriteToFile(cusp::array2d<float,cusp::host_memory,cusp::row_major> matrix, const char* file);
    static void WriteToFile(std::vector<std::vector<float>> xes, const char* str);
    static void WriteToFile(std::vector<int> xes, const char* str);

    // Old slow version of cpu decoding
    ErrorCode DecodeCPU(std::vector<int> xSyndrome, std::vector<int> zSyndrome, float errorProbability, 
        std::vector<int> &xErrors, std::vector<int> &zErrors, int maxIterations);

    // Fast version of cpu decoding
    ErrorCode DecodeCPU2(std::vector<int> xSyndrome, std::vector<int> zSyndrome, float errorProbability,
        std::vector<int> &xErrors, std::vector<int> &zErrors, int maxIterations);

    // Decode on the gpu using cuda
    ErrorCode QC_LDPC_CSS::DecodeCUDA(std::vector<int> syndromeX, std::vector<int> syndromeZ, float errorProbability,
        std::vector<int> &xErrors, std::vector<int> &zErrors, int maxIterations);

    std::vector<int> GetXSyndrome(std::vector<int> xErrors);
    std::vector<int> GetZSyndrome(std::vector<int> zErrors);
};
