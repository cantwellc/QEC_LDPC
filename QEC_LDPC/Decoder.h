#pragma once
#include "Quantum_LDPC_Code.h"
#include "HostDeviceArray.h"
#include <cusp/print.h>
#include "CodeStatistics.h"

class Decoder
{
private:
protected:
    Quantum_LDPC_Code _code;
public:

    static enum ErrorCode
    {
        SUCCESS = 0,
        SYNDROME_FAIL_X = 1 << 0,
        SYNDROME_FAIL_Z = 1 << 1,
        SYNDROME_FAIL_XZ = SYNDROME_FAIL_X | SYNDROME_FAIL_Z,
        CONVERGENCE_FAIL_X = 1 << 2,
        CONVERGENCE_FAIL_Z = 1 << 3,
        CONVERGENCE_FAIL_XZ = CONVERGENCE_FAIL_X | CONVERGENCE_FAIL_Z
    };

    friend inline ErrorCode operator|(const ErrorCode &a, const ErrorCode &b) {
        return static_cast<ErrorCode>(static_cast<int>(a) | static_cast<int>(b));
    }
    friend inline ErrorCode operator&(const ErrorCode &a, const ErrorCode &b) {
        return static_cast<ErrorCode>(static_cast<int>(a) & static_cast<int>(b));
    }

    explicit Decoder(Quantum_LDPC_Code code) : _code(code)
    {
    }

    virtual ~Decoder()
    {
    }

    virtual ErrorCode Decode(const IntArray1d_h& syndromeX, const IntArray1d_h& syndromeZ, float errorProbability, int maxIterations,
        IntArray1d_h& outErrorsX, IntArray1d_h& outErrorsZ) {
        return SUCCESS;
    }
    virtual CodeStatistics GetStatistics(int errorWeight, int numErrors, float errorProbability, 
        int maxIterations) = 0;
    virtual CodeStatistics GetStatistics(int errorWeight, int numErrors, float errorProbability,
        int maxIterations, unsigned int seed) = 0;
};

