#pragma once
#include <ostream>
#include "Quantum_LDPC_Code.h"

struct CodeStatistics
{
    Quantum_LDPC_Code code;
    unsigned int randSeed;
    unsigned int numErrorsTested;
    unsigned int numXErrorsTested;
    unsigned int numZErrorsTested;
    unsigned int errorWeight;
    unsigned int corrected;
    unsigned int syndromeErrorsX;
    unsigned int syndromeErrorsZ;
    unsigned int logicalErrors;
    unsigned int convergenceFailX;
    unsigned int convergenceFailZ;
    long long durationMicroSeconds;
};

inline std::ostream & operator <<(std::ostream & stream, CodeStatistics const & stats){
    stream << "Code: " << stats.code << std::endl
        << "Rand Seed: " << stats.randSeed << std::endl
        << "Duration(micro-s): " << stats.durationMicroSeconds << std::endl
        << "Errors Tested: " << stats.numErrorsTested << std::endl
        << "Errors With X: " << stats.numXErrorsTested << std::endl
        << "Errors With Z: " << stats.numZErrorsTested << std::endl
        << "Error Weight: " << stats.errorWeight << std::endl
        << "Corrected: " << stats.corrected << std::endl
        << "Syndrome Errors X: " << stats.syndromeErrorsX << std::endl
        << "Syndrome Errors Z: " << stats.syndromeErrorsZ << std::endl
        << "Logical Errors: " << stats.logicalErrors << std::endl
        << "Convergence Fail X: " << stats.convergenceFailX << std::endl
        << "Convergence Fail Z: " << stats.convergenceFailZ << std::endl;
    return stream;
}
