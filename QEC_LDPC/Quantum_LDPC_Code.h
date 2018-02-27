#pragma once
#include "HostDeviceArray.h"
#include <string>
#include <fstream>
#include "ArrayOutput.h"

class Quantum_LDPC_Code
{
public:
    const int J;
    const int K;
    const int L;
    const int P;
    const int sigma;
    const int tau;

    // number of physical qubits
    const int n; 
    const int numEqsX;
    const int numEqsZ;

    IntArray2d_h pcmX;
    IntArray2d_h pcmZ;
    IntArray2d_h iMinusP;

    static Quantum_LDPC_Code createFromFile(std::string file){
        std::cout << "Creating code from file " << file << std::endl;
        auto getArrayFromString = [](const std::string& cs, int nRows, int nCols){
            IntArray2d_h result(nRows, nCols);
            std::stringstream stream(cs);
            int index = 0;
            while (true)
            {
                int x;
                stream >> x;
                if (!stream) break;
                result.values[index] = x;
                ++index;
            }
            return result;
        };

        std::ifstream ifs;
        ifs.open(file.c_str());
        if (ifs.is_open())
        {
            int j, k, l, p, s, t;

            std::string parameters;
            getline(ifs, parameters);
            std::stringstream pStream(parameters);
            pStream >> j;
            pStream >> k;
            pStream >> l;
            pStream >> p;
            pStream >> s;
            pStream >> t;

            std::string hc;
            getline(ifs, hc);
            IntArray2d_h pcmX = getArrayFromString(hc, j*p, l*p);

            std::string hd;
            getline(ifs, hd);
            IntArray2d_h pcmZ = getArrayFromString(hd, k*p, l*p);

            std::string imp;
            getline(ifs, imp);
            // I-P matrix is stored in the full space
            // Hc 0
            // 0 Hd
            IntArray2d_h impMatrix = getArrayFromString(imp, 2 * l*p, 2 * l*p);

            return Quantum_LDPC_Code(j, k, l, p, s, t, pcmX, pcmZ, impMatrix);
        }
        else
        {
            throw std::string("Unable to find code file " + file);
        }
    }

    Quantum_LDPC_Code(int J, int K, int L, int P, int sigma, int tau, IntArray2d_h pcmX, IntArray2d_h pcmZ, IntArray2d_h imp) :
        J(J), K(K), L(L), P(P), sigma(sigma), tau(tau),
        n(L * P), numEqsX(J*P), numEqsZ(K*P),
        pcmX(pcmX), pcmZ(pcmZ), iMinusP(imp)
    {
        
    }

    ~Quantum_LDPC_Code()
    {
    }

    IntArray1d_h GetSyndromeX(IntArray1d_h errors)
    {
       IntArray1d_h syndrome(numEqsX);
        for (auto eqIdx = 0; eqIdx < numEqsX; ++eqIdx)
        {
            auto x = 0;
            for (auto varIdx = 0; varIdx < n; ++varIdx)
            {
                int pcmIdx = eqIdx * n + varIdx;
                x += pcmX.values[pcmIdx] * errors[varIdx];
            }
            syndrome[eqIdx] = x % 2;
        }
        return syndrome;
    }

    IntArray1d_h GetSyndromeZ(IntArray1d_h errors)
    {
        IntArray1d_h syndrome(numEqsZ);
        for (auto eqIdx = 0; eqIdx < numEqsZ; ++eqIdx)
        {
            auto x = 0;
            for (auto varIdx = 0; varIdx < n; ++varIdx)
            {
                int pcmIdx = eqIdx * n + varIdx;
                x += pcmZ.values[pcmIdx] * errors[varIdx];
            }
            syndrome[eqIdx] = x % 2;
        }
        return syndrome;
    }

    bool CheckLogicalError(IntArray1d_h errors)
    {
        // a set of e = {x1,x2,...,xn,z1,z2,...,zn} errors has a logical
        // error component if iMinusP.e != 0
        // if any element of the result vector is mod 2 non-zero return true (logical error)
        for(int i=0; i<iMinusP.num_rows; ++i)
        {
            int sum = 0;
            for(int j=0; j<iMinusP.num_cols; ++j)
            {
                int index = i*iMinusP.num_cols + j;
                sum += iMinusP.values[index] * errors[j];
            }
            if (sum % 2 != 0) return true;
        }
        return false;
    }
};

inline std::ostream & operator <<(std::ostream & stream, Quantum_LDPC_Code const & code) {
    stream << "[J=" << code.J << ",K=" << code.K << ",L=" << code.L << ",P=" << code.P 
        << ",s=" << code.sigma << ",t=" << code.tau 
        << "][[n=" << code.n << ",k=" << code.numEqsZ - code.numEqsX << "]]";
    return stream;
}