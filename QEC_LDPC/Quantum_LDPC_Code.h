#pragma once
#include "HostDeviceArray.h"

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

    Quantum_LDPC_Code(int J, int K, int L, int P, int sigma, int tau) :
        J(J), K(K), L(L), P(P), sigma(sigma), tau(tau),
        n(L * P), numEqsX(J*P), numEqsZ(K*P),
        pcmX(numEqsX,n), pcmZ(numEqsZ,n)
    {
        // only works for codes where k2 <= k1 
        // where k2 = rows(pcmZ)-rows(pcmZ) and k1 = rows(pcmX)-rows(pcmZ) 
        assert(numEqsZ >= numEqsX); 
        int i1, i2, i3, i4, t, p, invSigma;

        // index matrices for parity check matrices _pcmX_h and _pcmZ_h
        IntArray2d_h hHC(J, L);
        IntArray2d_h hHD(K, L);

        // construct the cyclic set from which HC and HD will be made
        cusp::array1d<int, cusp::host_memory> ZP(P - 1);
        for (i1 = 0; i1 < P - 1; ++i1) ZP[i1] = i1 + 1;

        // find sigma^(-1).  It is the element of ZP that when multiplied by sigma = 1
        for (i1 = 0; ZP[i1] * sigma % P != 1; ++i1); // loop through ZP until the inverse element is found.
        invSigma = ZP[i1];

        // Build parity check matrices for HC and HD on the host since this is a one shot operation.
        // Time to transfer data to the gpu will make this inefficient.
        for (i2 = 0; i2 < J; ++i2)
        {
            for (i4 = 0; i4 < L; ++i4)
            {
                t = 1;
                if (i4 < L / 2)
                {
                    p = -i2 + i4;
                    // find the correct power of sigma (or inverse sigma if p is negative)
                    if (p < 0) for (i1 = 0; i1 < -p; ++i1) t = (t * invSigma) % P; // sigma^(-k) = (sigma^(-1))^k
                    else for (i1 = 0; i1 < p; ++i1) t = (t * sigma) % P;
                }
                else
                {
                    p = i2 - 1 + i4;
                    // find the correct power of sigma (or inverse sigma if p is negative)
                    if (p < 0) for (i1 = 0; i1 < -p; ++i1) t = (t * invSigma) % P;
                    else for (i1 = 0; i1 < p; ++i1) t = (t * sigma) % P;
                    t = P - (tau * t) % P; // -(tau*sigma^p) = P - (tau*sigma^p)
                }
                hHC(i2, i4) = t;
            }
        }

        for (i3 = 0; i3 < K; ++i3)
        {
            for (i4 = 0; i4 < L; ++i4)
            {
                t = 1;
                if (i4 < L / 2)
                {
                    p = -i3 - 1 + i4;
                    // find the correct power of sigma (or inverse sigma if p is negative)
                    if (p < 0) for (i1 = 0; i1 < -p; ++i1) t = (t * invSigma) % P; // sigma^(-k) = (sigma^(-1))^k
                    else for (i1 = 0; i1 < p; ++i1) t = (t * sigma) % P;
                    t = (tau * t) % P;
                }
                else
                {
                    p = i3 + i4;
                    // find the correct power of sigma (or inverse sigma if p is negative)
                    if (p < 0) for (i1 = 0; i1 < -p; ++i1) t = (t * invSigma) % P;
                    else for (i1 = 0; i1 < p; ++i1) t = (t * sigma) % P;
                    t = P - (t); // -(sigma^p) = P - (sigma^p)
                }
                hHD(i3, i4) = t;
            }
        }

        int cj, ck, cjR, ckR, cl, c, row, col;
        // Construct the code matrix row by row.
        // The matrix is made up of JxL PxP blocks.
        // Each block is a circulant permutation matrix, I(1)^c with c given by HC calculated previously
        // see https://arxiv.org/pdf/quant-ph/0701020.pdf or https://en.wikipedia.org/wiki/Circulant_matrix
        for (row = 0; row < J * P; ++row)
        {
            cj = (int)(row / P); // the row index for HC is the integer part of j/P
            cjR = row % P; // the row within block cj is j%P.  P rows per block.
            for (cl = 0; cl < L; ++cl)
            {
                c = hHC(cj, cl); //this is the power for the circulant permutation matrix, I(1)^c
                                 // cjR=0, c=1, block column index for non-zero entry = 1
                                 // cjR=1, c=1, block column index for non-zero entry = 2
                                 // cjR=P, c=1, block column index for non-zero entry = 0
                                 // block column index = (c + cjR) % P
                                 // offset block column index by block width P: offset = cl * P
                                 // column index = (c + cjR) % P + (cl * P);
                col = (c + cjR) % P + (cl * P);
                int index = row * n + col;
                pcmX.values[index] = 1; // set value of non-zero value i
            }
        }

        for (row = 0; row < K * P; ++row)
        {
            ck = (int)(row / P); // the row index for HD is the integer part of k/P
            ckR = row % P; // the row within block ck is k%P.  P rows per block.
            for (cl = 0; cl < L; ++cl)
            {
                c = hHD(ck, cl); //this is the power for the circulant permutation matrix, I(1)^c
                col = (c + ckR) % P + (cl * P);
                int index = row * n + col;
                pcmZ.values[index] = 1; // set value of non-zero value i
            }
        }
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
};

inline std::ostream & operator <<(std::ostream & stream, Quantum_LDPC_Code const & code) {
    stream << "code: J=" << code.J << ",K=" << code.K << ",L=" << code.L << ",P=" << code.P 
        << ",sigma=" << code.sigma << ",tau=" << code.tau 
        << " [[n=" << code.n << ",k=" << code.numEqsZ - code.numEqsX << "]]";
    return stream;
}