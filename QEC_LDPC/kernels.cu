#include "kernels.cuh"

//	Build the parity check matrix for code C (i.e. H_1), and dual code D (i.e. H_2) given parameters
//	J: rows of C (must be > 1)
//	K: rows of D (must be < L/2, may be same as J)
//	L: columns (must be even, > 0)
//	P: degree of circulant matrix ( I(1)^P )
//	sigma: an element of Z_P
//	tau: an element in (Z_P)* \ {1,sigma,sigma^2,...}
//	This kernel builds a single element of matrix C
__global__ void buildHC_kernel(int J, int L, int P, int sigma, int invSigma, int tau, int *C)
{
	int j = threadIdx.x; // row index
	int l = threadIdx.y; // column index
	int index = j*L + l;
	int i, t, p;
	t = 1;
	if (l < L / 2) {
		p = -j + l;
		if (p<0) for (i = 0; i<-p; ++i) t = (t*invSigma) % P; // sigma^(-k) = (sigma^(-1))^k
		else for (i = 0; i < p; ++i) t = (t*sigma) % P;
	}
	else {
		p = j - 1 + l;
		if (p < 0) for (i = 0; i < -p; ++i) t = (t*invSigma) % P;
		else for (i = 0; i < p; ++i) t = (t*sigma) % P;
		t = P - (tau*t) % P; // -(tau*sigma^p) = P - (tau*sigma^p)
	}
	C[index] = t;
}

// Kernel for updating the check node belief matrix for
// a single variable within a single check node.
// queue kernel with a single block of dimension num_eqs x num_vars_per_eq
__global__ void eqNodeUpdate_kernel(int* eqNodeVarIndices, float* eqNodes, float* varNodes, int* syndrome, 
    int numVars, int numEqs, int numVarsPerEq)//, int numEqsPerVar)
{
    // For a check node interested in variables a,b,c,d to estimate the updated probability for variable a
    // syndrome = 0: even # of errors -> pa' = pb(1-pc)(1-pd) + pc(1-pb)(1-pd) + pd(1-pb)(1-pc) + pb*pc*pd
    //                                       = 0.5 * (1 - (1-2pb)(1-2pc)(1-2pd))
    // syndrome = 1: odd # of errors -> pa' = (1-pb)(1-pc)(1-pd) + pb*pc*(1-pd) + pb*(1-pc)*pd + (1-pb)*pc*pd
    //                                      = 0.5 * (1 + (1-2pb)(1-2pc)(1-2pd))

    int eqIdx = threadIdx.x; // rows of the block represent the parity check equations
    int i = threadIdx.y; // columns of the block represent the different variables involved in the eq.
    if (eqIdx < numEqs && i < numVarsPerEq) {
//    if(true){
        // location of the first variable index in the eqNodeVarIndices array
        int firstVarIdx = eqIdx*numVarsPerEq;

        // location of the variable index in the eqNodeVarIndices array
        int index = firstVarIdx + i;

        // variable index under investigation for this eq
        int varIdx = eqNodeVarIndices[index];

        float product = 1.0f; // reset product
        // loop over all other variables in the equation, accumulate (1-2p) terms
        for (auto j = 0; j < numVarsPerEq; ++j)
        {
            if (j == i) continue; // skip the variable being updated
            int otherIndex = firstVarIdx + j; // 1d array index to look up the variable index
            int otherVarIdx = eqNodeVarIndices[otherIndex];

            // the index holding the estimate being used for this eq
            int varNodesIndex = otherVarIdx * numEqs + eqIdx;

            // belief value this variable node holds for this eq
            float value = varNodes[varNodesIndex];
            product *= (1.0f - 2.0f*value);
        }
        int eqNodesIdx = eqIdx * numVars + varIdx; // index for value within the eq node array to update
        if (syndrome[eqIdx]) {
            eqNodes[eqNodesIdx] = 0.5 * (1.0f + product); // syndrome = 1 -> odd parity
        }
        else {
            eqNodes[eqNodesIdx] = 0.5f * (1.0f - product); // syndrome = 0 -> even parity
        }
    }
}

// kernel for updating the variable node matrix
// for a single variable belief corresponding to a single check node.
// varNodeEqIndices holds the indices of the equations this variable is part of
// eqNodes holds the belief estimates each equation node has for each variable node. (0 if equation doesn't use variable)
// varNodes holds the estimates each vaariable node has for each equation node. (0 if variable isn't used in equation)
// errorProbability is the a priori channel error probability
__global__ void varNodeUpdate_kernel(int *varNodeEqIndices, float *eqNodes, float * varNodes, 
    float errorProbability, bool last, int numVars, int numEqs, int numEqsPerVar)
{
    // For a variable node connected to equation nodes 1,2,3,4 use the following formula to send an estimate to var node 1
    // p1' = K*pch*p2*p3*p4   (pch is the channel error probability. ignore the estimate received from check node 1 unless last)
    // where K = 1/[(1-pch)(1-p2)(1-p3)(1-p4)... + pch*p2*p3*p4...]

    int varIdx = threadIdx.x; // variable index
    int i = threadIdx.y; // location of the equation index for which we are updating the value in varNodes

    if (varIdx < numVars && i < numEqsPerVar) {
        int firstVarNode = varIdx * numEqs; // start of entries in VarNodes array for this variable
        int firstEqIndices = varIdx * numEqsPerVar; // starting point for first equation in the index list for this var.

        int eqIdxLoc = firstEqIndices + i;
        // find the index of the equation estimate being updated
        int eqIdx = varNodeEqIndices[eqIdxLoc];

        // 1d index for var nodes entry being updated
        int varNodesIdx = firstVarNode + eqIdx;

        float prodP = errorProbability; // start with a priori channel error probability
        float prodOneMinusP = 1.0f - errorProbability;

        // calculate the updated probability for this check node based on belief estimates of all OTHER check nodes
        for (auto k = 0; k < numEqsPerVar; ++k)
        {
            int eqIndexLoc2 = firstEqIndices + k; // 1d index for entry in the index array
            int otherEQIdx = varNodeEqIndices[eqIndexLoc2];

            if (otherEQIdx == eqIdx && !last) continue;
           
            // 1d index for check nodes belief being used
            int checkNodesIdx = otherEQIdx * numVars + varIdx;
            float p = eqNodes[checkNodesIdx];

            prodOneMinusP *= (1.0f - p);
            prodP *= p;
        }

        float value = prodP / (prodOneMinusP + prodP);
        varNodes[varNodesIdx] = value;
    }
}

