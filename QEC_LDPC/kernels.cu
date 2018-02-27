#include "kernels.cuh"
#include <cstdio>

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

__global__ void beliefPropogation_kernel(float* eqNodes, float* varNodes, int* eqNodeVarIndices, int * varNodeEqIndices, 
    int* syndrome, float errorProbability, int numVars, int numEqs, int numVarsPerEq, int numEqsPerVar, int maxIterations)
{
    if (threadIdx.x == 0) {
        dim3 eqNodeGridDim(numEqs);  // number of blocks.
        dim3 eqNodeBlockDim(numVarsPerEq); // number of threads per block
        auto eqNodeMemSize = numVarsPerEq * sizeof(int);

        dim3 varNodeGridDim(numVars);
        dim3 varNodeBlockDim(numEqsPerVar);
        auto varNodeMemSize2 = numEqsPerVar * sizeof(int);
//        auto varNodeMemSize = numEqsPerVar * sizeof(float);

        float high = 0.99f;
        float low = 0.01f;
        bool converge = false;
        // we can potentially speed up here by performing the first eq node update outside the loop
        // this will allow us to synchronize only once.  child threads see the global memory of the
        // parent block at the time of launch. this gives us the potential to update var nodes and
        // eq nodes in parallel.  we could hit a race condition if the execution of all var node threads
        // completes before the eq nodes update is started.  this can be avoided by storing temporary 
        // values in a buffer                            
        

        for (auto n = 0; n < maxIterations; n++)
        {
            if (converge) break;
            // on the first execution we are doing the work twice for updating the eq nodes.
            eqNodeUpdate_kernel <<<eqNodeGridDim, eqNodeBlockDim, eqNodeMemSize >>> (eqNodeVarIndices, eqNodes, varNodes,
                syndrome, numVars, numEqs, numVarsPerEq);
            cudaDeviceSynchronize();

            varNodeUpdate2_kernel <<<varNodeGridDim, varNodeBlockDim, varNodeMemSize2 >>>
                (varNodeEqIndices, eqNodes, varNodes, errorProbability, n == maxIterations - 1, numVars, numEqs, numEqsPerVar);
            cudaDeviceSynchronize();

            if (n % 10 == 0) {
                converge = checkConvergence(varNodes, numVars, numEqs, high, low);
            }
        }
    }
}

__device__ bool checkConvergence(float* varNodes, int numVars, int numEqs, float high, float low)
{
    for(int i=0; i<numVars; ++i)
    {
        for(int j=0; j<numEqs; ++j)
        {
            int idx = i*numEqs + j;
            if(varNodes[idx] < high && varNodes[idx] > low)
            {
                return false;
            }
        }
    }
    return true;
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

    //int eqIdx = threadIdx.x; // rows of the block represent the parity check equations
    //int i = threadIdx.y; // columns of the block represent the different variables involved in the eq.
    int eqIdx = blockIdx.x; // block index is the equation node we're updating
    int i = threadIdx.x; // thread index is the column in the eqNodesVarIndices array we're looking at

    // copy index array into shared memory
    extern __shared__ int indices[];
    if (eqIdx < numEqs && i < numVarsPerEq) {

        // location of the first variable index in the eqNodeVarIndices array
        int firstVarIdx = eqIdx*numVarsPerEq;
        // location of the variable index in the eqNodeVarIndices array
        int index = firstVarIdx + i;
        // variable index under investigation for this eq
        int varIdx = eqNodeVarIndices[index];
        indices[i] = varIdx;
        __syncthreads();

        float product = 1.0f; // reset product
        // loop over all other variables in the equation, accumulate (1-2p) terms
        for (auto j = 0; j < numVarsPerEq; ++j)
        {
            if (j == i) continue; // skip the variable being updated
//            int otherIndex = firstVarIdx + j; // 1d array index to look up the variable index
//            int otherVarIdx = eqNodeVarIndices[otherIndex];
            int otherVarIdx = indices[j];

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
// launch n blocks, each block processes numEqsPerVar threads
__global__ void varNodeUpdate_kernel(int * varNodeEqIndices, float* eqNodes, float * varNodes, 
    float errorProbability, bool last, int numVars, int numEqs, int numEqsPerVar)
{
    // For a variable node connected to equation nodes 1,2,3,4 use the following formula to send an estimate to var node 1
    // p1' = K*pch*p2*p3*p4   (pch is the channel error probability. ignore the estimate received from check node 1 unless last)
    // where K = 1/[(1-pch)(1-p2)(1-p3)(1-p4)... + pch*p2*p3*p4...]

    // for efficiency, load memory into block shared memory.
    // array to hold ordered belief estimates that each equation has for this variable
    extern __shared__ float sharedEqBeliefs[]; 

    int varIdx = blockIdx.x; // variable index, row index for varNodeEqIndices array
    int i = threadIdx.x; // column index for varNodeEqIndices array

    if (varIdx < numVars && i < numEqsPerVar) { // skip if out of bounds

        int firstEqIndices = varIdx * numEqsPerVar; // starting point for first equation in the index list for this var.
        int eqIdxLoc = firstEqIndices + i; // location in varNodesEqIndices for this eq index.
        int eqIdx = varNodeEqIndices[eqIdxLoc]; // equation index being investigated in this thread
        int eqNodeIdx = eqIdx * numVars + varIdx; // index for the belief estimate in the eqNodes arraay for this var
        sharedEqBeliefs[i] = eqNodes[eqNodeIdx]; // store the belief value for this variable in shared memory
        __syncthreads();

        int firstVarNode = varIdx * numEqs; // start of entries in VarNodes array for this variable

        // 1d index for var nodes entry being updated (the estimate held for this eq)
        int varNodesIdx = firstVarNode + eqIdx;

        float prodP = errorProbability; // start with a priori channel error probability
        float prodOneMinusP = 1.0f - errorProbability;

        // calculate the updated probability for this eq node based on belief estimates of all OTHER eq nodes
        for (auto k = 0; k < numEqsPerVar; ++k)
        {
            if (k == i && !last) continue;
            float p = sharedEqBeliefs[k];
            prodOneMinusP *= (1.0f - p);
            prodP *= p;
        }

        float value = prodP / (prodOneMinusP + prodP);
        varNodes[varNodesIdx] = value;
    }
}

// kernel for updating the variable node matrix
// for a single variable belief corresponding to a single check node.
// varNodeEqIndices holds the indices of the equations this variable is part of
// eqNodes holds the belief estimates each equation node has for each variable node. (0 if equation doesn't use variable)
// varNodes holds the estimates each vaariable node has for each equation node. (0 if variable isn't used in equation)
// errorProbability is the a priori channel error probability
// launch n blocks, each block processes numEqsPerVar threads
__global__ void varNodeUpdate2_kernel(int * varNodeEqIndices, float* eqNodes, float * varNodes, 
    float errorProbability, bool last, int numVars, int numEqs, int numEqsPerVar)
{
    // For a variable node connected to equation nodes 1,2,3,4 use the following formula to send an estimate to var node 1
    // p1' = K*pch*p2*p3*p4   (pch is the channel error probability. ignore the estimate received from check node 1 unless last)
    // where K = 1/[(1-pch)(1-p2)(1-p3)(1-p4)... + pch*p2*p3*p4...]

    // for efficiency, load memory into block shared memory.
    // array to hold ordered belief estimates that each equation has for this variable
    extern __shared__ int indices[]; 

    int varIdx = blockIdx.x; // variable index, row index for varNodeEqIndices array
    int i = threadIdx.x; // column index for varNodeEqIndices array

    if (varIdx < numVars && i < numEqsPerVar) { // skip if out of bounds

        int firstEqIndices = varIdx * numEqsPerVar; // starting point for first equation in the index list for this var.
        int eqIdxLoc = firstEqIndices + i; // location in varNodesEqIndices for this eq index.
        int eqIdx = varNodeEqIndices[eqIdxLoc]; // equation index being investigated in this thread
        indices[i] = eqIdx;
        __syncthreads();

        int varNodesStart = varIdx * numEqs; // start of entries in VarNodes array for this variable
        // 1d index for var nodes entry being updated (the estimate held for this eq)
        int varNodesIdx = varNodesStart + eqIdx;

        float prodP = errorProbability; // start with a priori channel error probability
        float prodOneMinusP = 1.0f - errorProbability;

        // calculate the updated probability for this eq node based on belief estimates of all OTHER eq nodes
        for (auto k = 0; k < numEqsPerVar; ++k)
        {
            if (k == i && !last) continue;
            int otherEqIdx = indices[k];
            int eqNodeIdx = otherEqIdx * numVars + varIdx; // index for the belief estimate in the eqNodes arraay for this var
            float p = eqNodes[eqNodeIdx];
            prodOneMinusP *= (1.0f - p);
            prodP *= p;
        }

        float value = prodP / (prodOneMinusP + prodP);
        varNodes[varNodesIdx] = value;
    }
}

__device__ void generateSyndrome(int* errorString, int numQubits, int* pcm, int* syndrome)
{
    
}

__global__ void getStats_kernel(int* pcmX, int* pcmZ, int* eqNodeVarIndices, int* varNodeEqIndices, float* eqNodes, float* varNodes, 
    int* errorsX, int* errorsZ, int numErrors, int numVars, int numEqs, int numVarsPerEq)
{
    // loop over errors, generate a syndrome, decode, and update statistics
    for(int i=0; i<numErrors; ++i)
    {
        
    }
}