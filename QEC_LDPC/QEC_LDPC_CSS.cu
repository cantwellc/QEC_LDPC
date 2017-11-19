//#include "QC_LDPC_CSS.h"
//#include <chrono>
//// See https://arxiv.org/pdf/quant-ph/0701020.pdf for construction
//
//QC_LDPC_CSS::QC_LDPC_CSS(int J, int K, int L, int P, int sigma, int tau) :
//    _numEqsX(J * P), _numEqsZ(K * P), _numVars(L * P), _P(P),
//    _hHC_vec(J * P, std::vector<int>(L * P)), _hHD_vec(K * P, std::vector<int>(L * P)), 
//    // allocate host memory for parity check matrices
//    _pcmX_h(J * P, L * P), _pcmZ_h(K * P, L * P), 
//    // allocate host and device memory for syndromes
//    _syndromeX_h(_numEqsX,0), _syndromeX_d(_numEqsX,0),
//    _syndromeZ_h(_numEqsZ,0), _syndromeZ_d(_numEqsZ,0),
//    // allocate host and device memory for var node updates
//    _varNodesX(_numVars,_numEqsX,0), _varNodesZ(_numVars,_numEqsZ,0), 
//    _varNodesX_d(_numVars,_numEqsX,0), _varNodesZ_d(_numVars,_numEqsZ,0),
//    // allocate host and device memory for check node updates
//    _eqNodesX(_numEqsX,_numVars,0), _eqNodesZ(_numEqsZ,_numVars,0), 
//    _eqNodesX_d(_numEqsX,_numVars,0), _checkNodesZ_d(_numEqsZ,_numVars,0),
//    // allocate host and device memory for index matrices
//    _eqNodeVarIndicesX(_numEqsX, L), _eqNodeVarIndicesZ(_numEqsZ, L),
//    _eqNodeVarIndicesX_d(_numEqsX, L), _eqNodeVarIndicesZ_d(_numEqsZ, L),
//    _varNodeEqIndicesX(_numVars,J), _varNodeEqIndicesZ(_numVars, K),
//    _varNodeEqIndicesX_d(_numVars, J), _varNodeEqIndicesZ_d(_numVars, K),
//    _errorGenerator(_numVars)
//{
//    int i, j, k, l, t, p, invSigma;
//
//    // index matrices for parity check matrices _pcmX_h and _pcmZ_h
//    IntArray2d_h hHC(J, L);
//    IntArray2d_h hHD(K, L);
//
//    // construct the cyclic set from which HC and HD will be made
//    cusp::array1d<int, cusp::host_memory> ZP(P - 1);
//    for (i = 0; i < P - 1; ++i) ZP[i] = i + 1;
//    print(ZP);
//
//    // find sigma^(-1).  It is the element of ZP that when multiplied by sigma = 1
//    for (i = 0; ZP[i] * sigma % P != 1; ++i); // loop through ZP until the inverse element is found.
//    invSigma = ZP[i];
//
//    // Build parity check matrices for HC and HD on the host since this is a one shot operation.
//    // Time to transfer data to the gpu will make this inefficient.
//    for (j = 0; j < J; ++j)
//    {
//        for (l = 0; l < L; ++l)
//        {
//            t = 1;
//            if (l < L / 2)
//            {
//                p = -j + l;
//                // find the correct power of sigma (or inverse sigma if p is negative)
//                if (p < 0) for (i = 0; i < -p; ++i) t = (t * invSigma) % P; // sigma^(-k) = (sigma^(-1))^k
//                else for (i = 0; i < p; ++i) t = (t * sigma) % P;
//            }
//            else
//            {
//                p = j - 1 + l;
//                // find the correct power of sigma (or inverse sigma if p is negative)
//                if (p < 0) for (i = 0; i < -p; ++i) t = (t * invSigma) % P;
//                else for (i = 0; i < p; ++i) t = (t * sigma) % P;
//                t = P - (tau * t) % P; // -(tau*sigma^p) = P - (tau*sigma^p)
//            }
//            hHC(j, l) = t;
//        }
//    }
//
//    for (k = 0; k < K; ++k)
//    {
//        for (l = 0; l < L; ++l)
//        {
//            t = 1;
//            if (l < L / 2)
//            {
//                p = -k - 1 + l;
//                // find the correct power of sigma (or inverse sigma if p is negative)
//                if (p < 0) for (i = 0; i < -p; ++i) t = (t * invSigma) % P; // sigma^(-k) = (sigma^(-1))^k
//                else for (i = 0; i < p; ++i) t = (t * sigma) % P;
//                t = (tau * t) % P;
//            }
//            else
//            {
//                p = k + l;
//                // find the correct power of sigma (or inverse sigma if p is negative)
//                if (p < 0) for (i = 0; i < -p; ++i) t = (t * invSigma) % P;
//                else for (i = 0; i < p; ++i) t = (t * sigma) % P;
//                t = P - (t); // -(sigma^p) = P - (sigma^p)
//            }
//            hHD(k, l) = t;
//        }
//    }
////    print_matrix(hHC);
////    print_matrix(hHD);
//
//    int cj, ck, cjR, ckR, cl, c, row, col;
//    // Construct the parity check matrix matrix row by row.
//    // The matrix is made up of JxL PxP blocks.
//    // Each block is a circulant permutation matrix, I(1)^c with c given by HC calculated previously
//    // see https://arxiv.org/pdf/quant-ph/0701020.pdf or https://en.wikipedia.org/wiki/Circulant_matrix
//    for (row = 0; row < J * P; ++row)
//    {
//        cj = (int)(row / P); // the row index for HC is the integer part of j/P
//        cjR = row % P; // the row within block cj is j%P.  P rows per block.
//        for (cl = 0; cl < L; ++cl)
//        {
//            c = hHC(cj, cl); //this is the power for the circulant permutation matrix, I(1)^c
//            // cjR=0, c=1, block column index for non-zero entry = 1
//            // cjR=1, c=1, block column index for non-zero entry = 2
//            // cjR=P, c=1, block column index for non-zero entry = 0
//            // block column index = (c + cjR) % P
//            // offset block column index by block width P: offset = cl * P
//            // column index = (c + cjR) % P + (cl * P);
//            col = (c + cjR) % P + (cl * P);
//            int index = row * _numVars + col;
//            _hHC_vec[row][col] = 1;
//            _pcmX_h.values[index] = 1; // set value of non-zero value i
//        }
//    }
//
//    for (row = 0; row < K * P; ++row)
//    {
//        ck = (int)(row / P); // the row index for HD is the integer part of k/P
//        ckR = row % P; // the row within block ck is k%P.  P rows per block.
//        for (cl = 0; cl < L; ++cl)
//        {
//            c = hHD(ck, cl); //this is the power for the circulant permutation matrix, I(1)^c
//            col = (c + ckR) % P + (cl * P);
//            int index = row * _numVars + col;
//            _hHD_vec[row][col] = 1;
//            _pcmZ_h.values[index] = 1; // set value of non-zero value i
//        }
//    }
//
//    // set index arrays and device pointers
//    SetIndexArrays(_eqNodeVarIndicesX, _varNodeEqIndicesX, _pcmX_h);
//    thrust::copy(_eqNodeVarIndicesX.values.begin(), _eqNodeVarIndicesX.values.end(), _eqNodeVarIndicesX_d.values.begin());
//    thrust::copy(_varNodeEqIndicesX.values.begin(), _varNodeEqIndicesX.values.end(), _varNodeEqIndicesX_d.values.begin());
//    _eqNodeVarIndicesX_d_ptr = thrust::raw_pointer_cast(&_eqNodeVarIndicesX_d.values[0]);
//    _varNodeEqIndicesX_d_ptr = thrust::raw_pointer_cast(&_varNodeEqIndicesX_d.values[0]);
//
//    SetIndexArrays(_eqNodeVarIndicesZ, _varNodeEqIndicesZ, _pcmZ_h);
//    thrust::copy(_eqNodeVarIndicesZ.values.begin(), _eqNodeVarIndicesZ.values.end(), _eqNodeVarIndicesZ_d.values.begin());
//    thrust::copy(_varNodeEqIndicesZ.values.begin(), _varNodeEqIndicesZ.values.end(), _varNodeEqIndicesZ_d.values.begin());
//    _eqNodeVarIndicesZ_d_ptr = thrust::raw_pointer_cast(&_eqNodeVarIndicesZ_d.values[0]);
//    _varNodeEqIndicesZ_d_ptr = thrust::raw_pointer_cast(&_varNodeEqIndicesZ_d.values[0]);
//
//    _numEqsPerVarX = _varNodeEqIndicesX.num_cols;
//    _numVarsPerEqX = _eqNodeVarIndicesX.num_cols;
//    _numEqsPerVarZ = _varNodeEqIndicesZ.num_cols;
//    _numVarsPerEqZ = _eqNodeVarIndicesZ.num_cols;
//
//    // set device memory pointers for pre-allocated device matrices
//    _syndromeX_d_ptr = thrust::raw_pointer_cast(&_syndromeX_d[0]);
//    _syndromeZ_d_ptr = thrust::raw_pointer_cast(&_syndromeZ_d[0]);
//
//    _varNodesX_d_ptr = thrust::raw_pointer_cast(&_varNodesX_d.values[0]);
//    _varNodesZ_d_ptr = thrust::raw_pointer_cast(&_varNodesZ_d.values[0]);
//
//    _eqNodesX_d_ptr = thrust::raw_pointer_cast(&_eqNodesX_d.values[0]);
//    _eqNodesZ_d_ptr = thrust::raw_pointer_cast(&_checkNodesZ_d.values[0]);
//
//    // We now have parity check matrices hPHC and hPHD on the host.  https://arxiv.org/pdf/quant-ph/0701020.pdf
//    // These satisfy the constraints that the girth of their respective Tanner graphs are >= 6
//    //	and they have a "twisted relation", i.e.  dual(D) is in C.
//}
//
//QC_LDPC_CSS::~QC_LDPC_CSS()
//{
//}
//
//void QC_LDPC_CSS::SetIndexArrays(IntArray2d_h& checkNodeVarIndices, IntArray2d_h& varNodeEqIndices, IntArray2d_h& parityCheckMatrix)
//{
//    // set device index matrices for var node and check node updates
//    // each equation will include L variables.
//    // each variable will be involved in J equations
//    // loop over all check node equations in the parity check matrix for X errors    
//    int numEqs = parityCheckMatrix.num_rows;
//    int n = parityCheckMatrix.num_cols;
//    std::vector<std::vector<int>> cnVarIndices(numEqs, std::vector<int>());
//    std::vector<std::vector<int>> vnEqIndices(n, std::vector<int>());
//    // loop over all equations
//    for (int eqIdx = 0; eqIdx < numEqs; ++eqIdx)
//    {
//        // loop over all variables
//        for (int varIdx = 0; varIdx < n; ++varIdx)
//        {
//            int pcmIdx = eqIdx * n + varIdx;
//            // if the entry in the pcm is 1, this check node involves this variable.  set the index entry
//            if (parityCheckMatrix.values[pcmIdx])
//            {
//                cnVarIndices[eqIdx].push_back(varIdx);
//                vnEqIndices[varIdx].push_back(eqIdx);
//            }
//        }
//    }
//    // copy data into provided array containers
//    auto index = 0;
//    for (auto i = 0; i<cnVarIndices.size(); ++i)
//    {
//        for(auto j=0; j<cnVarIndices[0].size(); ++j)
//        {
//            checkNodeVarIndices.values[index] = cnVarIndices[i][j];
//            ++index;
//        }
//    }
//    index = 0;
//    for (auto i = 0; i<vnEqIndices.size(); ++i)
//    {
//        for (auto j = 0; j<vnEqIndices[0].size(); ++j)
//        {
//            varNodeEqIndices.values[index] = vnEqIndices[i][j];
//            ++index;
//        }
//    }
//}
//
//void QC_LDPC_CSS::WriteToFile(IntArray2d_h vec, const char* str)
//{
//    std::ofstream file;
//    file.open(str, std::ios::app);
//    if (file.is_open()) {
//        std::cout << "Writing to file " << str << std::endl;
//        for (auto i = 0; i < vec.num_rows; ++i)
//        {
//            for (auto j = 0; j < vec.num_cols; ++j) {
//                int index = i*vec.num_cols + j;
//                auto v = vec.values[index];
//                file << v << " ";
//            }
//            file << "\n";
//        }
//        file << "\n\n";
//        file.close();
//    }
//    else
//    {
//        std::cout << "Failed to open file " << str << std::endl;
//    }
//}
//
//void QC_LDPC_CSS::WriteToFile(cusp::array2d<float, cusp::host_memory, cusp::row_major> vec, const char* str)
//{
//    std::ofstream file;
//    file.open(str, std::ios::app);
//    if (file.is_open()) {
//        std::cout << "Writing to file " << str << std::endl;
//        for (auto i = 0; i < vec.num_rows; ++i)
//        {
//            for (auto j = 0; j < vec.num_cols; ++j) {
//                int index = i*vec.num_cols + j;
//                auto v = vec.values[index];
//                file << std::fixed << std::setprecision(3) << v << " ";
//            }
//            file << "\n";
//        }
//        file << "\n\n";
//        file.close();
//    }
//    else
//    {
//        std::cout << "Failed to open file " << str << std::endl;
//    }
//}
//
//void QC_LDPC_CSS::WriteToFile(std::vector<std::vector<float>> vec, const char* str)
//{
//    std::ofstream file;
//    file.open(str, std::ios::app);
//    if (file.is_open()) {
//        std::cout << "Writing to file " << str << std::endl;
//        for (auto i = 0; i < vec.size(); ++i)
//        {
//            for (auto j = 0; j < vec[i].size(); ++j) {
//                auto v = vec[i][j];
//                file << std::fixed << std::setprecision(3) << v << " ";
//            }
//            file << "\n";
//            //file << v << ",";
//        }
//        file << "\n\n";
//        file.close();
//    }else
//    {
//        std::cout << "Failed to open file " << str << std::endl;
//    }
//}
//
//void QC_LDPC_CSS::WriteToFile(std::vector<int> vec, const char* str)
//{
//    std::ofstream file;
//    file.open(str, std::ios::app);
//    if (file.is_open()) {
//        std::cout << "Writing to file " << str << std::endl;
//        for (auto i = 0; i < vec.size(); ++i)
//        {
//            auto v = vec[i];
//            file << std::fixed << std::setprecision(3) << v << " ";
//        }
//        file << "\n";
//        file.close();
//    }
//    else
//    {
//        std::cout << "Failed to open file " << str << std::endl;
//    }
//}
//
///*
//Given a set of x errors and z errors, this will attempt to decode the errors
//and will return a success / failure code.
//See paper for algorithm:
//We will use a Belief-propogation decoding scheme.
//*/
//QC_LDPC_CSS::ErrorCode QC_LDPC_CSS::DecodeCUDA(std::vector<int> syndromeX, std::vector<int> syndromeZ, float errorProbability,
//    std::vector<int> &xErrors, std::vector<int> &zErrors, int maxIterations)
//{
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
////    float varUpdateKernelTime = 0;
////    float eqUpdateKernelTime = 0;
////    std::chrono::microseconds memCopyTime(0);
//    std::chrono::microseconds checkConvergenceTime(0);
//    std::chrono::microseconds updateTime(0);
//    std::chrono::microseconds initTime(0);
//    std::chrono::microseconds decodeTime(0);
//    std::chrono::microseconds completeTime(0);
//
//    auto begin = std::chrono::high_resolution_clock::now();
//
//    // We will first decode xErrors and then zErrors
//    // An NxM parity check matrix H can be viewed as a bipartite graph with
//    // N symbol nodes and M parity check nodes.  Each symbol node is connected
//    // to ds parity-check nodes, and each parity-check node is connected to dc
//    // symbol nodes.
//    float p = 2.0f / 3.0f * errorProbability; // a priori probability for x/z OR y error
//    float high = 0.99f;
//    float low = 0.01f;
//    
//    // clear var node and check node arrays, and set syndrome arrays
//    for (int i = 0; i < _varNodesX.num_entries; ++i) _varNodesX.values[i] = 0;
//    for (int i = 0; i < _varNodesZ.num_entries; ++i) _varNodesZ.values[i] = 0;
//    int numVarsPerEq = _eqNodeVarIndicesX.num_cols;
//    for (int eqIdx = 0; eqIdx<_numEqsX; ++eqIdx)
//    {
//        for (int j = 0; j<numVarsPerEq; ++j)
//        {
//            int idx = eqIdx * numVarsPerEq + j;
//            int varIdx = _eqNodeVarIndicesX.values[idx];
//            int varNodeIdx = varIdx * _numEqsX + eqIdx;
//            _varNodesX.values[varNodeIdx] = p;
//        }
//    }
//    for (int eqIdx = 0; eqIdx<_numEqsZ; ++eqIdx)
//    {
//        for (int j = 0; j<_eqNodeVarIndicesZ.num_cols; ++j)
//        {
//            int idx = eqIdx * numVarsPerEq + j;
//            int varIdx = _eqNodeVarIndicesZ.values[idx];
//            int varNodeIdx = varIdx * _numEqsX + eqIdx;
//            _varNodesZ.values[varNodeIdx] = p;
//        }
//    }
//    for (int i = 0; i < _eqNodesX.num_entries; ++i) _eqNodesX.values[i] = 0.0f;
//    for (int i = 0; i < _eqNodesZ.num_entries; ++i) _eqNodesZ.values[i] = 0.0f;
//
//    // copy host data to device
//    thrust::copy(_varNodesX.values.begin(), _varNodesX.values.end(), _varNodesX_d.values.begin());
//    thrust::copy(_varNodesZ.values.begin(), _varNodesZ.values.end(), _varNodesZ_d.values.begin());
//    thrust::copy(_eqNodesX.values.begin(), _eqNodesX.values.end(), _eqNodesX_d.values.begin());
//    thrust::copy(_eqNodesZ.values.begin(), _eqNodesZ.values.end(), _checkNodesZ_d.values.begin());
//    thrust::copy(syndromeX.begin(), syndromeX.end(), _syndromeX_d.begin());
//    thrust::copy(syndromeZ.begin(), syndromeZ.end(), _syndromeZ_d.begin());
//
//
//    auto N = maxIterations; // maximum number of iterations
////    bool xConverge = false;
////    bool zConverge = false;
//
//    //dim3 eqNodeGridDimX(_numEqsX);  // number of blocks.
//    //dim3 eqNodeBlockDimX(_numEqsX,_numVarsPerEqX); // number of threads per block
//
//    //dim3 eqNodeGridDimZ(_numEqsZ);
//    //dim3 eqNodeBlockDimZ(_numEqsZ,_numVarsPerEqZ);
//
//    //dim3 varNodeGridDimX(_numVars);
//    //dim3 varNodeBlockDimX(_numEqsPerVarX);
//    //auto varNodeMemSizeX = _numEqsPerVarX * sizeof(float);
//
//    //dim3 varNodeGridDimZ(_numVars);
//    //dim3 varNodeBlockDimZ(_numEqsPerVarZ);
//    //auto varNodeMemSizeZ = _numEqsPerVarX * sizeof(float);
//
//    auto finish = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - begin);
//    initTime += duration;
//
//    begin = std::chrono::high_resolution_clock::now();
//    // launch a single warp of 32 threads.
//    beliefPropogation_kernel << <1, 1 >> > (_eqNodesX_d_ptr, _varNodesX_d_ptr, _eqNodeVarIndicesX_d_ptr, _varNodeEqIndicesX_d_ptr,
//        _syndromeX_d_ptr, p, _numVars, _numEqsX, _numVarsPerEqX, _numEqsPerVarX, N);
//    
//    // launch a single warp of 32 threads.
//    beliefPropogation_kernel << <1, 1 >> > (_eqNodesZ_d_ptr, _varNodesZ_d_ptr, _eqNodeVarIndicesZ_d_ptr, _varNodeEqIndicesZ_d_ptr,
//        _syndromeZ_d_ptr, p, _numVars, _numEqsZ, _numVarsPerEqZ, _numEqsPerVarZ, N);
//
//    cudaDeviceSynchronize();
//
//    finish = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - begin);
//    decodeTime += duration;
//
//    begin = std::chrono::high_resolution_clock::now();
//
//    thrust::copy(_varNodesX_d.values.begin(), _varNodesX_d.values.end(), _varNodesX.values.begin());
//    thrust::copy(_varNodesZ_d.values.begin(), _varNodesZ_d.values.end(), _varNodesZ.values.begin());
//
//    // accumulate the error estimates into a single vector
//    std::vector<int> finalEstimatesX(_varNodesX.num_rows, 0);
//    std::vector<int> finalEstimatesZ(_varNodesZ.num_rows, 0);
//
//    // check for correct error decoding
//    ErrorCode code = SUCCESS;
//    // check convergence errors
//    for (auto varIdx = 0; varIdx < _varNodesX.num_rows; ++varIdx) {
//        for (auto eqIdx = 0; eqIdx < _varNodesX.num_cols; ++eqIdx) {
//            int index = varIdx * _varNodesX.num_cols + eqIdx;
//            if (_varNodesX.values[index] >= 0.5f) // best guess of error
//            {
//                finalEstimatesX[varIdx] = 1;
//                break;
//            }
//        }
//    }
//    for (auto varIdx = 0; varIdx < _varNodesZ.num_rows; ++varIdx) {
//        for (auto eqIdx = 0; eqIdx < _varNodesZ.num_cols; ++eqIdx) {
//            int index = varIdx * _varNodesZ.num_cols + eqIdx;
//            if (_varNodesZ.values[index] >= 0.5f) // best guess of error
//            {
//                finalEstimatesZ[varIdx] = 1;
//                break;
//            }
//        }
//    }
//    // check for convergence failure
//    if (!CheckConvergence(_varNodesX, high, low)) {
//        code = code | CONVERGENCE_FAIL_X;
//    }
//    if (!CheckConvergence(_varNodesZ, high, low)) {
//        code = code | CONVERGENCE_FAIL_Z;
//    }
//    // check syndrome errors
//    auto xS = GetXSyndrome(finalEstimatesX);
//    if (!std::equal(syndromeX.begin(), syndromeX.end(), xS.begin())) { code = code | SYNDROME_FAIL_X; }
//
//    auto zS = GetZSyndrome(finalEstimatesZ);
//    if (!std::equal(syndromeZ.begin(), syndromeZ.end(), zS.begin())) { code = code | SYNDROME_FAIL_Z; }
//
//    xErrors = finalEstimatesX;
//    zErrors = finalEstimatesZ;
//    finish = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - begin);
//    completeTime += duration;
//
//
////    std::cout << "VarNode update kernel execution time: " << varUpdateKernelTime * 1000 << " micro-seconds." << std::endl;
////    std::cout << "EqNode update kernel execution time: " << eqUpdateKernelTime * 1000 << " micro-seconds." << std::endl;
////    std::cout << "MemCopyTime: " << memCopyTime.count() << " micro-seconds." << std::endl;
////    std::cout << "Check convergence time: " << checkConvergenceTime.count() << " micro-seconds." << std::endl;
//    std::cout << "Init time: " << initTime.count() << " micro-seconds." << std::endl;
//    std::cout << "Decode time: " << decodeTime.count() << " micro-seconds." << std::endl;
//    std::cout << "Complete time: " << completeTime.count() << " micro-seconds." << std::endl;
//    std::cout << "Check Convergence time: " << checkConvergenceTime.count() << " micro-seconds." << std::endl;
//    std::cout << "Update time: " << updateTime.count() << " micro-seconds." << std::endl;
//
//
//    return code;
//}
//
///*
//	Given a set of x errors and z errors, this will attempt to decode the errors
//	and will return a success / failure code.
//	See paper for algorithm:
//	We will use a Belief-propogation decoding scheme.
//*/
//QC_LDPC_CSS::ErrorCode QC_LDPC_CSS::DecodeCPU(std::vector<int> syndromeX, std::vector<int> syndromeZ, float errorProbability,
//    std::vector<int> &xErrors, std::vector<int> &zErrors, int maxIterations)
//{
//    // We will first decode xErrors and then zErrors
//    // An NxM parity check matrix H can be viewed as a bipartite graph with
//    // N symbol nodes and M parity check nodes.  Each symbol node is connected
//    // to ds parity-check nodes, and each parity-check node is connected to dc
//    // symbol nodes.
//    float p = 2.0f / 3.0f * errorProbability; // a priori probability for x/z OR y error
//    float high = 0.99f;
//    float low = 0.01f;
//    // array of probability estimates to send to each check node.  there are _numEqsX variables, and _numVars check nodes
//   /* std::vector<std::vector<float>> varNodeEstimatesX(_numEqsX, std::vector<float>(_numVars, p));
//    std::vector<std::vector<float>> varNodeEstimatesZ(_numEqsZ, std::vector<float>(_numVars, p));*/
//
//    // each var node has a list of estimates from each check node.
//    std::vector<std::vector<float>> varNodeEstimatesX(_numVars, std::vector<float>(_numEqsX, p));
//    std::vector<std::vector<float>> varNodeEstimatesZ(_numVars, std::vector<float>(_numEqsZ, p));
//
//    // each check node has a list of beliefs for the value of each var node.
//    std::vector<std::vector<float>> checkNodeBeliefsX(_numEqsX, std::vector<float>(_numVars));
//    std::vector<std::vector<float>> checkNodeBeliefsZ(_numEqsZ, std::vector<float>(_numVars));
//
//    //WriteToFile(varNodeEstimatesX, "results/xEstimates.txt");
//    //WriteToFile(checkNodeBeliefsX, "results/xBeliefs.txt");
//
//    auto N = maxIterations; // maximum number of iterations
//    bool xConverge = false;
//    bool zConverge = false;
//    for (auto n = 0; n < N; n++)
//    {
//        if (xConverge && zConverge) break;
//        if(!xConverge)
//        {
//            EqNodeUpdate(varNodeEstimatesX, checkNodeBeliefsX, _hHC_vec, syndromeX);
//            VarNodeUpdate(varNodeEstimatesX, checkNodeBeliefsX, _hHC_vec, p, n == N - 1);
//            //WriteToFile(varNodeEstimatesX, "results/xEstimates.txt");
//            //WriteToFile(checkNodeBeliefsX, "results/xBeliefs.txt");
//            if (n % 10 == 0)
//            {
//                xConverge = CheckConvergence(varNodeEstimatesX, high, low);
//            }
//        }
//        
//        if (!zConverge)
//        {
//            EqNodeUpdate(varNodeEstimatesZ, checkNodeBeliefsZ, _hHD_vec, syndromeZ);
//            VarNodeUpdate(varNodeEstimatesZ, checkNodeBeliefsZ, _hHD_vec, p, n == N - 1);
//            if (n % 10 == 0)
//            {
//                zConverge = CheckConvergence(varNodeEstimatesZ, high, low);
//            }
//        }
//
//        
//    }
//    // accumulate the error estimates into a single vector
//    std::vector<int> finalEstimatesX(varNodeEstimatesX.size(), 0);
//    std::vector<int> finalEstimatesZ(varNodeEstimatesZ.size(), 0);
//
//    // check for correct error decoding
//    ErrorCode code = SUCCESS;
//    // check convergence errors
//    for (auto i = 0; i < varNodeEstimatesX.size(); ++i) {
//        for (auto j = 0; j < varNodeEstimatesX[i].size(); ++j) {
//            if (varNodeEstimatesX[i][j] != 0.0f) {
//                if(varNodeEstimatesX[i][j] > high) finalEstimatesX[i] = 1;
//                else if (varNodeEstimatesX[i][j] < low) finalEstimatesX[i] = 0;
//                else {
//                    finalEstimatesX[i] = -1;
//                    code = code | CONVERGENCE_FAIL_X;
//                }
//                break;
//            }
//        }
//    }
//    for (auto i = 0; i < varNodeEstimatesZ.size(); ++i) {
//        for (auto j = 0; j < varNodeEstimatesZ[i].size(); ++j) {
//            if (varNodeEstimatesZ[i][j] != 0.0f) {
//                if (varNodeEstimatesZ[i][j] > high) finalEstimatesZ[i] = 1;
//                else if (varNodeEstimatesZ[i][j] < low) finalEstimatesZ[i] = 0;
//                else {
//                    finalEstimatesZ[i] = -1;
//                    code = code | CONVERGENCE_FAIL_Z;
//                }
//                break;
//            }
//        }
//    }
//    // check syndrome errors
//    if (code == SUCCESS) {
//        auto xS = GetXSyndrome(finalEstimatesX);
//        if (!std::equal(syndromeX.begin(), syndromeX.end(), xS.begin())) { code = code | SYNDROME_FAIL_X; }
//
//        auto zS = GetZSyndrome(finalEstimatesZ);
//        if (!std::equal(syndromeZ.begin(), syndromeZ.end(), zS.begin())) { code = code | SYNDROME_FAIL_Z; }
//    }
//
//    xErrors = finalEstimatesX;
//    zErrors = finalEstimatesZ;
//
//    return code;
//}
//
//QC_LDPC_CSS::ErrorCode QC_LDPC_CSS::DecodeCPU2(std::vector<int> xSyndrome, std::vector<int> zSyndrome, float errorProbability, std::vector<int>& xErrors, std::vector<int>& zErrors, int maxIterations)
//{
//    // We will first decode xErrors and then zErrors
//    // An NxM parity check matrix H can be viewed as a bipartite graph with
//    // N symbol nodes and M parity check nodes.  Each symbol node is connected
//    // to ds parity-check nodes, and each parity-check node is connected to dc
//    // symbol nodes.
//    float p = 2.0f / 3.0f * errorProbability; // a priori probability for x/z OR y error
//    float high = 0.99f;
//    float low = 0.01f;
//
//    // clear var node and check node arrays, and set syndrome arrays
//    for (int i = 0; i < _varNodesX.num_entries; ++i) _varNodesX.values[i] = 0;
//    for (int i = 0; i < _varNodesZ.num_entries; ++i) _varNodesZ.values[i] = 0;
//    int numVarsPerEq = _eqNodeVarIndicesX.num_cols;
//    for(int eqIdx=0; eqIdx<_numEqsX; ++eqIdx)
//    {
//        for(int j=0; j<numVarsPerEq; ++j)
//        {
//            int idx = eqIdx * numVarsPerEq + j;
//            int varIdx = _eqNodeVarIndicesX.values[idx];
//            int varNodeIdx = varIdx * _numEqsX + eqIdx;
//            _varNodesX.values[varNodeIdx] = p;
//        }
//    }
//    for (int eqIdx = 0; eqIdx<_numEqsZ; ++eqIdx)
//    {
//        for (int j = 0; j<_eqNodeVarIndicesZ.num_cols; ++j)
//        {
//            int idx = eqIdx * numVarsPerEq + j;
//            int varIdx = _eqNodeVarIndicesZ.values[idx];
//            int varNodeIdx = varIdx * _numEqsX + eqIdx;
//            _varNodesZ.values[varNodeIdx] = p;
//        }
//    }
//    for (int i = 0; i < _eqNodesX.num_entries; ++i) _eqNodesX.values[i] = 0.0f;
//    for (int i = 0; i < _eqNodesZ.num_entries; ++i) _eqNodesZ.values[i] = 0.0f;
//    for (int i = 0; i < xSyndrome.size(); ++i) _syndromeX_h[i] = xSyndrome[i];
//    for (int i = 0; i < zSyndrome.size(); ++i) _syndromeZ_h[i] = zSyndrome[i];
//
//    auto N = maxIterations; // maximum number of iterations
//    bool xConverge = false;
//    bool zConverge = false;
//    //WriteToFile(_varNodesX, "results/varX_CPU.txt");
//    //WriteToFile(_eqNodesX, "results/eqX_CPU.txt");
//    for (auto n = 0; n < N; n++)
//    {
//        if (xConverge && zConverge) break;
//        if (!xConverge)
//        {
//            EqNodeUpdate(_eqNodesX,_varNodesX,_eqNodeVarIndicesX, _syndromeX_h);
//            VarNodeUpdate(_eqNodesX, _varNodesX, _varNodeEqIndicesX ,p, n == N - 1);
//            //WriteToFile(_varNodesX, "results/varX_CPU.txt");
//            //WriteToFile(_eqNodesX, "results/eqX_CPU.txt");
//            if (n % 10 == 0)
//            {
//                xConverge = CheckConvergence(_varNodesX, high, low);
//            }
//        }
//
//        if (!zConverge)
//        {
//            EqNodeUpdate(_eqNodesZ, _varNodesZ, _eqNodeVarIndicesZ, _syndromeZ_h);
//            VarNodeUpdate(_eqNodesZ, _varNodesZ, _varNodeEqIndicesZ , p, n == N - 1);
//            if (n % 10 == 0)
//            {
//                zConverge = CheckConvergence(_varNodesZ, high, low);
//            }
//        }
//    }
//    // accumulate the error estimates into a single vector
//    std::vector<int> finalEstimatesX(_varNodesX.num_rows, 0);
//    std::vector<int> finalEstimatesZ(_varNodesZ.num_rows, 0);
//
//    // check for correct error decoding
//    ErrorCode code = SUCCESS;
//    // check convergence errors
//    for (auto varIdx = 0; varIdx < _varNodesX.num_rows; ++varIdx) {
//        for (auto eqIdx = 0; eqIdx < _varNodesX.num_cols; ++eqIdx) {
//            int index = varIdx * _varNodesX.num_cols + eqIdx;
//            if(_varNodesX.values[index] >= 0.5f) // best guess of error
//            {
//                finalEstimatesX[varIdx] = 1;
//                break;
//            }
//        }
//    }
//    for (auto varIdx = 0; varIdx < _varNodesZ.num_rows; ++varIdx) {
//        for (auto eqIdx = 0; eqIdx < _varNodesZ.num_cols; ++eqIdx) {
//            int index = varIdx * _varNodesZ.num_cols + eqIdx;
//            if (_varNodesZ.values[index] >= 0.5f) // best guess of error
//            {
//                finalEstimatesZ[varIdx] = 1;
//                break;
//            }
//        }
//    }
//    // check for convergence failure
//    if (!CheckConvergence(_varNodesX, high, low)) {
//        code = code | CONVERGENCE_FAIL_X;
////        WriteToFile(_varNodesX, "results/convXCPU.txt");
//    }
//    if (!CheckConvergence(_varNodesZ, high, low)) code = code | CONVERGENCE_FAIL_Z;
//    // check syndrome errors
//    auto xS = GetXSyndrome(finalEstimatesX);
//    if (!std::equal(xSyndrome.begin(), xSyndrome.end(), xS.begin())) { code = code | SYNDROME_FAIL_X; }
//
//    auto zS = GetZSyndrome(finalEstimatesZ);
//    if (!std::equal(zSyndrome.begin(), zSyndrome.end(), zS.begin())) { code = code | SYNDROME_FAIL_Z; }
//
//    xErrors = finalEstimatesX;
//    zErrors = finalEstimatesZ;
//
//    return code;
//}
//
//void QC_LDPC_CSS::EqNodeUpdate(FloatArray2d_h &eqNodes, FloatArray2d_h varNodes, IntArray2d_h eqNodeVarIndices, IntArray1d_h syndrome)
//{
//    // For a check node interested in variables a,b,c,d to estimate the updated probability for variable a
//    // syndrome = 0: even # of errors -> pa' = pb(1-pc)(1-pd) + pc(1-pb)(1-pd) + pd(1-pb)(1-pc) + pb*pc*pd
//    //                                       = 0.5 * (1 - (1-2pb)(1-2pc)(1-2pd))
//    // syndrome = 1: odd # of errors -> pa' = (1-pb)(1-pc)(1-pd) + pb*pc*(1-pd) + pb*(1-pc)*pd + (1-pb)*pc*pd
//    //                                      = 0.5 * (1 + (1-2pb)(1-2pc)(1-2pd))
//    int numEqs = eqNodes.num_rows;
//    int numVarsPerEq = eqNodeVarIndices.num_cols;
//    int n = varNodes.num_rows;
//    for (auto eqIdx = 0; eqIdx < numEqs; ++eqIdx) // loop over check nodes (parity equations)
//    {
//        int firstVarIdx = eqIdx*numVarsPerEq;
//        // loop over variables to be updated for this check node
//        for (auto i = 0; i < numVarsPerEq; ++i) 
//        {
//            int index = firstVarIdx + i; // 1d array index to look up the variable index
//            int varIdx = eqNodeVarIndices.values[index]; // variable index under investigation for this eq
//            float product = 1.0f; // reset product
//            // loop over all other variables in the equation, accumulate (1-2p) terms
//            for (auto k = 0; k < numVarsPerEq; ++k) 
//            {
//                if (k == i) continue; // skip the variable being updated
//                int otherIndex = firstVarIdx + k; // 1d array index to look up the variable index
//                int otherVarIdx = eqNodeVarIndices.values[otherIndex];
//
//                // the index holding the estimate beinng used for this eq
//                int varNodesIndex = otherVarIdx * numEqs + eqIdx; 
//                float value = varNodes.values[varNodesIndex]; // belief value for this variable and this eq
//                product *= (1.0f - 2.0f*value);
//            }
//            int cnIdx = eqIdx * n + varIdx; // index for value within the check node array to update
//            if (syndrome[eqIdx]) {
//                eqNodes.values[cnIdx] = 0.5 * (1.0f + product); // syndrome = 1 -> odd parity
//            }
//            else {
//                eqNodes.values[cnIdx] = 0.5f * (1.0f - product); // syndrome = 0 -> even parity
//            }
//        }
//    }
//    //    WriteToFile(eqNodeBeliefs, "results/CheckNodeBeliefs.txt");
//}
//
//void QC_LDPC_CSS::EqNodeUpdate(std::vector<std::vector<float>>& varNodeEstimates,
//                                  std::vector<std::vector<float>>& eqNodeBeliefs, 
//                                  std::vector<std::vector<int>> parityCheckMatrix,
//                                  std::vector<int> syndrome)
//{
//    // For a check node interested in variables a,b,c,d to estimate the updated probability for variable a
//    // syndrome = 0: even # of errors -> pa' = pb(1-pc)(1-pd) + pc(1-pb)(1-pd) + pd(1-pb)(1-pc) + pb*pc*pd
//    //                                       = 0.5 * (1 - (1-2pb)(1-2pc)(1-2pd))
//    // syndrome = 1: odd # of errors -> pa' = (1-pb)(1-pc)(1-pd) + pb*pc*(1-pd) + pb*(1-pc)*pd + (1-pb)*pc*pd
//    //                                      = 0.5 * (1 + (1-2pb)(1-2pc)(1-2pd))
//    int numEqs = eqNodeBeliefs.size();
//    int n = varNodeEstimates.size();
//
//    for (auto eqIdx = 0; eqIdx < numEqs; ++eqIdx) // loop over check nodes (parity equations)
//    {
//        for (auto varIdx = 0; varIdx < n; ++varIdx) // loop over variables to be updated for this check node
//        { 
//            eqNodeBeliefs[eqIdx][varIdx] = 0.0f; // not necessary, makes file output nicer.
//            if (!parityCheckMatrix[eqIdx][varIdx]) continue; // if the parity check matrix is 0, the eq doesn't involve this var
//            float product = 1.0f; // reset product
//            for (auto otherVarIdx = 0; otherVarIdx < n; ++otherVarIdx) // loop over all other variables, accumulate (1-2p) terms
//            { 
//                if (!parityCheckMatrix[eqIdx][otherVarIdx]) continue; // skip zeros
//                if (otherVarIdx == varIdx) continue; // skip the variable being updated
//                product *= (1.0f - 2.0f*varNodeEstimates[otherVarIdx][eqIdx]);
//            }
//            if(syndrome[eqIdx]) eqNodeBeliefs[eqIdx][varIdx] = 0.5 * (1.0f + product); // syndrome = 1 -> odd parity
//            else eqNodeBeliefs[eqIdx][varIdx] = 0.5f * (1.0f - product); // syndrome = 0 -> even parity
//        }
//    }
////    WriteToFile(eqNodeBeliefs, "results/CheckNodeBeliefs.txt");
//}
//
//void QC_LDPC_CSS::VarNodeUpdate(FloatArray2d_h eqNodes, FloatArray2d_h& varNodes, IntArray2d_h varNodeEqIndices, float errorProbability, bool last)
//{
//    // For a variable node connected to check nodes 1,2,3,4 use the following formula to send an estimate to var node 1
//    // p1' = K*pch*p2*p3*p4   (pch is the channel error probability. ignore the estimate received from check node 1 unless last)
//    // where K = 1/[(1-pch)(1-p2)(1-p3)(1-p4)... + pch*p2*p3*p4...]
//    int numEqs = eqNodes.num_rows;
//    int n = varNodes.num_rows;
//    int numEqsPerVar = varNodeEqIndices.num_cols;
//    
//    for (auto varIdx = 0; varIdx < n; ++varIdx) // loop over all variables
//    {
//        int firstVarNode = varIdx * numEqs; // start of entries in VarNodes array for this variable
//        int firstEqIndices = varIdx * numEqsPerVar; // starting point for first equation in the index list for this var.
//        for (auto j = 0; j < numEqsPerVar; ++j) // loop over all equations for this variable
//        {
//            // find the index of the equation estimate being updated
//            int index = firstEqIndices + j;
//            int eqIdx = varNodeEqIndices.values[index];
//
//            // 1d index for var nodes entry being updated
//            int varNodesIdx = firstVarNode + eqIdx; 
//
//            // start with a priori channel error probability
//            float prodP = errorProbability; 
//            float prodOneMinusP = 1.0f - errorProbability;
//
//            // calculate the updated probability for this check node based on belief estimates of all OTHER check nodes
//            for (auto k = 0; k < numEqsPerVar; ++k) 
//            {
//                int index2 = firstEqIndices + k; // 1d index for entry in the index array
//                int otherEQIdx = varNodeEqIndices.values[index2];
//
//                if (otherEQIdx == eqIdx && !last) continue; 
//                // 1d index for check nodes belief being used
//                int checkNodesIdx = otherEQIdx * n + varIdx; 
//                float p = eqNodes.values[checkNodesIdx];
//
//                prodOneMinusP *= (1.0f - p);
//                prodP *= p;
//            }
//            float value = prodP / (prodOneMinusP + prodP);
//            varNodes.values[varNodesIdx] = value;
//        }
//    }
//}
//
//void QC_LDPC_CSS::VarNodeUpdate(std::vector<std::vector<float>>& varNodeEstimates,
//                                     std::vector<std::vector<float>>& eqNodeBeliefs, 
//                                     std::vector<std::vector<int>> parityCheckMatrix,
//                                     float errorProbability, bool last)
//{
//    // For a variable node connected to check nodes 1,2,3,4 use the following formula to send an estimated probability to node 1
//    // p1' = K*pch*p2*p3*p4   (pch is the channel error probability. ignore the estimate received from check node 1)
//    // where K = 1/[(1-p1)(1-p2)(1-p3)... + p1*p2*p3...]
//    int numEqs = eqNodeBeliefs.size();
//    int n = varNodeEstimates.size();
//    for (auto varIdx = 0; varIdx < n; ++varIdx) // loop over all variables
//    {
//        for (auto eqIdx = 0; eqIdx < numEqs; ++eqIdx) // loop over all equations
//        {
//            varNodeEstimates[varIdx][eqIdx] = 0.0f; // not necessary, makes output nicer
//            if (!parityCheckMatrix[eqIdx][varIdx]) continue; // skip equations that this variable isn't involved in
//
//            float prodP = errorProbability; // start with a priori channel error probability
//            float prodOneMinusP = 1.0f - errorProbability;
//            // calculate the updated probability for this check node based on belief estimates of all OTHER check nodes
//            for (auto otherEqIdx = 0; otherEqIdx < numEqs; ++otherEqIdx) // loop over all equation estimates
//            {
//                if (otherEqIdx == eqIdx && !last) continue; // skip the belief estimate from j to update the probability sent to j
//                if (!parityCheckMatrix[otherEqIdx][varIdx]) continue; // skip equations that this variable isn't involved in
//                float p = eqNodeBeliefs[otherEqIdx][varIdx];
//
//                prodOneMinusP *= (1.0f - p);
//                prodP *= p;
//            }
//            float value = prodP / (prodOneMinusP + prodP);
////            std::cout << "Setting var: " << i << " eq: " << j << " value: " << value << std::endl;
//            varNodeEstimates[varIdx][eqIdx] = value;
//        }
//    }
////    WriteToFile(varNodeEstimates, "results/VariableNodeEstimates.txt");
//}
//
//std::vector<int> QC_LDPC_CSS::GetXSyndrome(std::vector<int> xErrors)
//{
//    std::vector<int> syndrome(_numEqsX);
//    for (int row = 0; row < _numEqsX; ++row)
//    {
//        auto x = 0;
//        for (int col = 0; col < _numVars; ++col)
//        {
//            x += _hHC_vec[row][col] * xErrors[col];
//        }
//        syndrome[row] = x % 2;
//    }
//    return syndrome;
//}
//
//std::vector<int> QC_LDPC_CSS::GetZSyndrome(std::vector<int> zErrors)
//{
//    std::vector<int> syndrome(_numEqsX);
//    for (int row = 0; row < _numEqsX; ++row)
//    {
//        auto x = 0;
//        for (int col = 0; col < _numVars; ++col)
//        {
//            x += _hHD_vec[row][col] * zErrors[col];
//        }
//        syndrome[row] = x % 2;
//    }
//    return syndrome;
//}
//
//void QC_LDPC_CSS::InitVarNodesArray(FloatArray2d_h& varNodes_h, FloatArray2d_d& varNodes_d, const IntArray2d_h& parityCheckMatrix, const int NUM_CONCURRENT_THREADS, float errorProbability)
//{
//    int n = parityCheckMatrix.num_cols;
//    int numEqs = parityCheckMatrix.num_rows;
//    int size = n*numEqs;
//    for (int varIdx = 0; varIdx < n; ++varIdx)
//    {
//        for (int eqIdx = 0; eqIdx < numEqs; ++eqIdx)
//        {
//            int pcmIdx = eqIdx * n + varIdx;
//            for (int n = 0; n<NUM_CONCURRENT_THREADS; ++n)
//            {
//                if (!parityCheckMatrix.values[pcmIdx]) continue;
//                int varNodesIdx = n*size + varIdx*numEqs + eqIdx;
//                varNodes_h.values[varNodesIdx] = errorProbability;
//            }
//        }
//    }
//    thrust::copy(varNodes_h.values.begin(), varNodes_h.values.end(), varNodes_d.values.begin());
//}
//
//void QC_LDPC_CSS::SetDeviceSyndrome(const std::vector<int>& syndrome_h, const IntArray2d_d& syndrome_d)
//{
//    for(int i=0; i<syndrome_h.size(); ++i)
//    {
//        syndrome_d.values[i] = syndrome_h[i];
//    }
//}
//
//QC_LDPC_CSS::Statistics QC_LDPC_CSS::GetStatistics(int errorWeight, int numErrors, float errorProbability, int maxIterations)
//{
//    //float p = 2 / 3 * errorProbability;
//    //const int NUM_CONCURRENT_THREADS = 32;
//    //std::vector<int> xErrors(_numVars, 0);
//    //std::vector<int> zErrors(_numVars, 0);
//
//    //// set up host and device memory for calculations
//    //IntArray2d_h xSyndromeArray_h(NUM_CONCURRENT_THREADS,_numEqsX);
//    //IntArray2d_d xSyndromeArray_d(NUM_CONCURRENT_THREADS,_numEqsX);
//    //int* xSyndrome_d_ptrs[NUM_CONCURRENT_THREADS];
//    //for (int i = 0; i < NUM_CONCURRENT_THREADS; ++i)
//    //    xSyndrome_d_ptrs[i] = thrust::raw_pointer_cast(&xSyndromeArray_d.values[i*_numEqsX]);
//
//    //IntArray2d_h zSyndromeArray_h(NUM_CONCURRENT_THREADS,_numEqsZ);
//    //IntArray2d_d zSyndromeArray_d(NUM_CONCURRENT_THREADS,_numEqsZ);   
//    //int* zSyndrome_d_ptrs[NUM_CONCURRENT_THREADS];
//    //for (int i = 0; i < NUM_CONCURRENT_THREADS; ++i) 
//    //    xSyndrome_d_ptrs[i] = thrust::raw_pointer_cast(&xSyndromeArray_d.values[i*_numEqsX]);
//
//    //int size = _numVars * _numEqsX;
//    //FloatArray2d_h varNodesX_h(NUM_CONCURRENT_THREADS, size,0.0f);
//    //FloatArray2d_d varNodesX_d(NUM_CONCURRENT_THREADS, size, 0.0f);
//    //float* varNodesX_d_ptrs[NUM_CONCURRENT_THREADS];
//    //for (int i = 0; i < NUM_CONCURRENT_THREADS; ++i)
//    //    varNodesX_d_ptrs[i] = thrust::raw_pointer_cast(&varNodesX_d.values[i*size]);
//
//    //FloatArray2d_h eqNodesX_h(NUM_CONCURRENT_THREADS, size, 0.0f);
//    //FloatArray2d_d eqNodesX_d(NUM_CONCURRENT_THREADS, size, 0.0f);
//    //float* eqNodesX_d_ptrs[NUM_CONCURRENT_THREADS];
//    //for (int i = 0; i < NUM_CONCURRENT_THREADS; ++i)
//    //    eqNodesX_d_ptrs[i] = thrust::raw_pointer_cast(&eqNodesX_d.values[i*size]);
//
//    //size = _numVars * _numEqsZ;
//    //FloatArray2d_h varNodesZ_h(NUM_CONCURRENT_THREADS, size,0.0f);
//    //FloatArray2d_d varNodesZ_d(NUM_CONCURRENT_THREADS, size, 0.0f);
//    //float* varNodesZ_d_ptrs[NUM_CONCURRENT_THREADS];
//    //for (int i = 0; i < NUM_CONCURRENT_THREADS; ++i)
//    //    varNodesZ_d_ptrs[i] = thrust::raw_pointer_cast(&varNodesZ_d.values[i*size]);    
//    //
//    //FloatArray2d_h eqNodesZ_h(NUM_CONCURRENT_THREADS, size, 0.0f);
//    //FloatArray2d_d eqNodesZ_d(NUM_CONCURRENT_THREADS, size, 0.0f);
//    //float* eqNodesZ_d_ptrs[NUM_CONCURRENT_THREADS];
//    //for (int i = 0; i < NUM_CONCURRENT_THREADS; ++i)
//    //    eqNodesZ_d_ptrs[i] = thrust::raw_pointer_cast(&eqNodesZ_d.values[i*size]);
//
//    //for (int i = 0; i < numErrors; ++i) {
//    //    InitVarNodesArray(varNodesX_h, varNodesX_d, _pcmX_h, NUM_CONCURRENT_THREADS, p);
//    //    InitVarNodesArray(varNodesZ_h, varNodesZ_d, _pcmZ_h, NUM_CONCURRENT_THREADS, p);
//    //    for (int j = 0; j < NUM_CONCURRENT_THREADS; ++j) {
//    //        _errorGenerator.GenerateError(xErrors, zErrors, errorWeight);
//    //        SetDeviceSyndrome(GetXSyndrome(xErrors), xSyndromeArray_d);
//    //        SetDeviceSyndrome(GetZSyndrome(zErrors), zSyndromeArray_d);
//    //    }
//    //}
//    return Statistics();
//}
//
//bool QC_LDPC_CSS::CheckConvergence(const std::vector<std::vector<float>>& estimates, float high, float low)
//{
//    // loop over all estimates
//    for (auto i = 0; i < estimates.size(); ++i) {
//        for (auto j = 0; j < estimates[i].size(); ++j) {
//            if (estimates[i][j] != 0.0f) {
//                // if any estimate is between the bounds we have failed to converge
//                if (estimates[i][j] > low && estimates[i][j] < high) return false;
//            }
//        }
//    }
//    return true;
//}
//
//bool QC_LDPC_CSS::CheckConvergence(const cusp::array2d<float,cusp::host_memory,cusp::row_major>& estimates, float high, float low)
//{
//    // loop over all estimates
//    for (auto i = 0; i < estimates.num_rows; ++i) {
//        for (auto j = 0; j < estimates.num_cols; ++j) {
//            int index = i * estimates.num_cols + j;
//            if (estimates.values[index] != 0.0f) {
//                // if any estimate is between the bounds we have failed to converge
//                if (estimates.values[index] > low && estimates.values[index] < high) return false;
//            }
//        }
//    }
//    return true;
//}