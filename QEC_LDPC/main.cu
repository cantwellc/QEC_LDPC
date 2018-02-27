#include <thrust/version.h>
#include <cusp/version.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime_api.h>
#include "Quantum_LDPC_Code.h"
#include "DecoderCPU.h"
#include "ArrayOutput.h"

void printCudaInfoToLog(std::ofstream& log)
{
    int thrust_major = THRUST_MAJOR_VERSION;
    int thrust_minor = THRUST_MINOR_VERSION;
    int cusp_major = CUSP_MAJOR_VERSION;
    int cusp_minor = CUSP_MINOR_VERSION;

    log << "Thrust v" << thrust_major << "." << thrust_minor << std::endl;
    log << "Cusp   v" << cusp_major << "." << cusp_minor << std::endl;

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
        size_t mem_tot;
        size_t mem_free;
        cudaMemGetInfo(&mem_free, &mem_tot);
        log << "Total Memory : " << mem_tot << std::endl;
        log << "Free memory : " << mem_free << std::endl;
    }
}

int main(int argc, char** argv)
{
    std::ofstream log;
    log.open("output_log.txt", std::ios::app);
    if(!log.is_open())
    {
        throw std::string ("Unable to open output log file");
    }
    std::time_t ts = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    log << std::endl << std::ctime(&ts);

    printCudaInfoToLog(log);

    if(argc != 2) {
        log << "Must provide initialization file." << std::endl;
        log.close();
        exit(0);
    }
    std::string initFile = argv[1];
    std::ifstream init(initFile);
    
    if(!init.is_open())
    {
        log << "Unable to open init file \"" << initFile 
        << "\". Please make sure the file exists in the current directory." << std::endl;
        log.close();
        exit(0);
    }
    
    log << "Initializing run from file " << initFile << std::endl;
    
    std::string codeFile;
    init >> codeFile;

    try {
        Quantum_LDPC_Code code = Quantum_LDPC_Code::createFromFile(codeFile);
        DecoderCPU decoder(code);

        int w, W, COUNT, MAX_ITERATIONS;
        float p;

        init >> w;
        init >> W;
        init >> COUNT;
        init >> MAX_ITERATIONS;
        init >> p;
        init.close();

        for (;w <= W;++w)
        {
            std::stringstream fileName;
            fileName << "results/" << code << "_W_" << w << "_MAX_" << MAX_ITERATIONS << "_p_" << p << ".txt";
            auto str = fileName.str();
            auto end = std::remove(str.begin(), str.end(), ' ');
            str.erase(end, str.end());
            std::cout << str << std::endl;
            std::ofstream outFile;
            outFile.open(fileName.str(), std::ios_base::app);
            auto stats = decoder.GetStatistics(w, COUNT, p, MAX_ITERATIONS);
            outFile << stats << std::endl << std::endl;
            outFile.close();
        }

    }catch(std::string s)
    {
        log << s << std::endl;
        log.close();
        init.close();
        exit(0);
    }

    log << "Run complete." << std::endl;
    log.close();

	return 1;
}
