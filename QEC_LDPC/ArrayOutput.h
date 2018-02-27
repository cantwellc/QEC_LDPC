#pragma once
#include "HostDeviceArray.h"
#include <fstream>
#include <iomanip>

void WriteToFile(IntArray2d_h vec, const char* str)
{
    std::ofstream file;
    file.open(str, std::ios::app);
    if (file.is_open()) {
        std::cout << "Writing to file " << str << std::endl;
        for (auto i = 0; i < vec.num_rows; ++i)
        {
            for (auto j = 0; j < vec.num_cols; ++j) {
                int index = i*vec.num_cols + j;
                auto v = vec.values[index];
                file << v << " ";
            }
            file << "\n";
        }
        file << "\n\n";
        file.close();
    }
    else
    {
        std::cout << "Failed to open file " << str << std::endl;
    }
}

void WriteToFile(IntArray1d_h vec, const char* str)
{
    std::ofstream file;
    file.open(str, std::ios::app);
    if (file.is_open()) {
        std::cout << "Writing to file " << str << std::endl;
        for (auto i = 0; i < vec.size(); ++i)
        {
            auto v = vec[i];
            file << v << " ";
        }
        file << "\n\n";
        file.close();
    }
    else
    {
        std::cout << "Failed to open file " << str << std::endl;
    }
}

void WriteToFile(cusp::array2d<float, cusp::host_memory, cusp::row_major> vec, const char* str)
{
    std::ofstream file;
    file.open(str, std::ios::app);
    if (file.is_open()) {
        std::cout << "Writing to file " << str << std::endl;
        for (auto i = 0; i < vec.num_rows; ++i)
        {
            for (auto j = 0; j < vec.num_cols; ++j) {
                int index = i*vec.num_cols + j;
                auto v = vec.values[index];
                file << std::fixed << std::setprecision(3) << v << " ";
            }
            file << "\n";
        }
        file << "\n\n";
        file.close();
    }
    else
    {
        std::cout << "Failed to open file " << str << std::endl;
    }
}

void WriteToFile(std::vector<float> vec, const char* str, int numRows, int numCols)
{
    std::ofstream file;
    file.open(str, std::ios::app);
    if (file.is_open()) {
        std::cout << "Writing to file " << str << std::endl;
        for (auto i = 0; i < numRows; ++i)
        {
            for (auto j = 0; j < numCols; ++j) {
                int index = i*numCols + j;
                auto v = vec[index];
                file << v << " ";
            }
            file << "\n";
        }
        file << "\n\n";
        file.close();
    }
    else
    {
        std::cout << "Failed to open file " << str << std::endl;
    }
}