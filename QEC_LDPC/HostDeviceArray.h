#pragma once
#include "thrust/device_vector.h"
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/array1d.h>
typedef cusp::array2d<int, cusp::host_memory, cusp::row_major> IntArray2d_h;
typedef cusp::array2d<int, cusp::device_memory, cusp::row_major> IntArray2d_d;

typedef cusp::array2d<float, cusp::host_memory, cusp::row_major> FloatArray2d_h;
typedef cusp::array2d<float, cusp::device_memory, cusp::row_major> FloatArray2d_d;

typedef cusp::array1d<int, cusp::host_memory> IntArray1d_h;
typedef cusp::array1d<int, cusp::device_memory> IntArray1d_d;
//
//template<typename hostType, typename deviceType, typename devicePtrType> class HostDeviceArray1d{
//public:
//    hostType hostArray;
//    deviceType deviceArray;
//    devicePtrType devicePtr;
//
//    HostDeviceArray1d(int numElements)
//    {
//        hostArray(numElements);
//        deviceArray(numElements);
//        devicePtr = thrust::raw_pointer_cast(&deviceArray[0]);
//    }
//};
//
//template<typename hostType, typename deviceType, typename devicePtrType> class HostDeviceArray2d {
//public:
//    hostType hostArray;
//    deviceType deviceArray;
//    devicePtrType devicePtr;
//
//    HostDeviceArray2d(int numRows, int numCols) : hostArray(numRows, numCols), deviceArray(numRows, numCols)
//    {
//        devicePtr = thrust::raw_pointer_cast(&deviceArray.values[0]);
//    }
//};
