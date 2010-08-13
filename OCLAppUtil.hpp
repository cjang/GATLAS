#ifndef _GATLAS_OCL_APP_UTIL_HPP_
#define _GATLAS_OCL_APP_UTIL_HPP_

//    Copyright 2010 Chris Jang
//
//    This file is part of GATLAS.
//
//    GATLAS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    GATLAS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with GATLAS.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <string>
#include <math.h>
#include "OCLApp.hpp"

#include "declare_namespace"

template <typename SCALAR_TYPE>
SCALAR_TYPE posrand() {
    SCALAR_TYPE a;
    while (0 == (a = drand48())) ;
    return a;
}

template <typename SCALAR_TYPE>
void fillrand(SCALAR_TYPE *outM, const size_t length) {
    for (size_t i = 0; i < length; i++)
        outM[i] = posrand<SCALAR_TYPE>();
}

template <typename SCALAR_TYPE>
void fillconst(SCALAR_TYPE *outM, const SCALAR_TYPE value, const size_t length) {
    for (size_t i = 0; i < length; i++)
        outM[i] = value;
}

template <typename SCALAR_TYPE>
long double absdiff(const SCALAR_TYPE *m1, const SCALAR_TYPE *m2, const size_t length) {
    long double accumdiff = 0;
    for (size_t i = 0; i < length; i++)
        accumdiff += fabsl(m1[i] - m2[i]);
    return accumdiff;
}

template <typename T>
bool checkArray(const T *ptr, const size_t length, const T testValue) {
    bool goodElements = true;
    for (size_t i = 0; i < length; i++)
        if (testValue != ptr[i]) goodElements = false;
    return goodElements;
}

template <typename T>
bool clearBuffer(OCLApp& oclApp, const size_t bufferIndex, const T value = 0) {
    oclApp.memsetBuffer<T>(bufferIndex, value);
    const int syncBuf = oclApp.enqueueWriteBuffer(bufferIndex);
    if (-1 == syncBuf || !oclApp.wait(syncBuf)) {
        std::cerr << "error: reset output buffer " << bufferIndex << std::endl;
        return false;
    }
    return true;
}

bool clearImage(OCLApp& oclApp, const size_t imageIndex);

template <typename T>
bool fillrandBuffer(OCLApp& oclApp, const size_t bufferIndex, const size_t length) {
    T *ptr = oclApp.bufferPtr<T>(bufferIndex);
    fillrand<T>(ptr, length);
    const int syncBuf = oclApp.enqueueWriteBuffer(bufferIndex);
    if (-1 == syncBuf || !oclApp.wait(syncBuf)) {
        std::cerr << "error: random fill buffer " << bufferIndex << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool fillrandImage(OCLApp& oclApp, const size_t imageIndex, const size_t length) {
    T *ptr = oclApp.imagePtr<T>(imageIndex);
    fillrand<T>(ptr, length);
    const int syncImg = oclApp.enqueueWriteImage(imageIndex);
    if (-1 == syncImg || !oclApp.wait(syncImg)) {
        std::cerr << "error: random fill image " << imageIndex << std::endl;
        return false;
    }
    return true;
}

template <typename T>
void printArray(const T *ptr, const size_t length) {
    for (size_t j = 0; j < length; j++) {
        if (0 != j) std::cerr << " ";
        std::cerr << ptr[j];
    }
    std::cerr << std::endl;
}

template <typename T>
void printArray(const T *ptr, const size_t width, const size_t height) {
    for (size_t i = 0; i < height; i++) {
        std::cerr << "[" << i << "]";
        for (size_t j = 0; j < width; j++)
            std::cerr << " " << ptr[i * width + j];
        std::cerr << std::endl;
    }
}

template <typename T>
void printDiff(const T *ptr1, const T *ptr2, const size_t length) {
    for (size_t j = 0; j < length; j++) {
        if (0 != j) std::cerr << " ";
        std::cerr << (ptr1[j] - ptr2[j]);
    }
    std::cerr << std::endl;
}

template <typename T>
void printDiff(const T *ptr1, const T *ptr2, const size_t width, const size_t height) {
    for (size_t i = 0; i < height; i++) {
        std::cerr << "[" << i << "]";
        for (size_t j = 0; j < width; j++)
            std::cerr << " " << (ptr1[i * width + j] - ptr2[i * width +j]);
        std::cerr << std::endl;
    }
}

template <typename T>
bool checkBuffer(OCLApp& oclApp,
                 const size_t bufferIndex,
                 const size_t length,
                 const T      testValue,
                 const bool   printOutput) {
    const T *ptr = oclApp.bufferPtr<T>(bufferIndex);

    // quick and primitive test here
    // e.g. if A and B are all 1s, then each element of C equals the matrix size
    const bool goodElements = checkArray(ptr, length, testValue);

    // print output matrix for debugging
    if (printOutput) printArray(ptr, length);

    return goodElements;
}

template <typename T>
bool checkBuffer(OCLApp& oclApp,
                 const size_t bufferIndex,
                 const size_t width,
                 const size_t height,
                 const T      testValue,
                 const bool   printOutput) {
    const T *ptr = oclApp.bufferPtr<T>(bufferIndex);

    // quick and primitive test here
    // e.g. if A and B are all 1s, then each element of C equals the matrix size
    const bool goodElements = checkArray(ptr, width * height, testValue);

    // print output matrix for debugging
    if (printOutput) printArray(ptr, width, height);

    return goodElements;
}

template <typename T>
bool checkImage(OCLApp& oclApp,
                const size_t imageIndex,
                const size_t length,
                const T      testValue,
                const bool   printOutput) {
    const T *ptr = oclApp.imagePtr<T>(imageIndex);

    // quick and primitive test here
    // e.g. if A and B are all 1s, then each element of C equals the matrix size
    const bool goodElements = checkArray(ptr, length, testValue);

    // print output matrix for debugging
    if (printOutput) printArray(ptr, length);

    return goodElements;
}

template <typename T>
bool checkImage(OCLApp& oclApp,
                const size_t imageIndex,
                const size_t width,
                const size_t height,
                const T      testValue,
                const bool   printOutput) {
    const T *ptr = oclApp.imagePtr<T>(imageIndex);

    // quick and primitive test here
    // e.g. if A and B are all 1s, then each element of C equals the matrix size
    const bool goodElements = checkArray(ptr, width * height, testValue);

    // print output matrix for debugging
    if (printOutput) printArray(ptr, width, height);

    return goodElements;
}

template <typename T>
bool checkBuffer(OCLApp&       oclApp,
                 const size_t  bufferIndex,
                 const size_t  length,
                 const T      *testBuffer,
                 const bool    printOutput) {
    const T *ptr = oclApp.bufferPtr<T>(bufferIndex);
    const long double diff = absdiff(ptr, testBuffer, length);
    std::cerr << "absdiff: " << diff << "\t";
    if (printOutput) printDiff(ptr, testBuffer, length);
    return diff < static_cast<double>(1) / (length);
}

template <typename T>
bool checkBuffer(OCLApp&       oclApp,
                 const size_t  bufferIndex,
                 const size_t  width,
                 const size_t  height,
                 const T      *testBuffer,
                 const bool    printOutput) {
    const T *ptr = oclApp.bufferPtr<T>(bufferIndex);
    const long double diff = absdiff(ptr, testBuffer, width * height);
    std::cerr << "absdiff: " << diff << "\t";
    if (printOutput) printDiff(ptr, testBuffer, width, height);
    return diff < static_cast<double>(1) / (width * height);
}

template <typename T>
bool checkImage(OCLApp&       oclApp,
                const size_t  imageIndex,
                const size_t  length,
                const T      *testImage,
                const bool    printOutput) {
    const T *ptr = oclApp.imagePtr<T>(imageIndex);
    const long double diff = absdiff(ptr, testImage, length);
    std::cerr << "absdiff: " << diff << "\t";
    if (printOutput) printDiff(ptr, testImage, length);
    return diff < static_cast<double>(1) / (length);
}

template <typename T>
bool checkImage(OCLApp&       oclApp,
                const size_t  imageIndex,
                const size_t  width,
                const size_t  height,
                const T      *testImage,
                const bool    printOutput) {
    const T *ptr = oclApp.imagePtr<T>(imageIndex);
    const long double diff = absdiff(ptr, testImage, width * height);
    std::cerr << "absdiff: " << diff << "\t";
    if (printOutput) printDiff(ptr, testImage, width, height);
    return diff < static_cast<double>(1) / (width * height);
}

template <typename T>
int createImageR(OCLApp& oclApp, const size_t width, const size_t height,
                 const std::string& argName,
                 const T value = 0, const bool pinned = false) {
    const size_t typeSize = sizeof(T) / sizeof(float);
    float *ptr = alloc_memalign<float, 4>(width * height * typeSize);
    if (NULL == ptr) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
        return -1;
    }
    fillconst<T>(reinterpret_cast<T*>(ptr), value, width * height);
    const int imgHandle = isfloat<T>()
                              ? oclApp.createImage(typeSize*width/4, height, OCLApp::READ, ptr, pinned)
                              : oclApp.createImage(typeSize*width/4, height, OCLApp::READ, reinterpret_cast<unsigned int*>(ptr), pinned);
    if (-1 == imgHandle) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
    } else {
        oclApp.ownImage(imgHandle);
    }
    return imgHandle;
}

template <typename T>
int createImageW(OCLApp& oclApp, const size_t width, const size_t height,
                 const std::string& argName,
                 const float value = 0, const bool pinned = false) {
    const size_t typeSize = sizeof(T) / sizeof(float);
    float *ptr = alloc_memalign<float, 4>(width * height * typeSize);
    if (NULL == ptr) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
        return -1;
    }
    fillconst<T>(reinterpret_cast<T*>(ptr), value, width * height);
    const int imgHandle = isfloat<T>()
                              ? oclApp.createImage(typeSize*width/4, height, OCLApp::WRITE, ptr, pinned)
                              : oclApp.createImage(typeSize*width/4, height, OCLApp::WRITE, reinterpret_cast<unsigned int*>(ptr), pinned);
    if (-1 == imgHandle) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
    } else {
        oclApp.ownImage(imgHandle);
    }
    return imgHandle;
}

template <typename T>
int createImageRW(OCLApp& oclApp, const size_t width, const size_t height,
                  const std::string& argName,
                  const float value = 0, const bool pinned = false) {
    const size_t typeSize = sizeof(T) / sizeof(float);
    float *ptr = alloc_memalign<float, 4>(width * height * typeSize);
    if (NULL == ptr) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
        return -1;
    }
    fillconst<T>(reinterpret_cast<T*>(ptr), value, width * height);
    const int imgHandle = isfloat<T>()
                              ? oclApp.createImage(typeSize*width/4, height, OCLApp::READWRITE, ptr, pinned)
                              : oclApp.createImage(typeSize*width/4, height, OCLApp::READWRITE, reinterpret_cast<unsigned int*>(ptr), pinned);
    if (-1 == imgHandle) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
    } else {
        oclApp.ownImage(imgHandle);
    }
    return imgHandle;
}

bool setArgImage(OCLApp& oclApp, const size_t kernelHandle, const size_t argIndex, const size_t imgHandle,
                 const std::string& argName);

template <typename SCALAR_TYPE, size_t VECTOR_LENGTH>
int createBufferR(OCLApp& oclApp, const size_t bufSize,
                  const std::string& argName,
                  const SCALAR_TYPE value = 0, const bool pinned = false) {
    SCALAR_TYPE *ptr = alloc_memalign<SCALAR_TYPE, VECTOR_LENGTH>(bufSize);
    if (NULL == ptr) {
        std::cerr << "error: OCL create buffer for " << argName << std::endl;
        return -1;
    }
    fillconst<SCALAR_TYPE>(ptr, value, bufSize);
    const int bufHandle = oclApp.createBuffer<SCALAR_TYPE>(bufSize, OCLApp::READ, ptr, pinned);
    if (-1 == bufHandle) {
        std::cerr << "error: OCL create buffer for " << argName << std::endl;
    } else {
        oclApp.ownBuffer(bufHandle);
    }
    return bufHandle;
}

template <typename SCALAR_TYPE, size_t VECTOR_LENGTH>
int createBufferW(OCLApp& oclApp, const size_t bufSize,
                  const std::string& argName,
                  const SCALAR_TYPE value = 0, const bool pinned = false) {
    SCALAR_TYPE *ptr = alloc_memalign<SCALAR_TYPE, 1>(bufSize);
    if (NULL == ptr) {
        std::cerr << "error: OCL create buffer for " << argName << std::endl;
        return -1;
    }
    fillconst<SCALAR_TYPE>(ptr, value, bufSize);
    const int bufHandle = oclApp.createBuffer<SCALAR_TYPE>(bufSize, OCLApp::WRITE, ptr, pinned);
    if (-1 == bufHandle) {
        std::cerr << "error: OCL create buffer for " << argName << std::endl;
    } else {
        oclApp.ownBuffer(bufHandle);
    }
    return bufHandle;
}

template <typename SCALAR_TYPE, size_t VECTOR_LENGTH>
int createBufferRW(OCLApp& oclApp, const size_t bufSize,
                   const std::string& argName,
                   const SCALAR_TYPE value = 0, const bool pinned = false) {
    SCALAR_TYPE *ptr = alloc_memalign<SCALAR_TYPE, VECTOR_LENGTH>(bufSize);
    if (NULL == ptr) {
        std::cerr << "error: OCL create buffer for " << argName << std::endl;
        return -1;
    }
    fillconst<SCALAR_TYPE>(ptr, value, bufSize);
    const int bufHandle = oclApp.createBuffer<SCALAR_TYPE>(bufSize, OCLApp::READWRITE, ptr, pinned);
    if (-1 == bufHandle) {
        std::cerr << "error: OCL create buffer for " << argName << std::endl;
    } else {
        oclApp.ownBuffer(bufHandle);
    }
    return bufHandle;
}

bool syncBufferToDevice(OCLApp& oclApp, const size_t bufferIndex);
bool syncBufferFromDevice(OCLApp& oclApp, const size_t bufferIndex);
bool syncImageToDevice(OCLApp& oclApp, const size_t imageIndex);
bool syncImageFromDevice(OCLApp& oclApp, const size_t imageIndex);

bool setArgGlobal(OCLApp& oclApp, const size_t kernelHandle, const size_t argIndex, const size_t bufHandle,
                  const std::string& argName);

template <typename SCALAR_TYPE>
bool setArgLocal(OCLApp& oclApp, const size_t kernelHandle, const size_t argIndex, const size_t bufSize,
                 const std::string& argName) {
    if (!oclApp.setArgLocal<SCALAR_TYPE>(kernelHandle, argIndex, bufSize)) {
        std::cerr << "error: set local kernel argument " << argName << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool setArgValue(OCLApp& oclApp, const size_t kernelHandle, const size_t argIndex, const T value,
                 const std::string& argName) {
    if (!oclApp.setArgValue<T>(kernelHandle, argIndex, value)) {
        std::cerr << "error: set private kernel argument " << argName << std::endl;
        return false;
    }
    return true;
}

}; // namespace

#endif
