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
double absdiff(const SCALAR_TYPE *m1, const SCALAR_TYPE *m2, const size_t length) {
    double accumdiff = 0;
    for (size_t i = 0; i < length; i++)
        accumdiff += fabs(m1[i] - m2[i]);
    return accumdiff;
}

int createImageR(OCLApp& oclApp, const size_t width, const size_t height,
                 const std::string& argName,
                 const float value = 0, const bool pinned = false);

int createImageW(OCLApp& oclApp, const size_t width, const size_t height,
                 const std::string& argName,
                 const float value = 0, const bool pinned = false);

int createImageRW(OCLApp& oclApp, const size_t width, const size_t height,
                  const std::string& argName,
                  const float value = 0, const bool pinned = false);

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
