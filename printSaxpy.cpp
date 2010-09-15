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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "KernelSaxpyBuffer.hpp"
#include "KernelSaxpyImage.hpp"

#include "using_namespace"

using namespace std;

bool parseOpts(int argc, char *argv[],
               size_t& packedKernels,
               bool& useMembufs,
               bool& useImages,
               bool& useFloat,
               bool& useDouble,
               size_t& vectorLength,
               int& M,
               int& N,
               int& groupHeight,
               int& groupWidth,
               int& blockHeight,
               int& blockWidth,
               int& extraParam,
               bool& vectorAttributeHint) {
    int opt;
    string kernelType = "<unspecified>";
    while ((opt = getopt(argc, argv, "hvC:T:m:n:G:Y:g:y:x:")) != -1) {
        switch (opt) {
            case ('h') :
                cerr << "usage: " << argv[0]
                     << " -T float1|float2|float4|double1|double2|double4|floatimg|doubleimg -m M [-n N]"
                        " [-C numKernels]"
                        " -Y groupHeight -G groupWidth -y blockHeight -g blockWidth -x extraParam"
                        " [-v] [-h]" << endl
                     << "\t-C number of coalesced kernels (default is 1)" << endl
                     << "\t-T kernel type: precision, vector length, memory buffers or images" << endl
                     << "\t-m matrix dimension M" << endl
                     << "\t-n matrix dimension N" << endl
                     << "\t-Y work item group height" << endl
                     << "\t-G work item group width" << endl
                     << "\t-y inner blocking height" << endl
                     << "\t-g inner blocking width" << endl
                     << "\t-x extra parameter" << endl
                     << "\t-v disable kernel vector attribute hint (default enabled)" << endl
                     << "\t-h help" << endl;
                exit(1);
            case ('C') : packedKernels = atoi(optarg); break;
            case ('T') : kernelType = optarg; break;
            case ('m') : M = atoi(optarg); break;
            case ('n') : N = atoi(optarg); break;
            case ('G') : groupWidth = atoi(optarg); break;
            case ('Y') : groupHeight = atoi(optarg); break;
            case ('g') : blockWidth = atoi(optarg); break;
            case ('y') : blockHeight = atoi(optarg); break;
            case ('x') : extraParam = atoi(optarg); break;
            case ('v') : vectorAttributeHint = false; break;
        }
    }

    // validate matrix dimensions
    bool rc = true;
    if (0 == packedKernels) {
        cerr << "error: number of kernels to coalesce must be at least one" << endl;
        rc = false;
    }
    vectorLength = 1;
    if ("float1" == kernelType) { useMembufs = true; useImages = false; useFloat = true; useDouble = false; vectorLength = 1; }
    else if ("float2" == kernelType) { useMembufs = true; useImages = false; useFloat = true; useDouble = false; vectorLength = 2; }
    else if ("float4" == kernelType) { useMembufs = true; useImages = false; useFloat = true; useDouble = false; vectorLength = 4; }
    else if ("double1" == kernelType) { useMembufs = true; useImages = false; useFloat = false; useDouble = true; vectorLength = 1; }
    else if ("double2" == kernelType) { useMembufs = true; useImages = false; useFloat = false; useDouble = true; vectorLength = 2; }
    else if ("double4" == kernelType) { useMembufs = true; useImages = false; useFloat = false; useDouble = true; vectorLength = 4; }
    else if ("floatimg" == kernelType) { useMembufs = false; useImages = true; useFloat = true; useDouble = false; vectorLength = 4; }
    else if ("doubleimg" == kernelType) { useMembufs = false; useImages = true; useFloat = false; useDouble = true; vectorLength = 2; }
    else {
        cerr << "error: invalid kernel type of " << kernelType << endl;
        rc = false;
    }
    const size_t VL = vectorLength;
    if (-1 == M) {
        cerr << "error: matrix dimension M must be specified" << endl;
        rc = false;
    } else {
        if (-1 == N) {
            N = VL;
        }
    }
    if (0 != M % VL) {
        cerr << "error: matrix dimension M must be multiple of " << VL << endl;
        rc = false;
    }
    if (0 != N % VL) {
        cerr << "error: matrix dimension N must be multiple of " << VL << endl;
        rc = false;
    }

    // kernel parameters
    if (-1 == groupHeight) {
        cerr << "error: must specify work item group height" << endl;
        rc = false;
    }
    if (-1 == groupWidth) {
        cerr << "error: must specify work item group width" << endl;
        rc = false;
    }
    if (-1 == blockHeight) {
        cerr << "error: must specify inner blocking height" << endl;
        rc = false;
    }
    if (-1 == blockWidth) {
        cerr << "error: must specify inner blocking width" << endl;
        rc = false;
    }
    if (-1 == extraParam) {
        cerr << "error: must specify extra parameter" << endl;
        rc = false;
    }

    return rc;
}

int main(int argc, char *argv[])
{
    size_t packedKernels = 1;
    bool useMembufs = false, useImages = false;
    bool useFloat = false, useDouble = false;
    size_t vectorLength = 0;
    int M = -1, N = -1;
    int groupHeight = -1, groupWidth = -1, blockHeight = -1, blockWidth = -1, extraParam = -1;
    bool vectorAttributeHint = true;

    if (!parseOpts(argc, argv,
                   packedKernels,
                   useMembufs,
                   useImages,
                   useFloat,
                   useDouble,
                   vectorLength,
                   M, N,
                   groupHeight, groupWidth, blockHeight, blockWidth, extraParam,
                   vectorAttributeHint))
        exit(1);

    // kernel generator
    KernelSaxpyBuffer < float, 1 > kernel_buf_sp_1;
    KernelSaxpyBuffer < float, 2 > kernel_buf_sp_2;
    KernelSaxpyBuffer < float, 4 > kernel_buf_sp_4;
    KernelSaxpyBuffer < double, 1 > kernel_buf_dp_1;
    KernelSaxpyBuffer < double, 2 > kernel_buf_dp_2;
    KernelSaxpyBuffer < double, 4 > kernel_buf_dp_4;
    KernelSaxpyImage < float, 4 > kernel_img_sp_4;
    KernelSaxpyImage < double, 2 > kernel_img_dp_2;
    KernelBaseSaxpy *ptrKernel = NULL;
    if (useMembufs) {
        if (useFloat) {
            if (1 == vectorLength) ptrKernel = &kernel_buf_sp_1;
            if (2 == vectorLength) ptrKernel = &kernel_buf_sp_2;
            if (4 == vectorLength) ptrKernel = &kernel_buf_sp_4;
        }
        if (useDouble) {
            if (1 == vectorLength) ptrKernel = &kernel_buf_dp_1;
            if (2 == vectorLength) ptrKernel = &kernel_buf_dp_2;
            if (4 == vectorLength) ptrKernel = &kernel_buf_dp_4;
        }
    }
    if (useImages) {
        if (useFloat) { ptrKernel = &kernel_img_sp_4; }
        if (useDouble) { ptrKernel = &kernel_img_dp_2; }
    }
    KernelBaseSaxpy& kernel = *ptrKernel;

    // kernel vector attribute hint?
    kernel.setUseAttrAutoVec(vectorAttributeHint);

    // packed kernel support
    kernel.setPackedCalc(packedKernels);

    // matrix dimensions, outer and inner blocking, extra parameters
    kernel.setSaxpyDimensions(M, N);
    kernel.setVectorLength(vectorLength);
    kernel.setWorkGroup(groupHeight, groupWidth);
    kernel.setInnerBlocking(blockHeight, blockWidth);
    kernel.setExtraParameter(extraParam);
    if (kernel.validParams()) {
        // print kernel source
        cout << kernel;

    } else {
        cerr << "error: invalid parameters (coalescedKernels, M, N, groupHeight, groupWidth, blockHeight, blockWidth, vectorLength, extraParam): "
             << "(" << packedKernels << ", "
                    << M << ", "
                    << N << ", "
                    << groupHeight << ", "
                    << groupWidth << ", "
                    << blockHeight << ", "
                    << blockWidth << ", "
                    << vectorLength << ", "
                    << extraParam
             << ")" << endl;
        exit(1);
    }

    return 0;
}
