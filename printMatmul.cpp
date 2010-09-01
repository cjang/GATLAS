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

#include "KernelMatmulBuffer.hpp"
#include "KernelMatmulImage.hpp"

#include "using_namespace"

using namespace std;

bool parseOpts(int argc, char *argv[],
               size_t& packedKernels,
               bool& useMembufs,
               bool& useImages,
               bool& useFloat,
               bool& useDouble,
               size_t& vectorLength,
               bool& useGEMM,
               int& M,
               int& N,
               int& K,
               int& groupSize,
               int& blockHeight,
               int& extraParam,
               bool& transposeA,
               bool& transposeB,
               bool& vectorAttributeHint) {
    int opt;
    string kernelType = "<unspecified>";
    while ((opt = getopt(argc, argv, "habvGC:T:m:n:k:g:y:x:")) != -1) {
        switch (opt) {
            case ('h') :
                cerr << "usage: " << argv[0]
                     << " -T float1|float2|float4|double1|double2|double4|floatimg|doubleimg -n N [-m M -k K]"
                        " [-C numKernels]"
                        " -g groupSize -y blockHeight -x extraParam"
                        " [-G] [-a] [-b] [-v] [-h]" << endl
                     << "\t-C number of coalesced kernels (default is 1)" << endl
                     << "\t-T kernel type: precision, vector length, memory buffers or images" << endl
                     << "\t-m matrix dimension M" << endl
                     << "\t-n matrix dimension N" << endl
                     << "\t-k matrix dimension K" << endl
                     << "\t-g work item group width and height" << endl
                     << "\t-y inner blocking height" << endl
                     << "\t-x extra parameter" << endl
                     << "\t-G use general matrix multiply (default no)" << endl
                     << "\t-a transpose A (default no)" << endl
                     << "\t-b transpose B (default no)" << endl
                     << "\t-v disable kernel vector attribute hint (default enabled)" << endl
                     << "\t-h help" << endl;
                exit(1);
            case ('C') : packedKernels = atoi(optarg); break;
            case ('T') : kernelType = optarg; break;
            case ('m') : M = atoi(optarg); break;
            case ('n') : N = atoi(optarg); break;
            case ('k') : K = atoi(optarg); break;
            case ('g') : groupSize = atoi(optarg); break;
            case ('y') : blockHeight = atoi(optarg); break;
            case ('x') : extraParam = atoi(optarg); break;
            case ('G') : useGEMM = true; break;
            case ('a') : transposeA = true; break;
            case ('b') : transposeB = true; break;
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
    if (-1 == N) {
        cerr << "error: matrix dimension N must be specified" << endl;
        rc = false;
    } else {
        if (-1 == M && -1 == K) {
            M = K = N;
        } else {
            if ( (-1 != M && -1 == K) || (-1 == M && -1 != K) ) {
                cerr << "error: all matrix dimensions M, N and K must be specified" << endl;
                rc = false;
            }
        }
    }
    if (0 != N % VL) {
        cerr << "error: matrix dimension N must be multiple of " << VL << endl;
        rc = false;
    }
    if (0 != M % VL) {
        cerr << "error: matrix dimension M must be multiple of " << VL << endl;
        rc = false;
    }
    if (0 != K % VL) {
        cerr << "error: matrix dimension K must be multiple of " << VL << endl;
        rc = false;
    }

    // kernel parameters
    if (-1 == groupSize) {
        cerr << "error: must specify work item group size" << endl;
        rc = false;
    }
    if (-1 == blockHeight) {
        cerr << "error: must specify inner blocking height" << endl;
        rc = false;
    }
    if (-1 == extraParam) {
        cerr << "error: must specify extra parameter" << endl;
        rc = false;
    }
    if (-1 != groupSize && (groupSize < 1 || groupSize > 16)) {
        cerr << "error: work item group size must be a number from 1 to 16 inclusive" << endl;
        rc = false;
    }
    if (transposeA && -1 != blockHeight && (0 != blockHeight % VL) ) {
        cerr << "error: inner blocking height must be multiple of " << VL << " when matrix A is transposed" << endl;
        rc = false;
    }
    if (-1 != blockHeight && blockHeight < VL) {
        cerr << "error: invalid inner blocking height" << endl;
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
    bool useGEMM = false;
    int M = -1, N = -1, K = -1;
    int groupSize = -1, blockHeight = -1, extraParam = -1;
    bool transposeA = false, transposeB = false;
    bool vectorAttributeHint = true;

    if (!parseOpts(argc, argv,
                   packedKernels,
                   useMembufs,
                   useImages,
                   useFloat,
                   useDouble,
                   vectorLength,
                   useGEMM,
                   M, N, K,
                   groupSize, blockHeight, extraParam,
                   transposeA, transposeB,
                   vectorAttributeHint))
        exit(1);

    // OpenCL parameterized kernel generator class
    KernelMatmulBuffer < float, 1 > kernel_buf_sp_1;
    KernelMatmulBuffer < float, 2 > kernel_buf_sp_2;
    KernelMatmulBuffer < float, 4 > kernel_buf_sp_4;
    KernelMatmulBuffer < double, 1 > kernel_buf_dp_1;
    KernelMatmulBuffer < double, 2 > kernel_buf_dp_2;
    KernelMatmulBuffer < double, 4 > kernel_buf_dp_4;
    KernelMatmulImage < float, 4 > kernel_img_sp_4;
    KernelMatmulImage < double, 2 > kernel_img_dp_2;
    KernelBaseMatmul *ptrKernel = NULL;
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
    KernelBaseMatmul& kernel = *ptrKernel;

    // kernel vector attribute hint?
    kernel.setUseAttrAutoVec(vectorAttributeHint);

    // packed kernel support
    kernel.setPackedCalc(packedKernels);

    // matrix dimensions, outer and inner blocking, extra parameters
    kernel.setGeneralizedMatmul( useGEMM );
    kernel.setMatrixDimensions(M, N, K);
    kernel.setDataLayout(transposeA, transposeB);
    kernel.setWorkGroup(groupSize);
    kernel.setInnerBlocking(blockHeight, vectorLength );
    kernel.setExtraParameter(extraParam);
    if (kernel.validParams()) {

        // print kernel source
        cout << kernel;

    } else {
        cerr << "error: invalid parameters (coalescedKernels, M, N, K, groupSize, blockHeight, extraParam): "
             << "(" << packedKernels << ", "
                    << M << ", "
                    << N << ", "
                    << K << ", "
                    << groupSize << ", "
                    << blockHeight << ", "
                    << extraParam
             << ")" << endl;
        exit(1);
    }

    return 0;
}
