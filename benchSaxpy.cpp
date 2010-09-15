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
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include "GatlasAppUtil.hpp"
#include "GatlasBenchmark.hpp"

#include "KernelSaxpyBuffer.hpp"
#include "KernelSaxpyImage.hpp"

#include "using_namespace"

using namespace std;

bool parseOpts(int argc, char *argv[],
               string& device,
               string& journalFile,
               size_t& packedKernels,
               bool& useMembufs,
               bool& useImages,
               bool& useFloat,
               bool& useDouble,
               size_t& vectorLength,
               int& M,
               int& N,
               size_t& numberTrials,
               int& topN,
               bool& emOptimization,
               bool& busTransferToDevice,
               bool& busTransferFromDevice,
               bool& paranoidCheck,
               bool& vectorAttributeHint,
               bool& printDebug) {
    int opt;
    string kernelType = "<unspecified>";
    while ((opt = getopt(argc, argv, "hesrpvzd:j:C:T:m:n:t:w:")) != -1) {
        switch (opt) {
            case ('h') :
                cerr << "usage: " << argv[0]
                     << " -d cpu|gpu|acc|cpuX|gpuX|accX -j journalFile -T float1|float2|float4|double1|double2|double4|floatimg|doubleimg -m M [-n N]"
                        " [-C numKernels]"
                        " [-t numberTrials]"
                        " [-w topN]"
                        " [-e] [-s] [-r] [-p] [-v] [-z] [-h]" << endl
                     << "\t-d cpu, gpu or accelerator device, optional X is the device number" << endl
                     << "\t-j journal file" << endl
                     << "\t-C number of coalesced kernels (default is 1)" << endl
                     << "\t-T kernel type: precision, vector length, memory buffers or images" << endl
                     << "\t-m matrix dimension M" << endl
                     << "\t-n matrix dimension N" << endl
                     << "\t-t number of trials (default is 1)" << endl
                     << "\t-w keep topN combinations" << endl
                     << "\t-e use faster expectation maximization optimization (default no)" << endl
                     << "\t-s include PCIe bus data transfer to device in timing (default no)" << endl
                     << "\t-r include PCIe bus data transfer from device in timing (default no)" << endl
                     << "\t-p paranoid output matrix check (default no)" << endl
                     << "\t-v disable kernel vector attribute hint (default enabled)" << endl
                     << "\t-z print matrix output (default no)" << endl
                     << "\t-h help" << endl
                     << "***DONE***" << endl; // needed for wrapper retry script
                exit(1);
            case ('d') : device = optarg; break;
            case ('j') : journalFile = optarg; break;
            case ('C') : packedKernels = atoi(optarg); break;
            case ('T') : kernelType = optarg; break;
            case ('m') : M = atoi(optarg); break;
            case ('n') : N = atoi(optarg); break;
            case ('t') : numberTrials = atoi(optarg); break;
            case ('w') : topN = atoi(optarg); break;
            case ('e') : emOptimization = true; break;
            case ('s') : busTransferToDevice = true; break;
            case ('r') : busTransferFromDevice = true; break;
            case ('p') : paranoidCheck = true; break;
            case ('v') : vectorAttributeHint = false; break;
            case ('z') : printDebug = true; break;
        }
    }

    // minimal validation of options
    bool rc = true;
    if (0 != device.find("cpu") && 0 != device.find("gpu") & 0 != device.find("acc")) {
        cerr << "error: invalid device " << device << endl;
        rc = false;
    }
    if (journalFile.empty()) {
        cerr << "error: journal file must be specified" << endl;
        rc = false;
    }
    if (0 == packedKernels) {
        cerr << "error: number of kernels to coalesce must be at least one" << endl;
        rc = false;
    }
    vectorLength = 1;
    if ("float1" == kernelType) {
        useMembufs = true; useImages = false; useFloat = true; useDouble = false; vectorLength = 1;
    } else if ("float2" == kernelType) {
        useMembufs = true; useImages = false; useFloat = true; useDouble = false; vectorLength = 2;
    } else if ("float4" == kernelType) {
        useMembufs = true; useImages = false; useFloat = true; useDouble = false; vectorLength = 4;
    } else if ("double1" == kernelType) {
        useMembufs = true; useImages = false; useFloat = false; useDouble = true; vectorLength = 1;
    } else if ("double2" == kernelType) {
        useMembufs = true; useImages = false; useFloat = false; useDouble = true; vectorLength = 2;
    } else if ("double4" == kernelType) {
        useMembufs = true; useImages = false; useFloat = false; useDouble = true; vectorLength = 4;
    } else if ("floatimg" == kernelType) {
        useMembufs = false; useImages = true; useFloat = true; useDouble = false; vectorLength = 4;
    } else if ("doubleimg" == kernelType) {
        useMembufs = false; useImages = true; useFloat = false; useDouble = true; vectorLength = 2;
    } else {
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

    return rc;
}

vector< vector<size_t> > getParams(OCLApp& oclApp,
                                   KernelBaseSaxpy & kernel,
                                   const size_t vectorLength,
                                   const size_t maxBlockHeight,
                                   const size_t maxGroupSize,
                                   const size_t M, const size_t N,
                                   const size_t groupHeight = -1, const size_t groupWidth = -1,
                                   const size_t blockHeight = -1, const size_t blockWidth = -1,
                                   const size_t extraParam = -1)
{
    vector< vector<size_t> > pargs;
    vector<size_t> a;

    kernel.setSaxpyDimensions(M, N);
    kernel.setVectorLength(vectorLength);
    if (-1 != groupHeight) kernel.setWorkGroup(groupHeight, groupWidth);
    if (-1 != blockHeight) kernel.setInnerBlocking(blockHeight, blockWidth);
    if (-1 != extraParam) kernel.setExtraParameter(extraParam);

    // all parameters
    if (-1 != groupHeight && -1 != blockHeight && -1 != extraParam) {
        if (kernel.getParams(a)) pargs.push_back(a);

    // work group is specified
    } else if (-1 != groupHeight && -1 == blockHeight && -1 == extraParam) {
        if (vectorLength == N) {
            // 1D work groups
            for (size_t bh = 1; bh <= maxBlockHeight; bh++) {
                kernel.setInnerBlocking(bh, 0);
                for (size_t xp = 0; xp < kernel.totalVariations(); xp++) {
                    kernel.setExtraParameter(xp);
                    if (kernel.getParams(a)) pargs.push_back(a);
                }
            }
        } else {
            // 2D work groups
            for (size_t bh = 1; bh <= maxBlockHeight; bh++)
            for (size_t bw = 1; bw <= 8; bw++) {
                kernel.setInnerBlocking(bh, bw);
                for (size_t xp = 0; xp < kernel.totalVariations(); xp++) {
                    kernel.setExtraParameter(xp);
                    if (kernel.getParams(a)) pargs.push_back(a);
                }
            }
        }

    // inner blocking and extra parameter are specified
    } else if (-1 == groupHeight && -1 != blockHeight && -1 != extraParam) {
        // maximum value of group size
        const size_t largestPossibleGroupSize = oclApp.maxWorkGroupSize();
        const size_t largestGroupSize = maxGroupSize < largestPossibleGroupSize
                                            ? maxGroupSize
                                            : largestPossibleGroupSize;
        if (vectorLength == N) {
            // 1D work groups
            for (size_t wg = 64; wg <= largestGroupSize; wg++) {
                kernel.setWorkGroup(wg, 0);
                if (kernel.getParams(a)) pargs.push_back(a);
            }
        } else {
            // 2D work groups
            for (size_t wgHeight = 1; wgHeight <= largestGroupSize; wgHeight++)
            for (size_t wgWidth = 1; wgWidth <= largestGroupSize; wgWidth++)
                if (wgHeight * wgWidth <= largestGroupSize) {
                    kernel.setWorkGroup(wgHeight, wgWidth);
                    if (kernel.getParams(a)) pargs.push_back(a);
                }
        }

    // non-EM case
    } else {
        // maximum value of group size
        const size_t largestPossibleGroupSize = oclApp.maxWorkGroupSize();
        const size_t largestGroupSize = maxGroupSize < largestPossibleGroupSize
                                            ? maxGroupSize
                                            : largestPossibleGroupSize;

        // inner blocking limits
        const size_t innerBlockingMin = (-1 != blockHeight) ? blockHeight : 1;
        const size_t innerBlockingMax = (-1 != blockHeight) ? blockHeight : maxBlockHeight;

        // extra parameter limits
        const size_t extraParamMin = (-1 != extraParam) ? extraParam : 0;
        const size_t extraParamMax = (-1 != extraParam) ? extraParam + 1 : kernel.totalVariations();

        if (vectorLength == N) {
            // 1D work groups
            for (size_t wg = 64; wg <= largestGroupSize; wg++) {
                kernel.setWorkGroup(wg, 0);
                for (size_t bh = innerBlockingMin; bh <= innerBlockingMax; bh++) {
                    kernel.setInnerBlocking(bh, 0);
                    for (size_t xp = extraParamMin; xp < extraParamMax; xp++) {
                        kernel.setExtraParameter(xp);
                        if (kernel.getParams(a)) pargs.push_back(a);
                    }
                }
            }

        } else {
            // 2D work groups
            for (size_t wgHeight = 1; wgHeight <= largestGroupSize; wgHeight++)
            for (size_t wgWidth = 1; wgWidth <= largestGroupSize; wgWidth++)
                if (wgHeight * wgWidth <= largestGroupSize) {
                    kernel.setWorkGroup(wgHeight, wgWidth);
                    for (size_t bh = innerBlockingMin; bh <= innerBlockingMax; bh++)
                    for (size_t bw = 1; bw <= 8; bw++) {
                        kernel.setInnerBlocking(bh, bw);
                        for (size_t xp = extraParamMin; xp < extraParamMax; xp++) {
                            kernel.setExtraParameter(xp);
                            if (kernel.getParams(a)) pargs.push_back(a);
                        }
                    }
                }
        }
    }

    return pargs;
}

// return number of benchmarked kernels that were ok
size_t mainLoop(KernelInterface& kernel,
                Bench& bench,
                Journal& journal,
                const vector< vector<size_t> >& pargs,
                vector<bool>& pargsOk,
                vector<double>& pargsAverage,
                const size_t numberTrials,
                const size_t topN,
                const bool busTransferToDevice,
                const bool busTransferFromDevice,
                const bool printDebug,
                const bool paranoidCheck) {

    size_t goodKernelCount = 0;

    vector<size_t> pargsTime;
    vector<size_t> pargsFlops;
    vector<double> pargsVariance;
    vector< vector<size_t> > pargsExtraDetail;

    for (size_t i = 0; i < pargs.size(); i++) {
        pargsTime.push_back(0);
        pargsFlops.push_back(0);
        pargsVariance.push_back(0);
        pargsExtraDetail.push_back(vector<size_t>());
    }

    // paranoid check
    if (paranoidCheck) kernel.paranoidCheck();

    // repeat main loop for number of trials
    for (size_t k = 0; k < numberTrials; k++) {

        const bool dummyRun = (0 == k);

        goodKernelCount = AppUtil::benchLoop(k,
                                             kernel,
                                             bench,
                                             journal,
                                             pargs,
                                             pargsOk,
                                             pargsTime,
                                             pargsFlops,
                                             pargsAverage,
                                             pargsVariance,
                                             pargsExtraDetail,
                                             busTransferToDevice,
                                             busTransferFromDevice,
                                             dummyRun,
                                             printDebug);

        // prune to top N parameter combinations
        // parameter combinations with the same time are treated together
        if (-1 != topN) AppUtil::markBench(topN, pargsOk, pargsTime);
    }

    AppUtil::printBench(numberTrials,
                        pargs,
                        pargsOk,
                        pargsTime,
                        pargsAverage,
                        pargsVariance,
                        pargsExtraDetail);

    return goodKernelCount;
}

size_t mainLoop(KernelInterface& kernel,
                Bench& bench,
                Journal& journal,
                const vector< vector<size_t> >& pargs,
                const size_t numberTrials,
                const size_t topN,
                const bool busTransferToDevice,
                const bool busTransferFromDevice,
                const bool printDebug,
                const bool paranoidCheck) {

    vector<bool> pargsOk;
    vector<double> pargsAverage;

    for (size_t i = 0; i < pargs.size(); i++) {
        pargsOk.push_back(true);
        pargsAverage.push_back(0);
    }

    return mainLoop(kernel,
                    bench,
                    journal,
                    pargs,
                    pargsOk,
                    pargsAverage,
                    numberTrials,
                    topN,
                    busTransferToDevice,
                    busTransferFromDevice,
                    printDebug,
                    paranoidCheck);
}

int main(int argc, char *argv[])
{
    string device = "<unspecified>";
    string journalFile;
    size_t packedKernels = 1;
    bool useMembufs = false, useImages = false;
    bool useFloat = false, useDouble = false;
    size_t vectorLength = 0;
    int M = -1, N = -1;
    size_t numberTrials = 1;
    int topN = -1;
    bool emOptimization = false;
    bool busTransferToDevice = false, busTransferFromDevice = false;
    bool paranoidCheck = false;
    bool vectorAttributeHint = true;
    bool printDebug = false;

    if (!parseOpts(argc, argv,
                   device,
                   journalFile,
                   packedKernels,
                   useMembufs,
                   useImages,
                   useFloat,
                   useDouble,
                   vectorLength,
                   M, N,
                   numberTrials,
                   topN,
                   emOptimization,
                   busTransferToDevice, busTransferFromDevice,
                   paranoidCheck,
                   vectorAttributeHint,
                   printDebug)) {
        cerr << "***DONE***" << endl; // needed for wrapper retry script
        exit(1);
    }

    const size_t maxBlockHeight = 16;
    const size_t maxGroupSize = 256;

    // initialize OpenCL
    OCLBase oclBase;
    const size_t device_index = AppUtil::getDeviceIndex(oclBase, device);
    OCLApp oclApp(oclBase, device_index);

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

    // journal and benchmark object
    Journal journal(journalFile);
    Bench bench(oclApp, kernel, journal);

    // kernel vector attribute hint?
    kernel.setUseAttrAutoVec(vectorAttributeHint);

    // packed kernel support
    kernel.setPackedCalc(packedKernels);

    // not using EM
    if (! emOptimization) {
        // brute force benchmark timings
        vector< vector<size_t> > pargs = getParams(oclApp,
                                                   kernel,
                                                   vectorLength,
                                                   maxBlockHeight,
                                                   maxGroupSize,
                                                   M, N);

        vector<bool> pargsOk;
        vector<double> pargsAverage;
        for (size_t i = 0; i < pargs.size(); i++) {
            pargsOk.push_back(true);
            pargsAverage.push_back(0);
        }

        journal.loadMemo();
        mainLoop(kernel,
                 bench,
                 journal,
                 pargs,
                 pargsOk,
                 pargsAverage,
                 numberTrials,
                 topN,
                 busTransferToDevice,
                 busTransferFromDevice,
                 printDebug,
                 paranoidCheck);

        // useful for parent process manager to know not to respawn process
        cout << "***DONE***" << endl;
        return 0;
    }

    // using EM
    size_t bestGroupHeight, bestGroupWidth;
    size_t bestBlockHeight = 1;
    size_t bestBlockWidth = (vectorLength == N) ? 0 : vectorLength;
    size_t bestExtraParam = 0;
    int bestIndex = -1;
    vector< vector<size_t> > pargs;
    bool foundMax = false;
    while (! foundMax) {

        // emStep of 0 is expectation lower bound
        // emStep of 1 is maximization of bound
        for (size_t emStep = 0; emStep <= 1; emStep++) {

            // need to handle case when all kernels are bad
            while (-1 == bestIndex) {

                const size_t groupHeight = (0 == emStep) ? -1 : bestGroupHeight;
                const size_t groupWidth  = (0 == emStep) ? -1 : bestGroupWidth;
                const size_t blockHeight = (0 == emStep) ? bestBlockHeight : -1;
                const size_t blockWidth  = (0 == emStep) ? bestBlockWidth : -1;
                const size_t extraParam  = (0 == emStep) ? bestExtraParam : -1;

                pargs = getParams(oclApp,
                                  kernel,
                                  vectorLength,
                                  maxBlockHeight,
                                  maxGroupSize,
                                  M, N,
                                  groupHeight, groupWidth,
                                  blockHeight, blockWidth,
                                  extraParam);

                vector<bool> pargsOk;
                vector<double> pargsAverage;
                for (size_t i = 0; i < pargs.size(); i++) {
                    pargsOk.push_back(true);
                    pargsAverage.push_back(0);
                }

                journal.loadMemo();
                mainLoop(kernel,
                         bench,
                         journal,
                         pargs,
                         pargsOk,
                         pargsAverage,
                         1, //numberTrials,
                         topN,
                         busTransferToDevice,
                         busTransferFromDevice,
                         printDebug,
                         paranoidCheck);

                // fastest kernel
                bestIndex = AppUtil::rankBench(0, pargsOk, pargsAverage);
                if (-1 == bestIndex) {
                    // there were no good kernels found!
                    cerr << "error: no good kernels found for group height " << bestGroupHeight
                         << ", group width " << bestGroupWidth
                         << ", block height " << bestBlockHeight
                         << ", block width " << bestBlockWidth
                         << " and extra parameter " << extraParam
                         << " so giving up" << endl
                         << "***DONE***" << endl;
                    exit(1);
                }
                kernel.setParams(pargs[bestIndex]);

                // stop when fastest kernel does not change
                if (bestGroupHeight == kernel.groupHeight() &&
                    bestGroupWidth == kernel.groupWidth() &&
                    bestBlockHeight == kernel.blockHeight() &&
                    bestBlockWidth == kernel.blockWidth() &&
                    bestExtraParam == kernel.extraParam())
                    foundMax = true;

                bestGroupHeight = kernel.groupHeight();
                bestGroupWidth = kernel.groupWidth();
                bestBlockHeight = kernel.blockHeight();
                bestBlockWidth = kernel.blockWidth();
                bestExtraParam = kernel.extraParam();
            }

            bestIndex = -1;
        }
    }

    // if more than one trial is specified, a final average
    if (numberTrials > 1) {
        pargs = getParams(oclApp,
                          kernel,
                          vectorLength,
                          maxBlockHeight,
                          maxGroupSize,
                          M, N,
                          bestGroupHeight, bestGroupWidth,
                          bestBlockHeight, bestBlockWidth,
                          bestExtraParam);
        mainLoop(kernel,
                 bench,
                 journal,
                 pargs,
                 numberTrials,
                 topN,
                 busTransferToDevice,
                 busTransferFromDevice,
                 printDebug,
                 paranoidCheck);
    }

    // useful for parent process manager to know not to respawn process
    cout << "***DONE***" << endl;

    return 0;
}
