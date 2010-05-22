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
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include "GatlasBenchmark.hpp"

#include "KernelProbeAutoVectorize.hpp"
#include "KernelFile.hpp"

#include "using_namespace"

using namespace std;

// explicitly vectorized kernels
typedef float scalar;
static const size_t VECTOR_LENGTH = 4;
typedef VecType<scalar, VECTOR_LENGTH> scalarN;

int getDeviceIndex(OCLBase& oclBase,
                   const string& device) {
    if ("cpu" == device || "cpu0" == device)
        return oclBase.cpuIndexes().empty() ? -1 : oclBase.cpuIndexes()[0];
    else if ("gpu" == device || "gpu0" == device)
        return oclBase.gpuIndexes().empty() ? -1 : oclBase.gpuIndexes()[0];
    else if ("acc" == device || "acc0" == device)
        return oclBase.accIndexes().empty() ? -1 : oclBase.accIndexes()[0];
    else {
        if (0 == device.find("cpu")) {
            stringstream ss;
            ss << device.substr(3);
            size_t index = 0;
            ss >> index;
            return oclBase.cpuIndexes().empty() ? -1 : oclBase.cpuIndexes()[index];
        } else if (0 == device.find("gpu")) {
            stringstream ss;
            ss << device.substr(3);
            size_t index = 0;
            ss >> index;
            return oclBase.gpuIndexes().empty() ? -1 : oclBase.gpuIndexes()[index];
        } else if (0 == device.find("acc")) {
            stringstream ss;
            ss << device.substr(3);
            size_t index = 0;
            ss >> index;
            return oclBase.accIndexes().empty() ? -1 : oclBase.accIndexes()[index];
        }
    }
    return -1;
}

bool parseOpts(int argc, char *argv[],
               string& device,
               int& M,
               int& N,
               int& K,
               int& groupSize,
               int& blockHeight,
               int& extraParam,
               size_t& numberTrials,
               int& topN,
               bool& nestedOptimization,
               bool& transposeA,
               bool& transposeB,
               bool& busTransferToDevice,
               bool& busTransferFromDevice,
               bool& paranoidCheck,
               bool& printDebug) {
    int opt;
    while ((opt = getopt(argc, argv, "heabsrpzd:m:n:k:g:y:x:t:w:")) != -1) {
        switch (opt) {
            case ('h') :
                cerr << "usage: " << argv[0]
                     << " -d cpu|gpu|acc|cpuX|gpuX|accX -n N [-m M -k K]"
                        " [-g groupSize [-y blockHeight [-x extraParam]]]"
                        " [-t numberTrials]"
                        " [-w topN]"
                        " [-e] [-a] [-b] [-s] [-r] [-p] [-z] [-h]" << endl
                     << "\t-d cpu, gpu or accelerator device, optional X is the device number" << endl
                     << "\t-m matrix dimension M" << endl
                     << "\t-n matrix dimension N" << endl
                     << "\t-k matrix dimension K" << endl
                     << "\t-g work item group width and height" << endl
                     << "\t-y inner blocking height" << endl
                     << "\t-x extra parameter" << endl
                     << "\t-t number of trials (default is 1)" << endl
                     << "\t-w keep topN (groupSize, blockHeight) combinations" << endl
                     << "\t-e use faster nested optimization (default no)" << endl
                     << "\t-a transpose A (default no)" << endl
                     << "\t-b transpose B (default no)" << endl
                     << "\t-s include PCIe bus data transfer to device in timing (default no)" << endl
                     << "\t-r include PCIe bus data transfer from device in timing (default no)" << endl
                     << "\t-p paranoid output matrix check (default no)" << endl
                     << "\t-z print matrix output (default no)" << endl
                     << "\t-h help" << endl;
                exit(1);
            case ('d') : device = optarg; break;
            case ('m') : M = atoi(optarg); break;
            case ('n') : N = atoi(optarg); break;
            case ('k') : K = atoi(optarg); break;
            case ('g') : groupSize = atoi(optarg); break;
            case ('y') : blockHeight = atoi(optarg); break;
            case ('x') : extraParam = atoi(optarg); break;
            case ('t') : numberTrials = atoi(optarg); break;
            case ('w') : topN = atoi(optarg); break;
            case ('e') : nestedOptimization = true; break;
            case ('a') : transposeA = true; break;
            case ('b') : transposeB = true; break;
            case ('s') : busTransferToDevice = true; break;
            case ('r') : busTransferFromDevice = true; break;
            case ('p') : paranoidCheck = true; break;
            case ('z') : printDebug = true; break;
        }
    }

    // minimal validation of options
    const size_t VL = KernelBaseMatmul::VECTOR_LENGTH;
    bool rc = true;
    if (0 != device.find("cpu") && 0 != device.find("gpu") & 0 != device.find("acc")) {
        cerr << "error: invalid device " << device << endl;
        rc = false;
    }
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
    if (-1 == groupSize && -1 != blockHeight) {
        cerr << "error: group size must be specified with block height" << endl;
        rc = false;
    }
    if (-1 == groupSize && -1 == blockHeight && -1 != extraParam) {
        cerr << "error: group size and block height must be specified with extra parameter" << endl;
        rc = false;
    }
    if (-1 != groupSize && (groupSize < 1 || groupSize > 16)) {
        cerr << "error: work item group size must be a number from 1 to 16 inclusive" << endl;
        rc = false;
    }
    if (transposeA && -1 != blockHeight && blockHeight != VL) {
        cerr << "error: inner blocking height is fixed to " << VL << " when matrix A is transposed" << endl;
        rc = false;
    }
    if (-1 != blockHeight && blockHeight < VL) {
        cerr << "error: invalid inner blocking height" << endl;
        rc = false;
    }

    // doesn't really make sense to specify blocking with nested optimization
    if (nestedOptimization && (-1 != groupSize || -1 != blockHeight || -1 != extraParam)) {
        cerr << "error: nested optimization will find optimal blocking" << endl;
        rc = false;
    }

    return rc;
}

ostream& printParams(ostream& os, const vector<size_t>& args) {
    for (size_t i = 0; i < args.size(); i++) {
        os << args[i];
        if (i != args.size() - 1) os << " ";
    }
}

vector< vector<size_t> > getParams(const KernelBaseMatmul& kernel,
                                   const size_t M, const size_t N, const size_t K,
                                   const size_t groupSize, const size_t blockHeight, const size_t extraParam,
                                   const int loopOrder)
{
    vector< vector<size_t> > pargs;

    // all parameters
    if (-1 != groupSize && -1 != blockHeight && -1 != extraParam) {
        vector<size_t> a;
        a.push_back(M);
        a.push_back(N);
        a.push_back(K);
        a.push_back(groupSize);
        a.push_back(blockHeight);
        a.push_back(extraParam);
        pargs.push_back(a);
    } else if (-1 != groupSize && -1 != blockHeight)
        pargs = kernel.parameters(M, N, K, groupSize, blockHeight);
    else if (-1 != groupSize)
        pargs = kernel.parameters(M, N, K, groupSize);
    else
        pargs = kernel.parameters(M, N, K);

    // constrain to specified loop order and use inlined matrix dimensions for probe trials
    if (-1 != loopOrder) {
        vector< vector<size_t> > probeargs;
        for (size_t i = 0; i < pargs.size(); i++) {
            const vector<size_t>& pa = pargs[i];
            const size_t xp = kernel.getExtraParam(pa);
            if (loopOrder == kernel._loopOrder(xp) && kernel._inlineMNK(xp)) {
                probeargs.push_back(pa);
            }
        }
        pargs = probeargs;
    }

    return pargs;
}

// return number of benchmarked kernels that were ok
size_t mainLoop(KernelInterface& kernel,
                Bench& bench,
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
    vector<string> pargsDesc;
    vector<double> pargsVariance;

    for (size_t i = 0; i < pargs.size(); i++) {
        pargsTime.push_back(0);
        pargsFlops.push_back(0);
        pargsDesc.push_back("");
        pargsVariance.push_back(0);
    }

    bool initRun = true;

    // repeat main loop for number of trials
    for (size_t k = 0; k < numberTrials; k++) {

        // main loop
        for (size_t i = 0; i < pargs.size(); i++) {

            const vector<size_t>& args = pargs[i];

            if (pargsOk[i]) {

                // dummy run to create buffers/images and flush to device
                if (initRun) {
                    bench.run(1, args, busTransferToDevice, busTransferFromDevice, printDebug);
                    initRun = false;
                    cout << endl << endl;

                    // paranoid check
                    if (paranoidCheck)
                        kernel.paranoidCheck();
                }

                cout << "[trial " << k << "] ";

                const size_t microsecs = bench.run(1, args, busTransferToDevice, busTransferFromDevice, printDebug);
                if (0 == microsecs) {
                    pargsOk[i] = false;
                    continue;
                }

                goodKernelCount++;

                const size_t numflops = kernel.numberFlops();

                pargsTime[i] += microsecs;
                pargsFlops[i] += numflops;
                pargsDesc[i] = kernel.desc();

                // single pass mean and variance
                const double avg = static_cast<double>(numflops) / microsecs / 1000;
                if (0 == k) {
                    pargsAverage[i] = avg;
                } else {
                    const double delta = avg - pargsAverage[i];
                    pargsVariance[i] += (static_cast<double>(k) / (k + 1)) * delta * delta;
                    pargsAverage[i] += (static_cast<double>(1) / (k + 1)) * delta;
                }

                cout << microsecs << "\t";
                printParams(cout, args);
                cout << endl;
            }
        }

        cout << endl;

        // prune to top N parameter combinations
        // parameter combinations with the same time are treated together
        if (-1 != topN) {

            // sort trial times, skip any parameters with errors
            set<size_t> trialTimes;
            for (size_t i = 0; i < pargs.size(); i++)
                if (pargsOk[i])
                    trialTimes.insert(pargsTime[i]);

            // only interested in topN fastest parameter combinations
            size_t count = 0;
            set<size_t> topTimes;
            for (set<size_t>::const_iterator iter = trialTimes.begin();
                 iter != trialTimes.end();
                 iter++) {
                if (count++ >= topN) break;
                topTimes.insert(*iter);
            }

            // mark all parameters as bad except for the top N fastest ones
            for (size_t i = 0; i < pargsOk.size(); i++)
                if (pargsOk[i] && 0 == topTimes.count(pargsTime[i]))
                    pargsOk[i] = false;
        }
    }

    // sort accumulated trial times, skip any parameters with errors
    map<size_t, size_t> timeToIdx;
    for (size_t i = 0; i < pargs.size(); i++)
        if (pargsOk[i])
            timeToIdx[pargsTime[i]] = i;

    // print results in descending order, so fastest kernels are first
    size_t count = 0;
    for (map<size_t, size_t>::const_iterator iter = timeToIdx.begin();
         iter != timeToIdx.end();
         iter++) {

        const size_t accumTime = (*iter).first;
        const size_t idx = (*iter).second;

        const vector<size_t>& args = pargs[idx];
        const string& desc = pargsDesc[idx];

        // show average of GFLOPS rate for individual trials, not average over all trials
        // the variance is of the average GFLOPS rate for individual trials
        //const double gflops = static_cast<double>(pargsFlops[idx]) / accumTime / 1000;
        const double gflops = pargsAverage[idx];
        const double variance = pargsVariance[idx];

        cout << "[" << count++ << "] "
             << accumTime << " usec"
             << "\tavg: " << gflops
             << "\tstddev: " << sqrt(variance/numberTrials)
             << "\t";
        printParams(cout, args);
        cout << "\t" << desc << endl;
    }

    return goodKernelCount;
}

size_t mainLoop(KernelInterface& kernel,
                Bench& bench,
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
    int M = -1, N = -1, K = -1;
    int groupSize = -1, blockHeight = -1, extraParam = -1;
    size_t numberTrials = 1;
    int topN = -1;
    bool nestedOptimization = false;
    bool transposeA = false, transposeB = false;
    bool busTransferToDevice = false, busTransferFromDevice = false;
    bool paranoidCheck = false;
    bool printDebug = false;

    if (!parseOpts(argc, argv,
                   device,
                   M, N, K,
                   groupSize, blockHeight, extraParam,
                   numberTrials,
                   topN,
                   nestedOptimization,
                   transposeA, transposeB,
                   busTransferToDevice, busTransferFromDevice,
                   paranoidCheck,
                   printDebug))
        exit(1);

    OCLBase oclBase;
    const size_t device_index = getDeviceIndex(oclBase, device);
    OCLApp oclApp(oclBase, device_index);

    KERNEL_CLASS kernel(transposeA, transposeB);
    Bench bench(oclApp, kernel);

    // does this device support vector attribute hint?
    // ATI    - ok
    // nVidia - ok but slow (needs scalar kernel)
    // CELL   - program build failure
    KernelProbeAutoVectorize<scalarN> kpav;
    Bench kpavBench(oclApp, kpav);
    vector< vector<size_t> > kpavArgs;
    kpavArgs.push_back(kpav.parameters(true));
    if (0 == mainLoop(kpav, kpavBench, kpavArgs, 1, -1, false, false, false, false)) {
        cout << "device does not support vector attribute hint" << endl;
        kernel.setUseAttrAutoVec(false);
    } else {
        cout << "vector attribute hint ok" << endl;
    }

    vector< vector<size_t> > pargs = getParams(kernel,
                                               M, N, K,
                                               groupSize, blockHeight, extraParam,
                                               (nestedOptimization ? 0 : -1));

    vector<bool> pargsOk;
    vector<double> pargsAverage;
    for (size_t i = 0; i < pargs.size(); i++) {
        pargsOk.push_back(true);
        pargsAverage.push_back(0);
    }

    mainLoop(kernel,
             bench,
             pargs,
             pargsOk,
             pargsAverage,
             numberTrials,
             topN,
             busTransferToDevice,
             busTransferFromDevice,
             printDebug,
             paranoidCheck);

    if (! nestedOptimization) return 0; // one full pass only

    // nested optimization needs second pass
    cout << endl << "*** nested optimization second pass ***" << endl;

    // sort based on average GFLOPS rate for individual trials
    map<double, size_t> gflopsToIdx;
    for (size_t i = 0; i < pargs.size(); i++)
        if (pargsOk[i])
            gflopsToIdx[pargsAverage[i]] = i;

    // best inner and outer blocking
    size_t bestGroupSize, bestBlockHeight;
    for (map<double, size_t>::const_reverse_iterator iter = gflopsToIdx.rbegin();
         iter != gflopsToIdx.rend();
         iter++) {

        const size_t idx = (*iter).second;
        const vector<size_t>& args = pargs[idx];
        bestGroupSize = kernel.getGroupSize(args);
        bestBlockHeight = kernel.getBlockHeight(args);
        break;
    }

    pargs = getParams(kernel,
                      M, N, K,
                      bestGroupSize, bestBlockHeight, extraParam,
                      -1);

    mainLoop(kernel,
             bench,
             pargs,
             numberTrials,
             topN,
             busTransferToDevice,
             busTransferFromDevice,
             printDebug,
             paranoidCheck);

    return 0;
}
