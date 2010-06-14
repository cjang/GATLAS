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

#include "GatlasAppUtil.hpp"
#include "GatlasBenchmark.hpp"
#include "KernelProbeAutoVectorize.hpp"
#include "KernelFile.hpp"

#include "using_namespace"

using namespace std;

// explicitly vectorized kernels
typedef float scalar;
static const size_t VECTOR_LENGTH = 4;
typedef VecType<scalar, VECTOR_LENGTH> scalarN;

bool parseOpts(int argc, char *argv[],
               string& device,
               int& M,
               int& N,
               int& K,
               int& groupSize,
               int& blockHeight,
               int& extraParam,
               bool& transposeA,
               bool& transposeB) {
    int opt;
    while ((opt = getopt(argc, argv, "habd:m:n:k:g:y:x:")) != -1) {
        switch (opt) {
            case ('h') :
                cerr << "usage: " << argv[0]
                     << " -d cpu|gpu|acc|cpuX|gpuX|accX -n N [-m M -k K]"
                        " -g groupSize -y blockHeight -x extraParam"
                        " [-a] [-b] [-h]" << endl
                     << "\t-d cpu, gpu or accelerator device, optional X is the device number" << endl
                     << "\t-m matrix dimension M" << endl
                     << "\t-n matrix dimension N" << endl
                     << "\t-k matrix dimension K" << endl
                     << "\t-g work item group width and height" << endl
                     << "\t-y inner blocking height" << endl
                     << "\t-x extra parameter" << endl
                     << "\t-a transpose A (default no)" << endl
                     << "\t-b transpose B (default no)" << endl
                     << "\t-h help" << endl;
                exit(1);
            case ('d') : device = optarg; break;
            case ('m') : M = atoi(optarg); break;
            case ('n') : N = atoi(optarg); break;
            case ('k') : K = atoi(optarg); break;
            case ('g') : groupSize = atoi(optarg); break;
            case ('y') : blockHeight = atoi(optarg); break;
            case ('x') : extraParam = atoi(optarg); break;
            case ('a') : transposeA = true; break;
            case ('b') : transposeB = true; break;
        }
    }

    // validate matrix dimensions
    const size_t VL = VECTOR_LENGTH; // FIXME
    bool rc = true;
    if (0 != device.find("cpu") && 0 != device.find("gpu") && 0 != device.find("acc")) {
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
    if (-1 != blockHeight && blockHeight < VL) {
        cerr << "error: invalid inner blocking height" << endl;
        rc = false;
    }
    if (transposeA && -1 != blockHeight && blockHeight != VL) {
        cerr << "error: inner blocking height is fixed to " << VL << " when matrix A is transposed" << endl;
        rc = false;
    }

    return rc;
}

int main(int argc, char *argv[])
{
    string device = "<unspecified>";
    int M = -1, N = -1, K = -1;
    int groupSize = -1, blockHeight = -1, extraParam = -1;
    bool transposeA = false, transposeB = false;

    if (!parseOpts(argc, argv,
                   device,
                   M, N, K,
                   groupSize, blockHeight, extraParam,
                   transposeA, transposeB))
        exit(1);

    OCLBase oclBase;
    const size_t device_index = AppUtil::getDeviceIndex(oclBase, device);
    OCLApp oclApp(oclBase, device_index);

    // OpenCL parameterized kernel generator class
    KERNEL_CLASS_MACRO kernel( GEMM_MACRO );

    Bench(oclApp, kernel);

    // does this device support vector attribute hint?
    // ATI    - ok
    // nVidia - ok but slow (needs scalar kernel)
    // CELL   - program build failure
    KernelProbeAutoVectorize<scalar, VECTOR_LENGTH> kpav;
    Bench kpavBench(oclApp, kpav);
    vector<size_t> kpavArgs;
    kpavArgs.push_back(true);
    if (0 == kpavBench.run(1, kpavArgs, false, false, false)) {
        cout << "device does not support vector attribute hint" << endl;
        kernel.setUseAttrAutoVec(false);
    } else {
        cout << "vector attribute hint ok" << endl;
    }

    // matrix dimensions, outer and inner blocking, extra parameters
    kernel.setMatrixDimensions(M, N, K);
    kernel.setDataLayout(transposeA, transposeB, false);
    kernel.setWorkGroup(groupSize);
    kernel.setInnerBlocking(blockHeight, VECTOR_LENGTH);
    kernel.setExtraParameter(extraParam);
    if (kernel.validParams()) {

        // print kernel source
        cout << kernel;

    } else {
        cerr << "error: invalid parameters (M, N, K, groupSize, blockHeight, extraParam): "
             << "(" << M << ", "
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
