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
#include <sstream>
#include <string>
#include <stdlib.h>
#include <unistd.h>

#include "GatlasAppUtil.hpp"
#include "KernelProbeAutoVectorize.hpp"

#include "using_namespace"

using namespace std;

void parseOpts(int argc, char *argv[],
               string& device,
               bool& isFloat,
               bool& isDouble,
               size_t& VECTOR_LENGTH) {
    string scalar;
    bool deviceOk = false, scalarOk = false, vectorOk = false;
    int opt;
    while ((opt = getopt(argc, argv, "hd:s:l:")) != -1) {
        switch (opt) {
            case ('h') :
                cerr << "usage: " << argv[0]
                     << " -d cpu|gpu|acc|cpuN|gpuN|accN"
                        " -s float|double"
                        " -l 1|2|4|8|16"
                        " [-h]"
                     << endl
                     << "\t-d cpu, gpu or accelerator device, optional N is the device number" << endl
                     << "\t-s scalar type is float or double" << endl
                     << "\t-l vector length is 1, 2, 4, 8 or 16" << endl
                     << "\t-h print help" << endl;
                exit(1);
            case ('d') :
                device = optarg;
                if (0 != device.find("cpu") && 0 != device.find("gpu") && 0 != device.find("acc")) {
                    cerr << "error: invalid device " << device << endl;
                    exit(1);
                }
                deviceOk = true;
                break;
            case ('s') :
                scalar = optarg;
                if ("float" == scalar) {
                    isFloat = true;
                    isDouble = false;
                } else if ("double" == scalar) {
                    isFloat = false;
                    isDouble = true;
                } else {
                    cerr << "error: invalid scalar type " << scalar << endl;
                    exit(1);
                }
                scalarOk = true;
                break;
            case ('l') :
                VECTOR_LENGTH = atoi(optarg);
                if (1 != VECTOR_LENGTH &&
                    2 != VECTOR_LENGTH &&
                    4 != VECTOR_LENGTH &&
                    8 != VECTOR_LENGTH &&
                    16 != VECTOR_LENGTH) {
                    cerr << "error: invalid vector length " << VECTOR_LENGTH << endl;
                    exit(1);
                }
                vectorOk = true;
                break;
        }
    }
    if (!deviceOk) cerr << "error: missing -d <device>" << endl;
    if (!scalarOk) cerr << "error: missing -s <scalar type>" << endl;
    if (!vectorOk) cerr << "error: missing -l <vector length>" << endl;
    if (!deviceOk || !scalarOk || !vectorOk) exit(1);
}

template <typename SCALAR, size_t VECTOR_LENGTH>
size_t probeAutoVectorize(OCLApp& oclApp, const bool isAutoVecHint) {

    // just try one positive test using auto vectorize kernel attribute
    vector< vector<size_t> > pargs;
    vector<size_t> useAutoVec;
    useAutoVec.push_back(isAutoVecHint ? 1 : 0);
    pargs.push_back(useAutoVec);

    vector<bool> pargsOk;
    vector<size_t> pargsTime;
    vector<size_t> pargsFlops;
    vector<double> pargsAverage;
    vector<double> pargsVariance;
    vector< vector<size_t> > pargsExtraDetail;
    AppUtil::benchInit(pargs, pargsOk, pargsTime, pargsFlops, pargsAverage, pargsVariance, pargsExtraDetail);

    KernelProbeAutoVectorize<SCALAR, VECTOR_LENGTH> kernel;
    Bench bench(oclApp, kernel);

    return AppUtil::benchLoop(0,
                              kernel,
                              bench,
                              pargs,
                              pargsOk,
                              pargsTime,
                              pargsFlops,
                              pargsAverage,
                              pargsVariance,
                              pargsExtraDetail,
                              false,
                              false);
}

template <typename SCALAR, size_t VECTOR_LENGTH>
string autoVectorizeHint() {
    AutoVectorize< VecType<SCALAR, VECTOR_LENGTH> > attrAutoVec;
    stringstream ss;
    ss << attrAutoVec;
    return ss.str();
}

int main(int argc, char *argv[])
{
    string device;
    bool isFloat, isDouble;
    size_t VECTOR_LENGTH;

    parseOpts(argc, argv, device, isFloat, isDouble, VECTOR_LENGTH);

    OCLBase oclBase;
    const int device_index = AppUtil::getDeviceIndex(oclBase, device);
    OCLApp oclApp(oclBase, device_index);

    // does this device support vector attribute hint?
    // ATI    - ok
    // nVidia - ok but slow (needs scalar kernel)
    // CELL   - program build failure
    size_t useHintCount = 0, noHintCount = 0;
    string autoVecHint;
    if (isFloat) {
        switch (VECTOR_LENGTH) {
            case (1) :
                useHintCount = probeAutoVectorize<float, 1>(oclApp, true);
                noHintCount = probeAutoVectorize<float, 1>(oclApp, false);
                autoVecHint = autoVectorizeHint<float, 1>();
                break;
            case (2) :
                useHintCount = probeAutoVectorize<float, 2>(oclApp, true);
                noHintCount = probeAutoVectorize<float, 2>(oclApp, false);
                autoVecHint = autoVectorizeHint<float, 2>();
                break;
            case (4) :
                useHintCount = probeAutoVectorize<float, 4>(oclApp, true);
                noHintCount = probeAutoVectorize<float, 4>(oclApp, false);
                autoVecHint = autoVectorizeHint<float, 4>();
                break;
            case (8) :
                useHintCount = probeAutoVectorize<float, 8>(oclApp, true);
                noHintCount = probeAutoVectorize<float, 8>(oclApp, false);
                autoVecHint = autoVectorizeHint<float, 8>();
                break;
            case (16) :
                useHintCount = probeAutoVectorize<float, 16>(oclApp, true);
                noHintCount = probeAutoVectorize<float, 16>(oclApp, false);
                autoVecHint = autoVectorizeHint<float, 16>();
                break;
        }
    } else if (isDouble) {
        switch (VECTOR_LENGTH) {
            case (1) :
                useHintCount = probeAutoVectorize<double, 1>(oclApp, true);
                noHintCount = probeAutoVectorize<double, 1>(oclApp, false);
                autoVecHint = autoVectorizeHint<double, 1>();
                break;
            case (2) :
                useHintCount = probeAutoVectorize<double, 2>(oclApp, true);
                noHintCount = probeAutoVectorize<double, 2>(oclApp, false);
                autoVecHint = autoVectorizeHint<double, 2>();
                break;
            case (4) :
                useHintCount = probeAutoVectorize<double, 4>(oclApp, true);
                noHintCount = probeAutoVectorize<double, 4>(oclApp, false);
                autoVecHint = autoVectorizeHint<double, 4>();
                break;
            case (8) :
                useHintCount = probeAutoVectorize<double, 8>(oclApp, true);
                noHintCount = probeAutoVectorize<double, 8>(oclApp, false);
                autoVecHint = autoVectorizeHint<double, 8>();
                break;
            case (16) :
                useHintCount = probeAutoVectorize<double, 16>(oclApp, true);
                noHintCount = probeAutoVectorize<double, 16>(oclApp, false);
                autoVecHint = autoVectorizeHint<double, 16>();
                break;
        }
    }

    cout << autoVecHint << " is ";

    if (1 == useHintCount && 1 == noHintCount)
        cout << "ok";
    else if (0 == useHintCount && 1 == noHintCount)
        cout << "fail";
    else
        cout << "unknown";

    cout << endl;

    return 0;
}
