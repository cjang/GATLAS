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
#include <string.h>
#include "OCLBase.hpp"
#include "OCLApp.hpp"

using namespace std;

#include "using_namespace"

int main(int argc, char *argv[])
{
    // platform, devices, contexts, command queues
    OCLBase oclBase;
    oclBase.print();

    // check that devices were found
    const vector<size_t> cpuidx = oclBase.cpuIndexes();
    const vector<size_t> gpuidx = oclBase.gpuIndexes();
    const vector<size_t> accidx = oclBase.accIndexes();
    if (cpuidx.empty() && gpuidx.empty() && accidx.empty())
    {
        cerr << "no CPU, GPU or ACCELERATOR devices found" << endl;
        return -1;
    }

    // order preference for devices, just take first device found
    // 1. ACCELERATOR
    // 2. GPU
    // 3. CPU
    const bool isACCELERATOR = !accidx.empty();
    const bool isGPU = !gpuidx.empty();
    const bool isCPU = !cpuidx.empty();
    size_t devidx;
    const char *devtype;
    if (isACCELERATOR) {
        devidx = accidx[0];
        devtype = "ACCELERATOR";
    } else if (isGPU) {
        devidx = gpuidx[0];
        devtype = "GPU";
    } else if (isCPU) {
        devidx = cpuidx[0];
        devtype = "CPU";
    }
    cout << "Using " << devtype << " device " << devidx << endl;

    // program, kernels and memory buffers for selected device
    OCLApp oclApp(oclBase, devidx);
    oclApp.print();

    return 0;
}
