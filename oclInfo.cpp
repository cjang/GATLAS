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
    if (cpuidx.empty() && gpuidx.empty())
    {
        cerr << "no CPU or GPU devices found" << endl;
        return -1;
    }

    // prefer GPU over CPU, just take first device found
    const size_t devidx = gpuidx.empty() ? cpuidx[0] : gpuidx[0];
    const char* devtype = gpuidx.empty() ? "CPU" : "GPU";
    cout << "Using " << devtype << " device " << devidx << endl;

    // program, kernels and memory buffers for selected device
    OCLApp oclApp(oclBase, devidx);
    oclApp.print();

    if (oclApp.setOutOfOrder())
        cout << "set command queue to out of order" << endl;
    else
    {
        cerr << "failed to set command queue to out of order (expected)"
             << endl;
    }

    return 0;
}
