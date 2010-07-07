#ifndef _GATLAS_BENCHMARK_HPP_
#define _GATLAS_BENCHMARK_HPP_

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

#include <map>
#include <string>
#include <vector>
#include "OCLApp.hpp"

#include "declare_namespace"

struct KernelInterface
{
    // return the kernel name
    virtual std::string kernelName() const = 0;

    // return number of flops
    virtual size_t numberFlops() const = 0;

    // kernel source is parameterized
    virtual void setParams(const std::vector<size_t>& params) = 0;

    // kernel extra parameter by dimensions
    virtual std::vector<size_t> extraParamDetail() const = 0;

    // allocate buffers and set kernel matrix arguments
    virtual bool setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) = 0;

    // check output, sometimes bad kernels do nothing
    virtual bool syncOutput(OCLApp& oclApp) = 0;
    virtual bool checkOutput(OCLApp& oclApp, const bool printOutput = false) = 0;

    // switches on paranoid checking
    virtual void paranoidCheck() = 0;

    // work items
    virtual std::vector<size_t> globalWorkItems() const = 0;
    virtual std::vector<size_t> localWorkItems() const = 0;

    // prints the kernel source
    virtual std::ostream& print(std::ostream&) const = 0;
};

std::ostream& operator<< (std::ostream& os, const KernelInterface& k);

class Journal
{
    const std::string _journalFile;

    std::map<std::string, int>                  _memoRunState; // contains all param keys
    std::map<std::string, std::vector<size_t> > _memoTime;     // only contains param keys in state KERNEL_OK

    std::string toString(const KernelInterface& kernel, const std::vector<size_t>& params) const;

public:
    enum RunState { MISSING           = 0,
                    BUILD_IN_PROGRESS = -1,
                    BUILD_OK          = -2,
                    RUN_IN_PROGRESS   = -3,
                    RUN_OK            = -4 };

    Journal(const std::string& journalFile);

    // load records from memo file
    bool loadMemo();

    // (assumes load memo has been called)
    // remove unnecessary records from memo file (kernel solutions that ran ok; keep only the last record for a key)
    // returns number of bad kernels (either crashed during build or hung while running)
    int purgeMemo(const bool deleteTimes = false);

    // read from memo
    size_t memoGood() const; // number of kernels than ran ok (even if check output failed)
    int    memoRunState(const KernelInterface& kernel, const std::vector<size_t>& params);
    int    memoTime(const KernelInterface& kernel, const std::vector<size_t>& params, const size_t trialNumber);

    // write to memo file
    bool takeMemo(const KernelInterface& kernel, const std::vector<size_t>& params, const int value) const;
};

class Bench
{
    OCLApp&                  _oclApp;
    KernelInterface&         _kernel;
    Journal*                 _journal;
    std::vector<std::string> _programSource;
    int                      _kernelHandle;

    const bool               _printStatus;

    bool rebuildProgram();

public:

    Bench(OCLApp& oclApp, KernelInterface& kernel, const bool printStatus = true);
    Bench(OCLApp& oclApp, KernelInterface& kernel, Journal& journal, const bool printStatus = true);

    bool printStatus() const;

    // returns elapsed time in microseconds, 0 if error
    size_t run(const size_t numTrials,
               const std::vector<size_t>& args,
               const bool busTransferToDevice,
               const bool busTransferFromDevice,
               const bool printDebug = false);
};

}; // namespace

#endif
