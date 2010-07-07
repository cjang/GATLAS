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
#include <ostream>
#include <sstream>
#include <stdlib.h>
#include <sys/time.h>
#include "GatlasType.hpp"
#include "GatlasQualifier.hpp"
#include "GatlasCodeText.hpp"

#include "GatlasBenchmark.hpp"

#include "declare_namespace"

using namespace std;

////////////////////////////////////////
// KernelInterface

ostream& operator<< (ostream& os, const KernelInterface& k) {
    return k.print(os);
}

////////////////////////////////////////
// Journal

string Journal::toString(const KernelInterface& kernel, const vector<size_t>& params) const {
    stringstream ss;
    ss << kernel.kernelName() << "_";
    for (size_t i = 0; i < params.size(); i++)
        ss << params[i] << "_";
    return ss.str();
}

Journal::Journal(const std::string& journalFile)
    : _journalFile(journalFile)
{ }

bool Journal::loadMemo() {
    ifstream journal(_journalFile.c_str());
    if (journal.is_open()) {
        string key;
        int value;
        _memoRunState.clear();
        _memoTime.clear();
        while (! journal.eof() && (journal >> key >> value)) {
            if (value < 0)
                _memoRunState[key] = value;
            else
                _memoTime[key].push_back(value);
        }
        return true;
    } else {
        return false;
    }
}

int Journal::purgeMemo(const bool deleteTimes) {
    int count = -1;
    ofstream journal(_journalFile.c_str());
    if (journal.is_open()) {
        count = 0;
        for (map<string, int>::const_iterator iter = _memoRunState.begin();
             iter != _memoRunState.end();
             iter++) {
            const string key = (*iter).first;
            const int value = (*iter).second;

            // always keep records for bad kernels
            if (0 == _memoTime.count(key)) {
                // this kernel has no benchmark time so must be bad
                journal << key << "\t" << value << endl;
                count++; // increment number of bad kernels

            // optionally keep benchmark time records for good kernels
            } else {
                if (! deleteTimes) {
                    journal << key << "\t" << value << endl; // this is always RUN_OK
                    for (size_t i = 0; i < _memoTime[key].size(); i++) {
                        const size_t benchtime = _memoTime[key][i];
                        journal << key << "\t" << benchtime << endl;
                    }
                }
            }
        }
    }
    return count;
}

size_t Journal::memoGood() const { return _memoTime.size(); }

int Journal::memoRunState(const KernelInterface& kernel, const vector<size_t>& params) {
    const string key = toString(kernel, params);
    if (0 == _memoRunState.count(key))
        return MISSING; // not in memo
    else
        return _memoRunState[key];
}

int Journal::memoTime(const KernelInterface& kernel, const vector<size_t>& params, const size_t trialNumber) {
    const string key = toString(kernel, params);
    if (0 == _memoTime.count(key))
        return -1; // not in memo
    else {
        if (trialNumber < _memoTime[key].size())
            return _memoTime[key][trialNumber];
        else
            return -1; 
    }
}

bool Journal::takeMemo(const KernelInterface& kernel, const std::vector<size_t>& params, const int value) const {
    ofstream journal(_journalFile.c_str(), ios::app);
    if (journal.is_open()) {
        journal << toString(kernel, params) << "\t" << value << endl;
        return true;
    } else {
        return false;
    }
}

////////////////////////////////////////
// Bench

Bench::Bench(OCLApp& oclApp, KernelInterface& kernel, const bool printStatus)
    : _oclApp(oclApp),
      _kernel(kernel),
      _journal(NULL),
      _kernelHandle(-1),
      _printStatus(printStatus)
{ }

Bench::Bench(OCLApp& oclApp, KernelInterface& kernel, Journal& journal, const bool printStatus)
    : _oclApp(oclApp),
      _kernel(kernel),
      _journal(&journal),
      _kernelHandle(-1),
      _printStatus(printStatus)
{ }

bool Bench::printStatus() const { return _printStatus; }

bool Bench::rebuildProgram() {
    // program source
    stringstream ss;
    ss << _kernel;
    _programSource.clear();
    _programSource.push_back(ss.str());

    // build program
    if (_oclApp.buildProgram(_programSource)) {
        // create kernel
        _kernelHandle = _oclApp.createKernel(_kernel.kernelName());
        return true;
    } else {
        return false;
    }
}

// returns elapsed time in microseconds, 0 if error
size_t Bench::run(const size_t numTrials,
                  const vector<size_t>& args,
                  const bool busTransferToDevice,
                  const bool busTransferFromDevice,
                  const bool printDebug) {

    // kernel parameter arguments
    _kernel.setParams(args);

    if (_printStatus && printDebug) cerr << _kernel << endl;

    if (_journal) _journal->takeMemo(_kernel, args, Journal::BUILD_IN_PROGRESS);

    // kernels change depending on arguments
    if (_printStatus) cerr << "rebuilding kernel...";
    if (! rebuildProgram()) return 0; // build program failed
    if (_printStatus) cerr << " done\t";

    if (_journal) _journal->takeMemo(_kernel, args, Journal::BUILD_OK);

    // set kernel arguments (exclude PCIe bus transfer cost)
    if (!busTransferToDevice)
        if (!_kernel.setArgs(_oclApp, _kernelHandle, busTransferToDevice)) return 0; // fail

    // work item dimensions
    const vector<size_t> globalDims = _kernel.globalWorkItems();
    const vector<size_t> localDims = _kernel.localWorkItems();

    if (_journal) _journal->takeMemo(_kernel, args, Journal::RUN_IN_PROGRESS);

    // start gettimeofday timer
    struct timeval start_time;
    if (-1 == gettimeofday(&start_time, 0)) {
        if (_printStatus) cerr << "error: start gettimeofday" << endl;
        return 0; // fail
    }

    // set kernel arguments (include PCIe bus transfer cost)
    if (busTransferToDevice)
        if (!_kernel.setArgs(_oclApp, _kernelHandle, busTransferToDevice)) return 0; // fail

    // execute kernel for specified number of trials
    int waitKernel;
    for (size_t i = 0; i < numTrials; i++)
    {
        if (0 == i)
            // first enqueued kernel
            waitKernel = _oclApp.enqueueKernel(_kernelHandle, globalDims, localDims);
        else
            // subsequent enqueued kernels depend on previous one
            waitKernel = _oclApp.enqueueKernel(_kernelHandle, globalDims, localDims, waitKernel);

        if (-1 == waitKernel) {
            if (_printStatus) cerr << "error: enqueue kernel " << i << endl;
            if (0 != i && !_oclApp.wait()) // blocking call
                if (_printStatus) cerr << "error: waiting for " << (i-1) << " enqueued kernels" << endl;
            return 0; // fail
        }
    }

    // wait for all kernels to finish
    if (!_oclApp.wait(waitKernel)) { // blocking call
        if (_printStatus) cerr << "error: waiting for all kernels" << endl;
        return 0; // fail
    }

    // read back output data from device (including PCIe data transfer cost)
    if (busTransferFromDevice) {
        if (!_kernel.syncOutput(_oclApp)) {
            if (_printStatus) cerr << "error: read output data from device" << endl;
            return 0; // fail
        }
    }

    // stop gettimeofday timer
    struct timeval stop_time;
    if (-1 == gettimeofday(&stop_time, 0)) {
        if (_printStatus) cerr << "error: stop gettimeofday" << endl;
        return 0; // fail
    }

    // read back output data from device (excluding PCIe data transfer cost)
    if (!busTransferFromDevice) {
        if (!_kernel.syncOutput(_oclApp)) {
            if (_printStatus) cerr << "error: read output data from device" << endl;
            return 0; // fail
        }
    }

    // calculate elapsed time in microseconds
    const size_t elapsed_time
        = 1000 * 1000 * ( stop_time.tv_sec - start_time.tv_sec )
              + stop_time.tv_usec
              + (1000 * 1000 - start_time.tv_usec)
              - 1000 * 1000;

    // allow kernel to check results, sometimes bad kernels do nothing
    const bool isOk = _kernel.checkOutput(_oclApp, printDebug);
    if (! isOk && _printStatus) cerr << "fail";

    // final cleanup
    if (!_oclApp.wait()) {
        if (_printStatus) cerr << "error: clean up wait events" << endl;
    }

    if (_journal) {
        _journal->takeMemo(_kernel, args, Journal::RUN_OK);
        _journal->takeMemo(_kernel, args, isOk ? elapsed_time : 0);
    }

    return isOk ? elapsed_time : 0;
}

}; // namespace
