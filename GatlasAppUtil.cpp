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
#include <math.h>
#include <set>

#include "GatlasAppUtil.hpp"

using namespace std;

#include "declare_namespace"

namespace AppUtil {

int getDeviceIndex(OCLBase& oclBase, const string& device)
{
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

void benchInit(const vector< vector<size_t> >& pargs,
               vector<bool>& pargsOk,
               vector<size_t>& pargsTime,
               vector<size_t>& pargsFlops,
               vector<double>& pargsAverage,
               vector<double>& pargsVariance,
               vector< vector<size_t> >& pargsExtraDetail)
{
    pargsOk.clear();
    pargsTime.clear();
    pargsFlops.clear();
    pargsAverage.clear();
    pargsVariance.clear();
    pargsExtraDetail.clear();
    for (size_t i = 0; i < pargs.size(); i++) {
        pargsOk.push_back(true);
        pargsTime.push_back(0);
        pargsFlops.push_back(0);
        pargsAverage.push_back(0);
        pargsVariance.push_back(0);
        pargsExtraDetail.push_back(vector<size_t>());
    }
}

size_t benchLoop(const size_t trialNumber,
                 KernelInterface& kernel,
                 Bench& bench,
                 const vector< vector<size_t> >& pargs,
                 vector<bool>& pargsOk,
                 vector<size_t>& pargsTime,
                 vector<size_t>& pargsFlops,
                 vector<double>& pargsAverage,
                 vector<double>& pargsVariance,
                 vector< vector<size_t> >& pargsExtraDetail,
                 const bool busTransferToDevice,
                 const bool busTransferFromDevice,
                 const bool dummyRun,
                 const bool printDebug)
{
    const bool printStatus = bench.printStatus();
    size_t goodKernelCount = 0;

    bool needDummyRun = dummyRun;

    // main loop
    for (size_t j = 0; j < pargs.size(); j++) {
        const vector<size_t>& args = pargs[j];
        if (pargsOk[j]) {

            if (needDummyRun) {
                cout << "[dummy run] ";
                bench.run(1, args, busTransferToDevice, busTransferFromDevice, printDebug);
                cout << endl;
                needDummyRun = false;
            }

            if (printStatus) cout << "[trial " << trialNumber << "] ";

            const size_t microsecs = bench.run(1, args, busTransferToDevice, busTransferFromDevice, printDebug);

            const vector<size_t>& extra = pargsExtraDetail[j] = kernel.extraParamDetail();

            if (0 == microsecs) {
                cout << "\t";
                for (size_t i = 0; i < args.size(); i++) {
                    cout << args[i];
                    if (i != args.size() - 1) cout << " ";
                }
                cout << "\t(";
                for (size_t i = 0; i < extra.size(); i++) {
                    cout << extra[i];
                    if (i != extra.size() -1) cout << " ";
                }
                cout << ")" << endl;
                pargsOk[j] = false;
                continue;
            }

            goodKernelCount++;

            const size_t numflops = kernel.numberFlops();
            pargsTime[j] += microsecs;
            pargsFlops[j] += numflops;

            // single pass mean and variance
            const double avg = static_cast<double>(numflops) / microsecs / 1000;
            if (0 == trialNumber) {
                pargsAverage[j] = avg;
            } else {
                const double delta = avg - pargsAverage[j];
                pargsAverage[j] += (static_cast<double>(1) / (trialNumber + 1)) * delta;
                pargsVariance[j] += (static_cast<double>(trialNumber) / (trialNumber + 1)) * delta * delta;
            }

            if (printStatus) {
                cout << microsecs << "\t";
                for (size_t i = 0; i < args.size(); i++) {
                    cout << args[i];
                    if (i != args.size() - 1) cout << " ";
                }
                cout << "\t(";
                for (size_t i = 0; i < extra.size(); i++) {
                    cout << extra[i];
                    if (i != extra.size() -1) cout << " ";
                }
                cout << ")" << endl;
            }
        }
    }

    return goodKernelCount;
}

size_t benchLoop(const size_t trialNumber,
                 KernelInterface& kernel,
                 Bench& bench,
                 Journal& journal,
                 const vector< vector<size_t> >& pargs,
                 vector<bool>& pargsOk,
                 vector<size_t>& pargsTime,
                 vector<size_t>& pargsFlops,
                 vector<double>& pargsAverage,
                 vector<double>& pargsVariance,
                 vector< vector<size_t> >& pargsExtraDetail,
                 const bool busTransferToDevice,
                 const bool busTransferFromDevice,
                 const bool dummyRun,
                 const bool printDebug)
{
    const bool printStatus = bench.printStatus();
    size_t goodKernelCount = 0;

    bool needDummyRun = dummyRun;

    // main loop
    for (size_t j = 0; j < pargs.size(); j++) {
        const vector<size_t>& args = pargs[j];
        if (pargsOk[j]) {

            // check memo
            const size_t memoState = journal.memoRunState(kernel, args);

            if (needDummyRun && Journal::MISSING == memoState) {
                cout << "[dummy run] ";
                bench.run(1, args, busTransferToDevice, busTransferFromDevice, printDebug);
                cout << endl;
                needDummyRun = false;
            }

            if (printStatus) cout << "[trial " << trialNumber << "] ";

            size_t microsecs;
            if (Journal::MISSING == memoState) {
                microsecs = bench.run(1, args, busTransferToDevice, busTransferFromDevice, printDebug);

            } else if (Journal::RUN_OK == memoState) {
                const int memoValue = journal.memoTime(kernel, args, trialNumber);
                microsecs = (-1 == memoValue)
                                ? bench.run(1, args, busTransferToDevice, busTransferFromDevice, printDebug)
                                : microsecs = memoValue;
                kernel.setParams(args);

            } else { // these kernel parameters cause seg fault or hang
                if (printStatus) cout << "bad";
                microsecs = 0;
                kernel.setParams(args);
            }

            const vector<size_t>& extra = pargsExtraDetail[j] = kernel.extraParamDetail();

            if (0 == microsecs) {
                if (printStatus) {
                    cout << "\t";
                    for (size_t i = 0; i < args.size(); i++) {
                        cout << args[i];
                        if (i != args.size() - 1) cout << " ";
                    }
                    cout << "\t(";
                    for (size_t i = 0; i < extra.size(); i++) {
                        cout << extra[i];
                        if (i != extra.size() -1) cout << " ";
                    }
                    cout << ")" << endl;
                }
                pargsOk[j] = false;
                continue;
            }

            goodKernelCount++;

            const size_t numflops = kernel.numberFlops();
            pargsTime[j] += microsecs;
            pargsFlops[j] += numflops;

            // single pass mean and variance
            const double avg = static_cast<double>(numflops) / microsecs / 1000;
            if (0 == trialNumber) {
                pargsAverage[j] = avg;
            } else {
                const double delta = avg - pargsAverage[j];
                pargsAverage[j] += (static_cast<double>(1) / (trialNumber + 1)) * delta;
                pargsVariance[j] += (static_cast<double>(trialNumber) / (trialNumber + 1)) * delta * delta;
            }

            if (printStatus) {
                cout << microsecs << "\t";
                for (size_t i = 0; i < args.size(); i++) {
                    cout << args[i];
                    if (i != args.size() - 1) cout << " ";
                }
                cout << "\t(";
                for (size_t i = 0; i < extra.size(); i++) {
                    cout << extra[i];
                    if (i != extra.size() -1) cout << " ";
                }
                cout << ")" << endl;
            }
        }
    }

    return goodKernelCount;
}

void markBench(const size_t topN,
               vector<bool>& pargsOk,
               const vector<size_t>& pargsTime)
{
    // prune to top N parameter combinations
    // parameter combinations with the same time are treated together

    // sort trial times, skip any parameters with errors
    set<size_t> trialTimes;
    for (size_t i = 0; i < pargsOk.size(); i++)
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

void markBench(const size_t topN,
               vector<bool>& pargsOk,
               const vector<double>& pargsAverage)
{
    // prune to top N parameter combinations
    // parameter combinations with the same time are treated together

    // sort trial average gigaFLOPS, skip any parameters with errors
    set<double> trialAvgs;
    for (size_t i = 0; i < pargsOk.size(); i++)
        if (pargsOk[i])
            trialAvgs.insert(pargsAverage[i]);

    // only interested in topN fastest parameter combinations
    size_t count = 0;
    set<double> topAvgs;
    for (set<double>::const_reverse_iterator iter = trialAvgs.rbegin();
         iter != trialAvgs.rend();
         iter++) {
        if (count++ >= topN) break;
        topAvgs.insert(*iter);
    }

    // mark all parameters as bad except for the top N fastest ones
    for (size_t i = 0; i < pargsOk.size(); i++)
        if (pargsOk[i] && 0 == topAvgs.count(pargsAverage[i]))
            pargsOk[i] = false;
}

int rankBench(const size_t nthPlace, // 0th is winner, 1st is runner up, etc
              vector<bool>& pargsOk,
              const vector<double>& pargsAverage)
{
    // sort based on average GFLOPS rate for individual trials
    map<double, size_t> gflopsToIdx;
    for (size_t i = 0; i < pargsOk.size(); i++)
        if (pargsOk[i])
            gflopsToIdx[pargsAverage[i]] = i;

    // best inner and outer blocking
    size_t count = 0;
    for (map<double, size_t>::const_reverse_iterator iter = gflopsToIdx.rbegin();
         iter != gflopsToIdx.rend();
         iter++) {

        const size_t idx = (*iter).second;
        if (nthPlace == count++) return idx;
    }

    return -1;
}

void printBench(const size_t numberTrials,
                const vector< vector<size_t> >& pargs,
                const vector<bool>& pargsOk,
                const vector<size_t>& pargsTime,
                const vector<double>& pargsAverage,
                const vector<double>& pargsVariance,
                const vector< vector<size_t> >& pargsExtraDetail)
{
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
        const vector<size_t>& extra = pargsExtraDetail[idx];

        // show average of GFLOPS rate for individual trials, not average over all trials
        // the variance is of the average GFLOPS rate for individual trials
        const double gflops = pargsAverage[idx];
        const double variance = pargsVariance[idx];

        // this is the average GFLOPS rate over all trials taken together
        //const double gflops = static_cast<double>(pargsFlops[idx]) / accumTime / 1000;

        cout << "[" << count++ << "] "
             << accumTime << " usec"
             << "\tavg: " << gflops
             << "\tstddev: " << sqrt(variance/numberTrials)
             << "\t";
        for (size_t i = 0; i < args.size(); i++) {
            cout << args[i];
            if (i != args.size() - 1) cout << " ";
        }
        cout << "\t(";
        for (size_t i = 0; i < extra.size(); i++) {
            cout << extra[i];
            if (i != extra.size() -1) cout << " ";
        }
        cout << ")" << endl;
    }
}

}; // namespace AppUtil

}; // namespace
