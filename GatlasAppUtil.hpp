#ifndef _GATLAS_APP_UTIL_HPP_
#define _GATLAS_APP_UTIL_HPP_

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

#include <string>
#include <vector>

#include "OCLBase.hpp"
#include "GatlasBenchmark.hpp"

#include "declare_namespace"

namespace AppUtil
{
    // device may be: cpu, gpu, acc or cpuN, gpuN, accN where N = 0, 1,...
    int getDeviceIndex(OCLBase& oclBase, const std::string& device);

    // initialize benchmark vectors
    void benchInit(const std::vector< std::vector<size_t> >& pargs,
                   std::vector<bool>& pargsOk,
                   std::vector<size_t>& pargsTime,
                   std::vector<size_t>& pargsFlops,
                   std::vector<double>& pargsAverage,
                   std::vector<double>& pargsVariance,
                   std::vector< std::vector<size_t> >& pargsExtraDetail);

    // return number of benchmarked kernels that were ok
    size_t benchLoop(const size_t trialNumber,
                     KernelInterface& kernel,
                     Bench& bench,
                     const std::vector< std::vector<size_t> >& pargs,
                     std::vector<bool>& pargsOk,
                     std::vector<size_t>& pargsTime,
                     std::vector<size_t>& pargsFlops,
                     std::vector<double>& pargsAverage,
                     std::vector<double>& pargsVariance,
                     std::vector< std::vector<size_t> >& pargsExtraDetail,
                     const bool busTransferToDevice,
                     const bool busTransferFromDevice,
                     const bool dummyRun = false,
                     const bool printDebug = false);

    // return number of benchmarked kernels that were ok
    size_t benchLoop(const size_t trialNumber,
                     KernelInterface& kernel,
                     Bench& bench,
                     Journal& journal,
                     const std::vector< std::vector<size_t> >& pargs,
                     std::vector<bool>& pargsOk,
                     std::vector<size_t>& pargsTime,
                     std::vector<size_t>& pargsFlops,
                     std::vector<double>& pargsAverage,
                     std::vector<double>& pargsVariance,
                     std::vector< std::vector<size_t> >& pargsExtraDetail,
                     const bool busTransferToDevice,
                     const bool busTransferFromDevice,
                     const bool dummyRun = false,
                     const bool printDebug = false);

    // keep top N fastest times
    void markBench(const size_t topN,
                   std::vector<bool>& pargsOk,
                   const std::vector<size_t>& pargsTime);

    // keep top N fastest average gigaFLOPS
    void markBench(const size_t topN,
                   std::vector<bool>& pargsOk,
                   const std::vector<double>& pargsAverage);

    // nth place fastest average gigaFLOPS, -1 if no such finisher
    int rankBench(const size_t nthPlace, // 0th is winner, 1st is runner up, etc
                  std::vector<bool>& pargsOk,
                  const std::vector<double>& pargsAverage);

    // print benchmark results
    void printBench(const size_t numberTrials,
                    const std::vector< std::vector<size_t> >& pargs,
                    const std::vector<bool>& pargsOk,
                    const std::vector<size_t>& pargsTime,
                    const std::vector<double>& pargsAverage,
                    const std::vector<double>& pargsVariance,
                    const std::vector< std::vector<size_t> >& pargsExtraDetail);
};

}; // namespace

#endif
