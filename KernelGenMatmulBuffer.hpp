#ifndef _GATLAS_KERNEL_GEN_MATMUL_BUFFER_HPP_
#define _GATLAS_KERNEL_GEN_MATMUL_BUFFER_HPP_

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

#include "KernelBaseMatmul.hpp"

#include "declare_namespace"

// general matrix multiply using memory buffers
class KernelGenMatmulBuffer : public KernelGenMatmul
{
    int _handleA;
    int _handleB;
    int _handleC;

    // compare matrix multiply to reference results
    bool _paranoidCheck;
    scalar *_paranoidC;

    static const size_t NUMBER_EXTRA_PARAM = 2  // matrix dimensions inline or passed as kernel arguments
                                           * 6; // inner product loop order

protected:
    std::string namePrefix() const;

public:
    KernelGenMatmulBuffer(const bool transposeA = false,
                          const bool transposeB = true);

    ~KernelGenMatmulBuffer();

    void paranoidCheck();

    std::string desc() const;

    size_t maxGroupSize(const size_t M, const size_t N, const size_t K) const;

    size_t maxBlockHeight() const;

    // matrix dimensions are inlined constants or passed as kernel arguments
    bool _inlineMNK(const size_t extraParam) const;

    // inner product accumulation loop order, 3! permutations of (j,k,l)
    size_t _loopOrder(const size_t extraParam) const;

    bool syncOutput(OCLApp& oclApp);
    bool checkOutput(OCLApp& oclApp, const bool printOutput);

    bool setArgs(OCLApp& oclApp, const size_t kernelHandle, bool syncInput);

    // prints the kernel source
    std::ostream& print(std::ostream& os) const;
};

}; // namespace

#endif
