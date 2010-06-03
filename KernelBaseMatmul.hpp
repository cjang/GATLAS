#ifndef _GATLAS_KERNEL_BASE_MATMUL_HPP_
#define _GATLAS_KERNEL_BASE_MATMUL_HPP_

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
#include <sstream>
#include "OCLApp.hpp"
#include "OCLAppUtil.hpp"
#include "GatlasBenchmark.hpp"

#include "declare_namespace"

////////////////////////////////////////
// MatmulMatrixDimensions

class MatmulMatrixDimensions
{
    // C = AB where A is MxK, B is KxN, C is MxN
    size_t _dimM;
    size_t _dimN;
    size_t _dimK;

    // indicates matrix memory must be resized
    bool _mnkChanged;

public:
    MatmulMatrixDimensions();

    void setMNK(const size_t M, const size_t N, const size_t K);

    bool mnkChanged() const;

    size_t dimM() const;
    size_t dimN() const;
    size_t dimK() const;
};

////////////////////////////////////////
// MatmulDataLayout

class MatmulDataLayout
{
    const bool _transposeA;
    const bool _transposeB;

public:
    MatmulDataLayout(const bool A, const bool B);

    bool transposeA() const;
    bool transposeB() const;
};

////////////////////////////////////////
// MatmulWorkGroup

class MatmulWorkGroup
{
    // work group dimensions, corresponds to warp/wavefront
    size_t _dimWidth;
    size_t _dimHeight;

    // add one to avoid local memory bank conflicts
    static const size_t LOCALMEM_PAD = 1;

public:
    // work item IDs
    const ConstantValue<std::string> globalCol;
    const ConstantValue<std::string> globalRow;
    const ConstantValue<std::string> blockRow;
    const ConstantValue<std::string> blockCol;
    const ConstantValue<std::string> row;
    const ConstantValue<std::string> col;

    MatmulWorkGroup();

    void setWorkGroup(const size_t width, const size_t height);
    void setWorkGroup(const size_t sz);

    size_t groupWidth() const;
    size_t groupHeight() const;
    size_t groupSize() const;

    // for memory buffer kernels
    size_t localWidth() const;
    size_t localHeight() const;
    size_t localSize() const;
};

////////////////////////////////////////
// MatmulQuadBlocking

class MatmulQuadBlocking
{
    size_t _blockWidth;
    size_t _blockHeight;

public:
    // explicitly vectorized kernels
    typedef float scalar;
    static const size_t VECTOR_LENGTH = 4;
    typedef VecType<scalar, VECTOR_LENGTH> scalarN;

    MatmulQuadBlocking();

    void setQuadBlocking(const size_t height);

    size_t blockWidth() const;
    size_t blockHeight() const;

    size_t wholeQuads() const;
    size_t fractQuads() const;

    ConstantValue<std::string> multQuads(const Value& valSize) const;
    size_t multQuads(const size_t valSize) const;
};

////////////////////////////////////////
// MatmulExtraParameter

class MatmulExtraParameter
{
    size_t _extraParam;

    const size_t _numExtraParam;

public:
    MatmulExtraParameter(const size_t numExtraParam);

    void setExtraParameter(const size_t value);

    size_t extraParam() const;
    size_t numExtraParam() const;
};

////////////////////////////////////////
// KernelBaseMatmul

class KernelBaseMatmul : public KernelInterface,
                         protected MatmulMatrixDimensions,
                         protected MatmulDataLayout,
                         protected MatmulWorkGroup,
                         protected MatmulQuadBlocking,
                         protected MatmulExtraParameter
{
    // kernel vector attribute hint is not supported on all platforms
    bool _useAttrAutoVec; // default value is true

public:
    void setUseAttrAutoVec(const bool value);

    using MatmulQuadBlocking::VECTOR_LENGTH;

    // matrix dimensions are inlined constants or passed as kernel arguments
    virtual bool _inlineMNK(const size_t) const = 0;
    bool inlineMNK() const;

    // inner product accumulation loop order, 3! permutations of (j,k,l)
    virtual size_t _loopOrder(const size_t extraParam) const = 0;
    size_t loopOrder() const;

protected:
    bool getUseAttrAutoVec() const;

    // inner product accumulation
    std::string assignMAD(const Vector< scalarN >& accum,
                          const Vector< scalarN >& valA,
                          const Vector< scalarN >& valB,
                          const size_t j,           // output row
                          const size_t k,           // output vector element component
                          const size_t l) const;    // inner product component

    // inner product loop reordering
    std::ostream& assignMAD(std::ostream& os,
                            const size_t loopOrder,
                            const Vector< scalarN >& accum,
                            const Vector< scalarN >& valA,
                            const Vector< scalarN >& valB) const;

    KernelBaseMatmul(const bool transposeA,
                     const bool transposeB,
                     const size_t numExtraParam);

    virtual ~KernelBaseMatmul();

    virtual std::string namePrefix() const = 0;

public:
    std::string kernelName() const;

    // initializes and activates paranoid checking of kernel output
    virtual void paranoidCheck() = 0;

    std::vector<size_t> globalWorkItems() const;

    std::vector<size_t> localWorkItems() const;

    // skip small work group sizes as they are always slow
    virtual size_t minGroupSize(const size_t M, const size_t N, const size_t K) const;

    // maximum work group size varies with matrix dimensions (to avoid kernel hangs)
    virtual size_t maxGroupSize(const size_t M, const size_t N, const size_t K) const = 0;

    // inner blocking depends on the kernel
    size_t minBlockHeight() const;
    virtual size_t maxBlockHeight() const = 0;
    size_t stepBlockHeight() const;

    // validate kernel parameters
    bool validateParams(const size_t M,
                        const size_t N,
                        const size_t K,
                        const size_t groupSize,
                        const size_t blockHeight,
                        const size_t extraParam) const;

    bool setParams(const size_t M,
                   const size_t N,
                   const size_t K,
                   const size_t groupSize,
                   const size_t blockHeight,
                   const size_t extraParam);

    bool setParams(const std::vector<size_t>& args);

    size_t getGroupSize(const std::vector<size_t>& args) const;
    size_t getBlockHeight(const std::vector<size_t>& args) const;
    size_t getExtraParam(const std::vector<size_t>& args) const;

    // generate parameter arguments
    std::vector< std::vector<size_t> > parameters(const size_t M,
                                                  const size_t N,
                                                  const size_t K,
                                                  const int GroupSize,
                                                  const int BlockHeight = -1) const;

    // generate parameter args likely to contain optimal solutions
    std::vector< std::vector<size_t> > parameters(const size_t M,
                                                  const size_t N,
                                                  const size_t K) const;

    // generate all parameters args
    std::vector< std::vector<size_t> > parametersAll(const size_t M,
                                                     const size_t N,
                                                     const size_t K) const;
};

// base for matrix multiply
class KernelMatmul : public KernelBaseMatmul
{
protected:
    KernelMatmul(const bool transposeA,
                 const bool transposeB,
                 const size_t numExtraParam);

    virtual ~KernelMatmul();

public:
    size_t numberFlops() const;
};

// base for GEMM
class KernelGenMatmul : public KernelBaseMatmul
{
protected:
    KernelGenMatmul(const bool transposeA,
                    const bool transposeB,
                    const size_t numExtraParam);

    virtual ~KernelGenMatmul();

public:
    size_t numberFlops() const;
};

}; // namespace

#endif
