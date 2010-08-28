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
#include <vector>
#include "OCLApp.hpp"
#include "OCLAppUtil.hpp"
#include "GatlasBenchmark.hpp"
#include "GatlasCodeText.hpp"

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

    void setMatrixDimensions(const size_t M, const size_t N, const size_t K);

    bool dimChanged() const;

    size_t dimM() const;
    size_t dimN() const;
    size_t dimK() const;
};

////////////////////////////////////////
// MatmulDataLayout

class MatmulDataLayout
{
    // transpose  matrix data layout
    // ---------  ------------------
    // true       column major
    // false      row major
    bool _transposeA;
    bool _transposeB;

    // indicates images must be reallocated
    bool _abcChanged;

public:
    MatmulDataLayout();

    void setDataLayout(const bool A, const bool B);

    bool layoutChanged() const;

    bool transposeA() const;
    bool transposeB() const;
};

////////////////////////////////////////
// MatmulWorkGroup

class MatmulWorkGroup
{
    // work group dimensions, corresponds to warp/wavefront
    size_t _dimHeight;
    size_t _dimWidth;

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

    void setWorkGroup(const size_t height, const size_t width);
    void setWorkGroup(const size_t sz);

    size_t groupHeight() const;
    size_t groupWidth() const;
    size_t groupSize() const;

    // for memory buffer kernels
    size_t localHeight() const;
    size_t localWidth() const;
    size_t localSize() const;
};

////////////////////////////////////////
// MatmulInnerBlocking

class MatmulInnerBlocking
{
    size_t _blockHeight;
    size_t _blockWidth;

public:
    MatmulInnerBlocking();

    // blocking is height x width
    void setInnerBlocking(const size_t height, const size_t width);

    size_t blockHeight() const;
    size_t blockWidth() const;

    size_t wholeHeight() const;
    size_t fractHeight() const;

    ConstantValue<std::string> multHeight(const Value& valSize) const;
    size_t multHeight(const size_t valSize) const;
};

////////////////////////////////////////
// MatmulExtraParameter

class MatmulExtraParameterObserver;

class MatmulExtraParameter
{
    size_t _extraParam;
    size_t _totalVariations; // 0 <= _extraParam < _totalVariations

    std::vector<size_t> _observerParams;

    std::vector<MatmulExtraParameterObserver*> _observers;
    std::vector<size_t> _numberVariations;

public:
    MatmulExtraParameter();

    MatmulExtraParameter& getExtraParameter();

    void addObserver(MatmulExtraParameterObserver* observer);
    void setExtraParameter(const size_t value);
    std::vector<size_t> extraParamDetail() const;
    size_t extraParam() const;
    size_t totalVariations() const;
};

class MatmulExtraParameterObserver
{
    const size_t _numberVariations;
    size_t _paramValue;

protected:
    size_t getParam() const;

public:
    MatmulExtraParameterObserver(const size_t numberVariations,
                                 MatmulExtraParameter& subject);

    size_t numberVariations() const;
    void setParam(const size_t value);
};

////////////////////////////////////////
// MatmulParamInlineMNK

struct MatmulParamInlineMNK : public MatmulExtraParameterObserver
{
    MatmulParamInlineMNK(MatmulExtraParameter& subject);

    // matrix dimensions are inlined constants or passed as kernel arguments
    bool inlineMNK() const;
};

////////////////////////////////////////
// MatmulParamLoopOrder

struct MatmulParamLoopOrder : public MatmulExtraParameterObserver
{
    MatmulParamLoopOrder(MatmulExtraParameter& subject);

    // inner product accumulation loop order, 3! permutations of (j,k,l)
    size_t loopOrder() const;
};

////////////////////////////////////////
// MatmulParamGlobalID

struct MatmulParamGlobalID : public MatmulExtraParameterObserver
{
    MatmulParamGlobalID(MatmulExtraParameter& subject);

    // use global or group/local ID
    bool globalID() const;
};

////////////////////////////////////////
// MatmulAttrAutoVec

class MatmulAttrAutoVec
{
    // kernel vector attribute hint is not supported on all platforms
    bool _useAttrAutoVec; // default value is true

public:
    MatmulAttrAutoVec();

    void setUseAttrAutoVec(const bool value);
    bool getUseAttrAutoVec() const;
};

////////////////////////////////////////
// MatmulGeneralized

class MatmulGeneralized
{
    // GEMM if true, pure matrix multiply if false (default)
    bool _generalizedMatmul;

    // buffers are different for matrix multiply and GEMM
    bool _gemmChanged;

public:
    MatmulGeneralized();

    void setGeneralizedMatmul(const bool GEMM);

    bool gemmChanged() const;

    bool generalizedMatmul() const;
};

////////////////////////////////////////
// MatmulPackedCalc

class MatmulPackedCalc
{
    // number of computational kernels to pack into one GPU kernel
    size_t _packedCalc;

    // did the number of kernels change?
    bool   _packedChanged;

public:
    MatmulPackedCalc();

    void setPackedCalc(const size_t numCalc);

    bool packedChanged() const;

    size_t packedCalc() const;
};

////////////////////////////////////////
// KernelBaseMatmul

class KernelBaseMatmul : public KernelInterface,
                         protected MatmulMatrixDimensions,
                         protected MatmulDataLayout,
                         protected MatmulWorkGroup,
                         protected MatmulInnerBlocking,
                         protected MatmulExtraParameter,
                         protected MatmulAttrAutoVec,
                         protected MatmulGeneralized,
                         protected MatmulPackedCalc
{
public:
    // some OpenCL platforms do not support auto vectorize attribute
    using MatmulAttrAutoVec::setUseAttrAutoVec;

    // packed kernel support
    using MatmulPackedCalc::setPackedCalc;

    // parameters for kernel code generation
    using MatmulGeneralized::setGeneralizedMatmul;
    using MatmulMatrixDimensions::setMatrixDimensions;
    using MatmulDataLayout::setDataLayout;
    using MatmulWorkGroup::setWorkGroup;
    using MatmulInnerBlocking::setInnerBlocking;
    using MatmulExtraParameter::setExtraParameter;

    // accessors
    using MatmulMatrixDimensions::dimM;
    using MatmulMatrixDimensions::dimN;
    using MatmulMatrixDimensions::dimK;
    using MatmulDataLayout::transposeA;
    using MatmulDataLayout::transposeB;
    using MatmulWorkGroup::groupHeight;
    using MatmulWorkGroup::groupWidth;
    using MatmulWorkGroup::groupSize;
    using MatmulInnerBlocking::blockHeight;
    using MatmulInnerBlocking::blockWidth;
    using MatmulExtraParameter::extraParam;

    // number of valid kernel extra parameter values
    using MatmulExtraParameter::totalVariations;

    // kernel extra parameter by each dimension
    //using MatmulExtraParameter::extraParamDetail;
    std::vector<size_t> extraParamDetail() const {
        return MatmulExtraParameter::extraParamDetail();
    }

protected:
    KernelBaseMatmul();
    virtual ~KernelBaseMatmul();

    // inner product accumulation
    template <typename SCALAR, size_t VECTOR_LENGTH>
    std::string assignMAD(const Vector< VecType<SCALAR, VECTOR_LENGTH> >& accum,
                          const Vector< VecType<SCALAR, VECTOR_LENGTH> >& valA,
                          const Vector< VecType<SCALAR, VECTOR_LENGTH> >& valB,
                          const size_t j,           // output row
                          const size_t k,           // output vector element component
                          const size_t l) const {   // inner product component
        if (transposeA()) {
            const size_t offset = VECTOR_LENGTH * (j / VECTOR_LENGTH);
            const size_t index = j % VECTOR_LENGTH;
            if (transposeB())
                // At Bt
                return true //isfloat<SCALAR>()
                    ? assign(accum[j][k], MADValue(valA[l + offset][index], valB[k][l], accum[j][k]))
                    : increment(accum[j][k],
                                ConstantValue<std::string>(valA[l + offset][index]) *
                                ConstantValue<std::string>(valB[k][l]));
            else
                // At B
                return true //isfloat<SCALAR>()
                    ? assign(accum[j][k], MADValue(valA[l + offset][index], valB[l][k], accum[j][k]))
                    : increment(accum[j][k],
                                ConstantValue<std::string>(valA[l + offset][index]) *
                                ConstantValue<std::string>(valB[l][k]));
        } else {
            if (transposeB())
                // A Bt
                return true //isfloat<SCALAR>()
                    ? assign(accum[j][k], MADValue(valA[j][l], valB[k][l], accum[j][k]))
                    : increment(accum[j][k],
                                ConstantValue<std::string>(valA[j][l]) *
                                ConstantValue<std::string>(valB[k][l]));
            else
                // A B
                return true //isfloat<SCALAR>()
                    ? assign(accum[j][k], MADValue(valA[j][l], valB[l][k], accum[j][k]))
                    : increment(accum[j][k],
                                ConstantValue<std::string>(valA[j][l]) *
                                ConstantValue<std::string>(valB[l][k]));
        }
    }

    // inner product loop reordering
    template <typename SCALAR, size_t VECTOR_LENGTH>
    std::ostream& assignMAD(std::ostream& os,
                            const size_t loopOrder,
                            const Vector< VecType<SCALAR, VECTOR_LENGTH> >& accum,
                            const Vector< VecType<SCALAR, VECTOR_LENGTH> >& valA,
                            const Vector< VecType<SCALAR, VECTOR_LENGTH> >& valB) const {
        switch (loopOrder) {
            case (0) : // (j, k, l)
                for (size_t j = 0; j < blockHeight(); j++) // vector element
                for (size_t k = 0; k < VECTOR_LENGTH; k++) // component of vector element
                for (size_t l = 0; l < VECTOR_LENGTH; l++) // component of temporary values
                    os << assignMAD(accum, valA, valB, j, k, l);
                break;
            case (1) : // (k, j, l)
                for (size_t k = 0; k < VECTOR_LENGTH; k++) // component of vector element
                for (size_t j = 0; j < blockHeight(); j++) // vector element
                for (size_t l = 0; l < VECTOR_LENGTH; l++) // component of temporary values
                    os << assignMAD(accum, valA, valB, j, k, l);
                break;
            case (2) : // (l, j, k)
                for (size_t l = 0; l < VECTOR_LENGTH; l++) // component of temporary values
                for (size_t j = 0; j < blockHeight(); j++) // vector element
                for (size_t k = 0; k < VECTOR_LENGTH; k++) // component of vector element
                    os << assignMAD(accum, valA, valB, j, k, l);
                break;
            case (3) : // (j, l, k)
                for (size_t j = 0; j < blockHeight(); j++) // vector element
                for (size_t l = 0; l < VECTOR_LENGTH; l++) // component of temporary values
                for (size_t k = 0; k < VECTOR_LENGTH; k++) // component of vector element
                    os << assignMAD(accum, valA, valB, j, k, l);
                break;
            case (4) : // (k, l, j)
                for (size_t k = 0; k < VECTOR_LENGTH; k++) // component of vector element
                for (size_t l = 0; l < VECTOR_LENGTH; l++) // component of temporary values
                for (size_t j = 0; j < blockHeight(); j++) // vector element
                    os << assignMAD(accum, valA, valB, j, k, l);
                break;
            case (5) : // (l, k, j)
                for (size_t l = 0; l < VECTOR_LENGTH; l++) // component of temporary values
                for (size_t k = 0; k < VECTOR_LENGTH; k++) // component of vector element
                for (size_t j = 0; j < blockHeight(); j++) // vector element
                    os << assignMAD(accum, valA, valB, j, k, l);
                break;
        }
        return os;
    }

public:
    bool validParams() const;
    bool getParams(std::vector<size_t>& params) const;
    void setParams(const std::vector<size_t>& params);

    std::vector<size_t> globalWorkItems() const;
    std::vector<size_t> localWorkItems() const;

    size_t numberFlops() const;
};

}; // namespace

#endif
