#ifndef _GATLAS_KERNEL_BASE_MATVEC_HPP_
#define _GATLAS_KERNEL_BASE_MATVEC_HPP_

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
#include "OCLApp.hpp"
#include "OCLAppUtil.hpp"
#include "GatlasBenchmark.hpp"
#include "GatlasCodeText.hpp"

#include "declare_namespace"

////////////////////////////////////////
// MatvecMatrixDimensions

class MatvecMatrixDimensions
{
    // c = Ab where A is MxN, b is Nx1, c is Nx1
    size_t _dimM;
    size_t _dimN;

    // indicates matrix memory must be resized
    bool _mnChanged;

public:
    MatvecMatrixDimensions();

    void setMatrixDimensions(const size_t M, const size_t N);

    bool dimChanged() const;

    size_t dimM() const;
    size_t dimN() const;
};

////////////////////////////////////////
// MatvecDataLayout

class MatvecDataLayout
{
    // transpose  matrix data layout
    // ---------  ------------------
    // true       column major
    // false      row major
    bool _transposeA;

    // indicates images must be reallocated
    bool _aChanged;

public:
    MatvecDataLayout();

    void setDataLayout(const bool A);

    bool layoutChanged() const;

    bool transposeA() const;
};

////////////////////////////////////////
// MatvecWorkGroup

class MatvecWorkGroup
{
    // work group dimensions, corresponds to warp/wavefront
    size_t _dimSize;

    // add one to avoid local memory bank conflicts
    static const size_t LOCALMEM_PAD = 1;

public:
    // work item IDs
    const ConstantValue<std::string> globalRow;
    const ConstantValue<std::string> blockRow;
    const ConstantValue<std::string> row;

    MatvecWorkGroup();

    void setWorkGroup(const size_t sz);

    size_t groupSize() const;

    // for memory buffer kernels
    size_t localSize() const;
};

////////////////////////////////////////
// MatvecInnerBlocking

class MatvecInnerBlocking
{
    size_t _blockHeight;
    size_t _vectorLength;

public:
    MatvecInnerBlocking();

    // blocking is height x vector length
    void setInnerBlocking(const size_t height, const size_t vectorLength);

    size_t blockHeight() const;
    size_t vectorLength() const;

    size_t wholeHeight() const;
    size_t fractHeight() const;

    ConstantValue<std::string> multHeight(const Value& valSize) const;
    size_t multHeight(const size_t valSize) const;
};

////////////////////////////////////////
// MatvecExtraParameter

class MatvecExtraParameterObserver;

class MatvecExtraParameter
{
    size_t _extraParam;
    size_t _totalVariations; // 0 <= _extraParam < _totalVariations

    std::vector<size_t> _observerParams;

    std::vector<MatvecExtraParameterObserver*> _observers;
    std::vector<size_t> _numberVariations;

public:
    MatvecExtraParameter();

    MatvecExtraParameter& getExtraParameter();

    void addObserver(MatvecExtraParameterObserver* observer);
    void setExtraParameter(const size_t value);
    std::vector<size_t> extraParamDetail() const;
    size_t extraParam() const;
    size_t totalVariations() const;
};

class MatvecExtraParameterObserver
{
    const size_t _numberVariations;
    size_t _paramValue;

protected:
    size_t getParam() const;

public:
    MatvecExtraParameterObserver(const size_t numberVariations,
                                 MatvecExtraParameter& subject);

    size_t numberVariations() const;
    void setParam(const size_t value);
};

////////////////////////////////////////
// MatvecParamInlineMN

struct MatvecParamInlineMN : public MatvecExtraParameterObserver
{
    MatvecParamInlineMN(MatvecExtraParameter& subject);

    // matrix dimensions are inlined constants or passed as kernel arguments
    bool inlineMN() const;
};

////////////////////////////////////////
// MatvecParamGlobalID

struct MatvecParamGlobalID : public MatvecExtraParameterObserver
{
    MatvecParamGlobalID(MatvecExtraParameter& subject);

    // use global or group/local ID
    bool globalID() const;
};

////////////////////////////////////////
// MatvecAttrAutoVec

class MatvecAttrAutoVec
{
    // kernel vector attribute hint is not supported on all platforms
    bool _useAttrAutoVec; // default value is true

public:
    MatvecAttrAutoVec();

    void setUseAttrAutoVec(const bool value);
    bool getUseAttrAutoVec() const;
};

////////////////////////////////////////
// MatvecGeneralized

class MatvecGeneralized
{
    // generalized if true, pure multiply if false (default)
    bool _generalizedMatvec;

    // buffers are different for pure multiply and generalized
    bool _gemvChanged;

public:
    MatvecGeneralized();

    void setGeneralizedMatvec(const bool GEMV);

    bool gemvChanged() const;

    bool generalizedMatvec() const;
};

////////////////////////////////////////
// MatvecPackedCalc

class MatvecPackedCalc
{
    // number of computational kernels to pack into one GPU kernel
    size_t _packedCalc;

    // did the number of kernels change?
    bool   _packedChanged;

public:
    MatvecPackedCalc();

    void setPackedCalc(const size_t numCalc);

    bool packedChanged() const;

    size_t packedCalc() const;
};

////////////////////////////////////////
// KernelBaseMatvec

class KernelBaseMatvec : public KernelInterface,
                         protected MatvecMatrixDimensions,
                         protected MatvecDataLayout,
                         protected MatvecWorkGroup,
                         protected MatvecInnerBlocking,
                         protected MatvecExtraParameter,
                         protected MatvecAttrAutoVec,
                         protected MatvecGeneralized,
                         protected MatvecPackedCalc
{
public:
    // some OpenCL platforms do not support auto vectorize attribute
    using MatvecAttrAutoVec::setUseAttrAutoVec;

    // packed kernel support
    using MatvecPackedCalc::setPackedCalc;

    // parameters for kernel code generation
    using MatvecGeneralized::setGeneralizedMatvec;
    using MatvecMatrixDimensions::setMatrixDimensions;
    using MatvecDataLayout::setDataLayout;
    using MatvecWorkGroup::setWorkGroup;
    using MatvecInnerBlocking::setInnerBlocking;
    using MatvecExtraParameter::setExtraParameter;

    // accessors
    using MatvecMatrixDimensions::dimM;
    using MatvecMatrixDimensions::dimN;
    using MatvecDataLayout::transposeA;
    using MatvecWorkGroup::groupSize;
    using MatvecInnerBlocking::blockHeight;
    using MatvecInnerBlocking::vectorLength;
    using MatvecExtraParameter::extraParam;

    // number of valid kernel extra parameter values
    using MatvecExtraParameter::totalVariations;

    // kernel extra parameter by each dimension
    //using MatvecExtraParameter::extraParamDetail;
    std::vector<size_t> extraParamDetail() const {
        return MatvecExtraParameter::extraParamDetail();
    }

protected:
    KernelBaseMatvec();
    virtual ~KernelBaseMatvec();

    // matrix vector product accumulation
    template <typename SCALAR, size_t VECTOR_LENGTH>
    std::string assignMAD(const Var< VecType<SCALAR, VECTOR_LENGTH> >& accum,
                          const Var< VecType<SCALAR, VECTOR_LENGTH> >& valA,
                          const Var< VecType<SCALAR, VECTOR_LENGTH> >& valB,
                          const size_t j,           // row/component of vector element
                          const size_t k) const {   // inner product column
        return transposeA()
                   ? assign(accum[k], MADValue(valA[k], valB[j], accum[k]))
                   : assign(accum[j], MADValue(valA[k], valB[k], accum[j]));
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
