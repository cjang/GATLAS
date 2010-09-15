#ifndef _GATLAS_KERNEL_BASE_SAXPY_HPP_
#define _GATLAS_KERNEL_BASE_SAXPY_HPP_

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
// SaxpyDimensions

class SaxpyDimensions
{
    // Z = alpha * X + Y
    size_t _dimM;
    size_t _dimN;

    // indicates memory must be resized
    bool _mnChanged;

public:
    SaxpyDimensions();

    void setSaxpyDimensions(const size_t M, const size_t N);

    bool dimChanged() const;

    size_t dimM() const;
    size_t dimN() const;
};

////////////////////////////////////////
// SaxpyWorkGroup

class SaxpyWorkGroup
{
    // work group dimensions, corresponds to warp/wavefront
    size_t _dimHeight;
    size_t _dimWidth;

public:
    // work item IDs
    const ConstantValue<std::string> globalRow;
    const ConstantValue<std::string> globalCol;
    const ConstantValue<std::string> blockRow;
    const ConstantValue<std::string> blockCol;
    const ConstantValue<std::string> row;
    const ConstantValue<std::string> col;

    SaxpyWorkGroup();

    void setWorkGroup(const size_t height, const size_t width = 0);

    size_t groupHeight() const;
    size_t groupWidth() const;
};

////////////////////////////////////////
// SaxpyInnerBlocking

class SaxpyInnerBlocking
{
    size_t _blockHeight;
    size_t _blockWidth;
    size_t _vectorLength;

public:
    SaxpyInnerBlocking();

    // blocking is height x width
    void setInnerBlocking(const size_t height, const size_t width = 0);
    void setVectorLength(const size_t length);

    size_t blockHeight() const;
    size_t blockWidth() const;
    size_t vectorLength() const;

    size_t wholeHeight() const;
    size_t fractHeight() const;
    size_t wholeWidth() const;

    ConstantValue<std::string> multHeight(const Value& valSize) const;
    size_t multHeight(const size_t valSize) const;
};

////////////////////////////////////////
// SaxpyExtraParameter

class SaxpyExtraParameterObserver;

class SaxpyExtraParameter
{
    size_t _extraParam;
    size_t _totalVariations; // 0 <= _extraParam < _totalVariations

    std::vector<size_t> _observerParams;

    std::vector<SaxpyExtraParameterObserver*> _observers;
    std::vector<size_t> _numberVariations;

public:
    SaxpyExtraParameter();

    SaxpyExtraParameter& getExtraParameter();

    void addObserver(SaxpyExtraParameterObserver* observer);
    void setExtraParameter(const size_t value);
    std::vector<size_t> extraParamDetail() const;
    size_t extraParam() const;
    size_t totalVariations() const;
};

class SaxpyExtraParameterObserver
{
    const size_t _numberVariations;
    size_t _paramValue;

protected:
    size_t getParam() const;

public:
    SaxpyExtraParameterObserver(const size_t numberVariations,
                                SaxpyExtraParameter& subject);

    size_t numberVariations() const;
    void setParam(const size_t value);
};

////////////////////////////////////////
// SaxpyParamInlineMN

struct SaxpyParamInlineMN : public SaxpyExtraParameterObserver
{
    SaxpyParamInlineMN(SaxpyExtraParameter& subject);

    // matrix dimensions are inlined constants or passed as kernel arguments
    bool inlineMN() const;
};

////////////////////////////////////////
// SaxpyParamGlobalID

struct SaxpyParamGlobalID : public SaxpyExtraParameterObserver
{
    SaxpyParamGlobalID(SaxpyExtraParameter& subject);

    // use global or group/local ID
    bool globalID() const;
};

////////////////////////////////////////
// SaxpyAttrAutoVec

class SaxpyAttrAutoVec
{
    // kernel vector attribute hint is not supported on all platforms
    bool _useAttrAutoVec; // default value is true

public:
    SaxpyAttrAutoVec();

    void setUseAttrAutoVec(const bool value);
    bool getUseAttrAutoVec() const;
};

////////////////////////////////////////
// SaxpyPackedCalc

class SaxpyPackedCalc
{
    // number of computational kernels to pack into one GPU kernel
    size_t _packedCalc;

    // did the number of kernels change?
    bool   _packedChanged;

public:
    SaxpyPackedCalc();

    void setPackedCalc(const size_t numCalc);

    bool packedChanged() const;

    size_t packedCalc() const;
};

////////////////////////////////////////
// KernelBaseSaxpy

class KernelBaseSaxpy : public KernelInterface,
                        protected SaxpyDimensions,
                        protected SaxpyWorkGroup,
                        protected SaxpyInnerBlocking,
                        protected SaxpyExtraParameter,
                        protected SaxpyAttrAutoVec,
                        protected SaxpyPackedCalc
{
public:
    // some OpenCL platforms do not support auto vectorize attribute
    using SaxpyAttrAutoVec::setUseAttrAutoVec;

    // packed kernel support
    using SaxpyPackedCalc::setPackedCalc;

    // parameters for kernel code generation
    using SaxpyDimensions::setSaxpyDimensions;
    using SaxpyWorkGroup::setWorkGroup;
    using SaxpyInnerBlocking::setInnerBlocking;
    using SaxpyInnerBlocking::setVectorLength;
    using SaxpyExtraParameter::setExtraParameter;

    // accessors
    using SaxpyDimensions::dimM;
    using SaxpyDimensions::dimN;
    using SaxpyWorkGroup::groupHeight;
    using SaxpyWorkGroup::groupWidth;
    using SaxpyInnerBlocking::blockHeight;
    using SaxpyInnerBlocking::blockWidth;
    using SaxpyInnerBlocking::vectorLength;
    using SaxpyExtraParameter::extraParam;

    // number of valid kernel extra parameter values
    using SaxpyExtraParameter::totalVariations;

    // kernel extra parameter by each dimension
    //using SaxpyExtraParameter::extraParamDetail;
    std::vector<size_t> extraParamDetail() const {
        return SaxpyExtraParameter::extraParamDetail();
    }

protected:
    KernelBaseSaxpy();
    virtual ~KernelBaseSaxpy();

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
