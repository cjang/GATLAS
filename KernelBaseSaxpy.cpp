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

#include "KernelBaseSaxpy.hpp"

using namespace std;

#include "declare_namespace"

////////////////////////////////////////
// SaxpyDimensions

SaxpyDimensions::SaxpyDimensions()
    : _dimM(0), _dimN(0), _mnChanged(false)
{ }

void SaxpyDimensions::setSaxpyDimensions(const size_t M, const size_t N) {
    _mnChanged = (M != _dimM) || (N != _dimN);
    _dimM = M;
    _dimN = N;
}

bool SaxpyDimensions::dimChanged() const { return _mnChanged; }

size_t SaxpyDimensions::dimM() const { return _dimM; }
size_t SaxpyDimensions::dimN() const { return _dimN; }

////////////////////////////////////////
// SaxpyWorkGroup

SaxpyWorkGroup::SaxpyWorkGroup()
    : _dimHeight(0), _dimWidth(0),
      globalRow(func_string<size_t>("get_global_id", 0)),
      globalCol(func_string<size_t>("get_global_id", 1)),
      blockRow(func_string<size_t>("get_group_id", 0)),
      blockCol(func_string<size_t>("get_group_id", 1)),
      row(func_string<size_t>("get_local_id", 0)),
      col(func_string<size_t>("get_local_id", 1))
{ }

void SaxpyWorkGroup::setWorkGroup(const size_t height, const size_t width) {
    _dimHeight = height;
    _dimWidth = width;
}

size_t SaxpyWorkGroup::groupHeight() const { return _dimHeight; }
size_t SaxpyWorkGroup::groupWidth() const { return _dimWidth; }

////////////////////////////////////////
// SaxpyInnerBlocking

SaxpyInnerBlocking::SaxpyInnerBlocking()
    : _blockHeight(0), _blockWidth(0), _vectorLength(0)
{ }

void SaxpyInnerBlocking::setInnerBlocking(const size_t height, const size_t width) {
    _blockHeight = height;
    _blockWidth = width;
}

void SaxpyInnerBlocking::setVectorLength(const size_t length) {
    _vectorLength = length;
}

size_t SaxpyInnerBlocking::blockHeight() const { return _blockHeight; }
size_t SaxpyInnerBlocking::blockWidth() const { return _blockWidth; }
size_t SaxpyInnerBlocking::vectorLength() const { return _vectorLength; }

size_t SaxpyInnerBlocking::wholeHeight() const { return blockHeight() / vectorLength(); }
size_t SaxpyInnerBlocking::fractHeight() const { return blockHeight() % vectorLength(); }
size_t SaxpyInnerBlocking::wholeWidth() const { return blockWidth() / vectorLength(); }

ConstantValue<string> SaxpyInnerBlocking::multHeight(const Value& valSize) const {
    switch (fractHeight()) {
        case (0) : return wholeHeight() * valSize;
        case (1) : return (wholeHeight() * valSize + valSize / vectorLength());
        case (2) : return (wholeHeight() * valSize + valSize / 2);
        case (3) : return (wholeHeight() * valSize + 3 * valSize / vectorLength());
    }
    return ConstantValue<string>(""); // should never happen
}

size_t SaxpyInnerBlocking::multHeight(const size_t valSize) const {
    switch (fractHeight()) {
        case (0) : return wholeHeight() * valSize;
        case (1) : return (wholeHeight() * valSize + valSize / vectorLength());
        case (2) : return (wholeHeight() * valSize + valSize / 2);
        case (3) : return (wholeHeight() * valSize + 3 * valSize / vectorLength());
    }
    return 0; // should never happen
}

////////////////////////////////////////
// SaxpyExtraParameter

SaxpyExtraParameterObserver::SaxpyExtraParameterObserver(const size_t numberVariations,
                                                         SaxpyExtraParameter& subject)
    : _numberVariations(numberVariations), _paramValue(0)
{
    subject.addObserver(this);
}

size_t SaxpyExtraParameterObserver::numberVariations() const { return _numberVariations; }
size_t SaxpyExtraParameterObserver::getParam() const { return _paramValue; }
void SaxpyExtraParameterObserver::setParam(const size_t value) { _paramValue = value; }

SaxpyExtraParameter::SaxpyExtraParameter()
    : _extraParam(0), _totalVariations(1)
{ }

SaxpyExtraParameter& SaxpyExtraParameter::getExtraParameter() { return *this; }

void SaxpyExtraParameter::addObserver(SaxpyExtraParameterObserver* observer) {
    _observers.push_back(observer);
    _observerParams.push_back(0);
    const size_t observerVariations = observer->numberVariations();
    _totalVariations *= observerVariations;
    _numberVariations.push_back(observerVariations);
}

void SaxpyExtraParameter::setExtraParameter(const size_t value) {
    _extraParam = value;
    size_t shiftValue = 1;
    for (size_t i = 0; i < _observers.size(); i++) {
        if (i > 0) shiftValue *= _numberVariations[i - 1];
        const size_t observerValue = (value / shiftValue) % _numberVariations[i];
        _observers[i]->setParam(observerValue);
        _observerParams[i] = observerValue;
    }
}

vector<size_t> SaxpyExtraParameter::extraParamDetail() const { return _observerParams; }

size_t SaxpyExtraParameter::extraParam() const { return _extraParam; }

size_t SaxpyExtraParameter::totalVariations() const { return _totalVariations; }

////////////////////////////////////////
// SaxpyParamInlineMN

SaxpyParamInlineMN::SaxpyParamInlineMN(SaxpyExtraParameter& subject)
    : SaxpyExtraParameterObserver(2, subject)
{ }

bool SaxpyParamInlineMN::inlineMN() const { return getParam(); }

////////////////////////////////////////
// SaxpyParamGlobalID

SaxpyParamGlobalID::SaxpyParamGlobalID(SaxpyExtraParameter& subject)
    : SaxpyExtraParameterObserver(2, subject)
{ }

bool SaxpyParamGlobalID::globalID() const { return getParam(); }

////////////////////////////////////////
// SaxpyAttrAutoVec

SaxpyAttrAutoVec::SaxpyAttrAutoVec()
    : _useAttrAutoVec(true)
{ }

void SaxpyAttrAutoVec::setUseAttrAutoVec(const bool value) { _useAttrAutoVec = value; }
bool SaxpyAttrAutoVec::getUseAttrAutoVec() const { return _useAttrAutoVec; }

////////////////////////////////////////
// SaxpyPackedCalc

SaxpyPackedCalc::SaxpyPackedCalc()
    : _packedCalc(1), _packedChanged(false)
{ }

void SaxpyPackedCalc::setPackedCalc(const size_t num) {
    _packedChanged = (num != _packedCalc);
    _packedCalc = num;
}

bool SaxpyPackedCalc::packedChanged() const { return _packedChanged; }

size_t SaxpyPackedCalc::packedCalc() const { return _packedCalc; }

////////////////////////////////////////
// KernelBaseSaxpy

KernelBaseSaxpy::KernelBaseSaxpy()
    : SaxpyDimensions(),
      SaxpyWorkGroup(),
      SaxpyInnerBlocking(),
      SaxpyExtraParameter(),
      SaxpyAttrAutoVec(),
      SaxpyPackedCalc()
{ }

KernelBaseSaxpy::~KernelBaseSaxpy() { }

bool KernelBaseSaxpy::validParams() const {

    return
        // packed kernels
        packedCalc() > 0 &&

        // all matrix dimensions must be a multiple of vector length
        0 == dimM() % vectorLength() &&
        0 == dimN() % vectorLength() &&

        // check for blocking compatible with matrix dimensions
        0 == dimM() % (groupHeight() * blockHeight()) &&
        (groupWidth() && blockWidth() ? 0 == dimN() % (groupWidth() * blockWidth()) : true) &&
        groupHeight() * blockHeight() <= dimM() &&
        groupWidth() * blockWidth() <= dimN() &&

        // inner blocking width must always be aligned with vector elements
        0 == blockWidth() % vectorLength() &&

        // extra parameter
        extraParam() < totalVariations();
}

bool KernelBaseSaxpy::getParams(vector<size_t>& params) const {
    bool rc;
    if (rc = validParams()) {
        params.clear();

        // packed kernels
        params.push_back(packedCalc());

        // matrix dimensions
        params.push_back(dimM());
        params.push_back(dimN());

        // work group
        params.push_back(groupHeight());
        params.push_back(groupWidth());

        // inner blocking
        params.push_back(blockHeight());
        params.push_back(blockWidth());
        params.push_back(vectorLength());

        // extra parameter
        params.push_back(extraParam());
    }
    return rc;
}

void KernelBaseSaxpy::setParams(const vector<size_t>& params) {
    size_t index = 0;

    // packed kernels
    const size_t numberKernels = params[index++];
    setPackedCalc(numberKernels);

    // matrix dimensions
    const size_t M = params[index++];
    const size_t N = params[index++];
    setSaxpyDimensions(M, N);

    // work group
    const size_t groupHeight = params[index++];
    const size_t groupWidth = params[index++];
    setWorkGroup(groupHeight, groupWidth);

    // inner blocking
    const size_t blockHeight = params[index++];
    const size_t blockWidth = params[index++];
    const size_t vectorLength = params[index++];
    setInnerBlocking(blockHeight, blockWidth);
    setVectorLength(vectorLength);

    // extra parameter
    const size_t extraParam = params[index++];
    setExtraParameter(extraParam);
}

vector<size_t> KernelBaseSaxpy::globalWorkItems() const {
    vector<size_t> dims;
    dims.push_back(dimM() / blockHeight());
    if (blockWidth()) dims.push_back(dimN() / blockWidth());
    return dims;
}

vector<size_t> KernelBaseSaxpy::localWorkItems() const {
    vector<size_t> dims;
    dims.push_back(groupHeight());
    if (groupWidth()) dims.push_back(groupWidth());
    return dims;
}

size_t KernelBaseSaxpy::numberFlops() const {
    return packedCalc() * 2 * dimM() * dimN();
}

}; // namespace
