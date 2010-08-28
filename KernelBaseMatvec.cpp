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

#include "KernelBaseMatvec.hpp"

using namespace std;

#include "declare_namespace"

////////////////////////////////////////
// MatvecMatrixDimensions

MatvecMatrixDimensions::MatvecMatrixDimensions()
    : _dimM(0), _dimN(0), _mnChanged(false)
{ }

void MatvecMatrixDimensions::setMatrixDimensions(const size_t M, const size_t N) {
    _mnChanged = (M != _dimM) || (N != _dimN);
    _dimM = M;
    _dimN = N;
}

bool MatvecMatrixDimensions::dimChanged() const { return _mnChanged; }

size_t MatvecMatrixDimensions::dimM() const { return _dimM; }
size_t MatvecMatrixDimensions::dimN() const { return _dimN; }

////////////////////////////////////////
// MatvecDataLayout

MatvecDataLayout::MatvecDataLayout()
    : _transposeA(false), _aChanged(false)
{ }

void MatvecDataLayout::setDataLayout(const bool A) {
    _aChanged = (A != _transposeA);
    _transposeA = A;
}

bool MatvecDataLayout::layoutChanged() const { return _aChanged; }

bool MatvecDataLayout::transposeA() const { return _transposeA; }

////////////////////////////////////////
// MatvecWorkGroup

MatvecWorkGroup::MatvecWorkGroup()
    : _dimSize(0),
      globalRow(func_string<size_t>("get_global_id", 0)),
      blockRow(func_string<size_t>("get_group_id", 0)),
      row(func_string<size_t>("get_local_id", 0))
{ }

void MatvecWorkGroup::setWorkGroup(const size_t sz) {
    _dimSize = sz;
}

size_t MatvecWorkGroup::groupSize() const { return _dimSize; }

size_t MatvecWorkGroup::localSize() const { return groupSize() + LOCALMEM_PAD; }

////////////////////////////////////////
// MatvecInnerBlocking

MatvecInnerBlocking::MatvecInnerBlocking()
    : _blockHeight(0), _vectorLength(0)
{ }

void MatvecInnerBlocking::setInnerBlocking(const size_t height, const size_t vectorLength) {
    _blockHeight = height;
    _vectorLength = vectorLength;
}

size_t MatvecInnerBlocking::blockHeight() const { return _blockHeight; }
size_t MatvecInnerBlocking::vectorLength() const { return _vectorLength; }

size_t MatvecInnerBlocking::wholeHeight() const { return blockHeight() / vectorLength(); }
size_t MatvecInnerBlocking::fractHeight() const { return blockHeight() % vectorLength(); }

ConstantValue<string> MatvecInnerBlocking::multHeight(const Value& valSize) const {
    switch (fractHeight()) {
        case (0) : return wholeHeight() * valSize;
        case (1) : return (wholeHeight() * valSize + valSize / vectorLength());
        case (2) : return (wholeHeight() * valSize + valSize / 2);
        case (3) : return (wholeHeight() * valSize + 3 * valSize / vectorLength());
    }
    return ConstantValue<string>(""); // should never happen
}

size_t MatvecInnerBlocking::multHeight(const size_t valSize) const {
    switch (fractHeight()) {
        case (0) : return wholeHeight() * valSize;
        case (1) : return (wholeHeight() * valSize + valSize / vectorLength());
        case (2) : return (wholeHeight() * valSize + valSize / 2);
        case (3) : return (wholeHeight() * valSize + 3 * valSize / vectorLength());
    }
    return 0; // should never happen
}

////////////////////////////////////////
// MatvecExtraParameter

MatvecExtraParameterObserver::MatvecExtraParameterObserver(const size_t numberVariations,
                                                           MatvecExtraParameter& subject)
    : _numberVariations(numberVariations), _paramValue(0)
{
    subject.addObserver(this);
}

size_t MatvecExtraParameterObserver::numberVariations() const { return _numberVariations; }
size_t MatvecExtraParameterObserver::getParam() const { return _paramValue; }
void MatvecExtraParameterObserver::setParam(const size_t value) { _paramValue = value; }

MatvecExtraParameter::MatvecExtraParameter()
    : _extraParam(0), _totalVariations(1)
{ }

MatvecExtraParameter& MatvecExtraParameter::getExtraParameter() { return *this; }

void MatvecExtraParameter::addObserver(MatvecExtraParameterObserver* observer) {
    _observers.push_back(observer);
    _observerParams.push_back(0);
    const size_t observerVariations = observer->numberVariations();
    _totalVariations *= observerVariations;
    _numberVariations.push_back(observerVariations);
}

void MatvecExtraParameter::setExtraParameter(const size_t value) {
    _extraParam = value;
    size_t shiftValue = 1;
    for (size_t i = 0; i < _observers.size(); i++) {
        if (i > 0) shiftValue *= _numberVariations[i - 1];
        const size_t observerValue = (value / shiftValue) % _numberVariations[i];
        _observers[i]->setParam(observerValue);
        _observerParams[i] = observerValue;
    }
}

vector<size_t> MatvecExtraParameter::extraParamDetail() const { return _observerParams; }

size_t MatvecExtraParameter::extraParam() const { return _extraParam; }

size_t MatvecExtraParameter::totalVariations() const { return _totalVariations; }

////////////////////////////////////////
// MatvecParamInlineMN

MatvecParamInlineMN::MatvecParamInlineMN(MatvecExtraParameter& subject)
    : MatvecExtraParameterObserver(2, subject)
{ }

bool MatvecParamInlineMN::inlineMN() const { return getParam(); }

////////////////////////////////////////
// MatvecParamGlobalID

MatvecParamGlobalID::MatvecParamGlobalID(MatvecExtraParameter& subject)
    : MatvecExtraParameterObserver(2, subject)
{ }

bool MatvecParamGlobalID::globalID() const { return getParam(); }

////////////////////////////////////////
// MatvecAttrAutoVec

MatvecAttrAutoVec::MatvecAttrAutoVec()
    : _useAttrAutoVec(true)
{ }

void MatvecAttrAutoVec::setUseAttrAutoVec(const bool value) { _useAttrAutoVec = value; }
bool MatvecAttrAutoVec::getUseAttrAutoVec() const { return _useAttrAutoVec; }

////////////////////////////////////////
// MatvecGeneralized

MatvecGeneralized::MatvecGeneralized()
    : _generalizedMatvec(false),
      _gemvChanged(false)
{ }

void MatvecGeneralized::setGeneralizedMatvec(const bool GEMV) {
    _gemvChanged = GEMV != _generalizedMatvec;
    _generalizedMatvec = GEMV;
}

bool MatvecGeneralized::gemvChanged() const { return _gemvChanged; }

bool MatvecGeneralized::generalizedMatvec() const { return _generalizedMatvec; }

////////////////////////////////////////
// MatvecPackedCalc

MatvecPackedCalc::MatvecPackedCalc()
    : _packedCalc(1), _packedChanged(false)
{ }

void MatvecPackedCalc::setPackedCalc(const size_t num) {
    _packedChanged = (num != _packedCalc);
    _packedCalc = num;
}

bool MatvecPackedCalc::packedChanged() const { return _packedChanged; }

size_t MatvecPackedCalc::packedCalc() const { return _packedCalc; }

////////////////////////////////////////
// KernelBaseMatvec

KernelBaseMatvec::KernelBaseMatvec()
    : MatvecMatrixDimensions(),
      MatvecDataLayout(),
      MatvecWorkGroup(),
      MatvecInnerBlocking(),
      MatvecExtraParameter(),
      MatvecAttrAutoVec(),
      MatvecGeneralized(),
      MatvecPackedCalc()
{ }

KernelBaseMatvec::~KernelBaseMatvec() { }

bool KernelBaseMatvec::validParams() const {

    return
        // packed kernels
        packedCalc() > 0 &&

        // all matrix dimensions must be a multiple of vector length
        0 == dimM() % vectorLength() &&
        0 == dimN() % vectorLength() &&

        // check for blocking compatible with matrix dimensions
        0 == dimM() % (groupSize() * blockHeight()) &&
        0 == dimN() % (groupSize() * vectorLength()) &&
        groupSize() * blockHeight() <= dimM() &&
        groupSize() * vectorLength() <= dimN() &&

        // inner blocking must be even number of squares (quads)
        0 == blockHeight() % vectorLength() &&

        // extra parameter
        extraParam() < totalVariations();
}

bool KernelBaseMatvec::getParams(vector<size_t>& params) const {
    bool rc;
    if (rc = validParams()) {
        params.clear();

        // packed kernels
        params.push_back(packedCalc());

        // GEMV or pure multiply only
        params.push_back(generalizedMatvec());

        // matrix dimensions
        params.push_back(dimM());
        params.push_back(dimN());

        // data layout
        params.push_back(transposeA());

        // work group
        params.push_back(groupSize());

        // inner blocking
        params.push_back(blockHeight());
        params.push_back(vectorLength());

        // extra parameter
        params.push_back(extraParam());
    }
    return rc;
}

void KernelBaseMatvec::setParams(const vector<size_t>& params) {
    size_t index = 0;

    // packed kernels
    const size_t numberKernels = params[index++];
    setPackedCalc(numberKernels);

    // GEMV or pure multiply only
    const size_t GEMV = params[index++];
    setGeneralizedMatvec(GEMV);

    // matrix dimensions
    const size_t M = params[index++];
    const size_t N = params[index++];
    setMatrixDimensions(M, N);

    // data layout
    const size_t transposeA = params[index++];
    setDataLayout(transposeA);

    // work group
    const size_t groupSize = params[index++];
    setWorkGroup(groupSize);

    // inner blocking
    const size_t blockHeight = params[index++];
    const size_t vectorLength = params[index++];
    setInnerBlocking(blockHeight, vectorLength);

    // extra parameter
    const size_t extraParam = params[index++];
    setExtraParameter(extraParam);
}

vector<size_t> KernelBaseMatvec::globalWorkItems() const {
    vector<size_t> dims;
    dims.push_back(dimM() / blockHeight());
    return dims;
}

vector<size_t> KernelBaseMatvec::localWorkItems() const {
    vector<size_t> dims;
    dims.push_back(groupSize());
    return dims;
}

size_t KernelBaseMatvec::numberFlops() const {
    if (generalizedMatvec())
        return packedCalc() * 2 * dimM() * (dimN() + 1);
    else
        return packedCalc() * dimM() * (2 * dimN() - 1);
}

}; // namespace
