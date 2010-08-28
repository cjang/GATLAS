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

using namespace std;

#include "declare_namespace"

////////////////////////////////////////
// MatmulMatrixDimensions

MatmulMatrixDimensions::MatmulMatrixDimensions()
    : _dimM(0), _dimN(0), _dimK(0), _mnkChanged(false)
{ }

void MatmulMatrixDimensions::setMatrixDimensions(const size_t M, const size_t N, const size_t K) {
    _mnkChanged = (M != _dimM) || (N != _dimN) || (K != _dimK);
    _dimM = M;
    _dimN = N;
    _dimK = K;
}

bool MatmulMatrixDimensions::dimChanged() const { return _mnkChanged; }

size_t MatmulMatrixDimensions::dimM() const { return _dimM; }
size_t MatmulMatrixDimensions::dimN() const { return _dimN; }
size_t MatmulMatrixDimensions::dimK() const { return _dimK; }

////////////////////////////////////////
// MatmulDataLayout

MatmulDataLayout::MatmulDataLayout()
    : _transposeA(false), _transposeB(false), _abcChanged(false)
{ }

void MatmulDataLayout::setDataLayout(const bool A, const bool B) {
    _abcChanged = (A != _transposeA) || (B != _transposeB);
    _transposeA = A;
    _transposeB = B;
}

bool MatmulDataLayout::layoutChanged() const { return _abcChanged; }

bool MatmulDataLayout::transposeA() const { return _transposeA; }
bool MatmulDataLayout::transposeB() const { return _transposeB; }

////////////////////////////////////////
// MatmulWorkGroup

MatmulWorkGroup::MatmulWorkGroup()
    : _dimHeight(0), _dimWidth(0),
      globalRow(func_string<size_t>("get_global_id", 1)),
      globalCol(func_string<size_t>("get_global_id", 0)),
      blockRow(func_string<size_t>("get_group_id", 1)),
      blockCol(func_string<size_t>("get_group_id", 0)),
      row(func_string<size_t>("get_local_id", 1)),
      col(func_string<size_t>("get_local_id", 0))
{ }

void MatmulWorkGroup::setWorkGroup(const size_t height, const size_t width) {
    _dimHeight = height;
    _dimWidth = width;
}

void MatmulWorkGroup::setWorkGroup(const size_t sz) {
    setWorkGroup(sz, sz);
}

size_t MatmulWorkGroup::groupHeight() const { return _dimHeight; }
size_t MatmulWorkGroup::groupWidth() const { return _dimWidth; }
size_t MatmulWorkGroup::groupSize() const {
    return (_dimWidth == _dimHeight) ? _dimWidth : 0;
}

size_t MatmulWorkGroup::localHeight() const { return groupHeight() + LOCALMEM_PAD; }
size_t MatmulWorkGroup::localWidth() const { return groupWidth() + LOCALMEM_PAD; }
size_t MatmulWorkGroup::localSize() const { return groupSize() + LOCALMEM_PAD; }

////////////////////////////////////////
// MatmulInnerBlocking

MatmulInnerBlocking::MatmulInnerBlocking()
    : _blockHeight(0), _blockWidth(0)
{ }

void MatmulInnerBlocking::setInnerBlocking(const size_t height, const size_t width) {
    _blockHeight = height;
    _blockWidth = width;
}

size_t MatmulInnerBlocking::blockHeight() const { return _blockHeight; }
size_t MatmulInnerBlocking::blockWidth() const { return _blockWidth; }

size_t MatmulInnerBlocking::wholeHeight() const { return blockHeight() / blockWidth(); }
size_t MatmulInnerBlocking::fractHeight() const { return blockHeight() % blockWidth(); }

ConstantValue<string> MatmulInnerBlocking::multHeight(const Value& valSize) const {
    switch (fractHeight()) {
        case (0) : return wholeHeight() * valSize;
        case (1) : return (wholeHeight() * valSize + valSize / blockWidth());
        case (2) : return (wholeHeight() * valSize + valSize / 2);
        case (3) : return (wholeHeight() * valSize + 3 * valSize / blockWidth());
    }
    return ConstantValue<string>(""); // should never happen
}

size_t MatmulInnerBlocking::multHeight(const size_t valSize) const {
    switch (fractHeight()) {
        case (0) : return wholeHeight() * valSize;
        case (1) : return (wholeHeight() * valSize + valSize / blockWidth());
        case (2) : return (wholeHeight() * valSize + valSize / 2);
        case (3) : return (wholeHeight() * valSize + 3 * valSize / blockWidth());
    }
    return 0; // should never happen
}

////////////////////////////////////////
// MatmulExtraParameter

MatmulExtraParameterObserver::MatmulExtraParameterObserver(const size_t numberVariations,
                                                           MatmulExtraParameter& subject)
    : _numberVariations(numberVariations), _paramValue(0)
{
    subject.addObserver(this);
}

size_t MatmulExtraParameterObserver::numberVariations() const { return _numberVariations; }
size_t MatmulExtraParameterObserver::getParam() const { return _paramValue; }
void MatmulExtraParameterObserver::setParam(const size_t value) { _paramValue = value; }

MatmulExtraParameter::MatmulExtraParameter()
    : _extraParam(0), _totalVariations(1)
{ }

MatmulExtraParameter& MatmulExtraParameter::getExtraParameter() { return *this; }

void MatmulExtraParameter::addObserver(MatmulExtraParameterObserver* observer) {
    _observers.push_back(observer);
    _observerParams.push_back(0);
    const size_t observerVariations = observer->numberVariations();
    _totalVariations *= observerVariations;
    _numberVariations.push_back(observerVariations);
}

void MatmulExtraParameter::setExtraParameter(const size_t value) {
    _extraParam = value;
    size_t shiftValue = 1;
    for (size_t i = 0; i < _observers.size(); i++) {
        if (i > 0) shiftValue *= _numberVariations[i - 1];
        const size_t observerValue = (value / shiftValue) % _numberVariations[i];
        _observers[i]->setParam(observerValue);
        _observerParams[i] = observerValue;
    }
}

vector<size_t> MatmulExtraParameter::extraParamDetail() const { return _observerParams; }

size_t MatmulExtraParameter::extraParam() const { return _extraParam; }

size_t MatmulExtraParameter::totalVariations() const { return _totalVariations; }

////////////////////////////////////////
// MatmulParamInlineMNK

MatmulParamInlineMNK::MatmulParamInlineMNK(MatmulExtraParameter& subject)
    : MatmulExtraParameterObserver(2, subject)
{ }

bool MatmulParamInlineMNK::inlineMNK() const { return getParam(); }

////////////////////////////////////////
// MatmulParamLoopOrder

MatmulParamLoopOrder::MatmulParamLoopOrder(MatmulExtraParameter& subject)
    : MatmulExtraParameterObserver(6, subject)
{ }

size_t MatmulParamLoopOrder::loopOrder() const { return getParam(); }

////////////////////////////////////////
// MatmulParamGlobalID

MatmulParamGlobalID::MatmulParamGlobalID(MatmulExtraParameter& subject)
    : MatmulExtraParameterObserver(2, subject)
{ }

bool MatmulParamGlobalID::globalID() const { return getParam(); }

////////////////////////////////////////
// MatmulAttrAutoVec

MatmulAttrAutoVec::MatmulAttrAutoVec()
    : _useAttrAutoVec(true)
{ }

void MatmulAttrAutoVec::setUseAttrAutoVec(const bool value) { _useAttrAutoVec = value; }
bool MatmulAttrAutoVec::getUseAttrAutoVec() const { return _useAttrAutoVec; }

////////////////////////////////////////
// MatmulGeneralized

MatmulGeneralized::MatmulGeneralized()
    : _generalizedMatmul(false),
      _gemmChanged(false)
{ }

void MatmulGeneralized::setGeneralizedMatmul(const bool GEMM) {
    _gemmChanged = GEMM != _generalizedMatmul;
    _generalizedMatmul = GEMM;
}

bool MatmulGeneralized::gemmChanged() const { return _gemmChanged; }

bool MatmulGeneralized::generalizedMatmul() const { return _generalizedMatmul; }

////////////////////////////////////////
// MatvecPackedCalc

MatmulPackedCalc::MatmulPackedCalc()
    : _packedCalc(1), _packedChanged(false)
{ }

void MatmulPackedCalc::setPackedCalc(const size_t num) {
    _packedChanged = (num != _packedCalc);
    _packedCalc = num;
}

bool MatmulPackedCalc::packedChanged() const { return _packedChanged; }

size_t MatmulPackedCalc::packedCalc() const { return _packedCalc; }

////////////////////////////////////////
// KernelBaseMatmul

KernelBaseMatmul::KernelBaseMatmul()
    : MatmulMatrixDimensions(),
      MatmulDataLayout(),
      MatmulWorkGroup(),
      MatmulInnerBlocking(),
      MatmulExtraParameter(),
      MatmulAttrAutoVec(),
      MatmulGeneralized(),
      MatmulPackedCalc()
{ }

KernelBaseMatmul::~KernelBaseMatmul() { }

bool KernelBaseMatmul::validParams() const {

    const size_t VECTOR_LENGTH = blockWidth();

    return
        // packed kernels
        packedCalc() > 0 &&

        // all matrix dimensions must be a multiple of VECTOR_LENGTH
        0 == dimM() % VECTOR_LENGTH &&
        0 == dimN() % VECTOR_LENGTH &&
        0 == dimK() % VECTOR_LENGTH &&

        // check for blocking compatible with matrix dimensions
        0 == dimM() % (groupHeight() * blockHeight()) &&
        0 == dimN() % (groupWidth() * blockWidth()) &&
        0 == dimK() % (groupHeight() * VECTOR_LENGTH) &&
        groupHeight() * blockHeight() <= dimM() &&
        groupWidth() * blockWidth() <= dimN() &&
        groupHeight() * VECTOR_LENGTH <= dimK() &&

        // if A is transposed, then inner blocking must be even number of squares (quads)
        ( transposeA()
              ? (0 == blockHeight() % VECTOR_LENGTH)
              : true ) &&

        // extra parameter
        extraParam() < totalVariations();
}

bool KernelBaseMatmul::getParams(vector<size_t>& params) const {
    bool rc;
    if (rc = validParams()) {
        params.clear();

        // packed kernels
        params.push_back(packedCalc());

        // GEMM or pure matrix multiply only
        params.push_back(generalizedMatmul());

        // matrix dimensions
        params.push_back(dimM());
        params.push_back(dimN());
        params.push_back(dimK());

        // data layout
        params.push_back(transposeA());
        params.push_back(transposeB());

        // work group
        params.push_back(groupHeight());
        params.push_back(groupWidth());

        // inner blocking
        params.push_back(blockHeight());
        params.push_back(blockWidth());

        // extra parameter
        params.push_back(extraParam());
    }
    return rc;
}

void KernelBaseMatmul::setParams(const vector<size_t>& params) {
    size_t index = 0;

    // packed kernels
    const size_t numberKernels = params[index++];
    setPackedCalc(numberKernels);

    // GEMM or pure matrix multiply only
    const size_t GEMM = params[index++];
    setGeneralizedMatmul(GEMM);

    // matrix dimensions
    const size_t M = params[index++];
    const size_t N = params[index++];
    const size_t K = params[index++];
    setMatrixDimensions(M, N, K);

    // data layout
    const size_t transposeA = params[index++];
    const size_t transposeB = params[index++];
    setDataLayout(transposeA, transposeB);

    // work group
    const size_t groupHeight = params[index++];
    const size_t groupWidth = params[index++];
    setWorkGroup(groupHeight, groupWidth);

    // inner blocking
    const size_t blockHeight = params[index++];
    const size_t blockWidth = params[index++];
    setInnerBlocking(blockHeight, blockWidth);

    // extra parameter
    const size_t extraParam = params[index++];
    setExtraParameter(extraParam);
}

vector<size_t> KernelBaseMatmul::globalWorkItems() const {
    vector<size_t> dims;
    dims.push_back(dimN() / blockWidth());
    dims.push_back(dimM() / blockHeight());
    return dims;
}

vector<size_t> KernelBaseMatmul::localWorkItems() const {
    vector<size_t> dims;
    dims.push_back(groupWidth());
    dims.push_back(groupHeight());
    return dims;
}

size_t KernelBaseMatmul::numberFlops() const {
    if (generalizedMatmul())
        return packedCalc() * 2 * dimM() * dimN() * (dimK() + 1);
    else
        return packedCalc() * dimM() * dimN() * (2 * dimK() - 1);
}

}; // namespace
