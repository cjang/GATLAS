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

#include "KernelGenMatmulImage.hpp"

using namespace std;

#include "declare_namespace"

// matrix dimensions are inlined constants or passed as kernel arguments
bool KernelGenMatmulImage::_inlineMNK(const size_t extraParam) const { return extraParam / 6; }

// inner product accumulation loop order, 3! permutations of (j,k,l)
size_t KernelGenMatmulImage::_loopOrder(const size_t extraParam) const { return extraParam % 6; }

size_t KernelGenMatmulImage::numberExtraParam() const {
    return 2   // matrix dimensions inlined or passed as kernel arguments
         * 6;  // inner product loop order
}

string KernelGenMatmulImage::namePrefix() const { return "KernelGenMatmulImage"; }

KernelGenMatmulImage::KernelGenMatmulImage(const bool transposeA,
                                           const bool transposeB)
    : KernelGenMatmul(transposeA, transposeB),
      _handleA(-1),
      _handleB(-1),
      _handleC(-1),
      _paranoidCheck(false),
      _paranoidC(NULL)
{ }

KernelGenMatmulImage::~KernelGenMatmulImage() {
    delete[] _paranoidC;
}

void KernelGenMatmulImage::paranoidCheck() {
    _paranoidCheck = true;
    delete[] _paranoidC;
    _paranoidC = new scalar[dimM() * dimN()];
}

string KernelGenMatmulImage::desc() const {
    stringstream ss;
    ss << (inlineMNK() ? "inlineMNK" : "argsMNK")
       << " ";
    switch (loopOrder()) {
        case (0) : ss << "jkl"; break;
        case (1) : ss << "kjl"; break;
        case (2) : ss << "ljk"; break;
        case (3) : ss << "jlk"; break;
        case (4) : ss << "klj"; break;
        case (5) : ss << "lkj"; break;
    }
    return ss.str();
}

size_t KernelGenMatmulImage::maxGroupSize(const size_t M, const size_t N, const size_t K) const {
    return 16;
}

size_t KernelGenMatmulImage::maxBlockHeight() const {
    // transposing A fixes inner blocking to square quads
    return transposeA()
               ? VECTOR_LENGTH
               : 3 * VECTOR_LENGTH;
}

bool KernelGenMatmulImage::syncOutput(OCLApp& oclApp) {
    return syncBufferFromDevice(oclApp, _handleC);
}

bool KernelGenMatmulImage::checkOutput(OCLApp& oclApp, const bool printOutput) {
    if (_paranoidCheck) {
        return checkBuffer<scalar>(oclApp, _handleC, dimN(), dimM(), _paranoidC, printOutput);
    } else {
        const scalar testValue = dimK();
        return checkBuffer<scalar>(oclApp, _handleC, dimN(), dimM(), testValue, printOutput);
    }
}

bool KernelGenMatmulImage::setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) {

    if (-1 == _handleA || -1 == _handleB || -1 == _handleC || mnkChanged()) {
        oclApp.releaseImages();
        oclApp.releaseBuffers();

        if (transposeA())
            _handleA = createImageR(oclApp, dimM(), dimK(), "matA", 1);
        else
            _handleA = createImageR(oclApp, dimK(), dimM(), "matA", 1);

        if (transposeB())
            _handleB = createImageR(oclApp, dimK(), dimN(), "matB", 1);
        else
            _handleB = createImageR(oclApp, dimN(), dimK(), "matB", 1);

        _handleC = createBufferRW<scalar, VECTOR_LENGTH>(oclApp, dimN() * dimM(), "matC", 0);

        if (-1 == _handleA || -1 == _handleB || -1 == _handleC) return false; // failure

    } else {
        // matrices A and B
        if (syncInput) {
            if (!syncImageToDevice(oclApp, _handleA)) return false;
            if (!syncImageToDevice(oclApp, _handleB)) return false;
        }
        // matrix C
        if (!clearBuffer<scalar>(oclApp, _handleC)) return false;
    }

    const scalar alpha = _paranoidCheck ? posrand<scalar>() : 1;
    const scalar beta = _paranoidCheck ? posrand<scalar>() : 1;

    // paranoid check
    if (_paranoidCheck) {

        // fill A, B and C matrices with random values
        if (fillrandImage(oclApp, _handleA, dimM() * dimK()) &&
            fillrandImage(oclApp, _handleB, dimK() * dimN()) &&
            fillrandBuffer<scalar>(oclApp, _handleC, dimM() * dimN())) {

            const scalar *ptrA = oclApp.imagePtr(_handleA);
            const scalar *ptrB = oclApp.imagePtr(_handleB);
            const scalar *ptrC = oclApp.bufferPtr<scalar>(_handleC);

            fillconst<scalar>(_paranoidC, 0, dimM() * dimN());

            // calculate C
            for (size_t i = 0; i < dimM(); i++)
            for (size_t j = 0; j < dimN(); j++)
            for (size_t k = 0; k < dimK(); k++)
                _paranoidC[i * dimN() + j] += (transposeA()
                                                   ? ptrA[k * dimM() + i]
                                                   : ptrA[i * dimK() + k])
                                            * (transposeB()
                                                   ? ptrB[j * dimK() + k]
                                                   : ptrB[k * dimN() + j]);

            // multiply AB product by alpha and add beta times C
            for (size_t i = 0; i < dimM() * dimN(); i++)
                _paranoidC[i] = alpha * _paranoidC[i] + beta * ptrC[i];
        } else {
            cerr << "error: failed to fill matrices with random values" << endl;
        }
    }

    size_t argIndex = 0;
    bool rc =
        setArgGlobal(oclApp, kernelHandle, argIndex++, _handleC, "matC") &&
        setArgImage(oclApp, kernelHandle, argIndex++, _handleA, "matA") &&
        setArgImage(oclApp, kernelHandle, argIndex++, _handleB, "matB");
    if (! inlineMNK()) {
        rc = rc &&
        setArgValue<int>(oclApp, kernelHandle, argIndex++, dimM(), "M") &&
        setArgValue<int>(oclApp, kernelHandle, argIndex++, dimN(), "N") &&
        setArgValue<int>(oclApp, kernelHandle, argIndex++, dimK(), "K");
    }
    rc = rc &&
        setArgValue<scalar>(oclApp, kernelHandle, argIndex++, alpha, "alpha") &&
        setArgValue<scalar>(oclApp, kernelHandle, argIndex++, beta, "beta");
    return rc;
}

// prints the kernel source
ostream& KernelGenMatmulImage::print(ostream& os) const {

    // kernel function attributes
    AutoVectorize< scalarN > attrAutoVec;
    FunctionDeclaration kernelDecl(kernelName());
    kernelDecl.returnType<void>();
    kernelDecl.qualify(KERNEL);
    if (getUseAttrAutoVec()) kernelDecl.qualify(attrAutoVec);

    // kernel arguments
    Var< scalarN* >  matC("matC", GLOBAL, kernelDecl);
    Var< image2d_t > matA("matA", READONLY, kernelDecl);
    Var< image2d_t > matB("matB", READONLY, kernelDecl);
    Var< const int > M("M", kernelDecl, inlineMNK(), dimM());
    Var< const int > N("N", kernelDecl, inlineMNK(), dimN());
    Var< const int > K("K", kernelDecl, inlineMNK(), dimK());
    Var< const scalar > alpha("alpha", kernelDecl);
    Var< const scalar > beta("beta", kernelDecl);

    // begin function body
    os << kernelDecl;

    // image sampler
    Var< const sampler_t > sampler("sampler");
    os << declare(sampler, ImageSampler());

    // accumulate inner product sum
    Vector< scalarN > accum("accum", blockHeight());
    os << declare(accum, CastValue<scalarN>(ConstantValue<scalar>(0)));

    // current values from matrix A (for inner product)
    Vector< scalarN > valA("valA", blockHeight());
    os << declare(valA);

    // current values from matrix B (for inner product)
    Vector< scalarN > valB("valB", VECTOR_LENGTH);
    os << declare(valB);

    // inner product loop
    Var< int > idx("idx");
    os << ForLoop(idx, K / VECTOR_LENGTH, 1);

        // read in values of matrix A
        for (size_t j = 0; j < blockHeight(); j++)
            if (transposeA())
                os << assign(valA[j],
                             ReadImage(matA, sampler,
                                       globalRow,
                                       VECTOR_LENGTH * idx + j));
            else
                os << assign(valA[j],
                             ReadImage(matA, sampler,
                                       idx,
                                       blockHeight() * globalRow + j));

        // read in values of matrix B
        for (size_t j = 0; j < VECTOR_LENGTH; j++)
            if (transposeB())
                os << assign(valB[j],
                             ReadImage(matB, sampler,
                                       idx,
                                       VECTOR_LENGTH * globalCol + j));
            else
                os << assign(valB[j],
                             ReadImage(matB, sampler,
                                       globalCol,
                                       VECTOR_LENGTH * idx + j));

        // inner product accumulation
        assignMAD(os, loopOrder(), accum, valA, valB);

    os << EndBlock();

    const ConstantValue<string> outC = matC + multQuads(N) * globalRow + globalCol;

    os << assign(*outC,
                 MADValue(CastValue<scalarN>(alpha),
                          accum[0],
                          CastValue<scalarN>(beta) * *outC));

    for (size_t i = 1; i < blockHeight(); i++)
        os << assign(*(outC + i * (N / VECTOR_LENGTH)),
                     MADValue(CastValue<scalarN>(alpha),
                              accum[i],
                              CastValue<scalarN>(beta) * *(outC + i * (N / VECTOR_LENGTH))));

    return os << EndBlock(); // end function body
}

}; // namespace
