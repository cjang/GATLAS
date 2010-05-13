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

#include "KernelGenMatmulBuffer.hpp"

#include "declare_namespace"

// matrix dimensions are inlined constants or passed as kernel arguments
bool KernelGenMatmulBuffer::_inlineMNK(const size_t extraParam) const { return extraParam / 6; }

// inner product accumulation loop order, 3! permutations of (j,k,l)
size_t KernelGenMatmulBuffer::_loopOrder(const size_t extraParam) const { return extraParam % 6; }

// extra configuration parameters
size_t KernelGenMatmulBuffer::numberExtraParam() const {
    return 2   // matrix dimensions inlined or passed as kernel arguments
         * 6;  // inner product loop order
}

std::string KernelGenMatmulBuffer::namePrefix() const { return "KernelGenMatmulBuffer"; }

KernelGenMatmulBuffer::KernelGenMatmulBuffer(const bool transposeA,
                                             const bool transposeB)
    : KernelGenMatmul(transposeA, transposeB),
      _handleA(-1),
      _handleB(-1),
      _handleC(-1),
      _paranoidCheck(false),
      _paranoidC(NULL)
{ }

KernelGenMatmulBuffer::~KernelGenMatmulBuffer() {
    delete[] _paranoidC;
}

void KernelGenMatmulBuffer::paranoidCheck() {
    _paranoidCheck = true;
    delete[] _paranoidC;
    _paranoidC = new scalar[dimM() * dimN()];
}

std::string KernelGenMatmulBuffer::desc() const {
    std::stringstream ss;
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

size_t KernelGenMatmulBuffer::maxGroupSize(const size_t M, const size_t N, const size_t K) const {
    // limit work group size to avoid kernel hangs
    if ( (0 == M % 10) && (0 == N % 10) )
        return 10;
    else
        return 8;
}

size_t KernelGenMatmulBuffer::maxBlockHeight() const {
    // transposing A fixes inner blocking to square quads
    return transposeA()
               ? VECTOR_LENGTH
               : 2 * VECTOR_LENGTH + VECTOR_LENGTH / 2; // risk of kernel hangs if larger
}

bool KernelGenMatmulBuffer::checkOutput(OCLApp& oclApp, const bool printOutput) {
    if (_paranoidCheck) {
        return checkBuffer(oclApp, _handleC, dimN(), dimM(), _paranoidC, printOutput);
    } else {
        const scalar testValue = dimK();
        return checkBuffer(oclApp, _handleC, dimN(), dimM(), testValue, printOutput);
    }
}

bool KernelGenMatmulBuffer::setArgs(OCLApp& oclApp, const size_t kernelHandle) {
    if (-1 == _handleA || -1 == _handleB || -1 == _handleC || mnkChanged()) {
        oclApp.releaseBuffers();
        _handleA = createBufferR<scalar, VECTOR_LENGTH>(oclApp, dimM() * dimK(), "matA", 1);
        _handleB = createBufferR<scalar, VECTOR_LENGTH>(oclApp, dimK() * dimN(), "matB", 1);
        _handleC = createBufferRW<scalar, VECTOR_LENGTH>(oclApp, dimM() * dimN(), "matC", 0);
        if (-1 == _handleA || -1 == _handleB || -1 == _handleC) return false; // failure
    } else {
        if (!clearBuffer(oclApp, _handleC)) return false;
    }

    const scalar alpha = _paranoidCheck ? posrand<scalar>() : 1;
    const scalar beta = _paranoidCheck ? posrand<scalar>() : 1;

    // paranoid check
    if (_paranoidCheck) {

        // fill A, B and C matrices with random values
        if (fillrandBuffer(oclApp, _handleA, dimM() * dimK()) &&
            fillrandBuffer(oclApp, _handleB, dimK() * dimN()) &&
            fillrandBuffer(oclApp, _handleC, dimM() * dimN())) {

            const scalar *ptrA = oclApp.bufferPtr<scalar>(_handleA);
            const scalar *ptrB = oclApp.bufferPtr<scalar>(_handleB);
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
            std::cerr << "error: failed to fill matrices with random values" << std::endl;
        }
    }

    const size_t numberElemsTmpA = localWidth() * groupSize() * VECTOR_LENGTH * blockHeight();
    const size_t numberElemsTmpB = localWidth() * groupSize() * VECTOR_LENGTH * VECTOR_LENGTH;
    size_t argIndex = 0;
    bool rc =
        setArgGlobal(oclApp, kernelHandle, argIndex++, _handleC, "matC") &&
        setArgGlobal(oclApp, kernelHandle, argIndex++, _handleA, "matA") &&
        setArgGlobal(oclApp, kernelHandle, argIndex++, _handleB, "matB") &&
        setArgLocal<scalar>(oclApp, kernelHandle, argIndex++, numberElemsTmpA, "tmpA") &&
        setArgLocal<scalar>(oclApp, kernelHandle, argIndex++, numberElemsTmpB, "tmpB");
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
std::ostream& KernelGenMatmulBuffer::print(std::ostream& os) const {

    // kernel function attributes
    AutoVectorize< scalarN > attrAutoVec;
    FunctionDeclaration kernelDecl(kernelName());
    kernelDecl.returnType<void>();
    kernelDecl.qualify(KERNEL);
    kernelDecl.qualify(attrAutoVec);

    // kernel arguments
    Var< scalarN* >       matC("matC", GLOBAL, kernelDecl);
    Var< const scalarN* > matA("matA", GLOBAL, kernelDecl);
    Var< const scalarN* > matB("matB", GLOBAL, kernelDecl);
    Var< scalarN* >       tmpA("tmpA", LOCAL, kernelDecl);
    Var< scalarN* >       tmpB("tmpB", LOCAL, kernelDecl);
    Var< const int >      M("M");
    Var< const int >      N("N");
    Var< const int >      K("K");
    if (! inlineMNK()) {
        kernelDecl.argument(M);
        kernelDecl.argument(N);
        kernelDecl.argument(K);
    }
    Var< const scalar > alpha("alpha", kernelDecl);
    Var< const scalar > beta("beta", kernelDecl);

    // begin function body
    os << kernelDecl;

    // accumulate inner product sum
    Vector< scalarN > accum("accum", blockHeight());
    os << declare(accum, CastValue<scalarN>(ConstantValue<scalar>(0)));

    // pointer to matA
    Var< const scalarN* > ptrMatA("ptrMatA", GLOBAL);
    if (inlineMNK()) {
        if (transposeA())
            os << declare(ptrMatA, matA + globalRow
                                        + dimM() * col);
        else
            os << declare(ptrMatA, matA + multQuads(dimK()) * groupSize() * blockRow
                                        + (dimK() / VECTOR_LENGTH) * row
                                        + col);
    } else {
        if (transposeA())
            os << declare(ptrMatA, matA + globalRow
                                        + M * col);
        else
            os << declare(ptrMatA, matA + multQuads(K) * groupSize() * blockRow
                                        + (K / VECTOR_LENGTH) * row
                                        + col);
    }

    // pointer to matB
    Var< const scalarN* > ptrMatB("ptrMatB", GLOBAL);
    if (inlineMNK()) {
        if (transposeB())
            os << declare(ptrMatB, matB + dimK() * groupSize() * blockCol
                                        + (dimK() / VECTOR_LENGTH) * row
                                        + col);
        else
            os << declare(ptrMatB, matB + globalCol
                                        + dimN() * row);
    } else {
        if (transposeB())
            os << declare(ptrMatB, matB + K * groupSize() * blockCol
                                        + (K / VECTOR_LENGTH) * row
                                        + col);
        else
            os << declare(ptrMatB, matB + globalCol
                                        + N * row);
    }

    // pointer to tmpA (for inner product)
    Var< const scalarN* > ptrA("ptrA", LOCAL);
    os << declare(ptrA);

    // pointer to tmpB (for inner product)
    Var< const scalarN* > ptrB("ptrB", LOCAL);
    os << declare(ptrB);

    // current values from tmpA (for inner product)
    Vector< scalarN > valA("valA", blockHeight());
    os << declare(valA);

    // current values from tmpB (for inner product)
    Vector< scalarN > valB("valB", VECTOR_LENGTH);
    os << declare(valB);

    // outer loop over blocks
    Var< int > idx("idx");
    if (inlineMNK())
        os << ForLoop(idx, dimK() / (groupSize() * VECTOR_LENGTH), 1);
    else
        os << ForLoop(idx, K / (groupSize() * VECTOR_LENGTH), 1);

        // copy block of A
        if (transposeA())
            os << assign(*(tmpA + localWidth() * VECTOR_LENGTH * row + col), *ptrMatA);
        else
            os << assign(*(tmpA + localWidth() * row + col), *ptrMatA);
        for (size_t i = 1; i < blockHeight(); i++) {
            if (inlineMNK()) {
                if (transposeA())
                    os << assign(*(tmpA + localWidth() * (VECTOR_LENGTH * row + i) + col),
                                 *(ptrMatA + i * dimM() / VECTOR_LENGTH));
                else
                    os << assign(*(tmpA + localWidth() * (row + i * groupSize()) + col),
                                 *(ptrMatA + i * groupSize() * dimK() / VECTOR_LENGTH));
            } else {
                if (transposeA())
                    os << assign(*(tmpA + localWidth() * (VECTOR_LENGTH * row + i) + col),
                                 *(ptrMatA + i * M / VECTOR_LENGTH));
                else
                    os << assign(*(tmpA + localWidth() * (row + i * groupSize()) + col),
                                 *(ptrMatA + i * groupSize() * K / VECTOR_LENGTH));
            }
        }

        // copy block of B
        if (transposeB())
            os << assign(*(tmpB + localWidth() * row + col), *ptrMatB);
        else
            os << assign(*(tmpB + localWidth() * VECTOR_LENGTH * col + row), *ptrMatB);
        for (size_t i = 1; i < VECTOR_LENGTH; i++) {
            if (inlineMNK()) {
                if (transposeB())
                    os << assign(*(tmpB + localWidth() * (row + i * groupSize()) + col),
                                 *(ptrMatB + i * groupSize() * dimK() / VECTOR_LENGTH));
                else
                    os << assign(*(tmpB + localWidth() * (VECTOR_LENGTH * col + i) + row),
                                 *(ptrMatB + i * dimN() / VECTOR_LENGTH));
            } else {
                if (transposeB())
                    os << assign(*(tmpB + localWidth() * (row + i * groupSize()) + col),
                                 *(ptrMatB + i * groupSize() * K / VECTOR_LENGTH));
                else
                    os << assign(*(tmpB + localWidth() * (VECTOR_LENGTH * col + i) + row),
                                 *(ptrMatB + i * N / VECTOR_LENGTH));
            }
        }

        // barrier
        os << LocalBarrier();

        // next block for A
        if (transposeA()) {
            if (inlineMNK())
                os << increment(ptrMatA, dimM() * groupSize());
            else
                os << increment(ptrMatA, M * groupSize());
        } else {
            os << increment(ptrMatA, groupSize());
        }

        // next block for B
        if (transposeB()) {
            os << increment(ptrMatB, groupSize());
        } else {
            if (inlineMNK())
                os << increment(ptrMatB, dimN() * groupSize());
            else
                os << increment(ptrMatB, N * groupSize());
        }

        // for inner product
        os << assign(ptrA, tmpA + localWidth() * blockHeight() * row);
        os << assign(ptrB, tmpB + localWidth() * VECTOR_LENGTH * col);

        // inner product accumulation
        Var< int > jdx("jdx");
        os << ForLoop(jdx, groupSize(), 1);

            // read in values of tmpA
            os << assign(valA[0], *ptrA);
            for (size_t j = 1; j < blockHeight(); j++) {
                os << assign(valA[j], *(ptrA + j * localWidth()));
            }

            os << increment(ptrA, 1);

            // read in values of tmpB
            os << assign(valB[0], *ptrB);
            for (size_t j = 1; j < VECTOR_LENGTH; j++) {
                os << assign(valB[j], *(ptrB + j * localWidth()));
            }

            os << increment(ptrB, 1);

            // inner product accumulation
            assignMAD(os, loopOrder(), accum, valA, valB);

        os << EndBlock();

    os << EndBlock();

    const ConstantValue<std::string> outC
        = inlineMNK()
              ? matC + multQuads(dimN()) * globalRow + globalCol
              : matC + multQuads(N) * globalRow + globalCol;

    os << assign(*outC,
                 MADValue(CastValue<scalarN>(alpha),
                          accum[0],
                          CastValue<scalarN>(beta) * *outC));
    for (size_t i = 1; i < blockHeight(); i++) {
        if (inlineMNK())
            os << assign(*(outC + i * (dimN() / VECTOR_LENGTH)),
                         MADValue(CastValue<scalarN>(alpha),
                                  accum[i],
                                  CastValue<scalarN>(beta) * *(outC + i * (dimN() / VECTOR_LENGTH))));
        else
            os << assign(*(outC + i * (N / VECTOR_LENGTH)),
                         MADValue(CastValue<scalarN>(alpha),
                                  accum[i],
                                  CastValue<scalarN>(beta) * *(outC + i * (N / VECTOR_LENGTH))));
    }

    return os << EndBlock(); // end function body
}

}; // namespace
