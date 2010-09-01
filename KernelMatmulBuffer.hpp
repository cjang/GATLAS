#ifndef _GATLAS_KERNEL_MATMUL_BUFFER_HPP_
#define _GATLAS_KERNEL_MATMUL_BUFFER_HPP_

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

// matrix multiply using memory buffers
template <typename SCALAR, size_t VECTOR_LENGTH>
class KernelMatmulBuffer : public KernelBaseMatmul,
                           protected MatmulParamInlineMNK,
                           protected MatmulParamLoopOrder
{
    typedef SCALAR scalar;
    typedef VecType< SCALAR, VECTOR_LENGTH > scalarN;

    int _handleA;
    int _handleB;
    int _handleC;

    // compare matrix multiply to reference results
    bool _paranoidCheck;
    scalar *_paranoidC;

public:
    KernelMatmulBuffer()
        : KernelBaseMatmul(),
          MatmulParamInlineMNK(getExtraParameter()),
          MatmulParamLoopOrder(getExtraParameter()),
          _handleA(-1),
          _handleB(-1),
          _handleC(-1),
          _paranoidCheck(false),
          _paranoidC(NULL)
    { }

    ~KernelMatmulBuffer() {
        delete[] _paranoidC;
    }

    std::string kernelName() const {
        std::stringstream ss;
        ss << "matmulbuffer" << nameof<SCALAR>() << VECTOR_LENGTH;
        return ss.str();
    }

    void paranoidCheck() {
        _paranoidCheck = true;
        delete[] _paranoidC;
        _paranoidC = new scalar[packedCalc() * dimM() * dimN()];
    }

    bool syncOutput(OCLApp& oclApp) {
        return syncBufferFromDevice(oclApp, _handleC);
    }

    bool checkOutput(OCLApp& oclApp, const bool printOutput) {
        if (_paranoidCheck) {
            return checkBuffer<scalar>(oclApp, _handleC, dimN(), packedCalc() * dimM(), _paranoidC, printOutput);
        } else {
            const scalar testValue = dimK();
            return checkBuffer<scalar>(oclApp, _handleC, dimN(), packedCalc() * dimM(), testValue, printOutput);
        }
    }

    bool setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) {

        // buffer allocation
        if (-1 == _handleA || -1 == _handleB || -1 == _handleC || dimChanged()) {
            oclApp.releaseBuffers();
            _handleA = createBufferR<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimM() * dimK(), "matA", 1);
            _handleB = createBufferR<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimK() * dimN(), "matB", 1);
            _handleC = generalizedMatmul()
                           ? createBufferRW<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimM() * dimN(), "matC", 0)
                           : createBufferW<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimM() * dimN(), "matC", 0);
            if (-1 == _handleA || -1 == _handleB || -1 == _handleC) return false; // failure
        } else {
            // matrices A and B
            if (syncInput) {
                if (!syncBufferToDevice(oclApp, _handleA)) return false;
                if (!syncBufferToDevice(oclApp, _handleB)) return false;
            }
            // matrix C
            if (!clearBuffer<scalar>(oclApp, _handleC)) return false;
        }

        const scalar alpha = (_paranoidCheck && generalizedMatmul()) ? posrand<scalar>() : 1;
        const scalar beta = (_paranoidCheck && generalizedMatmul()) ? posrand<scalar>() : 1;

        // paranoid check
        if (_paranoidCheck) {

            // fill A and B matrices with random values
            if (fillrandBuffer<scalar>(oclApp, _handleA, packedCalc() * dimM() * dimK()) &&
                fillrandBuffer<scalar>(oclApp, _handleB, packedCalc() * dimK() * dimN()) &&
                (generalizedMatmul()
                     ? fillrandBuffer<scalar>(oclApp, _handleC, packedCalc() * dimM() * dimN())
                     : true)) {

                const scalar *ptrA = oclApp.bufferPtr<scalar>(_handleA);
                const scalar *ptrB = oclApp.bufferPtr<scalar>(_handleB);
                const scalar *ptrC = generalizedMatmul() ? oclApp.bufferPtr<scalar>(_handleC) : NULL;

                fillconst<scalar>(_paranoidC, 0, packedCalc() * dimM() * dimN());

                // packed kernels
                for (size_t pIdx = 0; pIdx < packedCalc(); pIdx++) {

                    // calculate C
                    for (size_t i = 0; i < dimM(); i++)
                    for (size_t j = 0; j < dimN(); j++)
                    for (size_t k = 0; k < dimK(); k++)
                        _paranoidC[pIdx * dimM() * dimN() + i * dimN() + j]
                            += (transposeA()
                                    ? ptrA[pIdx * dimM() * dimK() + k * dimM() + i]
                                    : ptrA[pIdx * dimM() * dimK() + i * dimK() + k])
                             * (transposeB()
                                    ? ptrB[pIdx * dimK() * dimN() + j * dimK() + k]
                                    : ptrB[pIdx * dimK() * dimN() + k * dimN() + j]);

                    // multiply AB product by alpha and add beta times C
                    if (generalizedMatmul())
                        for (size_t i = 0; i < dimM() * dimN(); i++)
                            _paranoidC[pIdx * dimM() * dimN() + i] = alpha * _paranoidC[pIdx * dimM() * dimN() + i]
                                                                   + beta * ptrC[pIdx * dimM() * dimN() + i];
                }
            } else {
                std::cerr << "error: failed to fill input matrices with random values" << std::endl;
            }
        }

        // set kernel arguments
        const size_t numberElemsTmpA = localSize() * groupSize() * VECTOR_LENGTH * blockHeight();
        const size_t numberElemsTmpB = localSize() * groupSize() * VECTOR_LENGTH * VECTOR_LENGTH;
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
        if (generalizedMatmul()) {
            rc = rc &&
            setArgValue<scalar>(oclApp, kernelHandle, argIndex++, alpha, "alpha") &&
            setArgValue<scalar>(oclApp, kernelHandle, argIndex++, beta, "beta");
        }
        return rc;
    }

    // prints the kernel source
    std::ostream& print(std::ostream& os) const {

        pragma_extension<scalar>(os);

        // kernel function attributes
        AutoVectorize< scalarN > attrAutoVec;
        FunctionDeclaration kernelDecl(kernelName());
        kernelDecl.returnType<void>();
        kernelDecl.qualify(KERNEL);
        if (getUseAttrAutoVec()) kernelDecl.qualify(attrAutoVec);

        // kernel arguments
        Var< scalarN* >       matC("matC", GLOBAL, kernelDecl);
        Var< const scalarN* > matA("matA", GLOBAL, kernelDecl);
        Var< const scalarN* > matB("matB", GLOBAL, kernelDecl);
        Var< scalarN* >       tmpA("tmpA", LOCAL, kernelDecl);
        Var< scalarN* >       tmpB("tmpB", LOCAL, kernelDecl);
        Var< const int >      M("M", kernelDecl, inlineMNK(), dimM());
        Var< const int >      N("N", kernelDecl, inlineMNK(), dimN());
        Var< const int >      K("K", kernelDecl, inlineMNK(), dimK());
        Var< const scalar >   alpha("alpha", kernelDecl, generalizedMatmul());
        Var< const scalar >   beta("beta", kernelDecl, generalizedMatmul());

        // begin function body
        os << kernelDecl;

        // accumulate inner product sum
        Vector< scalarN > accum("accum", blockHeight());
        os << declare(accum);

        // pointer to matA
        Var< const scalarN* > ptrMatA("ptrMatA", GLOBAL);
        os << declare(ptrMatA);

        // pointer to matB
        Var< const scalarN* > ptrMatB("ptrMatB", GLOBAL);
        os << declare(ptrMatB);

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

        // packed kernel support
        Var< int > pIdx("pIdx", 1 == packedCalc(), 0);
        if (1 != packedCalc()) os << ForLoop(pIdx, packedCalc(), 1);

            // set accumulators to zero
            os << assign(accum, CastValue<scalarN>(ConstantValue<scalar>(0)));

            // start at left or top edge of matrix A
            if (transposeA())
                os << assign(ptrMatA, matA + wholeHeight() * globalRow
                                           + M * col
                                           + pIdx * M * K / VECTOR_LENGTH);
            else
                os << assign(ptrMatA, matA + multHeight(K) * groupSize() * blockRow
                                           + (K / VECTOR_LENGTH) * row
                                           + col
                                           + pIdx * M * K / VECTOR_LENGTH);

            // start at left or top edge of matrix B
            if (transposeB())
                os << assign(ptrMatB, matB + K * groupSize() * blockCol
                                           + (K / VECTOR_LENGTH) * row
                                           + col
                                           + pIdx * K * N / VECTOR_LENGTH);
            else
                os << assign(ptrMatB, matB + globalCol
                                           + N * row
                                           + pIdx * K * N / VECTOR_LENGTH);

            // outer loop over blocks
            Var< int > idx("idx");
            os << ForLoop(idx, K / (groupSize() * VECTOR_LENGTH), 1);

                // copy block of A
                if (transposeA())
                    for (size_t i = 0; i < blockHeight(); i++) {
                        const size_t blockNum = i / VECTOR_LENGTH;
                        const size_t blockIdx = i % VECTOR_LENGTH;
                        os << assign(*(tmpA + localSize() * (blockHeight() * row + i) + col),
                                     *(ptrMatA + blockNum + blockIdx * M / VECTOR_LENGTH));
                    }
                else
                    for (size_t i = 0; i < blockHeight(); i++)
                        os << assign(*(tmpA + localSize() * (row + i * groupSize()) + col),
                                     *(ptrMatA + i * groupSize() * K / VECTOR_LENGTH));

                // copy block of B
                for (size_t i = 0; i < VECTOR_LENGTH; i++)
                    if (transposeB())
                        os << assign(*(tmpB + localSize() * (row + i * groupSize()) + col),
                                     *(ptrMatB + i * groupSize() * K / VECTOR_LENGTH));
                    else
                        os << assign(*(tmpB + localSize() * (VECTOR_LENGTH * col + i) + row),
                                     *(ptrMatB + i * N / VECTOR_LENGTH));

                // barrier
                os << LocalBarrier();

                // next block for A
                if (transposeA())
                    os << increment(ptrMatA, M * groupSize());
                else
                    os << increment(ptrMatA, groupSize());

                // next block for B
                if (transposeB())
                    os << increment(ptrMatB, groupSize());
                else
                    os << increment(ptrMatB, N * groupSize());

                // for inner product
                os << assign(ptrA, tmpA + localSize() * blockHeight() * row);
                os << assign(ptrB, tmpB + localSize() * VECTOR_LENGTH * col);

                // inner product accumulation
                Var< int > jdx("jdx");
                os << ForLoop(jdx, groupSize(), 1);

                    // read in values of tmpA
                    for (size_t j = 0; j < blockHeight(); j++)
                        os << assign(valA[j], *(ptrA + j * localSize()));

                    os << increment(ptrA, 1);

                    // read in values of tmpB
                    for (size_t j = 0; j < VECTOR_LENGTH; j++)
                        os << assign(valB[j], *(ptrB + j * localSize()));

                    os << increment(ptrB, 1);

                    // inner product accumulation
                    assignMAD(os, loopOrder(), accum, valA, valB);

                os << EndBlock();

            os << EndBlock();

            const ConstantValue<std::string> outC = matC + multHeight(N) * globalRow + globalCol + pIdx * M * N / VECTOR_LENGTH;

            for (size_t i = 0; i < blockHeight(); i++)
                if (generalizedMatmul())
                    os << assign(*(outC + i * (N / VECTOR_LENGTH)),
                                 true //isfloat<SCALAR>()
                                     ? MADValue(CastValue<scalarN>(alpha),
                                                accum[i],
                                                CastValue<scalarN>(beta) * *(outC + i * (N / VECTOR_LENGTH)))
                                     : CastValue<scalarN>(alpha) * accum[i] +
                                       CastValue<scalarN>(beta) * *(outC + i * (N / VECTOR_LENGTH)));
                else
                    os << assign(*(outC + i * (N / VECTOR_LENGTH)), accum[i]);

        if (1 != packedCalc()) os << EndBlock();

        return os << EndBlock(); // end function body
    }
};

}; // namespace

#endif
