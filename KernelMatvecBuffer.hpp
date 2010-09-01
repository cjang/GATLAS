#ifndef _GATLAS_KERNEL_MATVEC_BUFFER_HPP_
#define _GATLAS_KERNEL_MATVEC_BUFFER_HPP_

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

#include "declare_namespace"

// matrix vector multiply using memory buffers
template <typename SCALAR, size_t VECTOR_LENGTH>
class KernelMatvecBuffer : public KernelBaseMatvec,
                           protected MatvecParamInlineMN
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
    KernelMatvecBuffer()
        : KernelBaseMatvec(),
          MatvecParamInlineMN(getExtraParameter()),
          _handleA(-1),
          _handleB(-1),
          _handleC(-1),
          _paranoidCheck(false),
          _paranoidC(NULL)
    { }

    ~KernelMatvecBuffer() {
        delete[] _paranoidC;
    }

    std::string kernelName() const {
        std::stringstream ss;
        ss << "matvecbuffer" << nameof<SCALAR>() << VECTOR_LENGTH;
        return ss.str();
    }

    void paranoidCheck() {
        _paranoidCheck = true;
        delete[] _paranoidC;
        _paranoidC = new scalar[packedCalc() * dimM()];
    }

    bool syncOutput(OCLApp& oclApp) {
        return syncBufferFromDevice(oclApp, _handleC);
    }

    bool checkOutput(OCLApp& oclApp, const bool printOutput) {
        if (_paranoidCheck) {
            return checkBuffer<scalar>(oclApp, _handleC, packedCalc() * dimM(), _paranoidC, printOutput);
        } else {
            const scalar testValue = dimN();
            return checkBuffer<scalar>(oclApp, _handleC, packedCalc() * dimM(), testValue, printOutput);
        }
    }

    bool setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) {

        // buffer allocation
        if (-1 == _handleA || -1 == _handleB || -1 == _handleC || dimChanged() || packedChanged()) {
            oclApp.releaseBuffers();
            _handleA = createBufferR<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimM() * dimN(), "matA", 1);
            _handleB = createBufferR<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimN(), "vecB", 1);
            _handleC = generalizedMatvec()
                           ? createBufferRW<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimM(), "vecC", 0)
                           : createBufferW<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimM(), "vecC", 0);
            if (-1 == _handleA || -1 == _handleB || -1 == _handleC) return false; // failure
        } else {
            // matrix A and vector B
            if (syncInput) {
                if (!syncBufferToDevice(oclApp, _handleA)) return false;
                if (!syncBufferToDevice(oclApp, _handleB)) return false;
            }
            // vector C
            if (!clearBuffer<scalar>(oclApp, _handleC)) return false;
        }

        const scalar alpha = (_paranoidCheck && generalizedMatvec()) ? posrand<scalar>() : 1;
        const scalar beta = (_paranoidCheck && generalizedMatvec()) ? posrand<scalar>() : 1;

        // paranoid check
        if (_paranoidCheck) {

            // fill matrix A and vector B with random values
            if (fillrandBuffer<scalar>(oclApp, _handleA, packedCalc() * dimM() * dimN()) &&
                fillrandBuffer<scalar>(oclApp, _handleB, packedCalc() * dimN()) &&
                (generalizedMatvec()
                     ? fillrandBuffer<scalar>(oclApp, _handleC, packedCalc() * dimM())
                     : true)) {

                const scalar *ptrA = oclApp.bufferPtr<scalar>(_handleA);
                const scalar *ptrB = oclApp.bufferPtr<scalar>(_handleB);
                const scalar *ptrC = generalizedMatvec() ? oclApp.bufferPtr<scalar>(_handleC) : NULL;

                fillconst<scalar>(_paranoidC, 0, packedCalc() * dimM());

                // packed kernels
                for (size_t pIdx = 0; pIdx < packedCalc(); pIdx++) {

                    // calculate C
                    for (size_t i = 0; i < dimM(); i++)
                    for (size_t j = 0; j < dimN(); j++)
                        _paranoidC[pIdx * dimM() + i]
                            += (transposeA()
                                    ? ptrA[pIdx * dimM() * dimN() + j * dimM() + i]
                                    : ptrA[pIdx * dimM() * dimN() + i * dimN() + j])
                             * ptrB[pIdx * dimN() + j];

                    // multiply AB product by alpha and add beta times C
                    if (generalizedMatvec())
                        for (size_t i = 0; i < dimM(); i++)
                            _paranoidC[pIdx * dimM() + i] = alpha * _paranoidC[pIdx * dimM() + i]
                                                          + beta * ptrC[pIdx * dimM() + i];
                }
            } else {
                std::cerr << "error: failed to fill input matrices with random values" << std::endl;
            }
        }

        // set kernel arguments
/*
 * copying global vector B to local memory causes failures with
 * vector elements on ATI and as scalars are slow, is bad
 * this does work on NVIDIA Fermi but is slower than direct reads
 * from global memory
 *
        const size_t numberElemsTmpB = dimN();
 */
        size_t argIndex = 0;
        bool rc =
            setArgGlobal(oclApp, kernelHandle, argIndex++, _handleC, "vecC") &&
            setArgGlobal(oclApp, kernelHandle, argIndex++, _handleA, "matA") &&
            setArgGlobal(oclApp, kernelHandle, argIndex++, _handleB, "vecB");
/*
 * copying global vector B to local memory causes failures with
 * vector elements on ATI and as scalars are slow, is bad
 * this does work on NVIDIA Fermi but is slower than direct reads
 * from global memory
 *
            setArgLocal<scalar>(oclApp, kernelHandle, argIndex++, numberElemsTmpB, "tmpB");
 */
        if (! inlineMN()) {
            rc = rc &&
            setArgValue<int>(oclApp, kernelHandle, argIndex++, dimM(), "M") &&
            setArgValue<int>(oclApp, kernelHandle, argIndex++, dimN(), "N");
        }
        if (generalizedMatvec()) {
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
        Var< scalarN* >       vecC("vecC", GLOBAL, kernelDecl);
        Var< const scalarN* > matA("matA", GLOBAL, kernelDecl);
        Var< const scalarN* > vecB("vecB", GLOBAL, kernelDecl);
/*
 * copying global vector B to local memory causes failures with
 * vector elements on ATI and as scalars are slow, is bad
 * this does work on NVIDIA Fermi but is slower than direct reads
 * from global memory
 *
        Var< scalarN* >       tmpB("tmpB", LOCAL, kernelDecl);
 */
        Var< const int >      M("M", kernelDecl, inlineMN(), dimM());
        Var< const int >      N("N", kernelDecl, inlineMN(), dimN());
        Var< const scalar >   alpha("alpha", kernelDecl, generalizedMatvec());
        Var< const scalar >   beta("beta", kernelDecl, generalizedMatvec());

        // begin function body
        os << kernelDecl;

        // accumulate inner product sum
        Vector< scalarN > accum("accum", wholeHeight());
        os << declare(accum);

        // current value from matA (for matrix vector outer product)
        Var< scalarN > valA("valA");
        os << declare(valA);

        // current value from tmpB (for matrix vector outer product)
        Var< scalarN > valB("valB");
        os << declare(valB);

        // pointer to matA
        Var< const scalarN* > ptrMatA("ptrMatA", GLOBAL);
        os << declare(ptrMatA);

        // packed kernel support
        Var< int > pIdx("pIdx", 1 == packedCalc(), 0);
        if (1 != packedCalc()) os << ForLoop(pIdx, packedCalc(), 1);

            // set accumulators to zero
            os << assign(accum, CastValue<scalarN>(ConstantValue<scalar>(0)));

            // start at left or top edge of matrix A
            if (transposeA())
                os << assign(ptrMatA, matA + wholeHeight() * globalRow
                                           + pIdx * M * N / VECTOR_LENGTH);
            else
                os << assign(ptrMatA, matA + multHeight(N) * globalRow
                                           + pIdx * M * N / VECTOR_LENGTH);

/*
 * copying global vector B to local memory causes failures with
 * vector elements on ATI and as scalars are slow, is bad
 * this does work on NVIDIA Fermi but is slower than direct reads
 * from global memory
 *
        // pointer to vecB
        Var< const scalarN* > ptrVecB("ptrVecB", GLOBAL);
        os << declare(ptrVecB, vecB + row);

        // pointer to tmpB
        Var< scalarN * > ptrTmpB("ptrTmpB", LOCAL);
        os << declare(ptrTmpB, tmpB + row);

        // copy entire vector B from global to local memory
        Var< int > idx("idx");
        os << ForLoop(idx, N / (groupSize() * VECTOR_LENGTH), 1);
            os << assign(*ptrTmpB, *ptrVecB);
            os << increment(ptrVecB, groupSize() * VECTOR_LENGTH);
            os << increment(ptrTmpB, groupSize() * VECTOR_LENGTH);
        os << EndBlock();

        // barrier
        os << LocalBarrier();
 */

            // outer loop over vector B
            Var< int > idx("idx");
            os << ForLoop(idx, N / VECTOR_LENGTH, 1);

                // read value from vector B
/*
 * copying global vector B to local memory causes failures with
 * vector elements on ATI and as scalars are slow, is bad
 * this does work on NVIDIA Fermi but is slower than direct reads
 * from global memory
 *
            os << assign(valB, *(tmpB + idx));
 */
                os << assign(valB, *(vecB + idx
                                          + pIdx * N / VECTOR_LENGTH));

                // inner loop over matrix A
                for (size_t j = 0; j < blockHeight(); j++) {
                    const size_t blockNum = j / VECTOR_LENGTH;
                    const size_t blockIdx = j % VECTOR_LENGTH;
                    if (transposeA()) {
                        os << assign(valA, *(ptrMatA + blockNum + blockIdx * M / VECTOR_LENGTH));
                        for (size_t k = 0; k < VECTOR_LENGTH; k++)
                            os << assignMAD(accum[blockNum], valA, valB, blockIdx, k);
                    } else {
                        os << assign(valA, *(ptrMatA + j * N / VECTOR_LENGTH));
                        for (size_t k = 0; k < VECTOR_LENGTH; k++)
                            os << assignMAD(accum[blockNum], valA, valB, blockIdx, k);
                    }
                }

                // next block column of A
                if (transposeA())
                    os << increment(ptrMatA, M);
                else
                    os << increment(ptrMatA, 1);

            os << EndBlock();

            const ConstantValue<std::string> outC = vecC + wholeHeight() * globalRow
                                                         + pIdx * M / VECTOR_LENGTH;

            for (size_t i = 0; i < wholeHeight(); i++)
                if (generalizedMatvec())
                    os << assign(*(outC + i),
                                 true //isfloat<SCALAR>()
                                     ? MADValue(CastValue<scalarN>(alpha),
                                                accum[i],
                                                CastValue<scalarN>(beta) * *(outC + i))
                                     : CastValue<scalarN>(alpha) * accum[i] +
                                       CastValue<scalarN>(beta) * *(outC + i));
                else
                    os << assign(*(outC + i), accum[i]);

        if (1 != packedCalc()) os << EndBlock();

        return os << EndBlock(); // end function body
    }
};

}; // namespace

#endif
