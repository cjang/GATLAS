#ifndef _GATLAS_KERNEL_MATMUL_IMAGE_HPP_
#define _GATLAS_KERNEL_MATMUL_IMAGE_HPP_

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

// matrix multiply using image textures
template <typename SCALAR, size_t VECTOR_LENGTH>
class KernelMatmulImage : public KernelBaseMatmul,
                          protected MatmulParamInlineMNK,
                          protected MatmulParamLoopOrder,
                          protected MatmulParamGlobalID
{
    typedef SCALAR scalar;
    typedef VecType< SCALAR, VECTOR_LENGTH > scalarN; // float4 or double2

    const bool _spQuad; // true when SCALAR is float and VECTOR_LENGTH is 4

    int _handleA;
    int _handleB;
    int _handleC;

    // compare matrix multiply to reference results
    bool _paranoidCheck;
    scalar *_paranoidC;

public:
    KernelMatmulImage()
        : KernelBaseMatmul(),
          MatmulParamInlineMNK(getExtraParameter()),
          MatmulParamLoopOrder(getExtraParameter()),
          MatmulParamGlobalID(getExtraParameter()),
          _spQuad(isfloat<SCALAR>() && 4 == VECTOR_LENGTH),
          _handleA(-1),
          _handleB(-1),
          _handleC(-1),
          _paranoidCheck(false),
          _paranoidC(NULL)
    { }

    ~KernelMatmulImage() {
        delete[] _paranoidC;
    }

    std::string kernelName() const {
        std::stringstream ss;
        ss << "matmulimage" << nameof<SCALAR>() << VECTOR_LENGTH;
        return ss.str();
    }

    void paranoidCheck() {
        _paranoidCheck = true;
        delete[] _paranoidC;
        _paranoidC = new scalar[packedCalc() * dimM() * dimN()];
    }

    bool syncOutput(OCLApp& oclApp) {
        return generalizedMatmul()
                   ? syncBufferFromDevice(oclApp, _handleC)
                   : syncImageFromDevice(oclApp, _handleC);
    }

    bool checkOutput(OCLApp& oclApp, const bool printOutput) {
        if (_paranoidCheck) {
            return generalizedMatmul()
                       ? checkBuffer<scalar>(oclApp, _handleC, dimN(), packedCalc() * dimM(), _paranoidC, printOutput)
                       : checkImage<scalar>(oclApp, _handleC, dimN(), packedCalc() * dimM(), _paranoidC, printOutput);
        } else {
            const scalar testValue = dimK();
            return generalizedMatmul()
                       ? checkBuffer<scalar>(oclApp, _handleC, dimN(), packedCalc() * dimM(), testValue, printOutput)
                       : checkImage<scalar>(oclApp, _handleC, dimN(), packedCalc() * dimM(), testValue, printOutput);
        }
    }

    bool setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) {

        // buffer allocation
        if (-1 == _handleA || -1 == _handleB || -1 == _handleC || dimChanged() || layoutChanged() || packedChanged()) {
            oclApp.releaseImages();
            if (generalizedMatmul()) oclApp.releaseBuffers();
            _handleA = transposeA()
                           ? createImageR<scalar>(oclApp, dimM(), packedCalc() * dimK(), "matA", 1)
                           : createImageR<scalar>(oclApp, dimK(), packedCalc() * dimM(), "matA", 1);
            _handleB = transposeB()
                           ? createImageR<scalar>(oclApp, dimK(), packedCalc() * dimN(), "matB", 1)
                           : createImageR<scalar>(oclApp, dimN(), packedCalc() * dimK(), "matB", 1);
            _handleC = generalizedMatmul()
                           ? createBufferRW<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimN() * dimM(), "matC", 0)
                           : createImageW<scalar>(oclApp, dimN(), packedCalc() * dimM(), "matC", 0);
            if (-1 == _handleA || -1 == _handleB || -1 == _handleC) return false; // failure
        } else {
            // matrices A and B
            if (syncInput) {
                if (!syncImageToDevice(oclApp, _handleA)) return false;
                if (!syncImageToDevice(oclApp, _handleB)) return false;
            }
            // matrix C
            if (generalizedMatmul()) {
                if (!clearBuffer<scalar>(oclApp, _handleC)) return false;
            } else {
                if (!clearImage(oclApp, _handleC)) return false;
            }
        }

        const scalar alpha = (_paranoidCheck && generalizedMatmul()) ? posrand<scalar>() : 1;
        const scalar beta = (_paranoidCheck && generalizedMatmul()) ? posrand<scalar>() : 1;

        // paranoid check
        if (_paranoidCheck) {

            // fill A and B matrices with random values
            if (fillrandImage<scalar>(oclApp, _handleA, packedCalc() * dimM() * dimK()) &&
                fillrandImage<scalar>(oclApp, _handleB, packedCalc() * dimK() * dimN()) &&
                (generalizedMatmul()
                     ? fillrandBuffer<scalar>(oclApp, _handleC, packedCalc() * dimM() * dimN())
                     : true)) {

                const scalar *ptrA = oclApp.imagePtr<scalar>(_handleA);
                const scalar *ptrB = oclApp.imagePtr<scalar>(_handleB);
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
        size_t argIndex = 0;
        bool rc =
            (generalizedMatmul()
                 ? setArgGlobal(oclApp, kernelHandle, argIndex++, _handleC, "matC")
                 : setArgImage(oclApp, kernelHandle, argIndex++, _handleC, "matC")) &&
            setArgImage(oclApp, kernelHandle, argIndex++, _handleA, "matA") &&
            setArgImage(oclApp, kernelHandle, argIndex++, _handleB, "matB");
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
        Var< image2d_t > matC_img("matC", WRITEONLY, kernelDecl, !generalizedMatmul());
        Var< scalarN* >  matC_buf("matC", GLOBAL, kernelDecl, generalizedMatmul());
        Var< image2d_t > matA("matA", READONLY, kernelDecl);
        Var< image2d_t > matB("matB", READONLY, kernelDecl);
        Var< const int > M("M", kernelDecl, inlineMNK(), dimM());
        Var< const int > N("N", kernelDecl, inlineMNK(), dimN());
        Var< const int > K("K", kernelDecl, inlineMNK(), dimK());
        Var< const scalar >   alpha("alpha", kernelDecl, generalizedMatmul());
        Var< const scalar >   beta("beta", kernelDecl, generalizedMatmul());

        // begin function body
        os << kernelDecl;

        // image sampler
        Var< const sampler_t > sampler("sampler");
        os << declare(sampler, ImageSampler());

        // accumulate inner product sum
        Vector< scalarN > accum("accum", blockHeight());
        os << declare(accum);

        // current values from matrix A (for inner product)
        Vector< scalarN > valA("valA", blockHeight());
        os << declare(valA);

        // current values from matrix B (for inner product)
        Vector< scalarN > valB("valB", VECTOR_LENGTH);
        os << declare(valB);

        // packed kernel support
        Var< int > pIdx("pIdx", 1 == packedCalc(), 0);
        if (1 != packedCalc()) os << ForLoop(pIdx, packedCalc(), 1);

            // set accumulators to zero
            os << assign(accum, CastValue<scalarN>(ConstantValue<scalar>(0)));

            // inner product loop
            Var< int > idx("idx");
            os << ForLoop(idx, K / VECTOR_LENGTH, 1);

                // read in values of matrix A
                if (transposeA())
                    for (size_t j = 0; j < blockHeight(); j++) {
                        const size_t blockNum = j / VECTOR_LENGTH;
                        const size_t blockIdx = j % VECTOR_LENGTH;
                        os << assign(valA[j],
                                     ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                         ReadImage<scalar>(matA, sampler,
                                                           wholeHeight() * globalRow + blockNum,
                                                           VECTOR_LENGTH * idx + blockIdx + pIdx * K),
                                         !_spQuad));
                    }
                else
                    for (size_t j = 0; j < blockHeight(); j++)
                        os << assign(valA[j],
                                     ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                         ReadImage<scalar>(matA, sampler,
                                                           idx,
                                                           blockHeight() * globalRow + j + pIdx * M),
                                         !_spQuad));

                // read in values of matrix B
                for (size_t j = 0; j < VECTOR_LENGTH; j++)
                    if (transposeB())
                        os << assign(valB[j],
                                     ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                         ReadImage<scalar>(matB, sampler,
                                                           idx,
                                                           VECTOR_LENGTH * globalCol + j + pIdx * N),
                                         !_spQuad));
                    else
                        os << assign(valB[j],
                                     ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                         ReadImage<scalar>(matB, sampler,
                                                           globalCol,
                                                           VECTOR_LENGTH * idx + j + pIdx * K),
                                         !_spQuad));

                // inner product accumulation
                assignMAD(os, loopOrder(), accum, valA, valB);

            os << EndBlock();

            if (generalizedMatmul()) {
                const ConstantValue<std::string> outC = matC_buf + multHeight(N) * globalRow + globalCol + pIdx * M * N / VECTOR_LENGTH;
                for (size_t i = 0; i < blockHeight(); i++)
                    os << assign(*(outC + i * (N / VECTOR_LENGTH)),
                                 true // isfloat<scalar>()
                                     ? MADValue(CastValue<scalarN>(alpha),
                                                accum[i],
                                                CastValue<scalarN>(beta) * *(outC + i * (N / VECTOR_LENGTH)))
                                     : CastValue<scalarN>(alpha) * accum[i] + CastValue<scalarN>(beta) * *(outC + i * (N / VECTOR_LENGTH)));
            } else {
                const ConstantValue<std::string> valueGlobalCol = globalID()
                                                                      ? globalCol
                                                                      : groupSize() * blockCol + col;
                const ConstantValue<std::string> valueGlobalRow = globalID()
                                                                      ? globalRow
                                                                      : groupSize() * blockRow + row;
                for (size_t i = 0; i < blockHeight(); i++)
                    os << WriteImage<scalar>(matC_img,
                                             valueGlobalCol,
                                             blockHeight() * valueGlobalRow + i + pIdx * M,
                                             ReinterpretValue<uint, 4>(accum[i], !_spQuad));
            }

        if (1 != packedCalc()) os << EndBlock();

        return os << EndBlock(); // end function body
    }
};

}; // namespace

#endif
