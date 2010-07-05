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
    typedef VecType< SCALAR, VECTOR_LENGTH > scalarN;

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
        return "KernelMatmulImage";
    }

    void paranoidCheck() {
        _paranoidCheck = true;
        delete[] _paranoidC;
        _paranoidC = new scalar[dimM() * dimN()];
    }

    bool syncOutput(OCLApp& oclApp) {
        return generalizedMatmul()
                   ? syncBufferFromDevice(oclApp, _handleC)
                   : syncImageFromDevice(oclApp, _handleC);
    }

    bool checkOutput(OCLApp& oclApp, const bool printOutput) {
        if (_paranoidCheck) {
            return generalizedMatmul()
                       ? checkBuffer<scalar>(oclApp, _handleC, dimN(), dimM(), _paranoidC, printOutput)
                       : checkImage(oclApp, _handleC, dimN(), dimM(), _paranoidC, printOutput);
        } else {
            const scalar testValue = dimK();
            return generalizedMatmul()
                       ? checkBuffer<scalar>(oclApp, _handleC, dimN(), dimM(), testValue, printOutput)
                       : checkImage(oclApp, _handleC, dimN(), dimM(), testValue, printOutput);
        }
    }

    bool setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) {

        // buffer allocation
        if (-1 == _handleA || -1 == _handleB || -1 == _handleC || dimChanged() || layoutChanged()) {
            oclApp.releaseImages();
            if (generalizedMatmul()) oclApp.releaseBuffers();
            _handleA = transposeA()
                           ? createImageR(oclApp, dimM(), dimK(), "matA", 1)
                           : createImageR(oclApp, dimK(), dimM(), "matA", 1);
            _handleB = transposeB()
                           ? createImageR(oclApp, dimK(), dimN(), "matB", 1)
                           : createImageR(oclApp, dimN(), dimK(), "matB", 1);
            _handleC = generalizedMatmul()
                           ? createBufferRW<scalar, VECTOR_LENGTH>(oclApp, dimN() * dimM(), "matC", 0)
                           : createImageW(oclApp, dimN(), dimM(), "matC", 0);
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
            if (fillrandImage(oclApp, _handleA, dimM() * dimK()) &&
                fillrandImage(oclApp, _handleB, dimK() * dimN()) &&
                (generalizedMatmul()
                     ? fillrandBuffer<scalar>(oclApp, _handleC, dimM() * dimN())
                     : true)) {

                const scalar *ptrA = oclApp.imagePtr(_handleA);
                const scalar *ptrB = oclApp.imagePtr(_handleB);
                const scalar *ptrC = generalizedMatmul() ? oclApp.bufferPtr<scalar>(_handleC) : NULL;

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
                if (generalizedMatmul())
                    for (size_t i = 0; i < dimM() * dimN(); i++)
                        _paranoidC[i] = alpha * _paranoidC[i] + beta * ptrC[i];
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
            if (transposeA())
                for (size_t j = 0; j < blockHeight(); j++) {
                    const size_t blockNum = j / VECTOR_LENGTH;
                    const size_t blockIdx = j % VECTOR_LENGTH;
                    os << assign(valA[j],
                                 ReadImage(matA, sampler,
                                           wholeHeight() * globalRow + blockNum,
                                           VECTOR_LENGTH * idx + blockIdx));
                }
            else
                for (size_t j = 0; j < blockHeight(); j++)
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

        if (generalizedMatmul()) {
            const ConstantValue<std::string> outC = matC_buf + multHeight(N) * globalRow + globalCol;
            for (size_t i = 0; i < blockHeight(); i++)
                os << assign(*(outC + i * (N / VECTOR_LENGTH)),
                             MADValue(CastValue<scalarN>(alpha),
                                      accum[i],
                                      CastValue<scalarN>(beta) * *(outC + i * (N / VECTOR_LENGTH))));
        } else {
            const ConstantValue<std::string> valueGlobalCol = globalID()
                                                                  ? globalCol
                                                                  : groupSize() * blockCol + col;
            const ConstantValue<std::string> valueGlobalRow = globalID()
                                                                  ? globalRow
                                                                  : groupSize() * blockRow + row;
            for (size_t i = 0; i < blockHeight(); i++)
                os << WriteImage(matC_img,
                                 valueGlobalCol,
                                 blockHeight() * valueGlobalRow + i,
                                 accum[i]);
        }

        return os << EndBlock(); // end function body
    }
};

}; // namespace

#endif
