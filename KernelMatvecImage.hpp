#ifndef _GATLAS_KERNEL_MATVEC_IMAGE_HPP_
#define _GATLAS_KERNEL_MATVEC_IMAGE_HPP_

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

// matrix vector multiply using images
template <typename SCALAR, size_t VECTOR_LENGTH>
class KernelMatvecImage : public KernelBaseMatvec,
                          protected MatvecParamInlineMN,
                          protected MatvecParamGlobalID
{
    typedef SCALAR scalar;
    typedef VecType< SCALAR, VECTOR_LENGTH > scalarN;

    const bool _spQuad; // true when SCALAR is float and VECTOR_LENGTH is 4

    int _handleA;
    int _handleB;
    int _handleC;

    // compare matrix multiply to reference results
    bool _paranoidCheck;
    scalar *_paranoidC;

public:
    KernelMatvecImage()
        : KernelBaseMatvec(),
          MatvecParamInlineMN(getExtraParameter()),
          MatvecParamGlobalID(getExtraParameter()),
          _spQuad(isfloat<SCALAR>() && 4 == VECTOR_LENGTH),
          _handleA(-1),
          _handleB(-1),
          _handleC(-1),
          _paranoidCheck(false),
          _paranoidC(NULL)
    { }

    ~KernelMatvecImage() {
        delete[] _paranoidC;
    }

    std::string kernelName() const {
        std::stringstream ss;
        ss << "matvecimage" << nameof<SCALAR>() << VECTOR_LENGTH;
        return ss.str();
    }

    void paranoidCheck() {
        _paranoidCheck = true;
        delete[] _paranoidC;
        _paranoidC = new scalar[packedCalc() * dimM()];
    }

    bool syncOutput(OCLApp& oclApp) {
        return generalizedMatvec()
                   ? syncBufferFromDevice(oclApp, _handleC)
                   : syncImageFromDevice(oclApp, _handleC);
    }

    bool checkOutput(OCLApp& oclApp, const bool printOutput) {
        if (_paranoidCheck) {
            return generalizedMatvec()
                       ? checkBuffer<scalar>(oclApp, _handleC, packedCalc() * dimM(), _paranoidC, printOutput)
                       : checkImage<scalar>(oclApp, _handleC, packedCalc() * dimM(), _paranoidC, printOutput);
        } else {
            const scalar testValue = dimN();
            return generalizedMatvec()
                       ? checkBuffer<scalar>(oclApp, _handleC, packedCalc() * dimM(), testValue, printOutput)
                       : checkImage<scalar>(oclApp, _handleC, packedCalc() * dimM(), testValue, printOutput);
        }
    }

    bool setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) {

        // buffer allocation
        if (-1 == _handleA || -1 == _handleB || -1 == _handleC || dimChanged() || layoutChanged() || packedChanged()) {
	    oclApp.releaseImages();
            if (generalizedMatvec()) oclApp.releaseBuffers();
            _handleA = transposeA()
                           // stack matrices in height so row major ordering is maintained
                           ? createImageR<scalar>(oclApp, dimM(), packedCalc() * dimN(), "matA", 1)
                           : createImageR<scalar>(oclApp, dimN(), packedCalc() * dimM(), "matA", 1);
            // stack vectors in height to maintain row major ordering
            _handleB = createImageR<scalar>(oclApp, dimN(), packedCalc(), "vecB", 1);
            _handleC = generalizedMatvec()
                           ? createBufferRW<scalar, VECTOR_LENGTH>(oclApp, packedCalc() * dimM(), "vecC", 0)
                           // stack vectors in height to maintain row major ordering
                           : createImageW<scalar>(oclApp, dimM(), packedCalc(), "vecC", 0);
            if (-1 == _handleA || -1 == _handleB || -1 == _handleC) return false; // failure
        } else {
            // matrix A and vector B
            if (syncInput) {
                if (!syncImageToDevice(oclApp, _handleA)) return false;
                if (!syncImageToDevice(oclApp, _handleB)) return false;
            }
            // vector C
            if (generalizedMatvec()) {
                if (!clearBuffer<scalar>(oclApp, _handleC)) return false;
            } else {
                if (!clearImage(oclApp, _handleC)) return false;
            }
        }

        const scalar alpha = (_paranoidCheck && generalizedMatvec()) ? posrand<scalar>() : 1;
        const scalar beta = (_paranoidCheck && generalizedMatvec()) ? posrand<scalar>() : 1;

        // paranoid check
        if (_paranoidCheck) {

            // fill matrix A and vector B with random values
            if (fillrandImage<scalar>(oclApp, _handleA, packedCalc() * dimM() * dimN()) &&
                fillrandImage<scalar>(oclApp, _handleB, packedCalc() * dimN()) &&
                (generalizedMatvec()
                     ? fillrandBuffer<scalar>(oclApp, _handleC, packedCalc() * dimM())
                     : true)) {

                const scalar *ptrA = oclApp.imagePtr<scalar>(_handleA);
                const scalar *ptrB = oclApp.imagePtr<scalar>(_handleB);
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
        const size_t numberElemsTmpB = dimN();
        size_t argIndex = 0;
        bool rc =
            (generalizedMatvec()
                 ? setArgGlobal(oclApp, kernelHandle, argIndex++, _handleC, "vecC")
                 : setArgImage(oclApp, kernelHandle, argIndex++, _handleC, "vecC")) &&
            setArgImage(oclApp, kernelHandle, argIndex++, _handleA, "matA") &&
            setArgImage(oclApp, kernelHandle, argIndex++, _handleB, "vecB");
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
        Var< image2d_t > vecC_img("vecC", WRITEONLY, kernelDecl, !generalizedMatvec());
        Var< scalarN* >  vecC_buf("vecC", GLOBAL, kernelDecl, generalizedMatvec());
        Var< image2d_t > matA("matA", READONLY, kernelDecl);
        Var< image2d_t > vecB("matB", READONLY, kernelDecl);
        Var< const int > M("M", kernelDecl, inlineMN(), dimM());
        Var< const int > N("N", kernelDecl, inlineMN(), dimN());
        Var< const scalar >   alpha("alpha", kernelDecl, generalizedMatvec());
        Var< const scalar >   beta("beta", kernelDecl, generalizedMatvec());

        // begin function body
        os << kernelDecl;

        // image sampler
        Var< const sampler_t > sampler("sampler");
        os << declare(sampler, ImageSampler());

        // accumulate inner product sum
        Vector< scalarN > accum("accum", wholeHeight());
        os << declare(accum);

        // current value from matA (for matrix vector outer product)
        Var< scalarN > valA("valA");
        os << declare(valA);

        // current value from vecB (for matrix vector outer product)
        Var< scalarN > valB("valB");
        os << declare(valB);

        // packed kernel support
        Var< int > pIdx("pIdx", 1 == packedCalc(), 0);
        if (1 != packedCalc()) os << ForLoop(pIdx, packedCalc(), 1);

            // set accumulators to zero
            os << assign(accum, CastValue<scalarN>(ConstantValue<scalar>(0)));

            // outer loop over vector B
            Var< int > idx("idx");
            os << ForLoop(idx, N / VECTOR_LENGTH, 1);

                // read value from vector B
                os << assign(valB,
                             ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                 ReadImage<scalar>(vecB, sampler,
                                                   idx,
                                                   pIdx),
                                 !_spQuad));

                // inner loop over matrix A
                for (size_t j = 0; j < blockHeight(); j++) {
                    const size_t blockNum = j / VECTOR_LENGTH;
                    const size_t blockIdx = j % VECTOR_LENGTH;
                    if (transposeA()) {
                        os << assign(valA,
                                     ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                         ReadImage<scalar>(matA, sampler,
                                                           wholeHeight() * globalRow + blockNum,
                                                           VECTOR_LENGTH * idx + blockIdx + pIdx * N),
                                         !_spQuad));
                        for (size_t k = 0; k < VECTOR_LENGTH; k++)
                            os << assignMAD(accum[blockNum], valA, valB, blockIdx, k);
                    } else {
                        os << assign(valA,
                                     ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                         ReadImage<scalar>(matA, sampler,
                                                           idx,
                                                           blockHeight() * globalRow + j + pIdx * M),
                                         !_spQuad));
                        for (size_t k = 0; k < VECTOR_LENGTH; k++)
                            os << assignMAD(accum[blockNum], valA, valB, blockIdx, k);
                    }
                }

            os << EndBlock();

            if (generalizedMatvec()) {
                const ConstantValue<std::string> outC = vecC_buf + wholeHeight() * globalRow + pIdx * M / VECTOR_LENGTH;
                for (size_t i = 0; i < wholeHeight(); i++)
                    os << assign(*(outC + i),
                                 MADValue(CastValue<scalarN>(alpha),
                                          accum[i],
                                          CastValue<scalarN>(beta) * *(outC +i)));
            } else {
                const ConstantValue<std::string> valueGlobalRow = globalID()
                                                                      ? globalRow
                                                                      : groupSize() * blockRow + row;
                for (size_t i = 0; i < wholeHeight(); i++)
                    os << WriteImage<scalar>(vecC_img,
                                             wholeHeight() * valueGlobalRow + i,
                                             pIdx,
                                             ReinterpretValue<uint, 4>(accum[i], !_spQuad));
            }

        if (1 != packedCalc()) os << EndBlock();

        return os << EndBlock(); // end function body
    }
};

}; // namespace

#endif
