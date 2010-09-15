#ifndef _GATLAS_KERNEL_SAXPY_IMAGE_HPP_
#define _GATLAS_KERNEL_SAXPY_IMAGE_HPP_

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

#include "declare_namespace"

// saxpy using images
template <typename SCALAR, size_t VECTOR_LENGTH>
class KernelSaxpyImage : public KernelBaseSaxpy,
                         protected SaxpyParamInlineMN,
                         protected SaxpyParamGlobalID
{
    typedef SCALAR scalar;
    typedef VecType< SCALAR, VECTOR_LENGTH > scalarN;

    const bool _spQuad; // true when SCALAR is float and VECTOR_LENGTH is 4

    int _handleX;
    int _handleY;
    int _handleZ;

    // compare matrix multiply to reference results
    bool _paranoidCheck;
    scalar *_paranoidZ;

    size_t bufferSize() const {
        return packedCalc() * dimM() * dimN();
    }

public:
    KernelSaxpyImage()
        : KernelBaseSaxpy(),
          SaxpyParamInlineMN(getExtraParameter()),
          SaxpyParamGlobalID(getExtraParameter()),
          _spQuad(isfloat<SCALAR>() && 4 == VECTOR_LENGTH),
          _handleX(-1),
          _handleY(-1),
          _handleZ(-1),
          _paranoidCheck(false),
          _paranoidZ(NULL)
    { }

    ~KernelSaxpyImage() {
        delete[] _paranoidZ;
    }

    std::string kernelName() const {
        std::stringstream ss;
        ss << "saxpyimage" << nameof<SCALAR>() << VECTOR_LENGTH;
        return ss.str();
    }

    void paranoidCheck() {
        _paranoidCheck = true;
        delete[] _paranoidZ;
        _paranoidZ = new scalar[bufferSize()];
    }

    bool syncOutput(OCLApp& oclApp) {
        return syncImageFromDevice(oclApp, _handleZ);
    }

    bool checkOutput(OCLApp& oclApp, const bool printOutput) {
        if (_paranoidCheck) {
            return checkImage<scalar>(oclApp, _handleZ, dimN(), packedCalc() * dimM(), _paranoidZ, printOutput);
        } else {
            const scalar testValue = 1 * 1 + 1;
            return checkImage<scalar>(oclApp, _handleZ, dimN(), packedCalc() * dimM(), testValue, printOutput);
        }
    }

    bool setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) {

        // buffer allocation
        if (-1 == _handleX || -1 == _handleY || -1 == _handleZ || dimChanged() || packedChanged()) {
            oclApp.releaseImages();
            _handleX = createImageR<scalar>(oclApp, dimN(), packedCalc() * dimM(), "X", 1);
            _handleY = createImageR<scalar>(oclApp, dimN(), packedCalc() * dimM(), "Y", 1);
            _handleZ = createImageW<scalar>(oclApp, dimN(), packedCalc() * dimM(), "Z", 0);
            if (-1 == _handleX || -1 == _handleY || -1 == _handleZ) return false; // failure
        } else {
            // X and Y
            if (syncInput) {
                if (!syncImageToDevice(oclApp, _handleX)) return false;
                if (!syncImageToDevice(oclApp, _handleY)) return false;
            }
            // Z
            if (!clearImage(oclApp, _handleZ)) return false;
        }

        const scalar alpha = _paranoidCheck ? posrand<scalar>() : 1;

        // paranoid check
        if (_paranoidCheck) {

            // fill X and Y with random values
            if (fillrandImage<scalar>(oclApp, _handleX, bufferSize()) &&
                fillrandImage<scalar>(oclApp, _handleY, bufferSize())) {

                const scalar *ptrX = oclApp.imagePtr<scalar>(_handleX);
                const scalar *ptrY = oclApp.imagePtr<scalar>(_handleY);

                fillconst<scalar>(_paranoidZ, 0, bufferSize());

                // calculate Z
                for (size_t i = 0; i < bufferSize(); i++)
                    _paranoidZ[i] = alpha * ptrX[i] + ptrY[i];

            } else {
                std::cerr << "error: failed to fill input matrices with random values" << std::endl;
            }
        }

        // set kernel arguments
        size_t argIndex = 0;
        bool rc =
            setArgImage(oclApp, kernelHandle, argIndex++, _handleZ, "Z") &&
            setArgImage(oclApp, kernelHandle, argIndex++, _handleX, "X") &&
            setArgImage(oclApp, kernelHandle, argIndex++, _handleY, "Y");
        if (! inlineMN()) {
            rc = rc && setArgValue<int>(oclApp, kernelHandle, argIndex++, dimM(), "M");
            if (vectorLength() != dimN()) rc = rc && setArgValue<int>(oclApp, kernelHandle, argIndex++, dimN(), "N");
        }
        rc = rc &&
            setArgValue<scalar>(oclApp, kernelHandle, argIndex++, alpha, "alpha");
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
        Var< image2d_t > Z("Z", WRITEONLY, kernelDecl);
        Var< image2d_t > X("X", READONLY, kernelDecl);
        Var< image2d_t > Y("Y", READONLY, kernelDecl);
        Var< const int > M("M", kernelDecl, inlineMN(), dimM());
        Var< const int > N("N", kernelDecl, inlineMN() || (vectorLength() == dimN()), dimN());
        Var< const scalar > alpha("alpha", kernelDecl);

        // begin function body
        os << kernelDecl;

        // image sampler
        Var< const sampler_t > sampler("sampler");
        os << declare(sampler, ImageSampler());

        // packed kernel support
        Var< int > pIdx("pIdx", 1 == packedCalc(), 0);
        if (1 != packedCalc()) os << ForLoop(pIdx, packedCalc(), 1);

            if (vectorLength() == dimN()) {
                // 1D work groups
                const ConstantValue<std::string> valueGlobalRow = globalID()
                                                                      ? globalRow
                                                                      : groupHeight() * blockRow + row;

                for (size_t i = 0; i < blockHeight(); i++)
                    os << WriteImage<scalar>(Z,
                                             ConstantValue<const int>(0),
                                             blockHeight() * valueGlobalRow + i + pIdx * M,
                                             ReinterpretValue<uint, 4>(
                                                 MADValue(CastValue<scalarN>(alpha),
                                                          ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                                              ReadImage<scalar>(X, sampler,
                                                                                ConstantValue<const int>(0), blockHeight() * valueGlobalRow + i + pIdx * M),
                                                              !_spQuad),
                                                          ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                                              ReadImage<scalar>(Y, sampler,
                                                                                ConstantValue<const int>(0), blockHeight() * valueGlobalRow + i + pIdx * M),
                                                              !_spQuad)),
                                                 !_spQuad));

            } else {
                // 2D work groups
                const ConstantValue<std::string> valueGlobalCol = globalID()
                                                                      ? globalCol
                                                                      : groupWidth() * blockCol + col;

                const ConstantValue<std::string> valueGlobalRow = globalID()
                                                                      ? globalRow
                                                                      : groupHeight() * blockRow + row;

                for (size_t j = 0; j < blockWidth(); j++)
                for (size_t i = 0; i < blockHeight(); i++)
                    os << WriteImage<scalar>(Z,
                                             blockWidth() * valueGlobalCol + j,
                                             blockHeight() * valueGlobalRow + i + pIdx * M,
                                             ReinterpretValue<uint, 4>(
                                                 MADValue(CastValue<scalarN>(alpha),
                                                          ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                                              ReadImage<scalar>(X, sampler,
                                                                                blockWidth() * valueGlobalCol + j,
                                                                                blockHeight() * valueGlobalRow + i + pIdx * M),
                                                              !_spQuad),
                                                          ReinterpretValue<SCALAR, VECTOR_LENGTH>(
                                                              ReadImage<scalar>(Y, sampler,
                                                                                blockWidth() * valueGlobalCol + j,
                                                                                blockHeight() * valueGlobalRow + i + pIdx * M),
                                                              !_spQuad)),
                                                 !_spQuad));
            }

        if (1 != packedCalc()) os << EndBlock();

        return os << EndBlock(); // end function body
    }
};

}; // namespace

#endif
