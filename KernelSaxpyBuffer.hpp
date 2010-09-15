#ifndef _GATLAS_KERNEL_SAXPY_BUFFER_HPP_
#define _GATLAS_KERNEL_SAXPY_BUFFER_HPP_

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

// saxpy using memory buffers
template <typename SCALAR, size_t VECTOR_LENGTH>
class KernelSaxpyBuffer : public KernelBaseSaxpy,
                          protected SaxpyParamInlineMN
{
    typedef SCALAR scalar;
    typedef VecType< SCALAR, VECTOR_LENGTH > scalarN;

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
    KernelSaxpyBuffer()
        : KernelBaseSaxpy(),
          SaxpyParamInlineMN(getExtraParameter()),
          _handleX(-1),
          _handleY(-1),
          _handleZ(-1),
          _paranoidCheck(false),
          _paranoidZ(NULL)
    { }

    ~KernelSaxpyBuffer() {
        delete[] _paranoidZ;
    }

    std::string kernelName() const {
        std::stringstream ss;
        ss << "saxpybuffer" << nameof<SCALAR>() << VECTOR_LENGTH;
        return ss.str();
    }

    void paranoidCheck() {
        _paranoidCheck = true;
        delete[] _paranoidZ;
        _paranoidZ = new scalar[bufferSize()];
    }

    bool syncOutput(OCLApp& oclApp) {
        return syncBufferFromDevice(oclApp, _handleZ);
    }

    bool checkOutput(OCLApp& oclApp, const bool printOutput) {
        if (_paranoidCheck) {
            return checkBuffer<scalar>(oclApp, _handleZ, bufferSize(), _paranoidZ, printOutput);
        } else {
            const scalar testValue = 1 * 1 + 1;
            return checkBuffer<scalar>(oclApp, _handleZ, bufferSize(), testValue, printOutput);
        }
    }

    bool setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) {

        // buffer allocation
        if (-1 == _handleX || -1 == _handleY || -1 == _handleZ || dimChanged() || packedChanged()) {
            oclApp.releaseBuffers();
            _handleX = createBufferR<scalar, VECTOR_LENGTH>(oclApp, bufferSize(), "X", 1);
            _handleY = createBufferR<scalar, VECTOR_LENGTH>(oclApp, bufferSize(), "Y", 1);
            _handleZ = createBufferW<scalar, VECTOR_LENGTH>(oclApp, bufferSize(), "Z", 0);
            if (-1 == _handleX || -1 == _handleY || -1 == _handleZ) return false; // failure
        } else {
            // X and Y
            if (syncInput) {
                if (!syncBufferToDevice(oclApp, _handleX)) return false;
                if (!syncBufferToDevice(oclApp, _handleY)) return false;
            }
            // Z
            if (!clearBuffer<scalar>(oclApp, _handleZ)) return false;
        }

        const scalar alpha = _paranoidCheck ? posrand<scalar>() : 1;

        // paranoid check
        if (_paranoidCheck) {

            // fill X and Y with random values
            if (fillrandBuffer<scalar>(oclApp, _handleX, bufferSize()) &&
                fillrandBuffer<scalar>(oclApp, _handleY, bufferSize())) {

                const scalar *ptrX = oclApp.bufferPtr<scalar>(_handleX);
                const scalar *ptrY = oclApp.bufferPtr<scalar>(_handleY);

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
            setArgGlobal(oclApp, kernelHandle, argIndex++, _handleZ, "Z") &&
            setArgGlobal(oclApp, kernelHandle, argIndex++, _handleX, "X") &&
            setArgGlobal(oclApp, kernelHandle, argIndex++, _handleY, "Y");
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
        Var< scalarN* >       Z("Z", GLOBAL, kernelDecl);
        Var< const scalarN* > X("X", GLOBAL, kernelDecl);
        Var< const scalarN* > Y("Y", GLOBAL, kernelDecl);
        Var< const int >      M("M", kernelDecl, inlineMN(), dimM());
        Var< const int >      N("N", kernelDecl, inlineMN() || (vectorLength() == dimN()), dimN());
        Var< const scalar >   alpha("alpha", kernelDecl);

        // begin function body
        os << kernelDecl;

        // packed kernel support
        Var< int > pIdx("pIdx", 1 == packedCalc(), 0);
        if (1 != packedCalc()) os << ForLoop(pIdx, packedCalc(), 1);

            if (vectorLength() == dimN()) {
                // 1D work groups
                const ConstantValue<std::string> outZ = Z
                                                      + blockHeight() * groupHeight() * blockRow
                                                      + row
                                                      + pIdx * M;

                const ConstantValue<std::string> inX = X
                                                     + blockHeight() * groupHeight() * blockRow
                                                     + row
                                                     + pIdx * M;

                const ConstantValue<std::string> inY = Y
                                                     + blockHeight() * groupHeight() * blockRow
                                                     + row
                                                     + pIdx * M;

                for (size_t i = 0; i < blockHeight(); i++)
                    os << assign(*(outZ + i * groupHeight()),
                                 MADValue(CastValue<scalarN>(alpha),
                                          *(inX + i * groupHeight()),
                                          *(inY + i * groupHeight())));
            } else {
                // 2D work groups
                const ConstantValue<std::string> outZ = Z
                                                      + multHeight(N) * groupHeight() * blockRow
                                                      + wholeWidth() * groupWidth() * blockCol
                                                      + row * N / VECTOR_LENGTH
                                                      + col
                                                      + pIdx * M * N / VECTOR_LENGTH;

                const ConstantValue<std::string> inX = X
                                                     + multHeight(N) * groupHeight() * blockRow
                                                     + wholeWidth() * groupWidth() * blockCol
                                                     + row * N / VECTOR_LENGTH
                                                     + col
                                                     + pIdx * M * N / VECTOR_LENGTH;

                const ConstantValue<std::string> inY = Y
                                                     + multHeight(N) * groupHeight() * blockRow
                                                     + wholeWidth() * groupWidth() * blockCol
                                                     + row * N / VECTOR_LENGTH
                                                     + col
                                                     + pIdx * M * N / VECTOR_LENGTH;

                for (size_t i = 0; i < blockHeight(); i++)
                    for (size_t j = 0; j < wholeWidth(); j++)
                        os << assign(*(outZ + i * groupHeight() * N / VECTOR_LENGTH + j * groupWidth()),
                                     MADValue(CastValue<scalarN>(alpha),
                                              *(inX + i * groupHeight() * N / VECTOR_LENGTH + j * groupWidth()),
                                              *(inY + i * groupHeight() * N / VECTOR_LENGTH + j * groupWidth())));
            }

        if (1 != packedCalc()) os << EndBlock();

        return os << EndBlock(); // end function body
    }
};

}; // namespace

#endif
