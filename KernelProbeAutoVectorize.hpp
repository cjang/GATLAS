#ifndef _GATLAS_KERNEL_PROBE_AUTO_VECTORIZE_HPP_
#define _GATLAS_KERNEL_PROBE_AUTO_VECTORIZE_HPP_

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

#include "OCLAppUtil.hpp"
#include "GatlasBenchmark.hpp"

#include "declare_namespace"

template <typename T>
class KernelProbeAutoVectorize : public KernelInterface
{
    int  _handle;
    bool _useAttrAutoVec;

public:
    // explicitly vectorized kernels
    typedef float scalar;
    static const size_t VECTOR_LENGTH = 4;
    typedef VecType<scalar, VECTOR_LENGTH> scalarN;

    KernelProbeAutoVectorize() : _handle(-1), _useAttrAutoVec(true) { }

    std::string kernelName() const { return "KernelProbeAutoVectorize"; }

    std::string desc() const {
        return _useAttrAutoVec
                   ? func_string("vec_type_hint", nameof<T>())
                   : "no_hint";
    }

    size_t numberFlops() const { return 0; }

    std::vector<size_t> parameters(const bool useAttrAutoVec) const {
        std::vector<size_t> args;
        args.push_back(useAttrAutoVec ? 1 : 0);
        return args;
    }

    bool setParams(const std::vector<size_t>& args) {
        if (args.size() < 1) return false;
        _useAttrAutoVec = (1 == args[0]);
        return true;
    }

    bool setArgs(OCLApp& oclApp, const size_t kernelHandle, const bool syncInput) {
        if (-1 == _handle) {
            oclApp.releaseBuffers();
            _handle = createBufferW<scalar, VECTOR_LENGTH>(oclApp, VECTOR_LENGTH, "outDummy", 0);
            if (-1 == _handle) return false; // failure
        }

        size_t argIndex = 0;
        return
            setArgGlobal(oclApp, kernelHandle, argIndex++, _handle, "outDummy") &&
            setArgValue<scalar>(oclApp, kernelHandle, argIndex++, 1, "inDummy");
    }

    bool syncOutput(OCLApp& oclApp) {
        return syncBufferFromDevice(oclApp, _handle);
    }

    bool checkOutput(OCLApp& oclApp, const bool printOutput) {
        return checkBuffer<scalar>(oclApp, _handle, VECTOR_LENGTH, 1, false);
    }

    void paranoidCheck() { }

    std::vector<size_t> globalWorkItems() const {
        std::vector<size_t> dims;
        dims.push_back(1);
        dims.push_back(1);
        return dims;
    }

    std::vector<size_t> localWorkItems() const {
        std::vector<size_t> dims;
        dims.push_back(1);
        dims.push_back(1);
        return dims;
    }

    // prints the kernel source
    std::ostream& print(std::ostream& os) const {

        // kernel function attributes
        AutoVectorize< scalarN > attrAutoVec;
        FunctionDeclaration kernelDecl(kernelName());
        kernelDecl.returnType<void>();
        kernelDecl.qualify(KERNEL);
        if (_useAttrAutoVec) kernelDecl.qualify(attrAutoVec);

        // kernel arguments
        Var< scalarN* > outDummy("outDummy", GLOBAL, kernelDecl);
        Var< scalar >   inDummy("inDummy", kernelDecl);

        // begin function body
        os << kernelDecl;

        // just one assignment so the kernel does something (even if trivial)
        os << assign(*outDummy, CastValue<scalarN>(inDummy));

        return os << EndBlock(); // end function body
    }
};

}; // namespace

#endif
