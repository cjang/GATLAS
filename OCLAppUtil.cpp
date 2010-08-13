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

#include <iostream>
#include <string>
#include "OCLAppUtil.hpp"

#include "declare_namespace"

bool clearImage(OCLApp& oclApp, const size_t imageIndex) {
    oclApp.memsetImage(imageIndex, 0);
    const int syncImg = oclApp.enqueueWriteImage(imageIndex);
    if (-1 == syncImg || !oclApp.wait(syncImg)) {
        std::cerr << "error: reset output image " << imageIndex << std::endl;
        return false;
    }
    return true;
}

bool syncBufferToDevice(OCLApp& oclApp, const size_t bufferIndex) {
    // retrieve output
    const int syncBuf = oclApp.enqueueWriteBuffer(bufferIndex);
    if (-1 == syncBuf || !oclApp.wait(syncBuf)) {
        std::cerr << "error: send input buffer " << bufferIndex << std::endl;
        return false;
    }
    return true;
}

bool syncBufferFromDevice(OCLApp& oclApp, const size_t bufferIndex) {
    // retrieve output
    const int syncBuf = oclApp.enqueueReadBuffer(bufferIndex);
    if (-1 == syncBuf || !oclApp.wait(syncBuf)) {
        std::cerr << "error: retrieve output buffer " << bufferIndex << std::endl;
        return false;
    }
    return true;
}

bool syncImageToDevice(OCLApp& oclApp, const size_t imageIndex) {
    // retrieve output
    const int syncImg = oclApp.enqueueWriteImage(imageIndex);
    if (-1 == syncImg || !oclApp.wait(syncImg)) {
        std::cerr << "error: send input image " << imageIndex << std::endl;
        return false;
    }
    return true;
}

bool syncImageFromDevice(OCLApp& oclApp, const size_t imageIndex) {
    // retrieve output
    const int syncImg = oclApp.enqueueReadImage(imageIndex);
    if (-1 == syncImg || !oclApp.wait(syncImg)) {
        std::cerr << "error: retrieve output image " << imageIndex << std::endl;
        return false;
    }
    return true;
}

bool setArgImage(OCLApp& oclApp, const size_t kernelHandle, const size_t argIndex, const size_t imgHandle,
                 const std::string& argName) {
    if (!oclApp.setArgImage(kernelHandle, argIndex, imgHandle)) {
        std::cerr << "error: set image kernel argument " << argName << std::endl;
        return false;
    }
    return true;
}

bool setArgGlobal(OCLApp& oclApp, const size_t kernelHandle, const size_t argIndex, const size_t bufHandle,
                  const std::string& argName) {
    if (!oclApp.setArgGlobal(kernelHandle, argIndex, bufHandle)) {
        std::cerr << "error: set global kernel argument " << argName << std::endl;
        return false;
    }
    return true;
}

}; // namespace
