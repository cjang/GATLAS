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

bool fillrandImage(OCLApp& oclApp, const size_t imageIndex, const size_t length) {
    float *ptr = oclApp.imagePtr(imageIndex);
    fillrand<float>(ptr, length);
    const int syncImg = oclApp.enqueueWriteImage(imageIndex);
    if (-1 == syncImg || !oclApp.wait(syncImg)) {
        std::cerr << "error: random fill image " << imageIndex << std::endl;
        return false;
    }
    return true;
}

bool checkImage(OCLApp& oclApp,
                const size_t imageIndex,
                const size_t length,
                const float  testValue,
                const bool   printOutput) {
    const float *ptr = oclApp.imagePtr(imageIndex);

    // quick and primitive test here
    // e.g. if A and B are all 1s, then each element of C equals the matrix size
    const bool goodElements = checkArray(ptr, length, testValue);

    // print output matrix for debugging
    if (printOutput) printArray(ptr, length);

    return goodElements;
}

bool checkImage(OCLApp& oclApp,
                const size_t imageIndex,
                const size_t width,
                const size_t height,
                const float  testValue,
                const bool   printOutput) {
    const float *ptr = oclApp.imagePtr(imageIndex);

    // quick and primitive test here
    // e.g. if A and B are all 1s, then each element of C equals the matrix size
    const bool goodElements = checkArray(ptr, width * height, testValue);

    // print output matrix for debugging
    if (printOutput) printArray(ptr, width, height);

    return goodElements;
}

bool checkImage(OCLApp&       oclApp,
                const size_t  imageIndex,
                const size_t  length,
                const float  *testImage,
                const bool    printOutput) {
    const float *ptr = oclApp.imagePtr(imageIndex);
    const double diff = absdiff(ptr, testImage, length);
    std::cerr << "absdiff: " << diff << "\t";
    if (printOutput) printDiff(ptr, testImage, length);
    return diff < static_cast<double>(1) / (length);
}

bool checkImage(OCLApp&       oclApp,
                const size_t  imageIndex,
                const size_t  width,
                const size_t  height,
                const float  *testImage,
                const bool    printOutput) {
    const float *ptr = oclApp.imagePtr(imageIndex);
    const double diff = absdiff(ptr, testImage, width * height);
    std::cerr << "absdiff: " << diff << "\t";
    if (printOutput) printDiff(ptr, testImage, width, height);
    return diff < static_cast<double>(1) / (width * height);
}

int createImageR(OCLApp& oclApp, const size_t width, const size_t height,
                 const std::string& argName,
                 const float value, const bool pinned) {
    float *ptr = alloc_memalign<float, 4>(width * height);
    if (NULL == ptr) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
        return -1;
    }
    fillconst<float>(ptr, value, width * height);
    const int imgHandle = oclApp.createImage(width/4, height, OCLApp::READ, ptr, pinned);
    if (-1 == imgHandle) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
    } else {
        oclApp.ownImage(imgHandle);
    }
    return imgHandle;
}

int createImageW(OCLApp& oclApp, const size_t width, const size_t height,
                 const std::string& argName,
                 const float value, const bool pinned) {
    float *ptr = alloc_memalign<float, 4>(width * height);
    if (NULL == ptr) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
        return -1;
    }
    fillconst<float>(ptr, value, width * height);
    const int imgHandle = oclApp.createImage(width/4, height, OCLApp::WRITE, ptr, pinned);
    if (-1 == imgHandle) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
    } else {
        oclApp.ownImage(imgHandle);
    }
    return imgHandle;
}

int createImageRW(OCLApp& oclApp, const size_t width, const size_t height,
                  const std::string& argName,
                  const float value, const bool pinned) {
    float *ptr = alloc_memalign<float, 4>(width * height);
    if (NULL == ptr) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
        return -1;
    }
    fillconst<float>(ptr, value, width * height);
    const int imgHandle = oclApp.createImage(width/4, height, OCLApp::READWRITE, ptr, pinned);
    if (-1 == imgHandle) {
        std::cerr << "error: OCL create image for " << argName << std::endl;
    } else {
        oclApp.ownImage(imgHandle);
    }
    return imgHandle;
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
