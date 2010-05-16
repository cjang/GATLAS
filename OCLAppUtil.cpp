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
