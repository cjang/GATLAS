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

#include "OCLUtil.hpp"

using namespace std;

#include "declare_namespace"

string
devtype(const cl_device_type device_type)
{
    stringstream ss;
    if (device_type & CL_DEVICE_TYPE_CPU) ss << "CPU, ";
    if (device_type & CL_DEVICE_TYPE_GPU) ss << "GPU, ";
    if (device_type & CL_DEVICE_TYPE_ACCELERATOR) ss << "ACCELERATOR, ";
    return ss.str();
}

const char*
devinfo(const cl_device_info device_info)
{
    switch (device_info)
    {
        // integral valued parameters
        case (CL_DEVICE_TYPE) :
            return "type";
        case (CL_DEVICE_VENDOR_ID) :
            return "vendor id";
        case (CL_DEVICE_MAX_COMPUTE_UNITS) :
            return "max compute units";
        case (CL_DEVICE_MAX_WORK_GROUP_SIZE) :
            return "max work group size";
        case (CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) :
            return "preferred vector width char";
        case (CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT) :
            return "preferred vector width short";
        case (CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) :
            return "preferred vector width int";
        case (CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG) :
            return "preferred vector width long";
        case (CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) :
            return "preferred vector width float";
        case (CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) :
            return "preferred vector width double";
        case (CL_DEVICE_MAX_CLOCK_FREQUENCY) :
            return "max clock frequency";
        case (CL_DEVICE_ADDRESS_BITS) :
            return "address bits";
        case (CL_DEVICE_MAX_MEM_ALLOC_SIZE) :
            return "max mem alloc size";
        case (CL_DEVICE_IMAGE_SUPPORT) :
            return "image support";
        case (CL_DEVICE_MAX_READ_IMAGE_ARGS) :
            return "max read image args";
        case (CL_DEVICE_MAX_WRITE_IMAGE_ARGS) :
            return "max write image args";
        case (CL_DEVICE_IMAGE2D_MAX_WIDTH) :
            return "image 2d max width";
        case (CL_DEVICE_IMAGE2D_MAX_HEIGHT) :
            return "image 2d max height";
        case (CL_DEVICE_IMAGE3D_MAX_WIDTH) :
            return "image 3d max width";
        case (CL_DEVICE_IMAGE3D_MAX_HEIGHT) :
            return "image 3d max height";
        case (CL_DEVICE_IMAGE3D_MAX_DEPTH) :
            return "image 3d max depth";
        case (CL_DEVICE_MAX_SAMPLERS) :
            return "max samplers";
        case (CL_DEVICE_MAX_PARAMETER_SIZE) :
            return "max parameter size";
        case (CL_DEVICE_MEM_BASE_ADDR_ALIGN) :
            return "mem base addr align";
        case (CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE) :
            return "min data type align size";
        case (CL_DEVICE_SINGLE_FP_CONFIG) :
            return "single fp config";
        case (CL_DEVICE_GLOBAL_MEM_CACHE_TYPE) :
            return "global mem cache type";
        case (CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) :
            return "global mem cacheline size";
        case (CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) :
            return "global mem cache size";
        case (CL_DEVICE_GLOBAL_MEM_SIZE) :
            return "global mem size";
        case (CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) :
            return "max constant buffer size";
        case (CL_DEVICE_MAX_CONSTANT_ARGS) :
            return "max constant args";
        case (CL_DEVICE_LOCAL_MEM_TYPE) :
            return "local mem type";
        case (CL_DEVICE_LOCAL_MEM_SIZE) :
            return "local mem size";
        case (CL_DEVICE_ERROR_CORRECTION_SUPPORT) :
            return "error correction support";
        case (CL_DEVICE_PROFILING_TIMER_RESOLUTION) :
            return "profiling timer resolution";
        case (CL_DEVICE_ENDIAN_LITTLE) :
            return "endian little";
        case (CL_DEVICE_AVAILABLE) :
            return "available";
        case (CL_DEVICE_COMPILER_AVAILABLE) :
            return "compiler available";
        case (CL_DEVICE_EXECUTION_CAPABILITIES) :
            return "execution capabilities";
        case (CL_DEVICE_QUEUE_PROPERTIES) :
            return "queue properties";

        // string valued parameters
        case (CL_DEVICE_NAME) :
            return "name";
        case (CL_DEVICE_VENDOR) :
            return "vendor";
        case (CL_DRIVER_VERSION) :
            return "driver version";
        case (CL_DEVICE_PROFILE) :
            return "profile";
        case (CL_DEVICE_VERSION) :
            return "version";
        case (CL_DEVICE_EXTENSIONS) :
            return "extensions";
    }
    return "UNKNOWN"; // this should never happen
}

vector<size_t>
idxlist(const int v0,
        const int v1,
        const int v2)
{
    vector<size_t> l;
    if (v0 >= 0) l.push_back(v0);
    if (v1 >= 0) l.push_back(v1);
    if (v2 >= 0) l.push_back(v2);
    return l;
}

template <> bool isfloat<double>() { return false; }
template <> bool isfloat<float>() { return true; }
template <> bool isfloat<unsigned int>() { return false; }

} // namespace
