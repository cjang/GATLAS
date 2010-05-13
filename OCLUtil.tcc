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

template <typename X>
bool checkFail(const cl_int status,
               const X msgX)
{
    if (CL_SUCCESS != status)
    {
        std::cerr << msgX << " " << status << std::endl;
        return true; // code was failure
    }

    return false; // did not fail
}

template <typename X, typename Y>
bool checkFail(const cl_int status,
               const X msgX, const Y msgY)
{
    if (CL_SUCCESS != status)
    {
        std::cerr << msgX << msgY << " " << status << std::endl;
        return true; // code was failure
    }

    return false; // did not fail
}

template <typename X, typename Y, typename Z>
bool checkFail(const cl_int status,
               const X msgX, const Y msgY, const Z msgZ)
{
    if (CL_SUCCESS != status)
    {
        std::cerr << msgX << msgY << msgZ << " " << status << std::endl;
        return true; // code was failure
    }

    return false; // did not fail
}

template <typename X, typename Y, typename Z,
          typename A>
bool checkFail(const cl_int status,
               const X msgX, const Y msgY, const Z msgZ,
               const A msgA)
{
    if (CL_SUCCESS != status)
    {
        std::cerr << msgX << msgY << msgZ
                  << msgA << " " << status << std::endl;
        return true; // code was failure
    }

    return false; // did not fail
}

template <typename X, typename Y, typename Z,
          typename A, typename B>
bool checkFail(const cl_int status,
               const X msgX, const Y msgY, const Z msgZ,
               const A msgA, const B msgB)
{
    if (CL_SUCCESS != status)
    {
        std::cerr << msgX << msgY << msgZ
                  << msgA << msgB << " " << status << std::endl;
        return true; // code was failure
    }

    return false; // did not fail
}

template <typename X, typename Y, typename Z,
          typename A, typename B, typename C>
bool checkFail(const cl_int status,
               const X msgX, const Y msgY, const Z msgZ,
               const A msgA, const B msgB, const C msgC)
{
    if (CL_SUCCESS != status)
    {
        std::cerr << msgX << msgY << msgZ
                  << msgA << msgB << msgC << " " << status << std::endl;
        return true; // code was failure
    }

    return false; // did not fail
}

template <typename T>
std::string devinfovalue(const cl_device_info device_info,
                         const T value)
{
    if (CL_DEVICE_SINGLE_FP_CONFIG == device_info)
    {
        std::string msg;
        if (CL_FP_DENORM & value) msg += "denorm, ";
        if (CL_FP_INF_NAN & value) msg += "inf and quiet NaNs, ";
        if (CL_FP_ROUND_TO_NEAREST & value) msg += "round to nearest, ";
        if (CL_FP_ROUND_TO_ZERO & value) msg += "round to zero, ";
        if (CL_FP_ROUND_TO_INF & value) msg += "round to inf, ";
        if (CL_FP_FMA & value) msg += "fused multiply-add, ";
        return msg;
    }

    if (CL_DEVICE_GLOBAL_MEM_CACHE_TYPE == device_info)
    {
        switch (value)
        {
            case (CL_NONE) : return "none";
            case (CL_READ_ONLY_CACHE) : return "read only cache";
            case (CL_READ_WRITE_CACHE) : return "read write cache";
        }
    }

    if (CL_DEVICE_LOCAL_MEM_TYPE == device_info)
    {
        switch (value)
        {
            case (CL_LOCAL) : return "local";
            case (CL_GLOBAL) : return "global";
        }
    }

    if (CL_DEVICE_EXECUTION_CAPABILITIES == device_info)
    {
        std::string msg;
        if (CL_EXEC_KERNEL & value) msg += "kernel, ";
        if (CL_EXEC_NATIVE_KERNEL & value) msg += "native kernel, ";
        return msg;
    }

    if (CL_DEVICE_QUEUE_PROPERTIES == device_info)
    {
        std::string msg;
        if (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE & value) msg += "out of order execution, ";
        if (CL_QUEUE_PROFILING_ENABLE & value) msg += "profiling, ";
        return msg;
    }

    switch (device_info)
    {
        case (CL_DEVICE_TYPE) : return devtype(value);
        case (CL_DEVICE_IMAGE_SUPPORT) : return value ? "true" : "false";
        case (CL_DEVICE_ERROR_CORRECTION_SUPPORT) : return value ? "true" : "false";
        case (CL_DEVICE_ENDIAN_LITTLE) : return value ? "true" : "false";
        case (CL_DEVICE_AVAILABLE) : return value ? "true" : "false";
        case (CL_DEVICE_COMPILER_AVAILABLE) : return value ? "true" : "value";
    }

    std::ostringstream msg;
    msg << value;
    return msg.str();
}

template <typename T>
T* alloc_memalign(const size_t n, const size_t ALIGNMENT)
{
    T* ptr;

    if (posix_memalign((void**)&ptr, ALIGNMENT, n * sizeof(T)))
        return NULL; // failure, could not allocate aligned memory

    return ptr;
}

template <typename T>
T* alloc_memalign(const size_t n)
{
    return alloc_memalign<T>(n, Type<T>::ALIGNMENT);
}

template <typename SCALAR, size_t N>
SCALAR* alloc_memalign(const size_t n)
{
    return alloc_memalign<SCALAR>(n, N * Type<SCALAR>::ALIGNMENT);
}
