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

#include "OCLBase.hpp"
#include "OCLUtil.hpp"
#include <iostream>

using namespace std;

#include "declare_namespace"

// It's ugly but safe to have a parameter name list for each type.
// Unfortunately, different platform SDKs use native integral types
// of different sizes. Instead of trying to find one largest type
// big enough to contain all of them, have separate lists using the
// OpenCL typedefs directly.
//
// Template metaprogramming could hide some of this. However, other
// ugliness would then be introduced. On balance, the most direct
// and simple-minded solution is preferable.

template <typename T>
void storeDeviceInfoParams(std::map<cl_device_info, std::vector<T> >& mapDevInfo,
                           const cl_device_info *listParams,
                           const size_t numParams,
                           const cl_device_id deviceId) {
    for (size_t i = 0; i < numParams; i++)
    {
        const cl_device_info key = listParams[i];

        T value;

        mapDevInfo[key].push_back(
            checkFail(
                clGetDeviceInfo(deviceId,
                                key,
                                sizeof(T),
                                &value,
                                NULL),
                "get device info ",
                devinfo(key))
                ? 0
                : value);
    }
}

template <typename T>
void printDeviceInfoParams(std::map<cl_device_info, std::vector<T> >& mapDevInfo,
                           const cl_device_info *listParams,
                           const size_t numParams,
                           const size_t device_index,
                           const char *prepend) {
    for (size_t i = 0; i < numParams; i++)
    {
        const cl_device_info key = listParams[i];

        cout
            << prepend << "\t"
            << devinfo(key) << "\t"
            << devinfovalue(key, mapDevInfo[key][device_index])
            << endl;
    }
}


// cl_device_type valued cl_device_info parameters
static const cl_device_info cl_device_type_params[] = {
    CL_DEVICE_TYPE
};

// cl_uint valued cl_device_info parameters
static const cl_device_info cl_uint_params[] = {
    CL_DEVICE_VENDOR_ID,
    CL_DEVICE_MAX_COMPUTE_UNITS,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
    CL_DEVICE_MAX_CLOCK_FREQUENCY,
    CL_DEVICE_ADDRESS_BITS,
    CL_DEVICE_MAX_READ_IMAGE_ARGS,
    CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
    CL_DEVICE_MAX_SAMPLERS,
    CL_DEVICE_MEM_BASE_ADDR_ALIGN,
    CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
    CL_DEVICE_MAX_CONSTANT_ARGS
};

// size_t valued cl_device_info parameters
static const cl_device_info size_t_params[] = {
    CL_DEVICE_MAX_WORK_GROUP_SIZE,
    CL_DEVICE_IMAGE2D_MAX_WIDTH,
    CL_DEVICE_IMAGE2D_MAX_HEIGHT,
    CL_DEVICE_IMAGE3D_MAX_WIDTH,
    CL_DEVICE_IMAGE3D_MAX_HEIGHT,
    CL_DEVICE_IMAGE3D_MAX_DEPTH,
    CL_DEVICE_MAX_PARAMETER_SIZE,
    CL_DEVICE_PROFILING_TIMER_RESOLUTION
};

// cl_ulong valued cl_device_info parameters
static const cl_device_info cl_ulong_params[] = {
    CL_DEVICE_MAX_MEM_ALLOC_SIZE,
    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
    CL_DEVICE_GLOBAL_MEM_SIZE,
    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
    CL_DEVICE_LOCAL_MEM_SIZE
};

// cl_bool valued cl_device_info parameters
static const cl_device_info cl_bool_params[] = {
    CL_DEVICE_IMAGE_SUPPORT,
    CL_DEVICE_ERROR_CORRECTION_SUPPORT,
    CL_DEVICE_ENDIAN_LITTLE,
    CL_DEVICE_AVAILABLE,
    CL_DEVICE_COMPILER_AVAILABLE
};

// cl_device_fp_config valued cl_device_info parameters
static const cl_device_info cl_device_fp_config_params[] = {
    CL_DEVICE_SINGLE_FP_CONFIG
};

// cl_device_mem_cache_type valued cl_device_info parameters
static const cl_device_info cl_device_mem_cache_type_params[] = {
    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE
};

// cl_device_local_mem_type valued cl_device_info parameters
static const cl_device_info cl_device_local_mem_type_params[] = {
    CL_DEVICE_LOCAL_MEM_TYPE
};

// cl_device_exec_capabilities valued cl_device_info parameters
static const cl_device_info cl_device_exec_capabilities_params[] = {
    CL_DEVICE_EXECUTION_CAPABILITIES
};

// cl_command_queue_properties values cl_device_info parameters
static const cl_device_info cl_command_queue_properties_params[] = {
    CL_DEVICE_QUEUE_PROPERTIES
};

// string valued cl_device_info parameters
static const cl_device_info string_params[] = {
    CL_DEVICE_NAME,
    CL_DEVICE_VENDOR,
    CL_DRIVER_VERSION,
    CL_DEVICE_PROFILE,
    CL_DEVICE_VERSION,
    CL_DEVICE_EXTENSIONS
};

void
OCLBase::init_platforms()
{
    // get number of platforms
    cl_uint num_platforms;
    if (checkFail(
        clGetPlatformIDs(0, NULL, &num_platforms),
        "get number of platforms")) return;

    // get platform ids
    cl_platform_id platform_ids[num_platforms];
    if (checkFail(
        clGetPlatformIDs(num_platforms, platform_ids, NULL),
        "get platform ids")) return;

    for (size_t i = 0; i < num_platforms; i++)
    {
        platforms.push_back(platform_ids[i]);
    }
}

int
OCLBase::insert_device(const cl_device_id device_id,
                       const size_t platform_index)
{
    // check if device already exists
    for (size_t i = 0; i < devices.size(); i++)
    {
        if (device_id == devices[i])
            return -1;
    }

    const int insert_index = devices.size();
    devices.push_back(device_id);
    device_platform.push_back(platform_index); // keep track of parent platform

    // only cl_device_type valued device info parameters
    storeDeviceInfoParams<cl_device_type>(
        device_info_cl_device_type,
        cl_device_type_params,
        sizeof(cl_device_type_params)/sizeof(cl_device_info),
        device_id);

    // only cl_uint valued device info parameters
    storeDeviceInfoParams<cl_uint>(
        device_info_cl_uint,
        cl_uint_params,
        sizeof(cl_uint_params)/sizeof(cl_device_info),
        device_id);

    // only size_t valued device info parameters
    storeDeviceInfoParams<size_t>(
        device_info_size_t,
        size_t_params,
        sizeof(size_t_params)/sizeof(cl_device_info),
        device_id);

    // only cl_ulong valued device info parameters
    storeDeviceInfoParams<cl_ulong>(
        device_info_cl_ulong,
        cl_ulong_params,
        sizeof(cl_ulong_params)/sizeof(cl_device_info),
        device_id);

    // only cl_bool valued device info parameters
    storeDeviceInfoParams<cl_bool>(
        device_info_cl_bool,
        cl_bool_params,
        sizeof(cl_bool_params)/sizeof(cl_device_info),
        device_id);

    // only cl_device_fp_config valued device info parameters
    storeDeviceInfoParams<cl_device_fp_config>(
        device_info_cl_device_fp_config,
        cl_device_fp_config_params,
        sizeof(cl_device_fp_config)/sizeof(cl_device_info),
        device_id);

    // only cl_device_mem_cache_type valued device info parameters
    storeDeviceInfoParams<cl_device_mem_cache_type>(
        device_info_cl_device_mem_cache_type,
        cl_device_mem_cache_type_params,
        sizeof(cl_device_mem_cache_type_params)/sizeof(cl_device_info),
        device_id);

    // only cl_device_local_mem_type valued device info parameters
    storeDeviceInfoParams<cl_device_local_mem_type>(
        device_info_cl_device_local_mem_type,
        cl_device_local_mem_type_params,
        sizeof(cl_device_local_mem_type_params)/sizeof(cl_device_info),
        device_id);

    // only cl_device_exec_capabilities valued device info parameters
    storeDeviceInfoParams<cl_device_exec_capabilities>(
        device_info_cl_device_exec_capabilities,
        cl_device_exec_capabilities_params,
        sizeof(cl_device_exec_capabilities_params)/sizeof(cl_device_info),
        device_id);

    // only string valued device info parameters
    for (size_t i = 0; i < sizeof(string_params)/sizeof(cl_device_info); i++)
    {
        const cl_device_info key = string_params[i];

        char buffer[512]; // arbitrary length

        device_stringinfo[key].push_back(
            checkFail(
                clGetDeviceInfo(device_id,
                                key,
                                sizeof(buffer),
                                buffer,
                                NULL),
                "get device info ",
                devinfo(key))
                ? ""
                : buffer);
    }

    // work item dimensions handled as a special case
    cl_uint dimnum = 0;
    if (checkFail(
            clGetDeviceInfo(device_id,
                            CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(dimnum),
                            &dimnum,
                            NULL),
            "get max work item dimensions"))
    {
        // failure
        device_workdim.push_back(vector<size_t>());
    }
    else
    {
        // ok
        vector<size_t> work_item_dim(dimnum, 0);

        checkFail(
            clGetDeviceInfo(device_id,
                            CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            dimnum * sizeof(size_t),
                            &work_item_dim[0],
                            NULL),
            "get max work item sizes");

        device_workdim.push_back(work_item_dim);
    }

    return insert_index;
}

void
OCLBase::init_devices()
{
    // the same device may have multiple types
//    const cl_device_type dtype[] = { CL_DEVICE_TYPE_CPU,
//                                     CL_DEVICE_TYPE_GPU,
//                                     CL_DEVICE_TYPE_ACCELERATOR,
//                                     CL_DEVICE_TYPE_DEFAULT,
//                                     CL_DEVICE_TYPE_ALL };
    const cl_device_type dtype[] = { CL_DEVICE_TYPE_ALL };

    // every platform
    for (size_t pi = 0; pi < platforms.size(); pi++)
    {
        const cl_platform_id platform_id = platforms[pi];

        // every device type
        for (size_t di = 0; di < sizeof(dtype)/sizeof(cl_device_type); di++)
        {
            const cl_device_type device_type = dtype[di];

            // get number of devices for this platform and device type
            cl_uint num_devices;
            if (checkFail(
                clGetDeviceIDs(platform_id,
                               device_type,
                               0,
                               NULL,
                               &num_devices),
                "get number devices")) continue; // ignore

            // get device ids
            cl_device_id device_ids[num_devices];
            if (checkFail(
                clGetDeviceIDs(platform_id,
                               device_type,
                               num_devices,
                               device_ids,
                               NULL),
                "get device ids")) return;

            // this platform has some devices
            for (size_t i = 0; i < num_devices; i++)
            {
                const cl_device_id device_id = device_ids[i];
                const int device_index = insert_device(device_id, pi);

                // keep track of devices for each platform
                if (-1 != device_index)
                    platform_devices[pi].insert(device_index);
            }
        }
    }
}

void
OCLBase::init_contexts()
{
    // every device has one context
    for (size_t i = 0; i < devices.size(); i++)
    {
        // fails to compile for NVIDIA OpenCL SDK if declared const
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)
                platforms[device_platform[i]],
            0 };

        // create context for device
        cl_int status;
        contexts.push_back(
            clCreateContext(props,
                            1,
                            &devices[i],
                            NULL,
                            NULL,
                            &status));

        // just print an error message if context creation fails
        checkFail(status, "create context for device ", i);
    }
}

void
OCLBase::init_queues()
{
    // devery device has a command queue
    for (size_t i = 0; i < devices.size(); i++)
    {
        // check if device has an associated context
        if (contexts[i])
        {
            // create command queue for device
            cl_int status;
            queues.push_back(
                clCreateCommandQueue(contexts[i],
                                     devices[i],
                // Note: ATI SDK device info returns profiling support only
                // for both Core i7 CPU and Radeon 5870 GPU but queue creation
                // still succeeds when passing out of order mode enable flag.
                // From AMD support forums, as of SDK v2.0, the Catalyst driver
                // does not support concurrent kernel execution although the
                // hardware has the capability (i.e. Eyefinity).
                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                     CL_QUEUE_PROFILING_ENABLE,
                                     &status));

            // just print an error message if queue creation fails
            checkFail(status, "create command queue for device ", i);
        }
    }
}

OCLBase::OCLBase()
{
    init_platforms();
    init_devices();
    init_contexts();
    init_queues();
}

OCLBase::~OCLBase()
{
    for (size_t i = 0; i < queues.size(); i++)
        if (queues[i])
            checkFail(clReleaseCommandQueue(queues[i]), "release queue ", i);

    for (size_t i = 0; i < contexts.size(); i++)
        if (contexts[i])
            checkFail(clReleaseContext(contexts[i]), "release context ", i);
}

cl_device_id&
OCLBase::getDevice(const size_t index)
{
    return devices[index];
}

cl_context&
OCLBase::getContext(const size_t index)
{
    return contexts[index];
}

cl_command_queue&
OCLBase::getQueue(const size_t index)
{
    return queues[index];
}

vector<size_t>
OCLBase::deviceIndexes(const cl_device_type dtype)
{
    vector<size_t> indexes;

    const vector<cl_device_type>& dev_type = device_info_cl_device_type[CL_DEVICE_TYPE];

    for (size_t i = 0; i < dev_type.size(); i++)
    {
        if (dtype & dev_type[i])
            indexes.push_back(i);
    }

    return indexes;
}

vector<size_t>
OCLBase::cpuIndexes()
{
    return deviceIndexes(CL_DEVICE_TYPE_CPU);
}

vector<size_t>
OCLBase::gpuIndexes()
{
    return deviceIndexes(CL_DEVICE_TYPE_GPU);
}

vector<size_t>
OCLBase::accIndexes()
{
    return deviceIndexes(CL_DEVICE_TYPE_ACCELERATOR);
}

size_t
OCLBase::maxWorkGroupSize(const size_t device_index)
{
    return device_info_size_t[CL_DEVICE_MAX_WORK_GROUP_SIZE][device_index];
}

size_t
OCLBase::maxComputeUnits(const size_t device_index)
{
    return device_info_cl_uint[CL_DEVICE_MAX_COMPUTE_UNITS][device_index];
}

size_t
OCLBase::maxMemAlloc(const size_t device_index)
{
    return device_info_cl_ulong[CL_DEVICE_MAX_MEM_ALLOC_SIZE][device_index];
}

size_t
OCLBase::maxConstBuffer(const size_t device_index)
{
    return device_info_cl_ulong[CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE][device_index];
}

size_t
OCLBase::localMemory(const size_t device_index)
{
    return device_info_cl_ulong[CL_DEVICE_LOCAL_MEM_SIZE][device_index];
}

size_t
OCLBase::globalMemory(const size_t device_index)
{
    return device_info_cl_ulong[CL_DEVICE_GLOBAL_MEM_SIZE][device_index];
}

void
OCLBase::print(const size_t device_index, const char *prepend)
{
    cout
        << prepend
        << "device[" << device_index << "] = " << devices[device_index]
        << "\tcontext " << contexts[device_index]
        << "\tqueue " << queues[device_index]
        << endl;

    // only cl_device_type valued device info parameters
    printDeviceInfoParams<cl_device_type>(
        device_info_cl_device_type,
        cl_device_type_params,
        sizeof(cl_device_type_params)/sizeof(cl_device_info),
        device_index,
        prepend);

    // only cl_uint valued device info parameters
    printDeviceInfoParams<cl_uint>(
        device_info_cl_uint,
        cl_uint_params,
        sizeof(cl_uint_params)/sizeof(cl_device_info),
        device_index,
        prepend);

    // only size_t valued device info parameters
    printDeviceInfoParams<size_t>(
        device_info_size_t,
        size_t_params,
        sizeof(size_t_params)/sizeof(cl_device_info),
        device_index,
        prepend);

    // only cl_ulong valued device info parameters
    printDeviceInfoParams<cl_ulong>(
        device_info_cl_ulong,
        cl_ulong_params,
        sizeof(cl_ulong_params)/sizeof(cl_device_info),
        device_index,
        prepend);

    // only cl_bool valued device info parameters
    printDeviceInfoParams<cl_bool>(
        device_info_cl_bool,
        cl_bool_params,
        sizeof(cl_bool_params)/sizeof(cl_device_info),
        device_index,
        prepend);

    // only cl_device_fp_config valued device info parameters
    printDeviceInfoParams<cl_device_fp_config>(
        device_info_cl_device_fp_config,
        cl_device_fp_config_params,
        sizeof(cl_device_fp_config_params)/sizeof(cl_device_info),
        device_index,
        prepend);

    // only cl_device_mem_cache_type valued device info parameters
    printDeviceInfoParams<cl_device_mem_cache_type>(
        device_info_cl_device_mem_cache_type,
        cl_device_mem_cache_type_params,
        sizeof(cl_device_mem_cache_type_params)/sizeof(cl_device_info),
        device_index,
        prepend);

    // only cl_device_local_mem_type valued device info parameters
    printDeviceInfoParams<cl_device_local_mem_type>(
        device_info_cl_device_local_mem_type,
        cl_device_local_mem_type_params,
        sizeof(cl_device_local_mem_type_params)/sizeof(cl_device_info),
        device_index,
        prepend);

    // only cl_device_exec_capabilities valued device info parameters
    printDeviceInfoParams<cl_device_exec_capabilities>(
        device_info_cl_device_exec_capabilities,
        cl_device_exec_capabilities_params,
        sizeof(cl_device_exec_capabilities_params)/sizeof(cl_device_info),
        device_index,
        prepend);

    // only string valued device info parameters
    for (size_t i = 0; i < sizeof(string_params)/sizeof(cl_device_info); i++)
    {
        const cl_device_info key = string_params[i];

        cout
            << prepend << "\t"
            << devinfo(key) << "\t" << device_stringinfo[key][device_index]
            << endl;
    }

    // work item dimensions
    cout
        << prepend << "\t"
        << "max work item dimensions";
    for (size_t i = 0; i < device_workdim[device_index].size(); i++)
        cout << "\t" << device_workdim[device_index][i];
    cout << endl;
}

void
OCLBase::print()
{
    for (size_t i = 0; i < platforms.size(); i++)
    {
        cout
            << "platforms[" << i << "] = " << platforms[i]
            << endl;

        for (set<int>::const_iterator iter = platform_devices[i].begin();
             iter != platform_devices[i].end();
             iter++)
        {
            print(*iter, "\t");
        }
    }
}

} // namespace
