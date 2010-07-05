#ifndef _GATLAS_OPENCL_BASE_HPP_
#define _GATLAS_OPENCL_BASE_HPP_

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

#include <CL/cl.h>
#include "OCLSTL.hpp"
#include "OCLUtil.hpp"

#include "declare_namespace"

class OCLBase
{
    // platforms
    vec_platform_id    platforms;         // platform index -> platform id
    map_setindex       platform_devices;  // platform index -> set of dev idx
    void init_platforms();

    // devices
    vec_device_id      devices;           // device index -> device id
    vec_size_t         device_platform;   // device index -> platform index
    vec_vec_size_t     device_workdim;    // device index -> work item dims

    // info type -> device index -> value
    map_devinfo_cl_device_type              device_info_cl_device_type;
    map_devinfo_cl_uint                     device_info_cl_uint;
    map_devinfo_size_t                      device_info_size_t;
    map_devinfo_cl_ulong                    device_info_cl_ulong;
    map_devinfo_cl_bool                     device_info_cl_bool;
    map_devinfo_cl_device_fp_config         device_info_cl_device_fp_config;
    map_devinfo_cl_device_mem_cache_type    device_info_cl_device_mem_cache_type;
    map_devinfo_cl_device_local_mem_type    device_info_cl_device_local_mem_type;
    map_devinfo_cl_device_exec_capabilities device_info_cl_device_exec_capabilities;

    // info type -> device index -> value
    map_devinfo_string                      device_stringinfo;

    int insert_device(const cl_device_id, const size_t);
    void init_devices();

    // contexts, one-to-one correspondence with devices
    vec_context        contexts; // device index -> context handle
    void init_contexts();

    // command queues, one-to-one correspondence with devices
    vec_command_queue  queues;   // device index -> queue handle
    void init_queues();

    vec_size_t deviceIndexes(const cl_device_type);

public:

    OCLBase();
    ~OCLBase();

    cl_device_id&     getDevice(const size_t);
    cl_context&       getContext(const size_t);
    cl_command_queue& getQueue(const size_t);

    // returns device indexes
    std::vector<size_t> cpuIndexes(); // CPU devices
    std::vector<size_t> gpuIndexes(); // GPU devices
    std::vector<size_t> accIndexes(); // ACCELERATOR devices

    // access device parameter values
    size_t maxWorkGroupSize(const size_t device_index);
    size_t maxComputeUnits(const size_t device_index);
    size_t maxMemAlloc(const size_t device_index);
    size_t maxConstBuffer(const size_t device_index);
    size_t localMemory(const size_t device_index);
    size_t globalMemory(const size_t device_index);

    // debugging
    void print(const size_t device_index, const char *prepend = "");
    void print();

}; // class OCLBase

}; // namespace

#endif
