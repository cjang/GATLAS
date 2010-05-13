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
    map_devinfo_int    device_intinfo;    // info type -> dev idx -> value
    map_devinfo_string device_stringinfo; // info type -> dev idx -> value
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

    // default is in-order command queues, use this for out of order
    bool setOutOfOrder(const size_t device_index,
                       const bool out_of_order = true);

    cl_device_id&     getDevice(const size_t);
    cl_context&       getContext(const size_t);
    cl_command_queue& getQueue(const size_t);

    // returns device indexes
    std::vector<size_t> cpuIndexes();
    std::vector<size_t> gpuIndexes();

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
