#ifndef _GATLAS_OCL_STL_HPP_
#define _GATLAS_OCL_STL_HPP_

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
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <stdlib.h>

#include "declare_namespace"

// (Almost) all OpenCL types must be memory aligned. If this
// isn't done, random segfaults happen when using STL collections.
template <typename T>
struct aligned_allocator : std::allocator<T>
{
    T* allocate(size_t n)
    {
        T *p;
        return posix_memalign((void**)&p, sizeof(T), n * sizeof(T))
            ? 0  // failure
            : p; // success
    }

    void deallocate(T *p, size_t)
    {
        free(p);
    }
};

// STL vectors of OpenCL types
template <typename T>
struct Type
{
    static const size_t ALIGNMENT = 8;
    typedef std::vector<T, aligned_allocator<T> > aligned_vector;
    typedef std::map<cl_device_info, std::vector<T> > map_device_info;
};
typedef Type<cl_platform_id>::aligned_vector   vec_platform_id;
typedef Type<cl_device_id>::aligned_vector     vec_device_id;
typedef Type<cl_device_info>::aligned_vector   vec_device_info;
typedef Type<cl_device_type>::aligned_vector   vec_device_type;
typedef Type<cl_context>::aligned_vector       vec_context;
typedef Type<cl_command_queue>::aligned_vector vec_command_queue;
typedef Type<cl_program>::aligned_vector       vec_program;
typedef Type<cl_kernel>::aligned_vector        vec_kernel;
typedef Type<cl_mem>::aligned_vector           vec_mem;
typedef Type<cl_event>::aligned_vector         vec_event;
typedef Type<cl_sampler>::aligned_vector       vec_sampler;

// device info integral/bitfield types
typedef Type<cl_device_type>::map_device_info              map_devinfo_cl_device_type;
typedef Type<cl_uint>::map_device_info                     map_devinfo_cl_uint;
typedef Type<size_t>::map_device_info                      map_devinfo_size_t;
typedef Type<cl_ulong>::map_device_info                    map_devinfo_cl_ulong;
typedef Type<cl_bool>::map_device_info                     map_devinfo_cl_bool;
typedef Type<cl_device_fp_config>::map_device_info         map_devinfo_cl_device_fp_config;
typedef Type<cl_device_mem_cache_type>::map_device_info    map_devinfo_cl_device_mem_cache_type;
typedef Type<cl_device_local_mem_type>::map_device_info    map_devinfo_cl_device_local_mem_type;
typedef Type<cl_device_exec_capabilities>::map_device_info map_devinfo_cl_device_exec_capabilities;
typedef Type<cl_command_queue_properties>::map_device_info map_devinfo_cl_command_queue_properties;

typedef Type<std::string>::map_device_info   map_devinfo_string;

}; // namespace

#endif
