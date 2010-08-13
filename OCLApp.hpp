#ifndef _GATLAS_OCL_APP_HPP_
#define _GATLAS_OCL_APP_HPP_

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
#include <string>
#include <vector>
#include <string.h>
#include "OCLSTL.hpp"
#include "OCLBase.hpp"
#include "OCLUtil.hpp"

#include "declare_namespace"

class OCLApp
{
    // base object is for all devices
    OCLBase& oclBase;

    // the device this program is for
    const size_t device_index;

    cl_program   program;       // program object may contain multiple kernels
    vec_kernel   kernels;       // kernel objects from the program
    vec_size_t   kernel_wgsize; // maximum work group size for each kernel

    vec_mem      membuffers;    // memory buffer objects
    vec_voidptr  memptrs;       // host pointers to allocated arrays for buffers
    vec_bool     memownptrs;    // does OCLApp have ownership of memory?
    vec_size_t   memsizeoftype; // sizeof(type) for the buffers
    vec_size_t   memsize;       // size of the memory buffer

    vec_mem      imgbuffers;    // image objects
    vec_floatptr imgptrs;       // host pointers to allocated arrays for images
    vec_bool     imgownptrs;    // does OCLApp have ownership of memory?
    vec_size_t   imgwidth;      // width of image in pixels
    vec_size_t   imgheight;     // height of image in pixels

    vec_sampler  imgsamplers;   // sampler objects

    vec_event    events;        // command queue events to wait for
    vec_bool     events_waited; // keep track of events already waited on
    size_t       events_pending;// number of events not waited on yet

    bool releaseKernels();
    bool releaseProgram();

    template <typename T> int createBufferWithPointer(const size_t n,
                                                      const cl_mem_flags,
                                                      T *ptr,
                                                      const bool own_memptr = true);
    template <typename T> int createBufferAllocMemory(const size_t n,
                                                      const cl_mem_flags,
                                                      const size_t ALIGNMENT);

    template <typename T> int createImageWithPointer(const size_t width,
                                                     const size_t height,
                                                     const cl_mem_flags flags,
                                                     T *ptr,
                                                     bool own_imgptr = true);
    template <typename T> int createImageAllocMemory(const size_t width,
                                                     const size_t height,
                                                     const cl_mem_flags flags,
                                                     const size_t ALIGNMENT);

public:

    OCLApp(OCLBase&, const size_t);
    ~OCLApp();

    // build program
    bool buildProgram(const std::vector<std::string>& program_source,
                      const std::string& options = "");
    std::string buildLog() const;

    // create kernels
    int createKernel(const std::string& kernel_name);

    // create and release memory buffers
    enum BUFFER_FLAGS { READ, WRITE, READWRITE };
    template <typename T> int createBuffer(const size_t n,
                                           BUFFER_FLAGS mode,
                                           bool pinned = false);
    template <typename T, size_t N> int createBuffer(const size_t n,
                                                     BUFFER_FLAGS mode,
                                                     bool pinned = false);
    template <typename T> int createBuffer(const size_t n,
                                           BUFFER_FLAGS mode,
                                           T *ptr,
                                           bool pinned = false);
    template <typename T> int createImage(const size_t width,
                                          const size_t height,
                                          BUFFER_FLAGS mode,
                                          bool pinned = false);
    template <typename T> int createImage(const size_t width,
                                          const size_t height,
                                          BUFFER_FLAGS mode,
                                          T *ptr,
                                          bool pinned = false);
    int createSampler();
    bool releaseBuffers();
    bool releaseImages();
    bool releaseSamplers();
    template <typename T> void memsetBuffer(const size_t buffer_index,
                                            const T value);
    void memsetImage(const size_t image_index, const float value);
    template <typename T> T* bufferPtr(const size_t buffer_index) const;
    template <typename T> T* imagePtr(const size_t image_index) const;
    void ownBuffer(const size_t buffer_index);
    void ownImage(const size_t image_index);

    // set kernel arguments
    bool setArgImage(const size_t kernel_index,
                     const size_t kernel_arg_index,
                     const size_t image_index);
    bool setArgSampler(const size_t kernel_index,
                       const size_t kernel_arg_index,
                       const size_t sampler_index);
    bool setArgGlobal(const size_t kernel_index,
                      const size_t kernel_arg_index,
                      const size_t buffer_index);
    template <typename T>
    bool setArgLocal(const size_t kernel_index,
                     const size_t kernel_arg_index,
                     const size_t n);
    template <typename T>
    bool setArgValue(const size_t kernel_index,
                     const size_t kernel_arg_index,
                     const T value);

    // enqueue kernel, returns handle to event
    int enqueueKernel(const size_t kernel_index,
                      const std::vector<size_t>& global_dim,
                      const std::vector<size_t>& local_dim);
    int enqueueKernel(const size_t kernel_index,
                      const std::vector<size_t>& global_dim,
                      const std::vector<size_t>& local_dim,
                      const size_t event_index);
    int enqueueKernel(const size_t kernel_index,
                      const std::vector<size_t>& global_dim,
                      const std::vector<size_t>& local_dim,
                      const size_t event_index_0,
                      const size_t event_index_1);
    int enqueueKernel(const size_t kernel_index,
                      const std::vector<size_t>& global_dim,
                      const std::vector<size_t>& local_dim,
                      const size_t event_index_0,
                      const size_t event_index_1,
                      const size_t event_index_2);
    int enqueueKernel(const size_t kernel_index,
                      const std::vector<size_t>& global_dim,
                      const std::vector<size_t>& local_dim,
                      const std::vector<size_t>& event_indexes);

    // asynchronous buffer copying, returns handle to event
    int enqueueReadBuffer(const size_t buffer_index);
    int enqueueReadBuffer(const size_t buffer_index,
                          const std::vector<size_t>& event_indexes);
    int enqueueReadBuffer(const size_t buffer_index,
                          const size_t offset,
                          const size_t n);
    int enqueueReadBuffer(const size_t buffer_index,
                          const size_t offset,
                          const size_t n,
                          const std::vector<size_t>& event_indexes);
    int enqueueWriteBuffer(const size_t buffer_index);
    int enqueueWriteBuffer(const size_t buffer_index,
                           const std::vector<size_t>& event_indexes);
    int enqueueWriteBuffer(const size_t buffer_index,
                           const size_t offset,
                           const size_t n);
    int enqueueWriteBuffer(const size_t buffer_index,
                           const size_t offset,
                           const size_t n,
                           const std::vector<size_t>& event_indexes);
    int enqueueCopyBuffer(const size_t src_buffer_index,
                          const size_t dest_buffer_index);
    int enqueueCopyBuffer(const size_t src_buffer_index,
                          const size_t dest_buffer_index,
                          const std::vector<size_t>& event_indexes);
    int enqueueCopyBuffer(const size_t src_buffer_index,
                          const size_t dest_buffer_index,
                          const size_t src_offset,
                          const size_t dest_offset,
                          const size_t n);
    int enqueueCopyBuffer(const size_t src_buffer_index,
                          const size_t dest_buffer_index,
                          const size_t src_offset,
                          const size_t dest_offset,
                          const size_t n,
                          const std::vector<size_t>& event_indexes);

    // asynchronous image copying, returns handle to event
    int enqueueReadImage(const size_t image_index);
    int enqueueReadImage(const size_t image_index,
                         const std::vector<size_t>& event_indexes);
    int enqueueReadImage(const size_t image_index,
                         const size_t origin_x,
                         const size_t origin_y,
                         const size_t region_width,
                         const size_t region_height);
    int enqueueReadImage(const size_t image_index,
                         const size_t origin_x,
                         const size_t origin_y,
                         const size_t region_width,
                         const size_t region_height,
                         const std::vector<size_t>& event_indexes);
    int enqueueWriteImage(const size_t image_index);
    int enqueueWriteImage(const size_t image_index,
                          const std::vector<size_t>& event_indexes);
    int enqueueWriteImage(const size_t image_index,
                          const size_t origin_x,
                          const size_t origin_y,
                          const size_t region_width,
                          const size_t region_height);
    int enqueueWriteImage(const size_t image_index,
                          const size_t origin_x,
                          const size_t origin_y,
                          const size_t region_width,
                          const size_t region_height,
                          const std::vector<size_t>& event_indexes);
    int enqueueCopyImage(const size_t src_image_index,
                         const size_t dest_image_index);
    int enqueueCopyImage(const size_t src_image_index,
                         const size_t dest_image_index,
                         const std::vector<size_t>& event_indexes);
    int enqueueCopyImage(const size_t src_image_index,
                         const size_t dest_image_index,
                         const size_t src_origin_x,
                         const size_t src_origin_y,
                         const size_t dest_origin_x,
                         const size_t dest_origin_y,
                         const size_t region_width,
                         const size_t region_height);
    int enqueueCopyImage(const size_t src_image_index,
                         const size_t dest_image_index,
                         const size_t src_origin_x,
                         const size_t src_origin_y,
                         const size_t dest_origin_x,
                         const size_t dest_origin_y,
                         const size_t region_width,
                         const size_t region_height,
                         const std::vector<size_t>& event_indexes);

    // asynchronous copies between buffers and images
    int enqueueCopyBufferToImage(const size_t src_buffer_index,
                                 const size_t dest_image_index);
    int enqueueCopyBufferToImage(const size_t src_buffer_index,
                                 const size_t dest_image_index,
                                 const std::vector<size_t>& event_indexes);
    int enqueueCopyBufferToImage(const size_t src_buffer_index,
                                 const size_t dest_image_index,
                                 const size_t src_offset,
                                 const size_t dest_origin_x,
                                 const size_t dest_origin_y,
                                 const size_t region_width,
                                 const size_t region_height);
    int enqueueCopyBufferToImage(const size_t src_buffer_index,
                                 const size_t dest_image_index,
                                 const size_t src_offset,
                                 const size_t dest_origin_x,
                                 const size_t dest_origin_y,
                                 const size_t region_width,
                                 const size_t region_height,
                                 const std::vector<size_t>& event_indexes);
    int enqueueCopyImageToBuffer(const size_t src_image_index,
                                 const size_t dest_buffer_index);
    int enqueueCopyImageToBuffer(const size_t src_image_index,
                                 const size_t dest_buffer_index,
                                 const std::vector<size_t>& event_indexes);
    int enqueueCopyImageToBuffer(const size_t src_image_index,
                                 const size_t dest_buffer_index,
                                 const size_t src_origin_x,
                                 const size_t src_origin_y,
                                 const size_t dest_offset,
                                 const size_t region_width,
                                 const size_t region_height);
    int enqueueCopyImageToBuffer(const size_t src_image_index,
                                 const size_t dest_buffer_index,
                                 const size_t src_origin_x,
                                 const size_t src_origin_y,
                                 const size_t dest_offset,
                                 const size_t region_width,
                                 const size_t region_height,
                                 const std::vector<size_t>& event_indexes);

    // flushing and waiting (blocking calls)
    bool wait();
    bool wait(const size_t event_index);
    bool wait(const size_t event_index_0,
              const size_t event_index_1);
    bool wait(const size_t event_index_0,
              const size_t event_index_1,
              const size_t event_index_2);
    bool wait(const std::vector<size_t>& event_indexes);
    bool finish() const; // ATI OpenCL SDK v2.0 does not support
                         // out of order execution so queues are
                         // inherently serialized anyway

    // return time of: enqueue, submit, start, end
    std::vector<unsigned long> profileEvent(const size_t event_index);

    // device properties
    size_t maxWorkGroupSize();
    size_t maxComputeUnits();
    size_t maxMemAlloc();
    size_t maxConstBuffer();
    size_t localMemory();
    size_t globalMemory();

    // debugging
    void print();

}; // class OCLApp

#include "OCLApp.tcc"

}; // namespace

#endif
