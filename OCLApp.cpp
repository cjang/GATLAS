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

#include "OCLApp.hpp"
#include "OCLUtil.hpp"
#include <iostream>
#include <string.h>

using namespace std;

#include "declare_namespace"

bool
OCLApp::releaseBuffers()
{
    bool allOk = true;

    for (size_t i = 0; i < membuffers.size(); i++)
    {
        if (checkFail(clReleaseMemObject(membuffers[i]), "release buffer", i))
            allOk = false;

        if (memownptrs[i]) free(memptrs[i]);
    }

    membuffers.clear();
    memptrs.clear();
    memownptrs.clear();
    memsizeoftype.clear();
    memsize.clear();

    return allOk;
}

bool
OCLApp::releaseImages()
{
    bool allOk = true;

    for (size_t i = 0; i < imgbuffers.size(); i++)
    {
        if (checkFail(clReleaseMemObject(imgbuffers[i]), "release image", i))
            allOk = false;

        if (imgownptrs[i]) free(imgptrs[i]);
    }

    imgbuffers.clear();
    imgptrs.clear();
    imgownptrs.clear();
    imgwidth.clear();
    imgheight.clear();

    return allOk;
}

bool
OCLApp::releaseSamplers()
{
    bool allOk = true;

    for (size_t i = 0; i < imgsamplers.size(); i++)
    {
        if (checkFail(clReleaseSampler(imgsamplers[i]), "release sampler", i))
            allOk = false;
    }

    imgsamplers.clear();

    return allOk;
}

void
OCLApp::ownBuffer(const size_t buffer_index)
{
    memownptrs[buffer_index] = true;
}

void
OCLApp::ownImage(const size_t image_index)
{
    imgownptrs[image_index] = true;
}

bool
OCLApp::releaseKernels()
{
    bool allOk = true;

    for (size_t i = 0; i < kernels.size(); i++)
        if (checkFail(clReleaseKernel(kernels[i]), "release kernel", i))
            allOk = false;

    kernels.clear();

    return allOk;
}

bool
OCLApp::releaseProgram()
{
    if (program)
        if (checkFail(clReleaseProgram(program), "release program") ||
            !releaseKernels())
            return false; // failure

    return true; // success

}

int
OCLApp::createSampler()
{
    cl_int status;
    const cl_sampler sampler = clCreateSampler(oclBase.getContext(device_index),
                                               CL_FALSE,          // CLK_NORMALIZED_COORDS_FALSE
                                               CL_ADDRESS_NONE,   // CLK_ADDRESS_NONE
                                               CL_FILTER_NEAREST, // CLK_FILTER_NEAREST
                                               &status);

    // check for failure
    if (checkFail(status, "create sampler"))
        return -1;

    // success
    const int sampler_index = imgsamplers.size();
    imgsamplers.push_back(sampler);

    return sampler_index;
}

void
OCLApp::memsetImage(const size_t image_index, const float value)
{
    memset(static_cast<void*>(imgptrs[image_index]),
           value,
           imgwidth[image_index] * imgheight[image_index] * 4 * sizeof(float));
}

OCLApp::OCLApp(OCLBase& ocl_base, const size_t index)
    : oclBase(ocl_base),
      device_index(index),
      program(NULL),
      events_pending(0)
{
}

OCLApp::~OCLApp()
{
    wait();            // events
    releaseProgram();  // programs and kernels
    releaseBuffers();  // buffers
    releaseImages();   // images
    releaseSamplers(); // image samplers
}

bool
OCLApp::buildProgram(const vector<string>& program_source,
                     const string& options)
{
    // release any pre-existing program and kernels
    if (!releaseProgram()) return false; // failure

    // program source in array of C strings form
    const char *source_array[program_source.size()];
    for (size_t i = 0; i < program_source.size(); i++)
    {
        source_array[i] = program_source[i].c_str();
    }

    // create program
    cl_int status;
    program = clCreateProgramWithSource(oclBase.getContext(device_index),
                                        program_source.size(),
                                        source_array,
                                        NULL,
                                        &status);
    if (checkFail(status, "create program")) return false; // failure

    // build program
    if (checkFail(clBuildProgram(program,
                                 1,
                                 &oclBase.getDevice(device_index),
                                 options.c_str(),
                                 NULL,
                                 NULL),
                  "build program")) {

        cerr << buildLog() << endl;
        return false; // failure
    }

    return true; // success
}

string
OCLApp::buildLog() const
{
    string msg;

    size_t msgsize = 0;
    char msgbuf[10240];
    memset(msgbuf, 0, sizeof(msgbuf));

    if (!checkFail(
        clGetProgramBuildInfo(
            program,
            oclBase.getDevice(device_index),
            CL_PROGRAM_BUILD_LOG,
            sizeof(msgbuf),
            msgbuf,
            &msgsize),
        "get program build log")) msg = msgbuf;

    return msg;
}

int
OCLApp::createKernel(const string& kernel_name)
{
    // create the kernel object
    cl_int status;
    const cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &status);

    // check for failure
    if (checkFail(status, "create kernel ", kernel_name))
        return -1; // failure

    // store the kernel handle
    const int kernel_index = kernels.size();
    kernels.push_back(kernel);

    // find maximum work group size for this kernel
    // not if call fails, the value 0 indicates failure (impossible value)
    size_t maxsize = 0;
    checkFail(
        clGetKernelWorkGroupInfo(kernel,
                                 oclBase.getDevice(device_index),
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(maxsize),
                                 &maxsize,
                                 NULL),
        "get kernel max work group size for ",
        kernel_name);
    kernel_wgsize.push_back(maxsize);

    return kernel_index;
}

bool
OCLApp::setArgImage(const size_t kernel_index,
                    const size_t kernel_arg_index,
                    const size_t image_index)
{
    return !checkFail(
        clSetKernelArg(kernels[kernel_index],
                       kernel_arg_index,
                       sizeof(cl_mem),
                       &imgbuffers[image_index]),
        "set kernel ", kernel_index,
        " argument ", kernel_arg_index,
        " to image ", image_index);
}

bool
OCLApp::setArgSampler(const size_t kernel_index,
                      const size_t kernel_arg_index,
                      const size_t sampler_index)
{
    return !checkFail(
        clSetKernelArg(kernels[kernel_index],
                       kernel_arg_index,
                       sizeof(cl_sampler),
                       &imgsamplers[sampler_index]),
        "set kernel ", kernel_index,
        " argument ", kernel_arg_index,
        " to sampler ", sampler_index);
}

bool
OCLApp::setArgGlobal(const size_t kernel_index,
                     const size_t kernel_arg_index,
                     const size_t buffer_index)
{
    return !checkFail(
        clSetKernelArg(kernels[kernel_index],
                       kernel_arg_index,
                       sizeof(cl_mem),
                       &membuffers[buffer_index]),
        "set kernel ", kernel_index,
        " argument ", kernel_arg_index,
        " to buffer ", buffer_index);
}

int
OCLApp::enqueueKernel(const size_t kernel_index,
                      const vector<size_t>& global_dim,
                      const vector<size_t>& local_dim)
{
    return enqueueKernel(kernel_index,
                         global_dim,
                         local_dim,
                         vector<size_t>());
}

int
OCLApp::enqueueKernel(const size_t kernel_index,
                      const vector<size_t>& global_dim,
                      const vector<size_t>& local_dim,
                      const size_t event_index)
{
    return enqueueKernel(kernel_index,
                         global_dim,
                         local_dim,
                         idxlist(event_index));
}

int
OCLApp::enqueueKernel(const size_t kernel_index,
                      const vector<size_t>& global_dim,
                      const vector<size_t>& local_dim,
                      const size_t event_index_0,
                      const size_t event_index_1)
{
    return enqueueKernel(kernel_index,
                         global_dim,
                         local_dim,
                         idxlist(event_index_0,
                                 event_index_1));
}

int
OCLApp::enqueueKernel(const size_t kernel_index,
                      const vector<size_t>& global_dim,
                      const vector<size_t>& local_dim,
                      const size_t event_index_0,
                      const size_t event_index_1,
                      const size_t event_index_2)
{
    return enqueueKernel(kernel_index,
                         global_dim,
                         local_dim,
                         idxlist(event_index_0,
                                 event_index_1,
                                 event_index_2));
}

int
OCLApp::enqueueKernel(const size_t kernel_index,
                      const vector<size_t>& global_dim,
                      const vector<size_t>& local_dim,
                      const vector<size_t>& event_indexes)
{
    cl_event event, event_wait_list[event_indexes.size()];

    for (size_t i = 0; i < event_indexes.size(); i++)
        event_wait_list[i] = events[event_indexes[i]];

    if (checkFail(
        clEnqueueNDRangeKernel(oclBase.getQueue(device_index),
                               kernels[kernel_index],
                               global_dim.size(),
                               NULL, // global work offset must be null
                               &global_dim[0],
                               &local_dim[0],
                               event_indexes.size(),
                               event_indexes.empty() ? NULL : event_wait_list,
                               &event),
        "enqueue kernel ", kernel_index)) return -1; // failure

    const size_t event_index = events.size();
    events.push_back(event);
    events_waited.push_back(false);
    events_pending++;

    return event_index;
}

int
OCLApp::enqueueReadBuffer(const size_t buffer_index)
{
    return enqueueReadBuffer(buffer_index, 0, memsize[buffer_index],
                             vector<size_t>());
}

int
OCLApp::enqueueReadBuffer(const size_t buffer_index,
                          const vector<size_t>& event_indexes)
{
    return enqueueReadBuffer(buffer_index, 0, memsize[buffer_index],
                             event_indexes);
}

int
OCLApp::enqueueReadBuffer(const size_t buffer_index,
                          const size_t offset,
                          const size_t n)
{
    return enqueueReadBuffer(buffer_index, offset, n, vector<size_t>());
}

int
OCLApp::enqueueReadBuffer(const size_t buffer_index,
                          const size_t offset,
                          const size_t n,
                          const vector<size_t>& event_indexes)
{
    cl_event event, event_wait_list[event_indexes.size()];

    for (size_t i = 0; i < event_indexes.size(); i++)
        event_wait_list[i] = events[event_indexes[i]];

    if (checkFail(
        clEnqueueReadBuffer(oclBase.getQueue(device_index),
                            membuffers[buffer_index],
                            CL_FALSE, // non-blocking
                            offset,
                            n * memsizeoftype[buffer_index],
                            memptrs[buffer_index],
                            event_indexes.size(),
                            event_indexes.empty() ? NULL : event_wait_list,
                            &event),
        "enqueue read buffer ", buffer_index,
        " offset ", offset,
        " n ", n)) return -1; // failure

    const size_t event_index = events.size();
    events.push_back(event);
    events_waited.push_back(false);
    events_pending++;

    return event_index;
}

int
OCLApp::enqueueWriteBuffer(const size_t buffer_index)
{
    return enqueueWriteBuffer(buffer_index, 0, memsize[buffer_index],
                              vector<size_t>());
}

int
OCLApp::enqueueWriteBuffer(const size_t buffer_index,
                           const vector<size_t>& event_indexes)
{
    return enqueueWriteBuffer(buffer_index, 0, memsize[buffer_index],
                              event_indexes);
}

int
OCLApp::enqueueWriteBuffer(const size_t buffer_index,
                           const size_t offset,
                           const size_t n)
{
    return enqueueWriteBuffer(buffer_index, offset, n, vector<size_t>());
}

int
OCLApp::enqueueWriteBuffer(const size_t buffer_index,
                           const size_t offset,
                           const size_t n,
                           const vector<size_t>& event_indexes)
{
    cl_event event, event_wait_list[event_indexes.size()];

    for (size_t i = 0; i < event_indexes.size(); i++)
        event_wait_list[i] = events[event_indexes[i]];

    if (checkFail(
        clEnqueueWriteBuffer(oclBase.getQueue(device_index),
                             membuffers[buffer_index],
                             CL_FALSE, // non-blocking
                             offset,
                             n * memsizeoftype[buffer_index],
                             memptrs[buffer_index],
                             event_indexes.size(),
                             event_indexes.empty() ? NULL : event_wait_list,
                             &event),
        "enqueue write buffer ", buffer_index,
        " offset ", offset,
        " n ", n)) return -1; // failure

    const size_t event_index = events.size();
    events.push_back(event);
    events_waited.push_back(false);
    events_pending++;

    return event_index;
}

int
OCLApp::enqueueCopyBuffer(const size_t src_buffer_index,
                          const size_t dest_buffer_index)
{
    return enqueueCopyBuffer(src_buffer_index,
                             dest_buffer_index,
                             0,
                             0,
                             memsize[src_buffer_index],
                             vector<size_t>());
}

int
OCLApp::enqueueCopyBuffer(const size_t src_buffer_index,
                          const size_t dest_buffer_index,
                          const vector<size_t>& event_indexes)
{
    return enqueueCopyBuffer(src_buffer_index,
                             dest_buffer_index,
                             0,
                             0,
                             memsize[src_buffer_index],
                             event_indexes);
}

int
OCLApp::enqueueCopyBuffer(const size_t src_buffer_index,
                          const size_t dest_buffer_index,
                          const size_t src_offset,
                          const size_t dest_offset,
                          const size_t n)
{
    return enqueueCopyBuffer(src_buffer_index,
                             dest_buffer_index,
                             src_offset,
                             dest_offset,
                             n,
                             vector<size_t>());
}

int
OCLApp::enqueueCopyBuffer(const size_t src_buffer_index,
                          const size_t dest_buffer_index,
                          const size_t src_offset,
                          const size_t dest_offset,
                          const size_t n,
                          const vector<size_t>& event_indexes)
{
    cl_event event, event_wait_list[event_indexes.size()];

    for (size_t i = 0; i < event_indexes.size(); i++)
        event_wait_list[i] = events[event_indexes[i]];

    if (checkFail(
        clEnqueueCopyBuffer(oclBase.getQueue(device_index),
                            membuffers[src_buffer_index],
                            membuffers[dest_buffer_index],
                            src_offset,
                            dest_offset,
                            // type size is always from source buffer
                            n * memsizeoftype[src_buffer_index],
                            event_indexes.size(),
                            event_indexes.empty() ? NULL : event_wait_list,
                            &event),
        "enqueue copy buffer from ", src_buffer_index,
        " to ", dest_buffer_index)) return -1; // failure

    const size_t event_index = events.size();
    events.push_back(event);
    events_waited.push_back(false);
    events_pending++;

    return event_index;
}

int
OCLApp::enqueueReadImage(const size_t image_index)
{
    return enqueueReadImage(image_index,
                            0, 0,
                            imgwidth[image_index], imgheight[image_index],
                            std::vector<size_t>());
}

int
OCLApp::enqueueReadImage(const size_t image_index,
                         const std::vector<size_t>& event_indexes)
{
    return enqueueReadImage(image_index,
                            0, 0,
                            imgwidth[image_index], imgheight[image_index],
                            event_indexes);
}

int
OCLApp::enqueueReadImage(const size_t image_index,
                         const size_t origin_x,
                         const size_t origin_y,
                         const size_t region_width,
                         const size_t region_height)
{
    return enqueueReadImage(image_index,
                            origin_x, origin_y,
                            region_width, region_height,
                            std::vector<size_t>());
}

int
OCLApp::enqueueReadImage(const size_t buffer_index,
                         const size_t origin_x,
                         const size_t origin_y,
                         const size_t region_width,
                         const size_t region_height,
                         const std::vector<size_t>& event_indexes)
{
    cl_event event, event_wait_list[event_indexes.size()];

    for (size_t i = 0; i < event_indexes.size(); i++)
        event_wait_list[i] = events[event_indexes[i]];

    size_t origin[3], region[3];
    origin[0] = origin_x;
    origin[1] = origin_y;
    origin[2] = 0;
    region[0] = region_width;
    region[1] = region_height;
    region[2] = 1;

    if (checkFail(
        clEnqueueReadImage(oclBase.getQueue(device_index),
                           imgbuffers[buffer_index],
                           CL_FALSE, // non-blocking
                           origin,
                           region,
                           0,
                           0,
                           imgptrs[buffer_index],
                           event_indexes.size(),
                           event_indexes.empty() ? NULL : event_wait_list,
                           &event),
        "enqueue read image ", buffer_index,
        " origin_x ", origin_x,
        " origin_y ", origin_y)) return -1; // failure

    const size_t event_index = events.size();
    events.push_back(event);
    events_waited.push_back(false);
    events_pending++;

    return event_index;
}

int
OCLApp::enqueueWriteImage(const size_t image_index)
{
    return enqueueWriteImage(image_index,
                             0, 0,
                             imgwidth[image_index], imgheight[image_index],
                             vector<size_t>());
}

int
OCLApp::enqueueWriteImage(const size_t image_index,
                          const vector<size_t>& event_indexes)
{
    return enqueueWriteImage(image_index,
                             0, 0,
                             imgwidth[image_index], imgheight[image_index],
                             event_indexes);
}

int
OCLApp::enqueueWriteImage(const size_t image_index,
                          const size_t origin_x,
                          const size_t origin_y,
                          const size_t region_width,
                          const size_t region_height)
{
    return enqueueWriteImage(image_index,
                             origin_x, origin_y,
                             region_width, region_height,
                             vector<size_t>());
}

int
OCLApp::enqueueWriteImage(const size_t image_index,
                          const size_t origin_x,
                          const size_t origin_y,
                          const size_t region_width,
                          const size_t region_height,
                          const vector<size_t>& event_indexes)
{
    cl_event event, event_wait_list[event_indexes.size()];

    for (size_t i = 0; i < event_indexes.size(); i++)
        event_wait_list[i] = events[event_indexes[i]];

    size_t origin[3], region[3];
    origin[0] = origin_x;
    origin[1] = origin_y;
    origin[2] = 0;
    region[0] = region_width;
    region[1] = region_height;
    region[2] = 1;

    if (checkFail(
        clEnqueueWriteImage(oclBase.getQueue(device_index),
                            imgbuffers[image_index],
                            CL_FALSE, // non-blocking
                            origin,
                            region,
                            0,
                            0,
                            imgptrs[image_index],
                            event_indexes.size(),
                            event_indexes.empty() ? NULL : event_wait_list,
                            &event),
        "enqueue write image ", image_index,
        " origin_x ", origin_x,
        " origin_y ", origin_y)) return -1; // failure

    const size_t event_index = events.size();
    events.push_back(event);
    events_waited.push_back(false);
    events_pending++;

    return event_index;
}

int
OCLApp::enqueueCopyImage(const size_t src_image_index,
                         const size_t dest_image_index)
{
    return enqueueCopyImage(src_image_index,
                            dest_image_index,
                            0, 0,
                            0, 0,
                            // always take region dimensions from source image
                            imgwidth[src_image_index], imgheight[src_image_index],
                            std::vector<size_t>());
}

int
OCLApp::enqueueCopyImage(const size_t src_image_index,
                         const size_t dest_image_index,
                         const std::vector<size_t>& event_indexes)
{
    return enqueueCopyImage(src_image_index,
                            dest_image_index,
                            0, 0,
                            0, 0,
                            // always take region dimensions from source image
                            imgwidth[src_image_index], imgheight[src_image_index],
                            event_indexes);
}

int
OCLApp::enqueueCopyImage(const size_t src_image_index,
                         const size_t dest_image_index,
                         const size_t src_origin_x,
                         const size_t src_origin_y,
                         const size_t dest_origin_x,
                         const size_t dest_origin_y,
                         const size_t region_width,
                         const size_t region_height)
{
    return enqueueCopyImage(src_image_index,
                            dest_image_index,
                            src_origin_x, src_origin_y,
                            dest_origin_x, dest_origin_y,
                            region_width, region_height,
                            std::vector<size_t>());
}

int
OCLApp::enqueueCopyImage(const size_t src_image_index,
                         const size_t dest_image_index,
                         const size_t src_origin_x,
                         const size_t src_origin_y,
                         const size_t dest_origin_x,
                         const size_t dest_origin_y,
                         const size_t region_width,
                         const size_t region_height,
                         const std::vector<size_t>& event_indexes)
{
    cl_event event, event_wait_list[event_indexes.size()];

    for (size_t i = 0; i < event_indexes.size(); i++)
        event_wait_list[i] = events[event_indexes[i]];

    size_t src_origin[3], dest_origin[3], region[3];
    src_origin[0] = src_origin_x;
    src_origin[1] = src_origin_y;
    src_origin[2] = 0;
    dest_origin[0] = dest_origin_x;
    dest_origin[1] = dest_origin_y;
    dest_origin[2] = 0;
    region[0] = region_width;
    region[1] = region_height;
    region[2] = 1;

    if (checkFail(
        clEnqueueCopyImage(oclBase.getQueue(device_index),
                           imgbuffers[src_image_index],
                           imgbuffers[dest_image_index],
                           src_origin,
                           dest_origin,
                           region,
                           event_indexes.size(),
                           event_indexes.empty() ? NULL : event_wait_list,
                           &event),
        "enqueue copy image from ", src_image_index,
        " to ", dest_image_index)) return -1; // failure

    const size_t event_index = events.size();
    events.push_back(event);
    events_waited.push_back(false);
    events_pending++;

    return event_index;
}

int
OCLApp::enqueueCopyBufferToImage(const size_t src_buffer_index,
                                 const size_t dest_image_index)
{
    return enqueueCopyBufferToImage(src_buffer_index,
                                    dest_image_index,
                                    0,
                                    0, 0,
                                    imgwidth[dest_image_index], imgheight[dest_image_index],
                                    std::vector<size_t>());
}

int
OCLApp::enqueueCopyBufferToImage(const size_t src_buffer_index,
                                 const size_t dest_image_index,
                                 const std::vector<size_t>& event_indexes)
{
    return enqueueCopyBufferToImage(src_buffer_index,
                                    dest_image_index,
                                    0,
                                    0, 0,
                                    imgwidth[dest_image_index], imgheight[dest_image_index],
                                    event_indexes);
}

int
OCLApp::enqueueCopyBufferToImage(const size_t src_buffer_index,
                                 const size_t dest_image_index,
                                 const size_t src_offset,
                                 const size_t dest_origin_x,
                                 const size_t dest_origin_y,
                                 const size_t region_width,
                                 const size_t region_height)
{
    return enqueueCopyBufferToImage(src_buffer_index,
                                    dest_image_index,
                                    src_offset,
                                    dest_origin_x, dest_origin_y,
                                    imgwidth[dest_image_index], imgheight[dest_image_index],
                                    std::vector<size_t>());
}

int
OCLApp::enqueueCopyBufferToImage(const size_t src_buffer_index,
                                 const size_t dest_image_index,
                                 const size_t src_offset,
                                 const size_t dest_origin_x,
                                 const size_t dest_origin_y,
                                 const size_t region_width,
                                 const size_t region_height,
                                 const std::vector<size_t>& event_indexes)
{
    cl_event event, event_wait_list[event_indexes.size()];

    for (size_t i = 0; i < event_indexes.size(); i++)
        event_wait_list[i] = events[event_indexes[i]];

    size_t dest_origin[3], region[3];
    dest_origin[0] = dest_origin_x;
    dest_origin[1] = dest_origin_y;
    dest_origin[2] = 0;
    region[0] = region_width;
    region[1] = region_height;
    region[2] = 1;

    if (checkFail(
        clEnqueueCopyBufferToImage(oclBase.getQueue(device_index),
                                   membuffers[src_buffer_index],
                                   imgbuffers[dest_image_index],
                                   src_offset,
                                   dest_origin,
                                   region,
                                   event_indexes.size(),
                                   event_indexes.empty() ? NULL : event_wait_list,
                                   &event),
        "enqueue copy buffer to image from ", src_buffer_index,
        " to ", dest_image_index)) return -1; // failure

    const size_t event_index = events.size();
    events.push_back(event);
    events_waited.push_back(false);
    events_pending++;

    return event_index;
}

int
OCLApp::enqueueCopyImageToBuffer(const size_t src_image_index,
                                 const size_t dest_buffer_index)
{
    return enqueueCopyImageToBuffer(src_image_index,
                                    dest_buffer_index,
                                    0, 0,
                                    0,
                                    imgwidth[src_image_index], imgheight[src_image_index],
                                    std::vector<size_t>());
}

int
OCLApp::enqueueCopyImageToBuffer(const size_t src_image_index,
                                 const size_t dest_buffer_index,
                                 const std::vector<size_t>& event_indexes)
{
    return enqueueCopyImageToBuffer(src_image_index,
                                    dest_buffer_index,
                                    0, 0,
                                    0,
                                    imgwidth[src_image_index], imgheight[src_image_index],
                                    event_indexes);
}

int
OCLApp::enqueueCopyImageToBuffer(const size_t src_image_index,
                                 const size_t dest_buffer_index,
                                 const size_t src_origin_x,
                                 const size_t src_origin_y,
                                 const size_t dest_offset,
                                 const size_t region_width,
                                 const size_t region_height)
{
    return enqueueCopyImageToBuffer(src_image_index,
                                    dest_buffer_index,
                                    src_origin_x, src_origin_y,
                                    dest_offset,
                                    imgwidth[src_image_index], imgheight[src_image_index],
                                    std::vector<size_t>());
}

int
OCLApp::enqueueCopyImageToBuffer(const size_t src_image_index,
                                 const size_t dest_buffer_index,
                                 const size_t src_origin_x,
                                 const size_t src_origin_y,
                                 const size_t dest_offset,
                                 const size_t region_width,
                                 const size_t region_height,
                                 const std::vector<size_t>& event_indexes)
{
    cl_event event, event_wait_list[event_indexes.size()];

    for (size_t i = 0; i < event_indexes.size(); i++)
        event_wait_list[i] = events[event_indexes[i]];

    size_t src_origin[3], region[3];
    src_origin[0] = src_origin_x;
    src_origin[1] = src_origin_y;
    src_origin[2] = 0;
    region[0] = region_width;
    region[1] = region_height;
    region[2] = 1;

    if (checkFail(
        clEnqueueCopyImageToBuffer(oclBase.getQueue(device_index),
                                   imgbuffers[src_image_index],
                                   membuffers[dest_buffer_index],
                                   src_origin,
                                   region,
                                   dest_offset,
                                   event_indexes.size(),
                                   event_indexes.empty() ? NULL : event_wait_list,
                                   &event),
        "enqueue copy image to buffer from ", src_image_index,
        " to ", dest_buffer_index)) return -1; // failure

    const size_t event_index = events.size();
    events.push_back(event);
    events_waited.push_back(false);
    events_pending++;

    return event_index;
}

bool
OCLApp::wait()
{
    vec_event event_list;

    // make list of all active events
    for (size_t i = 0; i < events.size(); i++)
        if (!events_waited[i])
            event_list.push_back(events[i]);

    bool isOk = true;

    // wait for everything
    if (!event_list.empty())
        isOk = ! checkFail(
            clWaitForEvents(event_list.size(), &event_list[0]),
            "wait for ", event_list.size(), " events");

    // release events
    //for (size_t i = 0; i < event_list.size(); i++)
    for (size_t i = 0; i < events.size(); i++)
        clReleaseEvent(events[i]); // events are explicitly released
                                   // when wait() is called, never
                                   // as a side-effect of waiting on
                                   // specific events

    events.clear();
    events_waited.clear();
    events_pending = 0;

    return isOk;
}

bool
OCLApp::wait(const size_t event_index)
{
    return wait(idxlist(event_index));
}

bool
OCLApp::wait(const size_t event_index_0,
             const size_t event_index_1)
{
    return wait(idxlist(event_index_0, event_index_1));
}

bool
OCLApp::wait(const size_t event_index_0,
             const size_t event_index_1,
             const size_t event_index_2)
{
    return wait(idxlist(event_index_0, event_index_1, event_index_2));
}

bool
OCLApp::wait(const vector<size_t>& event_indexes)
{
    vec_event event_list;

    for (size_t i = 0; i < event_indexes.size(); i++)
    {
        // just to be safe, check that event has not been waited on before
        if (! events_waited[event_indexes[i]])
            event_list.push_back(events[event_indexes[i]]);
    }

    const bool isOk = ! checkFail(
        clWaitForEvents(event_list.size(), &event_list[0]),
        "wait for ", event_list.size(), " events");

    for (size_t i = 0; i < event_indexes.size(); i++)
    {
        // do not release events here as profiling information will be lost
        //clReleaseEvent(events[event_indexes[i]]);
        events_waited[event_indexes[i]] = true;
        events_pending--;
    }

    //if (0 == events_pending)
    //{
    //    events.clear();
    //    events_waited.clear();
    //}

    return isOk;
}

bool
OCLApp::finish() const
{
    return ! checkFail(clFinish(oclBase.getQueue(device_index)),
                       "finish queue ", device_index);
}

vector<unsigned long>
OCLApp::profileEvent(const size_t event_index)
{
    vector<unsigned long> event_times;

    const cl_profiling_info params[] = { CL_PROFILING_COMMAND_QUEUED,
                                         CL_PROFILING_COMMAND_SUBMIT,
                                         CL_PROFILING_COMMAND_START,
                                         CL_PROFILING_COMMAND_END };

    cl_ulong value;

    for (size_t i = 0; i < sizeof(params)/sizeof(cl_profiling_info); i++)
    {
        event_times.push_back(
            checkFail(clGetEventProfilingInfo(events[event_index],
                                              params[i],
                                              sizeof(value),
                                              &value,
                                              NULL),
                      "get event profiling info")
            ? 0 // failure
            : value);
    }

    return event_times;
}

size_t
OCLApp::maxWorkGroupSize()
{
    return oclBase.maxWorkGroupSize(device_index);
}

size_t
OCLApp::maxComputeUnits()
{
    return oclBase.maxComputeUnits(device_index);
}

size_t
OCLApp::maxMemAlloc()
{
    return oclBase.maxMemAlloc(device_index);
}

size_t
OCLApp::maxConstBuffer()
{
    return oclBase.maxConstBuffer(device_index);
}

size_t
OCLApp::localMemory()
{
    return oclBase.localMemory(device_index);
}

size_t
OCLApp::globalMemory()
{
    return oclBase.globalMemory(device_index);
}

void
OCLApp::print()
{
    cout
        << "device " << device_index
        << "\tprogram " << program
        << endl;

    for (size_t i = 0; i < kernels.size(); i++)
    {
        cout
            << "\tkernel[" << i << "] = " << kernels[i]
            << "\tmax work group size " << kernel_wgsize[i]
            << endl;
    }

    for (size_t i = 0; i < membuffers.size(); i++)
    {
        cout
            << "\tbuffer[" << i << "] = " << membuffers[i]
            << "\t" << memsizeoftype[i]
            << " x " << memsize[i]
            << "\thost ptr " << memptrs[i];

        if (memownptrs[i]) cout << "\town";

        cout << endl;
    }
}

} // namespace
