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

template <typename T>
int
OCLApp::createBufferWithPointer(const size_t n,
                                const cl_mem_flags flags,
                                T *ptr,
                                bool own_memptr)
{
    // create buffer
    cl_int status;
    const cl_mem buffer = clCreateBuffer(oclBase.getContext(device_index),
                                         flags,
                                         n * sizeof(T),
                                         ptr,
                                         &status);

    // check for failure
    if (checkFail(status, "create buffer ", ptr))
        return -1;

    // success
    const int buffer_index = membuffers.size();
    membuffers.push_back(buffer);
    memptrs.push_back(ptr);
    memownptrs.push_back(own_memptr);
    memsizeoftype.push_back(sizeof(T));
    memsize.push_back(n);

    return buffer_index;
}

template <typename T>
int
OCLApp::createBufferAllocMemory(const size_t n,
                                const cl_mem_flags flags,
                                const size_t ALIGNMENT)
{
    T *ptr = alloc_memalign<T>(n, ALIGNMENT);

    if (!ptr) return -1; // failure, could not allocate aligned memory

    const int index = createBufferWithPointer(n, flags, ptr);
    if (-1 == index) free(ptr);

    return index;
}

template <typename T>
int
OCLApp::createBuffer(const size_t n,
                     BUFFER_FLAGS mode,
                     bool pinned)
{
    return createBuffer<T, 1>(n, mode, pinned);
}

template <typename T, size_t N>
int
OCLApp::createBuffer(const size_t n,
                     BUFFER_FLAGS mode,
                     bool pinned)
{
    int flags;

    switch (mode)
    {
        case (READ) : flags = CL_MEM_READ_ONLY; break;
        case (WRITE) : flags = CL_MEM_WRITE_ONLY; break;
        case (READWRITE) : flags = CL_MEM_READ_WRITE; break;
    }

    if (pinned)
    {
        flags |= CL_MEM_ALLOC_HOST_PTR;
        return createBufferWithPointer<T>(n, flags, NULL);
    }
    else
    {
        flags |= CL_MEM_USE_HOST_PTR;
        return createBufferAllocMemory<T>(n, flags, N * Type<T>::ALIGNMENT);
    }
}

template <typename T>
int
OCLApp::createBuffer(const size_t n,
                     BUFFER_FLAGS mode,
                     T *ptr,
                     bool pinned)
{
    int flags;

    switch (mode)
    {
        case (READ) : flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR; break;
        case (WRITE) : flags = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR; break;
        case (READWRITE) : flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR; break;
    }

    if (pinned) flags |= CL_MEM_ALLOC_HOST_PTR;

    // caller maintains ownership of ptr
    return createBufferWithPointer<T>(n, flags, ptr, false);
}

template <typename T>
void
OCLApp::memsetBuffer(const size_t buffer_index, const T value)
{
    memset(static_cast<void*>(memptrs[buffer_index]),
           value,
           memsize[buffer_index] * memsizeoftype[buffer_index]);

}

template <typename T>
T*
OCLApp::bufferPtr(const size_t buffer_index) const
{
    return static_cast<T*>(memptrs[buffer_index]);
}

template <typename T>
bool
OCLApp::setArgLocal(const size_t kernel_index,
                    const size_t kernel_arg_index,
                    const size_t n)
{
    return !checkFail(
        clSetKernelArg(kernels[kernel_index],
                       kernel_arg_index,
                       n * sizeof(T),
                       NULL),
        "set kernel ", kernel_index,
        " argument ", kernel_arg_index,
        " local memory size ", n * sizeof(T));
}

template <typename T>
bool
OCLApp::setArgValue(const size_t kernel_index,
                    const size_t kernel_arg_index,
                    const T value)
{
    return !checkFail(
        clSetKernelArg(kernels[kernel_index],
                       kernel_arg_index,
                       sizeof(T),
                       &value),
        "set kernel ", kernel_index,
        " argument ", kernel_arg_index,
        " to ", value);
}
