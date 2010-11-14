#ifndef _GATLAS_OCL_UTIL_HPP_
#define _GATLAS_OCL_UTIL_HPP_

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
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "OCLSTL.hpp"

#include "declare_namespace"

typedef std::vector<bool>                 vec_bool;
typedef std::vector<size_t>               vec_size_t;
typedef std::vector<std::vector<size_t> > vec_vec_size_t;
typedef std::vector<void*>                vec_voidptr;
typedef std::vector<float*>               vec_floatptr;
typedef std::map<int, std::set<int> >     map_setindex;

template <typename X>
bool checkFail(const cl_int,
               const X);
template <typename X, typename Y>
bool checkFail(const cl_int,
               const X, const Y);
template <typename X, typename Y, typename Z>
bool checkFail(const cl_int,
               const X, const Y, const Z);
template <typename X, typename Y, typename Z,
          typename A>
bool checkFail(const cl_int,
               const X, const Y, const Z, const A);
template <typename X, typename Y, typename Z,
          typename A, typename B>
bool checkFail(const cl_int,
               const X, const Y, const Z, const A, const B);
template <typename X, typename Y, typename Z,
          typename A, typename B, typename C>
bool checkFail(const cl_int,
               const X, const Y, const Z, const A, const B, const C);

std::string devtype(const cl_device_type);
const char *devinfo(const cl_device_info);
template <typename T> std::string devinfovalue(const cl_device_info, const T);

std::vector<size_t> idxlist(const int v0 = -1,
                            const int v1 = -1,
                            const int v2 = -1);

template <typename T> T* alloc_memalign(const size_t);
template <typename T> T* alloc_memalign(const size_t, const size_t ALIGNMENT);
template <typename SCALAR, size_t N> SCALAR* alloc_memalign(const size_t);

template <typename T> bool isfloat();

#include "OCLUtil.tcc"

}; // namespace

#endif
