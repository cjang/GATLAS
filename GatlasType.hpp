#ifndef _GATLAS_TYPE_HPP_
#define _GATLAS_TYPE_HPP_

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

#include <sstream>
#include <string>

#include "declare_namespace"

// length of a type, this is for handling scalar and vector types
template <typename T> size_t lengthof();

// base type of both scalar and vector types
template <typename T> std::string basenameof();

// string names of scalar types
template <typename T> std::string nameof();

// string names of scalar types without const qualifier
template <typename T> std::string castto();

#define DECL_TYPE_FUNCS(SCALAR) \
template <> size_t lengthof< SCALAR >() { return 1; } \
template <> size_t lengthof< SCALAR * >() { return 1; } \
template <> size_t lengthof< SCALAR * const >() { return 1; } \
template <> size_t lengthof< const SCALAR >() { return 1; } \
template <> size_t lengthof< const SCALAR * >() { return 1; } \
template <> size_t lengthof< const SCALAR * const >() { return 1; } \
template <> std::string basenameof< SCALAR >() { return #SCALAR ; } \
template <> std::string basenameof< SCALAR * >() { return #SCALAR ; } \
template <> std::string basenameof< SCALAR * const >() { return #SCALAR ; } \
template <> std::string basenameof< const SCALAR >() { return #SCALAR ; } \
template <> std::string basenameof< const SCALAR * >() { return #SCALAR ; } \
template <> std::string basenameof< const SCALAR * const >() { return #SCALAR ; } \
template <> std::string nameof< SCALAR >() { return #SCALAR ; } \
template <> std::string nameof< SCALAR * >() { return #SCALAR "*" ; } \
template <> std::string nameof< SCALAR * const >() { return #SCALAR "* const" ; } \
template <> std::string nameof< const SCALAR >() { return "const " #SCALAR ; } \
template <> std::string nameof< const SCALAR * >() { return "const " #SCALAR "*" ; } \
template <> std::string nameof< const SCALAR * const >() { return "const " #SCALAR "* const" ; } \
template <> std::string castto< SCALAR >() { return #SCALAR ; } \
template <> std::string castto< SCALAR * >() { return #SCALAR "*" ; } \
template <> std::string castto< SCALAR * const >() { return #SCALAR "*" ; } \
template <> std::string castto< const SCALAR >() { return #SCALAR ; } \
template <> std::string castto< const SCALAR * >() { return #SCALAR "*" ; } \
template <> std::string castto< const SCALAR * const >() { return #SCALAR "*" ; }

// 2D image type in OpenCL kernel language
class image2d_t { };

// image/texture sampler in OpenCL kernel language
class sampler_t { };

// string names of vector types
template <typename T, size_t N> std::string nameof() {
    std::stringstream ss;
    if (1 == N)
        ss << nameof<T>();
    else
        ss << nameof<T>() << N;
    return ss.str();
}

// string names of vector types suitable for casting (no const qualifier)
template <typename T, size_t N> std::string castto() {
    std::stringstream ss;
    if (1 == N)
        ss << castto<T>();
    else
        ss << castto<T>() << N;
    return ss.str();
}

template <typename SCALAR, size_t N> class VecType { };

#define DECL_VECTYPE_FUNCS(SCALAR, N) \
template <> size_t lengthof< VecType< SCALAR, N > >() { return N; } \
template <> size_t lengthof< VecType< SCALAR, N > * >() { return N; } \
template <> size_t lengthof< VecType< SCALAR, N > * const >() { return N; } \
template <> size_t lengthof< const VecType< SCALAR, N > >() { return N; } \
template <> size_t lengthof< const VecType< SCALAR, N > * >() { return N; } \
template <> size_t lengthof< const VecType< SCALAR, N > * const >() { return N; } \
template <> std::string basenameof< VecType< SCALAR, N > >() { return basenameof< SCALAR >(); } \
template <> std::string basenameof< VecType< SCALAR, N > * >() { return basenameof< SCALAR >(); } \
template <> std::string basenameof< VecType< SCALAR, N > * const >() { return basenameof< SCALAR >(); } \
template <> std::string basenameof< const VecType< SCALAR, N > >() { return basenameof< SCALAR >(); } \
template <> std::string basenameof< const VecType< SCALAR, N > * >() { return basenameof< SCALAR >(); } \
template <> std::string basenameof< const VecType< SCALAR, N > * const >() { return basenameof< SCALAR >(); } \
template <> std::string nameof< VecType< SCALAR, N > >() { return nameof< SCALAR, N >(); } \
template <> std::string nameof< VecType< SCALAR, N > * >() { return nameof< SCALAR, N >().append("*"); } \
template <> std::string nameof< VecType< SCALAR, N > * const >() { return nameof< SCALAR, N >().append("* const"); } \
template <> std::string nameof< const VecType< SCALAR, N > >() { return nameof< SCALAR, N >().insert(0, "const "); } \
template <> std::string nameof< const VecType< SCALAR, N > * >() { return nameof< SCALAR, N >().insert(0, "const ").append("*"); } \
template <> std::string nameof< const VecType< SCALAR, N > * const >() { return nameof< SCALAR, N >().insert(0, "const ").append("* const"); } \
template <> std::string castto< VecType< SCALAR, N > >() { return castto< SCALAR, N >(); } \
template <> std::string castto< VecType< SCALAR, N > * >() { return castto< SCALAR, N >().append("*"); } \
template <> std::string castto< VecType< SCALAR, N > * const >() { return castto< SCALAR, N >().append("*"); } \
template <> std::string castto< const VecType< SCALAR, N > >() { return castto< SCALAR, N >(); } \
template <> std::string castto< const VecType< SCALAR, N > * >() { return castto< SCALAR, N >().append("*"); } \
template <> std::string castto< const VecType< SCALAR, N > * const >() { return castto< SCALAR, N >().append("*"); }

#define DECL_VECTYPES_FUNCS(SCALAR) \
DECL_VECTYPE_FUNCS(SCALAR, 1) \
DECL_VECTYPE_FUNCS(SCALAR, 2) \
DECL_VECTYPE_FUNCS(SCALAR, 4) \
DECL_VECTYPE_FUNCS(SCALAR, 8) \
DECL_VECTYPE_FUNCS(SCALAR, 16)

// OpenCL pragmas required for specific scalar types
template <typename SCALAR> std::ostream& pragma_extension(std::ostream& os);

}; // namespace

#endif
