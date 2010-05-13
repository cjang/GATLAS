#ifndef _GATLAS_QUALIFIER_HPP_
#define _GATLAS_QUALIFIER_HPP_

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

#include <ostream>
#include <string>
#include "GatlasType.hpp"
#include "GatlasFormatting.hpp"

#include "declare_namespace"

// address space qualifiers
struct AddressSpace { virtual std::string str() const = 0; };

std::ostream& operator<< (std::ostream& os, const AddressSpace& qualifier);

#define DECL_ADDRESS_SPACE(TYPE, QUALIFIER) \
struct TYPE ## AddressSpace : public AddressSpace { std::string str() const; }; \
static const TYPE ## AddressSpace TYPE;

#define DEFN_ADDRESS_SPACE(TYPE, QUALIFIER) \
std::string TYPE ## AddressSpace::str() const { return #QUALIFIER " " ; }

DECL_ADDRESS_SPACE(GLOBAL, __global)
DECL_ADDRESS_SPACE(LOCAL, __local)
DECL_ADDRESS_SPACE(CONSTANT, __constant)
DECL_ADDRESS_SPACE(PRIVATE, __private)

// these are really not address space qualifiers but image access qualifiers
DECL_ADDRESS_SPACE(READONLY, __read_only)
DECL_ADDRESS_SPACE(WRITEONLY, __write_only)

DECL_ADDRESS_SPACE(DEFAULT, "")

// function qualifiers
struct FunctionAttribute { virtual std::string str() const = 0; };

std::ostream& operator<< (std::ostream& os, const FunctionAttribute& qualifier);

#define DECL_FUNCTION_ATTRIBUTE(TYPE, QUALIFIER) \
struct TYPE ## FunctionAttribute : public FunctionAttribute { std::string str() const; }; \
static const TYPE ## FunctionAttribute TYPE;

#define DEFN_FUNCTION_ATTRIBUTE(TYPE, QUALIFIER) \
std::string TYPE ## FunctionAttribute::str() const { return #QUALIFIER " " ; }

DECL_FUNCTION_ATTRIBUTE(KERNEL, __kernel)

template <typename T>
struct AutoVectorize : public FunctionAttribute
{
    std::string str() const {
        std::string s = "__attribute__((";
        s.append(func_string("vec_type_hint", nameof<T>()))
         .append(")) ");
        return s;
    }
};

class WorkGroupSizeHint : public FunctionAttribute
{
    const size_t _x;
    const size_t _y;
    const size_t _z;
public:
    WorkGroupSizeHint(const size_t X, const size_t Y, const size_t Z);
    std::string str() const;
};

class RequiredWorkGroupSize : public FunctionAttribute
{
    const size_t _x;
    const size_t _y;
    const size_t _z;
public:
    RequiredWorkGroupSize(const size_t X, const size_t Y, const size_t Z);
    std::string str() const;
};

}; // namespace

#endif
