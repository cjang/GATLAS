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

#include "GatlasQualifier.hpp"

#include "declare_namespace"

std::ostream& operator<< (std::ostream& os, const AddressSpace& qualifier) {
    return os << qualifier.str();
}

DEFN_ADDRESS_SPACE(GLOBAL, __global)
DEFN_ADDRESS_SPACE(LOCAL, __local)
DEFN_ADDRESS_SPACE(CONSTANT, __constant)
DEFN_ADDRESS_SPACE(PRIVATE, __private)

// these are really not address space qualifiers but image access qualifiers
DEFN_ADDRESS_SPACE(READONLY, __read_only)
DEFN_ADDRESS_SPACE(WRITEONLY, __write_only)

std::string DEFAULTAddressSpace::str() const { return ""; }

std::ostream& operator<< (std::ostream& os, const FunctionAttribute& qualifier) {
    return os << qualifier.str();
}

DEFN_FUNCTION_ATTRIBUTE(KERNEL, __kernel)

WorkGroupSizeHint::WorkGroupSizeHint(const size_t X, const size_t Y, const size_t Z)
    : _x(X), _y(Y), _z(Z)
    { }

std::string WorkGroupSizeHint::str() const {
    std::string s = "__attribute__((";
    s.append(func_string("work_group_size_hint", _x, _y, _z))
     .append(")) ");
    return s;
}

RequiredWorkGroupSize::RequiredWorkGroupSize(const size_t X, const size_t Y, const size_t Z)
    : _x(X), _y(Y), _z(Z)
    { }

std::string RequiredWorkGroupSize::str() const {
    std::string s = "__attribute__((";
    s.append(func_string("reqd_work_group_size", _x, _y, _z))
     .append(")) ");
    return s;
}

}; // namespace
