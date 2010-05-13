#ifndef _GATLAS_FORMATTING_HPP_
#define _GATLAS_FORMATTING_HPP_

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
#include <sstream>
#include <string>

#include "declare_namespace"

template <typename T>
std::string as_string(const T& value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template <typename T1>
std::string func_string(const char *func, const T1& arg1) {
    std::string s = func;
    return s.append("(").append(as_string<T1>(arg1)).append(")");
}

template <typename T1, typename T2>
std::string func_string(const char *func, const T1& arg1, const T2& arg2) {
    std::string s = func;
    return s.append("(")
            .append(as_string<T1>(arg1))
            .append(", ")
            .append(as_string<T2>(arg2))
            .append(")");
}

template <typename T1, typename T2, typename T3>
std::string func_string(const char *func, const T1& arg1, const T2& arg2, const T3& arg3) {
    std::string s = func;
    return s.append("(")
            .append(as_string<T1>(arg1))
            .append(", ")
            .append(as_string<T2>(arg2))
            .append(", ")
            .append(as_string<T3>(arg3))
            .append(")");
}

// convert ordinal digits to hex string for vector type components
const char* hex(const size_t n);

class Indent
{
    size_t _tabs;

public:
    Indent();
    Indent(const size_t num);
    Indent(const Indent& other);
    Indent& more();
    Indent& less();
    Indent& set(const Indent& other);
    std::string str() const;
    static Indent& obj();
};

static const Indent TAB0;
static const Indent TAB1(1);
static const Indent TAB2(2);
static const Indent TAB3(3);
static const Indent TAB4(4);
static const Indent TAB5(5);
static const Indent TAB6(6);
static const Indent TAB7(7);
static const Indent TAB8(8);

std::ostream& operator<< (std::ostream& os, const Indent& tabs);

std::ostream& endline(std::ostream& os);

}; // namespace

#endif
