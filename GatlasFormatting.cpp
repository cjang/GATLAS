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

#include "GatlasFormatting.hpp"

#include "declare_namespace"

// convert ordinal digits to hex string for vector type components
const char* hex(const size_t n) {
    static const char *lut[] = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F"
    };
    return (n < 16)
               ? lut[n]
               : "X";   // deliberately invalid value
}

std::ostream& operator<< (std::ostream& os, const Indent& tabs) {
    return os << tabs.str();
}

std::ostream& endline(std::ostream& os) {
    return os << ";" << std::endl;
}

Indent::Indent() : _tabs(0) { }
Indent::Indent(const size_t num) : _tabs(num) { }
Indent::Indent(const Indent& other) : _tabs(other._tabs) { }
Indent& Indent::more() { _tabs++; return *this; }
Indent& Indent::less() { _tabs--; return *this; }
Indent& Indent::set(const Indent& other) { _tabs = other._tabs; return *this; }
std::string Indent::str() const {
    std::string s;
    for (size_t i = 0; i < _tabs; i++)
        s.append("    ");
    return s;
}
Indent& Indent::obj() {
    static Indent defaultObj;
    return defaultObj;
}

}; // namespace
