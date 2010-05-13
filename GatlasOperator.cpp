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

#include "GatlasOperator.hpp"

#include "declare_namespace"

std::ostream& operator<< (std::ostream& os, const Operator& op) {
    return os << op.str();
}

DEFN_OPERATOR(ADD, +)
DEFN_OPERATOR(SUB, -)
DEFN_OPERATOR(MUL, *)
DEFN_OPERATOR(DIV, /)
DEFN_OPERATOR(INC, ++)
DEFN_OPERATOR(DEC, --)
DEFN_OPERATOR(ADDEQ, +=)
DEFN_OPERATOR(SUBEQ, -=)
DEFN_OPERATOR(MULEQ, *=)
DEFN_OPERATOR(DIVEQ, /=)
DEFN_OPERATOR(CMPLT, <)
DEFN_OPERATOR(CMPGT, >)
DEFN_OPERATOR(CMPLTE, <=)
DEFN_OPERATOR(CMPGTE, >=)
DEFN_OPERATOR(CMPEQ, ==)
DEFN_OPERATOR(CMPNE, !=)
DEFN_OPERATOR(DEREF, *)
DEFN_OPERATOR(ADDOF, &)

}; // namespace
