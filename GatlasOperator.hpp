#ifndef _GATLAS_OPERATOR_HPP_
#define _GATLAS_OPERATOR_HPP_

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

#include "declare_namespace"

// operators
struct Operator { virtual std::string str() const = 0; };

std::ostream& operator<< (std::ostream& os, const Operator& op);

#define DECL_OPERATOR(TYPE, TOKEN) \
struct TYPE ## Operator : public Operator { std::string str() const; }; \
static const TYPE ## Operator TYPE;

#define DEFN_OPERATOR(TYPE, TOKEN) \
std::string TYPE ## Operator::str() const { return #TOKEN ; }

DECL_OPERATOR(ADD, +)
DECL_OPERATOR(SUB, -)
DECL_OPERATOR(MUL, *)
DECL_OPERATOR(DIV, /)
DECL_OPERATOR(INC, ++)
DECL_OPERATOR(DEC, --)
DECL_OPERATOR(ADDEQ, +=)
DECL_OPERATOR(SUBEQ, -=)
DECL_OPERATOR(MULEQ, *=)
DECL_OPERATOR(DIVEQ, /=)
DECL_OPERATOR(CMPLT, <)
DECL_OPERATOR(CMPGT, >)
DECL_OPERATOR(CMPLTE, <=)
DECL_OPERATOR(CMPGTE, >=)
DECL_OPERATOR(CMPEQ, ==)
DECL_OPERATOR(CMPNE, !=)
DECL_OPERATOR(DEREF, *)
DECL_OPERATOR(ADDOF, &)

}; // namespace

#endif
