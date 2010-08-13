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

#include "GatlasCodeText.hpp"

#include "declare_namespace"

Printable::Printable(Indent& indent) : _indent(indent) { }

std::ostream& operator<< (std::ostream& os, const Printable& v) {
    return v.print(os);
}

///////////////////////////////////////////////////////////////////////////////
// constants and value transformations

DerefValue::DerefValue(const Value& value) : _value(value) { }
std::string DerefValue::name() const {
    std::stringstream ss;
    ss << "*" << _value.name();
    return ss.str();
}

ConstantValue<std::string> operator* (const Value& right) {
    return ConstantValue<std::string>(DerefValue(right).name());
}

PostIncValue::PostIncValue(const Value& value) : _value(value) { }
std::string PostIncValue::name() const {
    std::stringstream ss;
    ss << _value.name() << "++";
    return ss.str();
}

ConstantValue<std::string> operator++ (const Value& left, int dummy) {
    return ConstantValue<std::string>(PostIncValue(left).name());
}

std::string BinOpValue::left() const {
    if (_left) return _left->name();
    else return as_string<size_t>(_numLeft);
}

std::string BinOpValue::right() const {
    if (_right) return _right->name();
    else return as_string<size_t>(_numRight);
}

std::string BinOpValue::value(const std::string& op) const {
    std::stringstream ss;
    ss << "(" << left() << op << right() << ")";
    return ss.str();
}

BinOpValue::BinOpValue(const Value& left, const Value& right)
    : _left(&left),
      _right(&right),
      _numLeft(0),
      _numRight(0)
    { }

BinOpValue::BinOpValue(const Value& left, const size_t right)
    : _left(&left),
      _right(NULL),
      _numLeft(0),
      _numRight(right)
    { }

BinOpValue::BinOpValue(const size_t left, const Value& right)
    : _left(NULL),
      _right(&right),
      _numLeft(left),
      _numRight(0)
    { }

AddValue::AddValue(const Value& left, const Value& right) : BinOpValue(left, right) { }
AddValue::AddValue(const Value& left, const size_t right) : BinOpValue(left, right) { }
AddValue::AddValue(const size_t left, const Value& right) : BinOpValue(left, right) { }

std::string AddValue::name() const {
    if ("0" == left()) return right();
    else if ("0" == right()) return left();
    else return value(" + ");
}

SubValue::SubValue(const Value& left, const Value& right) : BinOpValue(left, right) { }
SubValue::SubValue(const Value& left, const size_t right) : BinOpValue(left, right) { }
SubValue::SubValue(const size_t left, const Value& right) : BinOpValue(left, right) { }

std::string SubValue::name() const {
    if ("0" == right()) return left();
    else return value(" - ");
}

MulValue::MulValue(const Value& left, const Value& right) : BinOpValue(left, right) { }
MulValue::MulValue(const Value& left, const size_t right) : BinOpValue(left, right) { }
MulValue::MulValue(const size_t left, const Value& right) : BinOpValue(left, right) { }

std::string MulValue::name() const {
    if ("1" == left()) return right();
    else if ("1" == right()) return left();
    else if ("0" == left()) return left();
    else if ("0" == right()) return right();
    else return value(" * ");
}

DivValue::DivValue(const Value& left, const Value& right) : BinOpValue(left, right) { }
DivValue::DivValue(const Value& left, const size_t right) : BinOpValue(left, right) { }
DivValue::DivValue(const size_t left, const Value& right) : BinOpValue(left, right) { }

std::string DivValue::name() const {
    if ("1" == right()) return left();
    else if ("0" == left()) return left();
    else return value(" / ");
}

ModValue::ModValue(const Value& left, const Value& right) : BinOpValue(left, right) { }
ModValue::ModValue(const Value& left, const size_t right) : BinOpValue(left, right) { }
ModValue::ModValue(const size_t left, const Value& right) : BinOpValue(left, right) { }

std::string ModValue::name() const {
    return value("%");
}

RightShiftValue::RightShiftValue(const Value& left, const Value& right) : BinOpValue(left, right) { }
RightShiftValue::RightShiftValue(const Value& left, const size_t right) : BinOpValue(left, right) { }
RightShiftValue::RightShiftValue(const size_t left, const Value& right) : BinOpValue(left, right) { }

std::string RightShiftValue::name() const {
    return value(">>");
}

LeftShiftValue::LeftShiftValue(const Value& left, const Value& right) : BinOpValue(left, right) { }
LeftShiftValue::LeftShiftValue(const Value& left, const size_t right) : BinOpValue(left, right) { }
LeftShiftValue::LeftShiftValue(const size_t left, const Value& right) : BinOpValue(left, right) { }

std::string LeftShiftValue::name() const {
    return value("<<");
}

ConstantValue<std::string> operator+ (const Value& left, const Value& right) {
    return ConstantValue<std::string>(AddValue(left, right).name());
}
ConstantValue<std::string> operator+ (const Value& left, const size_t right) {
    return ConstantValue<std::string>(AddValue(left, right).name());
}
ConstantValue<std::string> operator+ (const size_t left, const Value& right) {
    return ConstantValue<std::string>(AddValue(left, right).name());
}

ConstantValue<std::string> operator- (const Value& left, const Value& right) {
    return ConstantValue<std::string>(SubValue(left, right).name());
}
ConstantValue<std::string> operator- (const Value& left, const size_t right) {
    return ConstantValue<std::string>(SubValue(left, right).name());
}
ConstantValue<std::string> operator- (const size_t left, const Value& right) {
    return ConstantValue<std::string>(SubValue(left, right).name());
}

ConstantValue<std::string> operator* (const Value& left, const Value& right) {
    return ConstantValue<std::string>(MulValue(left, right).name());
}
ConstantValue<std::string> operator* (const Value& left, const size_t right) {
    return ConstantValue<std::string>(MulValue(left, right).name());
}
ConstantValue<std::string> operator* (const size_t left, const Value& right) {
    return ConstantValue<std::string>(MulValue(left, right).name());
}

ConstantValue<std::string> operator/ (const Value& left, const Value& right) {
    return ConstantValue<std::string>(DivValue(left, right).name());
}
ConstantValue<std::string> operator/ (const Value& left, const size_t right) {
    return ConstantValue<std::string>(DivValue(left, right).name());
}
ConstantValue<std::string> operator/ (const size_t left, const Value& right) {
    return ConstantValue<std::string>(DivValue(left, right).name());
}

ConstantValue<std::string> operator% (const Value& left, const Value& right) {
    return ConstantValue<std::string>(ModValue(left, right).name());
}
ConstantValue<std::string> operator% (const Value& left, const size_t right) {
    return ConstantValue<std::string>(ModValue(left, right).name());
}
ConstantValue<std::string> operator% (const size_t left, const Value& right) {
    return ConstantValue<std::string>(ModValue(left, right).name());
}

ConstantValue<std::string> operator>> (const Value& left, const Value& right) {
    return ConstantValue<std::string>(RightShiftValue(left, right).name());
}
ConstantValue<std::string> operator>> (const Value& left, const size_t right) {
    return ConstantValue<std::string>(RightShiftValue(left, right).name());
}
ConstantValue<std::string> operator>> (const size_t left, const Value& right) {
    return ConstantValue<std::string>(RightShiftValue(left, right).name());
}

ConstantValue<std::string> operator<< (const Value& left, const Value& right) {
    return ConstantValue<std::string>(LeftShiftValue(left, right).name());
}
ConstantValue<std::string> operator<< (const Value& left, const size_t right) {
    return ConstantValue<std::string>(LeftShiftValue(left, right).name());
}
ConstantValue<std::string> operator<< (const size_t left, const Value& right) {
    return ConstantValue<std::string>(LeftShiftValue(left, right).name());
}

MADValue::MADValue(const Value& a, const Value& b, const Value& c)
    : _a(&a),
      _b(&b),
      _c(&c),
      _useStrings(false)
    { }

MADValue::MADValue(const std::string& a, const std::string& b, const std::string& c)
    : _aStr(a),
      _bStr(b),
      _cStr(c),
      _useStrings(true)
    { }

MADValue::MADValue(const std::string& a, const std::string& b, const Value& c)
    : _aStr(a),
      _bStr(b),
      _cStr(c.name()),
      _useStrings(true)
    { }

std::string MADValue::name() const {
    std::stringstream ss;
    if (_useStrings)
        ss << "mad(" << _aStr << ", " << _bStr << ", " << _cStr << ")";
    else
        ss << "mad(" << _a->name() << ", " << _b->name() << ", " << _c->name() << ")";
    return ss.str();
}

///////////////////////////////////////////////////////////////////////////////
// function declaration

FunctionDeclaration::FunctionDeclaration(const std::string functionName, Indent& indent)
    : Printable(indent),
      _functionName(functionName)
    { }

void FunctionDeclaration::qualify(const FunctionAttribute& qualifier) {
    _qualifiers.push_back(&qualifier);
}

void FunctionDeclaration::argument(const Variable& variable) {
    _arguments.push_back(&variable);
}

std::ostream& FunctionDeclaration::print(std::ostream& os) const {
    os << _indent;
    // __kernel and __attribute__()
    for (std::vector<const FunctionAttribute*>::const_iterator iter = _qualifiers.begin();
         iter != _qualifiers.end();
         iter++) {
        os << **iter;
    }
    // return type and function name
    os << _returnType << " " << _functionName << " (";

    _indent.more();
    // function arguments
    for (size_t i = 0; i < _arguments.size(); i++) {
        os << std::endl << _indent << _arguments[i]->declaredName();
        if (i < _arguments.size()-1)
            os << ",";
    }
    _indent.less();
    os << " )" << std::endl << _indent << "{" << std::endl;
    _indent.more();
    return os;
}

///////////////////////////////////////////////////////////////////////////////
// global and local work items

WorkItems::WorkItems(Indent& indent)
    : Printable(indent)
    { }

void WorkItems::addDimension(const Var<const int>& groupVar,
                             const Var<const int>& localVar) {
    _global.push_back(&groupVar);
    _local.push_back(&localVar);
}

std::ostream& WorkItems::print(std::ostream& os) const {
    for (size_t i = 0; i < _global.size(); i++) {
        endline(os << _indent << _global[i]->declaredName() << " = " << func_string<size_t>("get_group_id", i));
    }
    for (size_t i = 0; i < _local.size(); i++) {
        endline(os << _indent << _local[i]->declaredName() << " = " << func_string<size_t>("get_local_id", i));
    }
    return os;
}

WorkItemGlobalSize::WorkItemGlobalSize(const size_t dimindex) : _dimindex(dimindex) { }
std::string WorkItemGlobalSize::name() const { return func_string<size_t>("get_global_size", _dimindex); }

WorkItemLocalSize::WorkItemLocalSize(const size_t dimindex) : _dimindex(dimindex) { }
std::string WorkItemLocalSize::name() const { return func_string<size_t>("get_local_size", _dimindex); }

///////////////////////////////////////////////////////////////////////////////
// special things like barriers and loops

LocalBarrier::LocalBarrier(Indent& indent) : Printable(indent) { }

std::ostream& LocalBarrier::print(std::ostream& os) const {
    return endline(os << _indent << "barrier(CLK_LOCAL_MEM_FENCE)");
}

ForLoop::ForLoop(const Variable& index, const Value& limit, const size_t step, Indent& indent)
    : Printable(indent),
      _index(index),
      _limit(&limit),
      _numLimit(0),
      _increment(step)
    { }

ForLoop::ForLoop(const Variable& index, const size_t limit, const size_t step, Indent& indent)
    : Printable(indent),
      _index(index),
      _limit(NULL),
      _numLimit(limit),
      _increment(step)
    { }

std::ostream& ForLoop::print(std::ostream& os) const {
    os << _indent << "for (int " << _index.name() << " = 0; " << _index.name() << " < ";
    if (_limit) {
        os << _limit->name();
    } else {
        os << _numLimit;
    }
    os << "; " << _index.name();
    if (1 == _increment) {
        os << "++";
    } else {
        os << " += " << _increment;
    }
    os << ")" << std::endl << _indent << "{" << std::endl;
    _indent.more();
    return os;
}

IfThen::IfThen(const Value& lhs, const std::string& op, const Value& rhs, Indent& indent)
    : Printable(indent),
      _lhs(lhs),
      _rhs(rhs),
      _op(op)
    { }

std::ostream& IfThen::print(std::ostream& os) const {
    os << _indent << "if (" << _lhs.name() << " " << _op << " " << _rhs.name() << ")" << std::endl
       << _indent << "{" << std::endl;
    _indent.more();
    return os;
}

EndBlock::EndBlock(Indent& indent) : Printable(indent) { }

std::ostream& EndBlock::print(std::ostream& os) const {
    return os << _indent.less() << "}" << std::endl;
}

// required sampler settings for 2D images
// use like this: Var< const sampler_t > sampler;
//                os << declare(sampler, ImageSampler());
std::string ImageSampler::name() const {
    return "CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE";
}

///////////////////////////////////////////////////////////////////////////////
// declaring, assigning and incrementing variables

std::string declare(const Variable& lhs, const Indent& indent) {
    std::stringstream ss;
    ss << indent << lhs.declaredName() << ";" << std::endl;
    return ss.str();
}

std::string declare(const Variable& lhs, const Value& rhs, const Indent& indent) {
    std::stringstream ss;
    ss << indent << lhs.declaredName() << " = " << rhs.name() << ";" << std::endl;
    return ss.str();
}

std::string declare(const Variable& lhs, const size_t rhs, const Indent& indent) {
    return declare(lhs, ConstantValue<size_t>(rhs), indent);
}

std::string assign(const std::string& lhs, const Value& rhs, const Indent& indent) {
    std::stringstream ss;
    ss << indent << lhs << " = " << rhs.name() << ";" << std::endl;
    return ss.str();
}

std::string assign(const std::string& lhs, const size_t rhs, const Indent& indent) {
    return assign(lhs, ConstantValue<size_t>(rhs), indent);
}

std::string assign(const Value& lhs, const Value& rhs, const Indent& indent) {
    std::stringstream ss;
    ss << indent << lhs.name() << " = " << rhs.name() << ";" << std::endl;
    return ss.str();
}

std::string assign(const Value& lhs, const size_t rhs, const Indent& indent) {
    return assign(lhs, ConstantValue<size_t>(rhs), indent);
}

std::string increment(const std::string& lhs, const Value& rhs, const Indent& indent) {
    std::stringstream ss;
    ss << indent << lhs << " += " << rhs.name() << ";" << std::endl;
    return ss.str();
}

std::string increment(const std::string& lhs, const size_t rhs, const Indent& indent) {
    return increment(lhs, ConstantValue<size_t>(rhs), indent);
}

std::string increment(const Variable& lhs, const Value& rhs, const Indent& indent) {
    std::stringstream ss;
    ss << indent << lhs.name() << " += " << rhs.name() << ";" << std::endl;
    return ss.str();
}

std::string increment(const Variable& lhs, const size_t rhs, const Indent& indent) {
    return increment(lhs, ConstantValue<size_t>(rhs), indent);
}

}; // namespace
