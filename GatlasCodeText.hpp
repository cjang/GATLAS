#ifndef _GATLAS_CODE_TEXT_HPP_
#define _GATLAS_CODE_TEXT_HPP_

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
#include <vector>
#include "OCLUtil.hpp"
#include "GatlasFormatting.hpp"
#include "GatlasOperator.hpp"
#include "GatlasQualifier.hpp"
#include "GatlasType.hpp"

#include "declare_namespace"

///////////////////////////////////////////////////////////////////////////////
// interfaces

struct Value
{
    virtual std::string name() const = 0;
};

struct Variable : public Value
{
    virtual std::string name() const = 0;
    virtual std::string declaredName() const = 0;
};

class Printable
{
protected:
    Indent& _indent;
    Printable(Indent& indent);
public:
    virtual std::ostream& print(std::ostream&) const = 0;
};

std::ostream& operator<< (std::ostream& os, const Printable& v);

///////////////////////////////////////////////////////////////////////////////
// constants and value transformations

template <typename T>
class ConstantValue : public Value
{
    const T _value;
public:
    ConstantValue(const T& value) : _value(value) { }
    ConstantValue(const Value& value) : _value(value.name()) { }
    std::string name() const {
        std::stringstream ss;
        ss << _value;
        return ss.str();
    }
};

template <typename T>
class CastValue : public Value
{
    const Value& _value;
public:
    CastValue(const Value& value) : _value(value) { }
    std::string name() const {
        std::stringstream ss;
        ss << "(" << castto<T>() << ")(" << _value.name() << ")";
        return ss.str();
    }
};

template <typename SCALAR, size_t VECTOR_LENGTH = 1>
class ReinterpretValue : public Value
{
    const Value& _value;
    const bool _doApply;
public:
    ReinterpretValue(const Value& value, const bool doApply = true) : _value(value), _doApply(doApply) { }
    std::string name() const {
        std::stringstream ss;
        if (_doApply)
            ss << "as_" << nameof<SCALAR, VECTOR_LENGTH>() << "(" << _value.name() << ")";
        else
            ss << _value.name();
        return ss.str();
    }
};

class DerefValue : public Value
{
    const Value& _value;
public:
    DerefValue(const Value& value);
    std::string name() const;
};

ConstantValue<std::string> operator* (const Value& right);

class PostIncValue : public Value
{
    const Value& _value;
public:
    PostIncValue(const Value& value);
    std::string name() const;
};

ConstantValue<std::string> operator++ (const Value& left, int dummy);

class BinOpValue : public Value
{
    const Value *_left;
    const Value *_right;
    const size_t _numLeft;
    const size_t _numRight;

protected:
    std::string left() const;
    std::string right() const;
    std::string value(const std::string& op) const;

    BinOpValue(const Value& left, const Value& right);
    BinOpValue(const Value& left, const size_t right);
    BinOpValue(const size_t left, const Value& right);

public:
    virtual std::string name() const = 0;
};

struct AddValue : public BinOpValue
{
    AddValue(const Value& left, const Value& right);
    AddValue(const Value& left, const size_t right);
    AddValue(const size_t left, const Value& right);

    std::string name() const;
};

struct SubValue : public BinOpValue
{
    SubValue(const Value& left, const Value& right);
    SubValue(const Value& left, const size_t right);
    SubValue(const size_t left, const Value& right);

    std::string name() const;
};

struct MulValue : public BinOpValue
{
    MulValue(const Value& left, const Value& right);
    MulValue(const Value& left, const size_t right);
    MulValue(const size_t left, const Value& right);

    std::string name() const;
};

struct DivValue : public BinOpValue
{
    DivValue(const Value& left, const Value& right);
    DivValue(const Value& left, const size_t right);
    DivValue(const size_t left, const Value& right);

    std::string name() const;
};

struct ModValue : public BinOpValue
{
    ModValue(const Value& left, const Value& right);
    ModValue(const Value& left, const size_t right);
    ModValue(const size_t left, const Value& right);

    std::string name() const;
};

struct RightShiftValue : public BinOpValue
{
    RightShiftValue(const Value& left, const Value& right);
    RightShiftValue(const Value& left, const size_t right);
    RightShiftValue(const size_t left, const Value& right);

    std::string name() const;
};

struct LeftShiftValue : public BinOpValue
{
    LeftShiftValue(const Value& left, const Value& right);
    LeftShiftValue(const Value& left, const size_t right);
    LeftShiftValue(const size_t left, const Value& right);

    std::string name() const;
};

ConstantValue<std::string> operator+ (const Value& left, const Value& right);
ConstantValue<std::string> operator+ (const Value& left, const size_t right);
ConstantValue<std::string> operator+ (const size_t left, const Value& right);

ConstantValue<std::string> operator- (const Value& left, const Value& right);
ConstantValue<std::string> operator- (const Value& left, const size_t right);
ConstantValue<std::string> operator- (const size_t left, const Value& right);

ConstantValue<std::string> operator* (const Value& left, const Value& right);
ConstantValue<std::string> operator* (const Value& left, const size_t right);
ConstantValue<std::string> operator* (const size_t left, const Value& right);

ConstantValue<std::string> operator/ (const Value& left, const Value& right);
ConstantValue<std::string> operator/ (const Value& left, const size_t right);
ConstantValue<std::string> operator/ (const size_t left, const Value& right);

ConstantValue<std::string> operator% (const Value& left, const Value& right);
ConstantValue<std::string> operator% (const Value& left, const size_t right);
ConstantValue<std::string> operator% (const size_t left, const Value& right);

ConstantValue<std::string> operator>> (const Value& left, const Value& right);
ConstantValue<std::string> operator>> (const Value& left, const size_t right);
ConstantValue<std::string> operator>> (const size_t left, const Value& right);

ConstantValue<std::string> operator<< (const Value& left, const Value& right);
ConstantValue<std::string> operator<< (const Value& left, const size_t right);
ConstantValue<std::string> operator<< (const size_t left, const Value& right);

// MAD operation
class MADValue : public Value
{
    const Value *_a;
    const Value *_b;
    const Value *_c;

    const std::string _aStr;
    const std::string _bStr;
    const std::string _cStr;

    const bool _useStrings;

public:
    MADValue(const Value& a, const Value& b, const Value& c);
    MADValue(const std::string& a, const std::string& b, const std::string& c);
    MADValue(const std::string& a, const std::string& b, const Value& c);

    std::string name() const;
};

///////////////////////////////////////////////////////////////////////////////
// function declaration

class FunctionDeclaration : public Printable
{
    const std::string                     _functionName;
    std::string                           _returnType;
    std::vector<const FunctionAttribute*> _qualifiers;
    std::vector<const Variable*>          _arguments;

public:
    FunctionDeclaration(const std::string functionName, Indent& indent = Indent::obj());

    template <typename T>
    void returnType() { _returnType = std::string(nameof<T>()); }

    void qualify(const FunctionAttribute& qualifier);

    void argument(const Variable& variable);

    std::ostream& print(std::ostream& os) const;
};

///////////////////////////////////////////////////////////////////////////////
// scalar variable

template <typename T>
class Var : public Variable
{
    const std::string   _identifier;
    const AddressSpace& _qualifier;
    const bool          _inlineValue;
    const int           _value;

public:
    Var(const std::string& name,
        const AddressSpace& q = DEFAULT)
        : _identifier(name),
          _qualifier(q),
          _inlineValue(false),
          _value(0)
    { }

    Var(const std::string& name,
        const bool inlineValue,
        const int value,
        const AddressSpace& q = DEFAULT)
        : _identifier(name),
          _qualifier(q),
          _inlineValue(inlineValue),
          _value(value)
    {
    }

    Var(const std::string& name,
        const AddressSpace& q,
        FunctionDeclaration& funcDecl)
        : _identifier(name),
          _qualifier(q),
          _inlineValue(false),
          _value(0)
    {
        funcDecl.argument(*this);
    }

    Var(const std::string& name,
        const AddressSpace& q,
        FunctionDeclaration& funcDecl,
        const bool isArg)
        : _identifier(name),
          _qualifier(q),
          _inlineValue(false),
          _value(0)
    {
        if (isArg) funcDecl.argument(*this);
    }

    Var(const std::string& name,
        FunctionDeclaration& funcDecl,
        const AddressSpace& q = DEFAULT)
        : _identifier(name),
          _qualifier(q),
          _inlineValue(false),
          _value(0)
    {
        funcDecl.argument(*this);
    }

    Var(const std::string& name,
        FunctionDeclaration& funcDecl,
        const bool isArg,
        const AddressSpace& q = DEFAULT)
        : _identifier(name),
          _qualifier(q),
          _inlineValue(false),
          _value(0)
    {
        if (isArg) funcDecl.argument(*this);
    }

    Var(const std::string& name,
        FunctionDeclaration& funcDecl,
        const bool inlineValue,
        const int value,
        const AddressSpace& q = DEFAULT)
        : _identifier(name),
          _qualifier(q),
          _inlineValue(inlineValue),
          _value(value)
    {
        if (! inlineValue) funcDecl.argument(*this);
    }

    std::string name() const {
        if (_inlineValue) {
            std::stringstream ss;
            ss << _value;
            return ss.str();
        } else {
            return _identifier;
        }
    }

    std::string declaredName() const {
        std::stringstream ss;
        ss << _qualifier << nameof<T>() << " " << name();
        return ss.str();
    }

    std::string operator[] (const size_t i) const {
        std::stringstream ss;
        ss << name();
        if (lengthof<T>() != 1)
            ss << ".s" << hex(i);
        return ss.str();
    }
};

///////////////////////////////////////////////////////////////////////////////
// global and local work items

class WorkItems : public Printable
{
    std::vector< const Var<const int>* > _global;
    std::vector< const Var<const int>* > _local;

public:
    WorkItems(Indent& indent = Indent::obj());

    void addDimension(const Var<const int>& groupVar,
                      const Var<const int>& localVar);

    std::ostream& print(std::ostream& os) const;
};

class WorkItemGlobalSize : public Value
{
    const size_t _dimindex;
public:
    WorkItemGlobalSize(const size_t dimindex);
    std::string name() const;
};

class WorkItemLocalSize : public Value
{
    const size_t _dimindex;
public:
    WorkItemLocalSize(const size_t dimindex);
    std::string name() const;
};

///////////////////////////////////////////////////////////////////////////////
// special things like barriers and loops

struct LocalBarrier : public Printable
{
    LocalBarrier(Indent& indent = Indent::obj());

    std::ostream& print(std::ostream& os) const;
};

class ForLoop : public Printable
{
    const Variable& _index;
    const Value*    _limit;
    const size_t    _numLimit;
    const size_t    _increment;

public:
    ForLoop(const Variable& index, const Value& limit, const size_t step, Indent& indent = Indent::obj());
    ForLoop(const Variable& index, const size_t limit, const size_t step, Indent& indent = Indent::obj());

    std::ostream& print(std::ostream& os) const;
};

class IfThen : public Printable
{
    const Value&      _lhs;
    const Value&      _rhs;
    const std::string _op;
public:
    IfThen(const Value& lhs, const std::string& op, const Value& rhs, Indent& indent = Indent::obj());

    std::ostream& print(std::ostream& os) const;
};

class EndBlock : public Printable
{

public:
    EndBlock(Indent& indent = Indent::obj());

    std::ostream& print(std::ostream& os) const;
};

// required sampler settings for 2D images
// use like this: Var< const sampler_t > sampler;
//                os << declare(sampler, ImageSampler());
struct ImageSampler : public Value
{
    std::string name() const;
};

// read float and uint32 quad from a 2D image
template <typename T>
class ReadImage : public Value
{
    const Var< image2d_t >&       _image;
    const Var< const sampler_t >& _sampler;
    const Value&                  _x;
    const Value&                  _y;

public:
    ReadImage(const Var< image2d_t >& image,
              const Var< const sampler_t >& sampler,
              const Value& x,
              const Value& y)
        : _image(image),
          _sampler(sampler),
          _x(x),
          _y(y)
    { }

    std::string name() const {
        std::stringstream ss;
        ss << (isfloat<T>() ? "read_imagef(" : "read_imageui(")
           << _image.name() << ", "
           << _sampler.name() << ", (int2)("
           << _x.name() << ", "
           << _y.name() << "))";
        return ss.str();
    }
};

// write float and uint32 quad to a 2D image
template <typename T>
class WriteImage : public Printable
{
    const Var< image2d_t >& _image;
    const Value&            _x;
    const Value&            _y;
    const Value&            _value;

public:
    WriteImage(const Var< image2d_t >& image,
               const Value& x,
               const Value& y,
               const Value& value,
               Indent& indent = Indent::obj())
        : Printable(indent),
          _image(image),
          _x(x),
          _y(y),
          _value(value)
    { }

    std::ostream& print(std::ostream& os) const {
        return endline(os << _indent
                          << (isfloat<T>() ? "write_imagef(" : "write_imageui(")
                          << _image.name() << ", (int2)("
                          << _x.name() << ", "
                          << _y.name() << "), "
                          << _value.name()
                          << ")");
    }
};

///////////////////////////////////////////////////////////////////////////////
// vector variables

template <typename T>
struct VectorInterface
{
    virtual size_t length() const = 0;
    virtual std::string name(const size_t i) const = 0;
    virtual std::string declaredName(const size_t i) const = 0;
};

template <typename T>
class Vector : public VectorInterface<T>
{
    const Var<T> _scalar;
    const size_t _length;

    // for operator[]
    const std::string   _name;
    const AddressSpace& _qualifier;

public:
    Vector(const std::string& name,
           const size_t length,
           const AddressSpace& qualifier = DEFAULT)
        : _scalar( Var<T>(name, qualifier) ),
          _length(length),
          _name(name),
          _qualifier(qualifier)
    { }

    size_t length() const { return _length; }

    std::string name(const size_t i) const {
        std::stringstream ss;
        ss << _scalar.name() << i;
        return ss.str();
    }

    std::string declaredName(const size_t i) const {
        std::stringstream ss;
        ss << _scalar.declaredName() << i;
        return ss.str();
    }

    Var<T> operator[] (const size_t i) const {
        return Var<T>(_name + as_string<size_t>(i), _qualifier);
    }
};

template <typename T, size_t N>
class NVector
{
    const Vector<T>       _vector;
    const NVector<T, N-1> _next;

public:
    NVector(const std::string& name,
            const size_t length,
            const AddressSpace& qualifier = DEFAULT)
        : _vector(Vector<T>(name + as_string<size_t>(N-1), length, qualifier)),
          _next(NVector<T, N-1>(name, length, qualifier))
    { }

    const Vector<T>& operator[] (const size_t i) const {
        if (N-1 == i) return _vector;
        return _next[i];
    }
};

template<typename T>
class NVector<T, 0>
{
public:
    NVector(const std::string& name,
            const size_t length,
            const AddressSpace& qualifier = DEFAULT)
    { }

    const Vector<T>& operator[] (const size_t i) const {
        return Vector<T>("", 0); // just return a dangling reference as this should never happen
    }
};

template <typename T>
class DerefVector : public VectorInterface<T>
{
    const VectorInterface<T>& _vector;

public:
    DerefVector(const VectorInterface<T>& v) : _vector(v) { }

    size_t length() const { return _vector.length(); }

    std::string name(const size_t i) const {
        std::stringstream ss;
        ss << "*" << _vector.name(i);
        return ss.str();
    }

    std::string declaredName(const size_t i) const {
        return _vector.declaredName(i);
    }
};

template <typename T>
DerefVector<T> operator* (const VectorInterface<T>& v) {
    return DerefVector<T>(v);
}

template <typename T>
class PostIncVector : public VectorInterface<T>
{
    const VectorInterface<T>& _vector;

public:
    PostIncVector(const VectorInterface<T>& v) : _vector(v) { }

    size_t length() const { return _vector.length(); }

    std::string name(const size_t i) const {
        std::stringstream ss;
        ss << _vector.name(i) << "++";
        return ss.str();
    }

    std::string declaredName(const size_t i) const {
        return _vector.declaredName(i);
    }
};

template <typename T>
PostIncVector<T> operator++ (const VectorInterface<T>& v, int dummy) {
    return PostIncVector<T>(v);
}

template <typename T>
class BinOpVector : public VectorInterface<T>
{
    const VectorInterface<T>& _vector;
    const Value&              _value;

protected:
    BinOpVector(const VectorInterface<T>& v, const Value& value) : _vector(v), _value(value) { }
    std::string value(const size_t i, const std::string& op) const {
        std::stringstream ss;
        ss << "(" << _vector.name(i) << op << _value.name() << ")";
        return ss.str();
    }

public:
    size_t length() const { return _vector.length(); }
    std::string name(const size_t i) const = 0;
    std::string declaredName(const size_t i) const { return _vector.declaredName(i); }
};

template <typename T>
struct AddVector : public BinOpVector<T>
{
    AddVector(const VectorInterface<T>& v, const Value& value) : BinOpVector<T>(v, value) { }
    std::string name(const size_t i) const { return BinOpVector<T>::value(i, " + "); }
};

template <typename T>
struct SubVector : public BinOpVector<T>
{
    SubVector(const VectorInterface<T>& v, const Value& value) : BinOpVector<T>(v, value) { }
    std::string name(const size_t i) const { return BinOpVector<T>::value(i, " - "); }
};

template <typename T>
struct MulVector : public BinOpVector<T>
{
    MulVector(const VectorInterface<T>& v, const Value& value) : BinOpVector<T>(v, value) { }
    std::string name(const size_t i) const { return BinOpVector<T>::value(i, " * "); }
};

template <typename T>
struct DivVector : public BinOpVector<T>
{
    DivVector(const VectorInterface<T>& v, const Value& value) : BinOpVector<T>(v, value) { }
    std::string name(const size_t i) const { return BinOpVector<T>::value(i, " / "); }
};

template <typename T>
AddVector<T> operator+ (const VectorInterface<T>& left, const Value& right) {
    return AddVector<T>(left, right);
}

template <typename T>
SubVector<T> operator- (const VectorInterface<T>& left, const Value& right) {
    return SubVector<T>(left, right);
}

template <typename T>
MulVector<T> operator* (const VectorInterface<T>& left, const Value& right) {
    return MulVector<T>(left, right);
}

template <typename T>
DivVector<T> operator/ (const VectorInterface<T>& left, const Value& right) {
    return DivVector<T>(left, right);
}

template <typename T>
class AccumValue : public Value
{
    const VectorInterface<T>& _vector;

public:
    AccumValue(const VectorInterface<T>& v) : _vector(v) { }

    std::string name() const {
        const size_t L = _vector.length();
        std::stringstream ss;
        ss << "(" << castto<T>() << ")(";
        for (size_t i = 0; i < L; i++) {
            for (size_t j = 0; j < L; j++) {
                ss << _vector.name(i) << ".s" << hex(j);
                if (L-1 == i && L-1 == j)
                    ss << ")";
                else if (L-1 == j)
                    ss << ", ";
                else
                    ss << " + ";
            }
        }
        return ss.str();
    }
};

template <typename T>
class SumValue : public Value
{
    const Value& _value;

public:
    SumValue(const Value& v) : _value(v) { }

    std::string name() const {
        std::stringstream ss;
        ss << "(";
        for (size_t i = 0; i < lengthof<T>(); i++) {
            ss << _value.name() << ".s" << hex(i);
            if (lengthof<T>()-1 != i) ss << " + ";
        }
        ss << ")";
        return ss.str();
    }
};

///////////////////////////////////////////////////////////////////////////////
// declaring, assigning and incrementing variables

std::string declare(const Variable& lhs, const Indent& indent = Indent::obj());
std::string declare(const Variable& lhs, const Value& rhs, const Indent& indent = Indent::obj());
std::string declare(const Variable& lhs, const size_t rhs, const Indent& indent = Indent::obj());

template <typename T>
std::string declare(const VectorInterface<T>& lhs, const Indent& indent = Indent::obj()) {
    std::stringstream ss;
    for (size_t i = 0; i < lhs.length(); i++) {
        ss << indent << lhs.declaredName(i) << ";" << std::endl;
    }
    return ss.str();
}

template <typename T>
std::string declare(const VectorInterface<T>& lhs, const Value& rhs, const Indent& indent = Indent::obj()) {
    std::stringstream ss;
    for (size_t i = 0; i < lhs.length(); i++) {
        ss << indent << lhs.declaredName(i) << " = " << rhs.name() << ";" << std::endl;
    }
    return ss.str();
}

template <typename T>
std::string declare(const VectorInterface<T>& lhs, const size_t rhs, const Indent& indent = Indent::obj()) {
    return declare<T>(lhs, ConstantValue<size_t>(rhs), indent);
}

template <typename T>
std::string declare(const VectorInterface<T>& lhs, const Value& rhs, const Value& step, const Indent& indent = Indent::obj()) {
    std::stringstream ss;
    for (size_t i = 0; i < lhs.length(); i++) {
        ss << indent << lhs.declaredName(i) << " = ";
        if (0 == i)
            ss << rhs.name();
        else
            ss << lhs.name(i-1) << " + " << step.name();
        ss << ";" << std::endl;
    }
    return ss.str();
}

std::string assign(const std::string& lhs, const Value& rhs, const Indent& indent = Indent::obj());
std::string assign(const std::string& lhs, const size_t rhs, const Indent& indent = Indent::obj());
std::string assign(const Value& lhs, const Value& rhs, const Indent& indent = Indent::obj());
std::string assign(const Value& lhs, const size_t rhs, const Indent& indent = Indent::obj());

template <typename T>
std::string assign(const VectorInterface<T>& lhs, const Value& rhs, const Indent& indent = Indent::obj()) {
    std::stringstream ss;
    for (size_t i = 0; i < lhs.length(); i++) {
        ss << indent << lhs.name(i) << " = " << rhs.name() << ";" << std::endl;
    }
    return ss.str();
}

template <typename T>
std::string assign(const VectorInterface<T>& lhs, const size_t rhs, const Indent& indent = Indent::obj()) {
    return assign<T>(lhs, ConstantValue<size_t>(rhs), indent);
}

template <typename T, typename T2>
std::string assign(const VectorInterface<T>& lhs, const VectorInterface<T2>& rhs, const Indent& indent = Indent::obj()) {
    const size_t len = (lhs.length() < rhs.length()) ? lhs.length() : rhs.length();
    std::stringstream ss;
    for (size_t i = 0; i < len; i++) {
        ss << indent << lhs.name(i) << " = " << rhs.name(i) << ";" << std::endl;
    }
    return ss.str();
}

template <typename T>
std::string assign(const VectorInterface<T>& lhs, const Value& rhs, const Value& step, const Indent& indent = Indent::obj()) {
    std::stringstream ss;
    for (size_t i = 0; i < lhs.length(); i++) {
        ss << indent << lhs.name(i) << " = ";
        if (0 == i)
            ss << rhs.name();
        else
            ss << lhs.name(i-1) << " + " << step.name();
        ss << ";" << std::endl;
    }
    return ss.str();
}

template <typename T>
std::string assign(const VectorInterface<T>& lhs, const Value& rhs, const size_t step, const Indent& indent = Indent::obj()) {
    return assign<T>(lhs, rhs, ConstantValue<size_t>(step), indent);
}

std::string increment(const std::string& lhs, const Value& rhs, const Indent& indent = Indent::obj());
std::string increment(const std::string& lhs, const size_t rhs, const Indent& indent = Indent::obj());
std::string increment(const Variable& lhs, const Value& rhs, const Indent& indent = Indent::obj());
std::string increment(const Variable& lhs, const size_t rhs, const Indent& indent = Indent::obj());

template <typename T>
std::string increment(const VectorInterface<T>& lhs, const Value& rhs, const Indent& indent = Indent::obj()) {
    std::stringstream ss;
    for (size_t i = 0; i < lhs.length(); i++) {
        ss << indent << lhs.name(i) << " += " << rhs.name() << ";" << std::endl;
    }
    return ss.str();
}

template <typename T>
std::string increment(const VectorInterface<T>& lhs, const size_t rhs, const Indent& indent = Indent::obj()) {
    return increment<T>(lhs, ConstantValue<size_t>(rhs), indent);
}

template <typename T, typename T2>
std::string increment(const VectorInterface<T>& lhs, const VectorInterface<T2>& rhs, const Indent& indent = Indent::obj()) {
    const size_t len = (lhs.length() < rhs.length()) ? lhs.length() : rhs.length();
    std::stringstream ss;
    for (size_t i = 0; i < len; i++) {
        ss << indent << lhs.name(i) << " += " << rhs.name(i) << ";" << std::endl;
    }
    return ss.str();
}

}; // namespace

#endif
