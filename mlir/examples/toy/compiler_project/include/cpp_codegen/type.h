#pragma once
#include <ostream>
#include "symbol.h"

namespace mlirgen {
  class MLIRGen;
}

namespace mlir {
  class Type;
}

namespace gen {
// List of primitive types.
#define PRIM_TYPE(F) \
F(I8, "int8_t") \
F(I16, "int16_t") \
F(I32, "int32_t") \
F(Int, "int") \
F(I64, "int64_t") \
F(U8, "uint8_t") \
F(U16, "uint16_t") \
F(U32, "uint32_t") \
F(U64, "uint64_t") \
F(F32, "float") \
F(F64, "double") \
F(Void, "void") \
F(Bool, "bool") \
F(Auto, "auto")

// Define the enum of primitive types.
#define PRIM_TYPE_ENTRY(type, ident) type,
enum class PrimType : uint32_t {
  PRIM_TYPE(PRIM_TYPE_ENTRY)
};
#undef PRIM_TYPE_ENTRY


// List of builtin types.
#define BUILTIN_TYPE(F) \
F(StdVector, "std::vector") \
F(StdString, "std::string")

// Define the enum of builtin types.
#define BUILTIN_TYPE_ENTRY(type, ident) type,
enum class BuiltinType : uint32_t {
  BUILTIN_TYPE(BUILTIN_TYPE_ENTRY)
};
#undef BUILTIN_TYPE_ENTRY

// Generic type
class Type {
 public:
  virtual ~Type() = default;

  virtual void Visit(std::ostream *os) const = 0;

  virtual mlir::Type Visit(mlirgen::MLIRGen* mlir_gen) const;
};

class PrimitiveType : public Type {
  public:
   explicit PrimitiveType(PrimType prim_type, Symbol symbol) : prim_types_(prim_type), symbol_(symbol) {}

   virtual ~PrimitiveType() = default;

   void Visit(std::ostream *os) const override {
     *os << symbol_.ident_;
   }

  mlir::Type Visit(mlirgen::MLIRGen* mlir_gen) const override;
  private:
   PrimType prim_types_;
   Symbol symbol_;
};

// A type that's just an identifier
class SimpleType : public Type {
 public:
  explicit SimpleType(Symbol symbol) : symbol_(symbol) {}

  virtual ~SimpleType() = default;

  void Visit(std::ostream *os) const override {
    *os << symbol_.ident_;
  }
 private:
  Symbol symbol_;
};

// A const type
class ConstType : public Type {
 public:
  explicit ConstType(const Type *child) : Type(), child_(child) {}

  virtual ~ConstType() = default;

  void Visit(std::ostream *os) const override {
    *os << "const ";
    child_->Visit(os);
  }
 private:
  const Type *child_;
};

// A ref type
class RefType : public Type {
 public:
  explicit RefType(const Type *child) : Type(), child_(child) {}

  virtual ~RefType() = default;

  void Visit(std::ostream *os) const override {
    child_->Visit(os);
    *os << "&";
  }
 private:
  const Type *child_;
};

// A pointer type
class PointerType : public Type {
 public:
  explicit PointerType(const Type *child) : Type(), child_(child) {}

  virtual ~PointerType() = default;

  void Visit(std::ostream *os) const override {
    child_->Visit(os);
    *os << "*";
  }
 private:
  const Type *child_;
};

// Represents root<special1, special2, ...>
class TemplateType : public Type {
 public:
  TemplateType(const Type *root, std::vector<const Type *> &&specials)
      : Type(), root_(root), specials_(std::move(specials)) {}

  virtual ~TemplateType() = default;

  void Visit(std::ostream *os) const override {
    root_->Visit(os);
    *os << "<";
    for (uint32_t i = 0; i < specials_.size(); i++) {
      specials_[i]->Visit(os);
      if (i < specials_.size() - 1) {
        *os << ", ";
      }
    }
    *os << ">";
  }
 private:
  const Type *root_;
  std::vector<const Type *> specials_;
};
}
