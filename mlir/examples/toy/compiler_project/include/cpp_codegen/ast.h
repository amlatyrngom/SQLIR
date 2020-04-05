// CppCodegen.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <unordered_set>
#include <ostream>
#include <variant>
#include <sstream>
#include <unordered_map>

#include "type.h"
#include "expr.h"
#include "node.h"

namespace gen {

// Information about current executing code.
class CodegenContext {
 public:
  // Constructor
  CodegenContext() {
    InitBuiltinSymbols();
    InitPrimTypes();
    InitBuiltinTypes();
  }

  void InitBuiltinSymbols() {
#define ADD_ENTRY(sym, ident) builtin_symbols_[BuiltinSymbol::sym] = GetSymbol(ident);
    BUILTIN_SYMBOL(ADD_ENTRY)
#undef ADD_ENTRY
  }

  // Add primitive types
  void InitPrimTypes() {
#define ADD_ENTRY(type, ident) prim_types_[PrimType::type] = AddType(std::make_unique<PrimitiveType>(PrimType::type, GetSymbol(ident)));
    PRIM_TYPE(ADD_ENTRY)
#undef ADD_ENTRY
  }

  // Add builtin types
  void InitBuiltinTypes() {
#define ADD_ENTRY(type, ident) builtin_types_[BuiltinType::type] = AddType(std::make_unique<SimpleType>(GetSymbol(ident)));
    BUILTIN_TYPE(ADD_ENTRY)
#undef ADD_ENTRY
  }

  // Return a primitive type
  const Type *GetType(PrimType prim_type) {
    return prim_types_[prim_type];
  }

  // Return a builtin type
  const Type *GetType(BuiltinType builtin) {
    return builtin_types_[builtin];
  }

  // Add an expression to the allocated expressions and get its pointer.
  const Expr *AddExpr(std::unique_ptr<Expr> &&expr) {
    auto ret = expr.get();
    allocated_exprs_.emplace_back(std::move(expr));
    return ret;
  }

  // Add an type to the allocated type and get its pointer.
  const Type *AddType(std::unique_ptr<Type> &&type) {
    auto ret = type.get();
    allocated_types_.emplace_back(std::move(type));
    return ret;
  }

  // Add a node to the allocated nodes and get its pointer.
  const Node *AddNode(std::unique_ptr<Node> &&node) {
    auto ret = node.get();
    allocated_nodes_.emplace_back(std::move(node));
    return ret;
  }

  // Return a unique symbol with the given prefix
  Symbol NewSymbol(std::string_view ident) {
    std::stringstream ss;
    ss << ident;
    ss << counter_;
    auto sym = GetSymbol(ss.str());

    counter_++;
    return sym;
  }

  // Return the symbol associated with the given identifier.
  Symbol GetSymbol(std::string_view ident) {
    // Check if the symbol already exists
    auto it = symbols.find(ident.data());
    if (it != symbols.end()) {
      // If so return it directly
      return Symbol{*it};
    }
    // Otherwise, insert the symbol.
    auto ret = symbols.emplace(ident.data());
    return Symbol(*ret.first);
  }

  // Return the builtin symbol
  Symbol GetSymbol(BuiltinSymbol sym) {
    return builtin_symbols_[sym];
  }
 private:
  std::unordered_set<std::string> symbols;
  std::vector<std::unique_ptr<Expr>> allocated_exprs_;
  std::vector<std::unique_ptr<Type>> allocated_types_;
  std::vector<std::unique_ptr<Node>> allocated_nodes_;
  std::unordered_map<PrimType, const Type *> prim_types_;
  std::unordered_map<BuiltinType, const Type *> builtin_types_;
  std::unordered_map<BuiltinSymbol, Symbol> builtin_symbols_;
  uint64_t counter_{0};
};

}
