#pragma once
#include <memory>
#include <ostream>
#include <cstdint>
#include <vector>
#include <string>

#include "symbol.h"
#include "expr.h"
#include "type.h"

namespace mlirgen {
  class MLIRGen;
}

namespace gen {

// A generic gen node
class Node {
 public:
  // Constructor
  Node() = default;

  virtual ~Node() = default;

  // Visit and output node to stream.
  virtual void Visit(std::ostream *os) const = 0;


  virtual void Visit(mlirgen::MLIRGen *mlir_gen) const {}
};

// An include directive.
class Include : public Node {
 public:
  explicit Include(Symbol path, bool system = false) : Node(), path_(path), system_(system) {}

  virtual ~Include() = default;

  void Visit(std::ostream *os) const override;
 private:
  Symbol path_;
  bool system_;
};

// A file
class File : public Node {
 public:
  explicit File(std::vector<const Node*> includes, std::vector<const Node *> &&top_level) : Node(), includes_(std::move(includes)), top_level_(std::move(top_level)) {}

  virtual ~File() = default;

  // Print file content one by one.
  void Visit(std::ostream *os) const override;

 private:
  std::vector<const Node *> includes_;
  std::vector<const Node *> top_level_;
};

// A field in a struct or function declaration.
struct Field {
  // Constructor
  Field(const Type *type, Symbol symbol) : type_(type), symbol_(symbol) {}

  // Type of the field
  const Type *type_;
  // Name of the field
  Symbol symbol_;
};

// A function
class Function : public Node {
 public:
  Function(Symbol name, const Type *ret_type, std::vector<Field> &&params, std::vector<const Node *> &&body)
      : Node(),
        name_(name),
        ret_type_(ret_type),
        params_(std::move(params)),
        body_(std::move(body)) {}

  virtual ~Function() = default;

  // Print function declaration and body.
  void Visit(std::ostream *os) const override;

  virtual void Visit(mlirgen::MLIRGen *mlir_gen) const override;

 private:
  Symbol name_;
  const Type *ret_type_;
  std::vector<Field> params_;
  std::vector<const Node *> body_;
};

// A Struct
class Struct : public Node {
 public:
  Struct(Symbol name, std::vector<Field> &&fields)
      : Node(), name_(name), fields_(std::move(fields)) {}

  virtual ~Struct() = default;

  // Print struct declaration and fields
  void Visit(std::ostream *os) const override;

 private:
  Symbol name_;
  std::vector<Field> fields_;
};

// An expression statement.
class ExprStmt : public Node {
 public:
  explicit ExprStmt(const Expr *expr)
      : Node(), expr_(expr) {}

  virtual ~ExprStmt() = default;

  // Just print expr with semicolon.
  void Visit(std::ostream *os) const override {
    expr_->Visit(os);
    *os << ";" << std::endl;
  }
 private:
  const Expr *expr_;
};

// Declare a variable
class VarDecl : public Node {
 public:
  VarDecl(const Type *type, Symbol sym, const Expr *expr)
      : Node(), type_(type), sym_(sym), expr_(expr) {}

  virtual ~VarDecl() = default;

  // Print the variable's declaration.
  void Visit(std::ostream *os) const override;

  virtual void Visit(mlirgen::MLIRGen *mlir_gen) const override;

 private:
  const Type *type_;
  Symbol sym_;
  const Expr *expr_;
};

// An if stmt with an else clause
class IfStmt : public Node {
 public:
  IfStmt(const Expr *cond, std::vector<const Node *> &&then_stmts, std::vector<const Node *> &&else_stmts)
      : Node(), cond_(cond), then_stmts_(std::move(then_stmts)), else_stmts_(std::move(else_stmts)) {}

  virtual ~IfStmt() = default;

  // Print: if(cond) then {...} else {...} .
  void Visit(std::ostream *os) const override;

 private:
  const Expr *cond_;
  const std::vector<const Node *> then_stmts_;
  const std::vector<const Node *> else_stmts_;
};

// A while loop
class WhileStmt : public Node {
 public:
  WhileStmt(const Expr *cond, std::vector<const Node *> &&body)
      : cond_(cond), body_(std::move(body)) {}

  virtual ~WhileStmt() = default;

  // Print: while (cond) {...}
  void Visit(std::ostream *os) const override;

 private:
  const Expr *cond_;
  const std::vector<const Node *> body_;
};

// A for loop
class ForStmt : public Node {
 public:
  ForStmt(const Node *init, const Expr *cond, const Expr *update, std::vector<const Node *> &&body)
      : init_(init), cond_(cond), update_(update), body_(std::move(body)) {}

  // Print for(init; cond; update) {...}
  void Visit(std::ostream *os) const override;

  virtual ~ForStmt() = default;

 private:
  const Node *init_;
  const Expr *cond_;
  const Expr *update_;
  const std::vector<const Node *> body_;
};

// A for in loop
class ForInStmt : public Node {
 public:
  ForInStmt(const Type *type, Symbol var, const Expr *iter, std::vector<const Node *> &&body)
      : Node(), type_(type), var_(var), iter_(iter), body_(std::move(body)) {}

  virtual ~ForInStmt() = default;

  // Print for (type var: iter) {...}
  void Visit(std::ostream *os) const override;

 private:
  const Type *type_;
  Symbol var_;
  const Expr *iter_;
  const std::vector<const Node *> body_;
};

// A return statement
class ReturnStmt : public Node {
 public:
  explicit ReturnStmt(const Expr *ret) : Node(), ret_(ret) {}

  virtual ~ReturnStmt() = default;

  // Print return ret;
  void Visit(std::ostream *os) const override;

  virtual void Visit(mlirgen::MLIRGen *mlir_gen) const override;

 private:
  const Expr *ret_;
};

// A jump Statement
enum JumpType {
  Break, Continue
};

// A break or continue
class JumpStmt : public Node {
 public:
  explicit JumpStmt(JumpType jump_type) : Node(), jump_type_(jump_type) {}

  virtual ~JumpStmt() = default;

  // Print break or continue.
  void Visit(std::ostream *os) const override;

 private:
  JumpType jump_type_;
};
}
