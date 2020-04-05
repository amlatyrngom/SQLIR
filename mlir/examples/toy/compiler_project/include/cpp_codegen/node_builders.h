#include "ast.h"

namespace gen {

class NodeFactory {
 public:
  // Constructor
  explicit NodeFactory(CodegenContext *cg) : cg_{cg} {}

  // Make an include node.
  const Node *MakeInclude(Symbol sym, bool system) {
    return cg_->AddNode(std::make_unique<Include>(sym, system));
  }

  // Make a function
  const Node *MakeFunction(Symbol fn_name,
                           const Type *ret_type,
                           std::vector<Field> &&params,
                           std::vector<const Node *> &&body) {
    return cg_->AddNode(std::make_unique<Function>(fn_name, ret_type, std::move(params), std::move(body)));
  }

  // Make a struct
  const Node *MakeStruct(Symbol name, std::vector<Field> &&fields) {
    return cg_->AddNode(std::make_unique<Struct>(name, std::move(fields)));
  }

  // Make a file
  const Node *MakeFile(const std::vector<const Node*> includes, std::vector<const Node *> &&content) {
    return cg_->AddNode(std::make_unique<File>(std::move(includes), std::move(content)));
  }

  // Make a stmt
  const Node *MakeStmt(const Expr *expr) {
    return cg_->AddNode(std::make_unique<ExprStmt>(expr));
  }

  // Make a variable declaration
  const Node *Declare(const Type *type, Symbol sym, const Expr *expr) {
    return cg_->AddNode(std::make_unique<VarDecl>(type, sym, expr));
  }

  // Make an if statement
  const Node *MakeIf(const Expr *cond,
                     std::vector<const Node *> &&then_stmts,
                     std::vector<const Node *> &&else_stmts = {}) {
    return cg_->AddNode(std::make_unique<IfStmt>(cond, std::move(then_stmts), std::move(else_stmts)));
  }

  // Make a while statement
  const Node *MakeWhile(const Expr *cond, std::vector<const Node *> &&body) {
    return cg_->AddNode(std::make_unique<WhileStmt>(cond, std::move(body)));
  }

  // Make a for statement
  const Node *MakeFor(const Node *init, const Expr *cond, const Expr *update, std::vector<const Node *> &&body) {
    return cg_->AddNode(std::make_unique<ForStmt>(init, cond, update, std::move(body)));
  }

  // Make a for in statement
  const Node *MakeForIn(const Type *type, Symbol var, const Expr *iter, std::vector<const Node *> &&body) {
    return cg_->AddNode(std::make_unique<ForInStmt>(type, var, iter, std::move(body)));
  }

  // Make a return statement
  const Node *MakeReturn(const Expr *ret) {
    return cg_->AddNode(std::make_unique<ReturnStmt>(ret));
  }

  // Make a break statement
  const Node *MakeBreak() {
    return cg_->AddNode(std::make_unique<JumpStmt>(JumpType::Break));
  }

  // Make a continue statement
  const Node *MakeContinue() {
    return cg_->AddNode(std::make_unique<JumpStmt>(JumpType::Continue));
  }
 private:
  CodegenContext *cg_;
};


// Generic builder
class Builder {
 public:
  // Constructor
  explicit Builder(CodegenContext* cg) : cg_(cg), fac_(cg) {}

  virtual ~Builder() = default;

  // Add an item to the function body
  void Add(const Node* node) {
    body_.emplace_back(node);
  }

  // Add an expression statement
  void Add(const Expr* expr) {
    Add(fac_.MakeStmt(expr));
  }

  // Return an expression
  void Return(const Expr* ret) {
    Add(fac_.MakeReturn(ret));
  }

  // Make a variable declaration with auto type
  void Declare(Symbol sym, const Expr *expr) {
    Add(cg_->AddNode(std::make_unique<VarDecl>(cg_->GetType(PrimType::Auto), sym, expr)));
  }

  // Make a variable declaration
  void Declare(const Type *type, Symbol sym, const Expr *expr) {
    Add(cg_->AddNode(std::make_unique<VarDecl>(type, sym, expr)));
  }

  // Finish constructing.
  virtual const Node* Finish() = 0;

 protected:
  CodegenContext* cg_;
  NodeFactory fac_;
  std::vector<const Node*> body_;
};

// File builder
class FileBuilder : public Builder {
 public:
  // Constructor
  explicit FileBuilder(CodegenContext* cg) : Builder(cg) {}

  virtual ~FileBuilder() = default;

  // Include the given path
  void Include(const Symbol& path, bool system=false) {
    includes_.emplace_back(fac_.MakeInclude(path, system));
  }

  const Node* Finish() override {
    return fac_.MakeFile(std::move(includes_), std::move(body_));
  }
 private:
  std::vector<const Node*> includes_;
};

// Function builder
class FunctionBuilder : public Builder {
 public:
  // Constructor
  explicit FunctionBuilder(CodegenContext* cg) : Builder(cg) {}

  virtual ~FunctionBuilder() = default;


  // Set the return type.
  void SetRetType(const Type* ret_type) {
    ret_type_ = ret_type;
  }

  // Set the function name
  void SetName(const Symbol& fn_name) {
    fn_name_ = fn_name;
  }

  // Add a field
  void AddField(const Field& param) {
    params_.emplace_back(param);
  }

  // Add a bunch of fields
  void AddFields(const std::vector<Field>& params) {
    params_.insert(params_.end(), params.begin(), params.end());
  }

  // Finish the function
  const Node* Finish() override {
    return fac_.MakeFunction(fn_name_, ret_type_, std::move(params_), std::move(body_));
  }

 private:
  const Type* ret_type_{nullptr};
  Symbol fn_name_;
  std::vector<Field> params_;
};

// While builder
class WhileBuilder : public Builder {
 public:
  // Constructor
  explicit WhileBuilder(CodegenContext* cg) : Builder(cg) {}

  virtual ~WhileBuilder() = default;


  // Set the condition.
  void SetCondition(const Expr* expr) {
    cond_ = expr;
  }

  const Node* Finish() override {
    return fac_.MakeWhile(cond_, std::move(body_));
  }
 private:
  const Expr* cond_{nullptr};
};

// If builder
class IfBuilder : public Builder {
 public:
  // Constructor
  explicit IfBuilder(CodegenContext* cg) : Builder(cg) {}

  virtual ~IfBuilder() = default;


  // Set the condition.
  void SetCondition(const Expr* expr) {
    cond_ = expr;
  }

  void StartElse() {
    has_else_ = true;
    then_stmts_ = std::move(body_);
    body_.clear();
  }


  const Node* Finish() override {
    if (has_else_) {
      else_stmts_ = std::move(body_);
    } else {
      then_stmts_ = std::move(body_);
    }
    return fac_.MakeIf(cond_, std::move(then_stmts_), std::move(else_stmts_));
  }
 private:
  const Expr* cond_{nullptr};
  bool has_else_{false};
  std::vector<const gen::Node*> then_stmts_;
  std::vector<const gen::Node*> else_stmts_;
};

// For builder
class ForBuilder : public Builder {
 public:
  // Constructor
  explicit ForBuilder(CodegenContext* cg) : Builder(cg) {}

  virtual ~ForBuilder() = default;


  // Set the initial stmt
  void SetInit(const Node* init) {
    init_ = init;
  }

  // Set the condition.
  void SetCondition(const Expr* expr) {
    cond_ = expr;
  }

  // Set the update expression.
  void SetUpdate(const Expr* expr) {
    update_ = expr;
  }

  const Node* Finish() override {
    return fac_.MakeFor(init_, cond_, update_, std::move(body_));
  }

 private:
  const Node* init_{nullptr};
  const Expr* cond_{nullptr};
  const Expr* update_{nullptr};
};

// For in builder
class ForInBuilder : public Builder {
 public:
  // Constructor
  explicit ForInBuilder(CodegenContext* cg) : Builder(cg) {}

  virtual ~ForInBuilder() = default;

  // Set variable type
  void SetVarType(const Type* type) {
    var_type_ = type;
  }

  // Set variable symbol
  void SetSymbol(const Symbol& var) {
    var_ = var;
  }

  // Set iteration
  void SetIter(const Expr* expr) {
    iter_ = expr;
  }

  const Node* Finish() override {
    return fac_.MakeForIn(var_type_, var_, iter_, std::move(body_));
  }

 private:
  const Type* var_type_{nullptr};
  Symbol var_;
  const Expr* iter_{nullptr};
};

}
