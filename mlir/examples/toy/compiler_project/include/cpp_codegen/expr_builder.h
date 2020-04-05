#pragma once
#include "ast.h"

namespace gen {
// Helper to build expressions
class ExprBuilder {
 public:
  // Constructor
  explicit ExprBuilder(CodegenContext* cg) : cg_(cg) {}

  // Make an expression from a symbol
  const Expr *MakeExpr(Symbol symbol) {
    return cg_->AddExpr(std::make_unique<IdentExpr>(symbol));
  }

  // Make an expression from a symbol
  const Expr* MakeExpr(gen::BuiltinSymbol sym) {
    return MakeExpr(cg_->GetSymbol(sym));
  }

  ////////////////////////////////////////////
  /// Literals
  /////////////////////////////////////////////

  // Integer literal
  const Expr *IntLiteral(int64_t val) {
    return cg_->AddExpr(std::make_unique<LiteralExpr>(val));
  }

  // Float literal
  const Expr *FloatLiteral(double val) {
    return cg_->AddExpr(std::make_unique<LiteralExpr>(val));
  }

  // Bool literal
  const Expr *BoolLiteral(bool val) {
    return cg_->AddExpr(std::make_unique<LiteralExpr>(val));
  }

  // Char literal
  const Expr *CharLiteral(char val) {
    return cg_->AddExpr(std::make_unique<LiteralExpr>(val));
  }

  // String literal
  const Expr *StringLiteral(std::string_view val) {
    return cg_->AddExpr(std::make_unique<LiteralExpr>(val));
  }

  //////////////////////////////////////////
  /// Object access & calls
  //////////////////////////////////////////

  // Gen: lhs.rhs or lhs->rhs
  const Expr *Member(const Expr *lhs, const Expr *rhs, bool ptr) {
    return cg_->AddExpr(std::make_unique<MemberOp>(ptr ? ExprType::Arrow : ExprType::Dot, lhs, rhs));
  }

  // Gen: lhs.rhs
  const Expr *Dot(const Expr *lhs, const Expr *rhs) {
    return Member(lhs, rhs, false);
  }

  // Gen: lhs->rhs
  const Expr *Arrow(const Expr *lhs, const Expr *rhs) {
    return Member(lhs, rhs, true);
  }

  // Gen: fn(args...)
  const Expr *Call(const Expr *fn, std::vector<const Expr *> &&args) {
    return cg_->AddExpr(std::make_unique<CallOp>(fn, std::move(args)));
  }

  // Gen: fn<types...>(args...)
  const Expr *TemplateCall(const Expr *fn, std::vector<const Type *> &&types, std::vector<const Expr *> &&args) {
    return cg_->AddExpr(std::make_unique<TemplateCallOp>(fn, std::move(types), std::move(args)));
  }

  // Gen: obj.fn(args...)
  const Expr *MemberCall(const Expr *obj, const Expr *fn, std::vector<const Expr *> &&args, bool ptr) {
    auto call = Call(fn, std::move(args));
    return Member(obj, call, ptr);
  }

  // Gen obj.fn<types...>(args...)
  const Expr *TemplateMemberCall(const Expr *obj,
                                 const Expr *fn,
                                 std::vector<const Type *> &&types,
                                 std::vector<const Expr *> &&args,
                                 bool ptr) {
    auto call = TemplateCall(fn, std::move(types), std::move(args));
    return Member(obj, call, ptr);
  }

  // Gen: obj.fn(args...)
  const Expr *DotCall(const Expr *obj, const Expr *fn, std::vector<const Expr *> &&children) {
    return MemberCall(obj, fn, std::move(children), false);
  }

  // Gen: obj->fn(args...)
  const Expr *ArrowCall(const Expr *obj, const Expr *fn, std::vector<const Expr *> &&children) {
    return MemberCall(obj, fn, std::move(children), true);
  }

  // Gen: obj.fn<types...>(args...)
  const Expr *TemplateDotCall(const Expr *obj,
                              const Expr *fn,
                              std::vector<const Type *> &&types,
                              std::vector<const Expr *> &&args) {
    return TemplateMemberCall(obj, fn, std::move(types), std::move(args), false);
  }

  // Gen: obj->fn<types...>(args...)
  const Expr *TemplateArrowCall(const Expr *obj,
                                const Expr *fn,
                                std::vector<const Type *> &&types,
                                std::vector<const Expr *> &&children) {
    return TemplateMemberCall(obj, fn, std::move(types), std::move(children), true);
  }

  // Gen: lhs[rhs]
  const Expr *Subsript(const Expr *lhs, const Expr *rhs) {
    return cg_->AddExpr(std::make_unique<SubscriptOp>(lhs, rhs));
  }

  // Gen: *obj
  const Expr *Deref(const Expr *obj) {
    return cg_->AddExpr(std::make_unique<PointerOp>(ExprType::Deref, obj));
  }

  // Gen: &obj
  const Expr *AddressOf(const Expr *obj) {
    return cg_->AddExpr(std::make_unique<PointerOp>(ExprType::Ref, obj));
  }

  ///////////////////////////////////////////////////
  /// Unary expressions
  //////////////////////////////////////////////////

  // Make a unary expression
  const Expr *Unary(ExprType op, const Expr *operand) {
    return cg_->AddExpr(std::make_unique<UnaryOp>(op, operand));
  }

  // Gen: +operand
  const Expr* UnaryPlus(const Expr *operand) {
    return cg_->AddExpr(std::make_unique<UnaryOp>(ExprType::Plus, operand));
  }

  // Gen: -operand
  const Expr* UnaryMinus(const Expr *operand) {
    return cg_->AddExpr(std::make_unique<UnaryOp>(ExprType::Minus, operand));
  }

  // Gen: ++operand
  const Expr* PreIncr(const Expr *operand) {
    return cg_->AddExpr(std::make_unique<UnaryOp>(ExprType::PreIncr, operand));
  }

  // Gen: --operand
  const Expr* PreDecr(const Expr *operand) {
    return cg_->AddExpr(std::make_unique<UnaryOp>(ExprType::PreDecr, operand));
  }

  // Gen: operand++
  const Expr* PostIncr(const Expr *operand) {
    return cg_->AddExpr(std::make_unique<UnaryOp>(ExprType::PostIncr, operand));
  }

  // Gen: operand--
  const Expr* PostDecr(const Expr *operand) {
    return cg_->AddExpr(std::make_unique<UnaryOp>(ExprType::PostDecr, operand));
  }

  // Gen: !operand
  const Expr* Not(const Expr *operand) {
    return cg_->AddExpr(std::make_unique<UnaryOp>(ExprType::Not, operand));
  }

  // Gen: ~operand
  const Expr* BitNot(const Expr *operand) {
    return cg_->AddExpr(std::make_unique<UnaryOp>(ExprType::BitNot, operand));
  }

  ///////////////////////////////////////////////////
  /// Binary expressions
  //////////////////////////////////////////////////

  // Make a binary expression
  const Expr *Binary(ExprType op, const Expr *lhs, const Expr *rhs) {
    return cg_->AddExpr(std::make_unique<BinaryOp>(op, lhs, rhs));
  }

  /////////////////////////////
  /// Arithmetic
  ////////////////////////////

  // Gen: lhs + rhs
  const Expr* IAdd(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::IAdd, lhs, rhs);
  }

  // Gen: lhs + rhs
  const Expr* FAdd(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::FAdd, lhs, rhs);
  }

  // Gen: lhs - rhs
  const Expr* ISub(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::ISub, lhs, rhs);
  }

  // Gen: lhs * rhs
  const Expr* IMul(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::IMul, lhs, rhs);
  }

  // Gen: lhs * rhs
  const Expr* FMul(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::FMul, lhs, rhs);
  }

  // Gen: lhs / rhs
  const Expr* IDiv(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::IDiv, lhs, rhs);
  }

  // Gen: lhs % rhs
  const Expr* IMod(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::IMod, lhs, rhs);
  }

  /////////////////////////////
  /// Comparison
  ////////////////////////////

  // Gen: lhs == rhs
  const Expr* Eq(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::Eq, lhs, rhs);
  }

  // Gen: lhs < rhs
  const Expr* Lt(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::Lt, lhs, rhs);
  }

  // Gen: lhs <= rhs
  const Expr* Le(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::Le, lhs, rhs);
  }

  // Gen: lhs > rhs
  const Expr* Gt(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::Gt, lhs, rhs);
  }

  // Gen: lhs >= rhs
  const Expr* Ge(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::Ge, lhs, rhs);
  }

  // Gen: lhs != rhs
  const Expr* Neq(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::Neq, lhs, rhs);
  }

  /////////////////////////////
  /// Logical
  ////////////////////////////

  // Gen: lhs && rhs
  const Expr* And(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::And, lhs, rhs);
  }

  // Gen: lhs || rhs
  const Expr* Or(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::Or, lhs, rhs);
  }

  /////////////////////////////
  /// Bitwise
  ////////////////////////////

  // Gen: lhs & rhs
  const Expr* BitAnd(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::BitAnd, lhs, rhs);
  }

  // Gen: lhs | rhs
  const Expr* BitOr(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::BitOr, lhs, rhs);
  }

  // Gen: lhs ^ rhs
  const Expr* BitXor(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::BitXor, lhs, rhs);
  }

  // Gen: lhs >> rhs
  const Expr* Shr(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::Shr, lhs, rhs);
  }

  // Gen: lhs << rhs
  const Expr* Shl(const Expr *lhs, const Expr *rhs) {
    return Binary(ExprType::Shl, lhs, rhs);
  }

  /////////////////////////////
  /// Assignments
  ////////////////////////////

  // Compound Assign
  const Expr *CompoundAssign(ExprType assign_type, const Expr *lhs, const Expr *rhs) {
    return cg_->AddExpr(std::make_unique<AssignOp>(assign_type, lhs, rhs));
  }

  // Gen: lhs = rhs
  const Expr *Assign(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::Assign, lhs, rhs);
  }

  // Gen: lhs += rhs
  const Expr *PlusEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::PlusEqual, lhs, rhs);
  }

  // Gen: lhs -= rhs
  const Expr *MinusEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::MinusEqual, lhs, rhs);
  }

  // Gen: lhs *= rhs
  const Expr *MulEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::MulEqual, lhs, rhs);
  }

  // Gen: lhs /= rhs
  const Expr *DivEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::DivEqual, lhs, rhs);
  }

  // Gen: lhs %= rhs
  const Expr *ModEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::ModEqual, lhs, rhs);
  }

  // Gen: lhs >>= rhs
  const Expr *ShrEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::ShrEqual, lhs, rhs);
  }

  // Gen: lhs <<= rhs
  const Expr *ShlEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::ShlEqual, lhs, rhs);
  }

  // Gen: lhs &= rhs
  const Expr *BitAndEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::BitAndEqual, lhs, rhs);
  }

  // Gen: lhs |= rhs
  const Expr *BitOrEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::BitOrEqual, lhs, rhs);
  }

  // Gen: lhs ^= rhs
  const Expr *BitXorEqual(const Expr *lhs, const Expr *rhs) {
    return CompoundAssign(ExprType::BitXorEqual, lhs, rhs);
  }

 private:
  CodegenContext *cg_;
};
}
