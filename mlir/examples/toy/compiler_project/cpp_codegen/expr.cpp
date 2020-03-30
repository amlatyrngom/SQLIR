#include "expr.h"
#include <iostream>

namespace gen {

void LiteralExpr::Visit(std::ostream *os) const {
  switch (Op()) {
    case ExprType::Int:*os << std::get<int64_t>(val_);
      break;
    case ExprType::String:*os << '"' << std::get<std::string_view>(val_) << '"';
      break;
    case ExprType::Float:*os << std::get<double>(val_);
      break;
    case ExprType::Char:*os << "'" << (std::get<char>(val_)) << "'";
      break;
    case ExprType::Bool:*os << (std::get<bool>(val_) ? "true" : "false");
      break;
    default:
      std::cout << "Not a literal expression!" << std::endl;
      abort();
  }
}


void AssignOp::Visit(std::ostream *os) const {
  // Gen member access
  auto lhs = Child(0);
  auto rhs = Child(1);
  lhs->Visit(os);
  switch (Op()) {
    case ExprType::Assign:*os << " = ";
      break;
    case ExprType::PlusEqual:*os << " += ";
      break;
    case ExprType::MinusEqual:*os << " -= ";
      break;
    case ExprType::MulEqual:*os << " *= ";
      break;
    case ExprType::DivEqual:*os << " /= ";
      break;
    case ExprType::ModEqual:*os << " %= ";
      break;
    case ExprType::ShrEqual:*os << " >>= ";
      break;
    case ExprType::ShlEqual:*os << " <<= ";
      break;
    case ExprType::BitAndEqual:*os << " &= ";
      break;
    case ExprType::BitOrEqual:*os << " |= ";
      break;
    case ExprType::BitXorEqual:*os << " ^= ";
      break;
    default:
      std::cout << "Not an assign op!" << std::endl;
      abort();
  }
  rhs->Visit(os);
}


void MemberOp::Visit(std::ostream *os) const {
  // Gen member access
  auto lhs = Child(0);
  auto rhs = Child(1);
  lhs->Visit(os);
  switch (Op()) {
    case ExprType::Dot:*os << ".";
      break;
    case ExprType::Arrow:*os << "->";
      break;
    default:
      std::cout << "Not a call expr!" << std::endl;
      abort();
  }
  rhs->Visit(os);
}


void CallOp::Visit(std::ostream *os) const {
  // Gen function
  auto fn = Child(0);
  fn->Visit(os);
  // Add args
  *os << "(";
  auto num_children = NumChildren();
  for (uint32_t i = 1; i < num_children; i++) {
    auto child = Child(i);
    child->Visit(os);
    if (i < num_children - 1) {
      *os << ", ";
    }
  }
  *os << ")";
}


void TemplateCallOp::Visit(std::ostream *os) const {
  // Gen function
  auto fn = Child(0);
  fn->Visit(os);

  // Add Types
  *os << "<";
  for (uint32_t i = 0; i < types_.size(); i++) {
    types_[i]->Visit(os);
    if (i < types_.size() - 1) {
      *os << ", ";
    }
  }
  *os << ">";

  // Add args
  *os << "(";
  auto num_children = NumChildren();
  for (uint32_t i = 1; i < num_children; i++) {
    auto child = Child(i);
    child->Visit(os);
    if (i < num_children - 1) {
      *os << ", ";
    }
  }
  *os << ")";
}


void SubscriptOp::Visit(std::ostream *os) const {
  // Open paren
  *os << "(";

  // Gen subscript op.
  auto lhs = Child(0);
  auto rhs = Child(1);
  lhs->Visit(os);
  *os << "[";
  rhs->Visit(os);
  *os << "]";

  // Close paren
  *os << ")";
}


void PointerOp::Visit(std::ostream *os) const {
  // Open paren
  *os << '(';

  // Gen pointer op
  auto operand = Child(0);
  switch (Op()) {
    case ExprType::Ref:*os << '&';
      break;
    case ExprType::Deref:*os << '*';
      break;
    default:
      std::cout << "Not a pointer operation" << std::endl;
      abort();
  }
  operand->Visit(os);

  // Close paren
  *os << ')';
}


void UnaryOp::Visit(std::ostream *os) const {
  // Open paren
  *os << '(';

  // Gen unary op
  auto operand = Child(0);
  switch (Op()) {
    case ExprType::Plus:*os << "+";
      operand->Visit(os);
      break;
    case ExprType::Minus:*os << "-";
      operand->Visit(os);
      break;
    case ExprType::Not:*os << "!";
      operand->Visit(os);
      break;
    case ExprType::BitNot:*os << "~";
      operand->Visit(os);
      break;
    case ExprType::PreIncr:*os << "++";
      operand->Visit(os);
      break;
    case ExprType::PreDecr:*os << "--";
      operand->Visit(os);
      break;
    case ExprType::PostIncr:operand->Visit(os);
      *os << "++";
      break;
    case ExprType::PostDecr:operand->Visit(os);
      *os << "--";
      break;
    default:
      std::cout << "Not a unary expr!" << std::endl;
      abort();
  }

  // Close paren
  *os << ')';
}


void BinaryOp::Visit(std::ostream *os) const {
  // Open paren
  *os << '(';

  // Gen binary op
  auto lhs = Child(0);
  auto rhs = Child(1);
  lhs->Visit(os);
  switch (Op()) {
    case ExprType::Add:*os << "+";
      break;
    case ExprType::Sub:*os << "-";
      break;
    case ExprType::Mul:*os << "*";
      break;
    case ExprType::Div:*os << "/";
      break;
    case ExprType::Mod:*os << "%";
      break;
    case ExprType::Lt:*os << "<";
      break;
    case ExprType::Le:*os << "<=";
      break;
    case ExprType::Gt:*os << ">";
      break;
    case ExprType::Ge:*os << ">=";
      break;
    case ExprType::Eq:*os << "==";
      break;
    case ExprType::Neq:*os << "!=";
      break;
    case ExprType::And:*os << "&&";
      break;
    case ExprType::Or:*os << "||";
      break;
    case ExprType::BitAnd:*os << "&";
      break;
    case ExprType::BitOr:*os << "|";
      break;
    case ExprType::BitXor:*os << "^";
      break;
    case ExprType::Shr:*os << ">>";
      break;
    case ExprType::Shl:*os << "<<";
      break;
    default:
      std::cout << "Not binary op!" << std::endl;
      abort();
  }
  rhs->Visit(os);

  // Close paren
  *os << ')';
}

}
