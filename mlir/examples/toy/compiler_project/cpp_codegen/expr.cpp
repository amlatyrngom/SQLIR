#include "cpp_codegen/expr.h"
#include <iostream>
#include "mlir/MLIRGen.h"

using namespace mlir;

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

mlir::Value LiteralExpr::Visit(mlirgen::MLIRGen* mlir_gen) const {
  switch (Op()) {
    case ExprType::Int: {
      mlir::Type type = mlir_gen->Builder()->getIntegerType(64, true);
      auto val = std::get<int64_t>(val_);
      auto mlir_attr = mlir_gen->Builder()->getIntegerAttr(type, val);
      return mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), type, mlir_attr);
    }
    case ExprType::Float: {
      mlir::Type type = mlir_gen->Builder()->getF64Type();
      auto val = std::get<double>(val_);
      auto mlir_attr = mlir_gen->Builder()->getFloatAttr(type, val);
      return mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), type, mlir_attr);
    }
    default:
      std::cout << "Not a literal expression!" << std::endl;
      abort();
  }
}

mlir::Value SelectExpr::Visit(mlirgen::MLIRGen* mlir_gen) const {
  auto block1 = mlir_gen->Builder()->createBlock(mlir_gen->Builder()->getBlock()->getParent(),
    mlir_gen->Builder()->getBlock()->getParent()->end());

  mlir_gen->Builder()->setInsertionPointToStart(block1);

  for(auto id: column_ids_) {
    llvm::StringRef callee("getcolumn");
    auto location = mlir_gen->Loc();

    // Codegen the operands first.
    SmallVector<mlir::Value, 2> operands;
    mlir::Type type = mlir_gen->Builder()->getIntegerType(64, true);
    auto mlir_attr = mlir_gen->Builder()->getIntegerAttr(type, table_id_);
    auto arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), type, mlir_attr);
    operands.push_back(arg);

    auto mlir_attr = mlir_gen->Builder()->getIntegerAttr(type, id);
    arg = mlir_gen->Builder()->create<ConstantOp>(mlir_gen->Loc(), type, mlir_attr);
    operands.push_back(arg);

    builder.create<GenericCallOp>(location, callee, operands);
  }


}

mlir::Value IdentExpr::Visit(mlirgen::MLIRGen* mlir_gen) const {
  if (auto variable = mlir_gen->SymTable()->lookup(symbol_.ident_))
    return variable;

  emitError(mlir_gen->Loc(), "error: unknown variable '")
      << std::string(symbol_.ident_) << "'";
  return nullptr;
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
    case ExprType::IAdd:
    case ExprType::FAdd:
      *os << "+";
      break;
    case ExprType::ISub:
    case ExprType::FSub:
      *os << "-";
      break;
    case ExprType::IMul:
    case ExprType::FMul:
      *os << "*";
      break;
    case ExprType::IDiv:*os << "/";
      break;
    case ExprType::IMod:*os << "%";
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

mlir::Value BinaryOp::Visit(mlirgen::MLIRGen *mlir_gen) const {
  // Gen binary op
  auto lhs = Child(0)->Visit(mlir_gen);
  auto rhs = Child(1)->Visit(mlir_gen);

  switch (Op()) {
    case ExprType::IAdd: {
      // Otherwise, this return operation has zero operands.
      return mlir_gen->Builder()->create<mlir::AddIOp>(mlir_gen->Loc(), lhs, rhs);
    }
    case ExprType::FAdd: {
      // Otherwise, this return operation has zero operands.
      return mlir_gen->Builder()->create<mlir::AddFOp>(mlir_gen->Loc(), lhs, rhs);
    }
    case ExprType::IMul: {
      // Otherwise, this return operation has zero operands.
      return mlir_gen->Builder()->create<mlir::MulIOp>(mlir_gen->Loc(), lhs, rhs);
    }
    case ExprType::FMul: {
      // Otherwise, this return operation has zero operands.
      return mlir_gen->Builder()->create<mlir::MulFOp>(mlir_gen->Loc(), lhs, rhs);
    }
    default: {
      std::cout << "Unsupported Binary Op" << std::endl;
      return nullptr;
    }
  }

}

}
