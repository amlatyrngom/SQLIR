#include "node.h"

namespace gen {

void Include::Visit(std::ostream *os) const {
  char first_token = system_ ? '<' : '"';
  char last_token = system_ ? '>' : '"';
  *os << "#include " << first_token << path_.ident_ << last_token << std::endl;
}


void File::Visit(std::ostream *os) const {
  for (const auto& incl: includes_) {
    incl->Visit(os);
  }

  *os << "extern \"C\" {" << std::endl;

  for (const auto &child : top_level_) {
    child->Visit(os);
  }
  *os << "}" << std::endl;
}


void Function::Visit(std::ostream *os) const {
  // Print return type
  ret_type_->Visit(os);
  // Print name
  *os << " " << name_.ident_ << '(';
  // Print parameters
  for (uint32_t i = 0; i < params_.size(); i++) {
    const auto &param = params_[i];
    param.type_->Visit(os);
    *os << " " << param.symbol_.ident_;
    if (i < params_.size() - 1) {
      *os << ", ";
    }
  }
  *os << ") {" << std::endl;
  // Print body
  for (const auto &stmt : body_) {
    stmt->Visit(os);
  }
  // Close function
  *os << "}" << std::endl << std::endl;
}


void Struct::Visit(std::ostream *os) const {
  // Print struct name
  *os << "struct " << name_.ident_ << " { " << std::endl;
  // Print fields
  for (uint32_t i = 0; i < fields_.size(); i++) {
    const auto &field = fields_[i];
    field.type_->Visit(os);
    *os << " " << field.symbol_.ident_;
    if (i < fields_.size() - 1) {
      *os << ",";
    }
    *os << std::endl;
  }
  // Close struct
  *os << "};" << std::endl << std::endl;
}


void VarDecl::Visit(std::ostream *os) const {
  // Print type
  type_->Visit(os);
  // Print variable name
  *os << " " << sym_.ident_;
  if (expr_ != nullptr) {
    // Print expression
    *os << " = ";
    expr_->Visit(os);
  }
  *os << ";" << std::endl;
}


void IfStmt::Visit(std::ostream *os) const {
  // Print if clause
  *os << "if (";
  cond_->Visit(os);
  *os << ") {" << std::endl;
  // Print if body
  for (const auto &stmt : then_stmts_) {
    stmt->Visit(os);
  }
  // Close if
  *os << "}" << std::endl;
  // Print else
  if (!else_stmts_.empty()) {
    // Print else clause
    *os << "else {" << std::endl;
    for (const auto &stmt : else_stmts_) {
      stmt->Visit(os);
    }
    // Close else
    *os << "}" << std::endl;
  }
}


void WhileStmt::Visit(std::ostream *os) const {
  // Print while clause
  *os << "while (";
  cond_->Visit(os);
  *os << ") {" << std::endl;
  // Print while body
  for (const auto &stmt : body_) {
    stmt->Visit(os);
  }
  // Close while
  *os << "}" << std::endl;
}


void ForStmt::Visit(std::ostream *os) const {
  // Print for clause
  *os << "for (";
  // Print initialization. Should automatically have a semicolon
  if (init_ == nullptr) {
    *os << ";";
  } else {
    init_->Visit(os);
  }
  // Print condition
  if (cond_ != nullptr) cond_->Visit(os);
  *os << ";";
  // Print advance
  if (update_ != nullptr) update_->Visit(os);
  *os << ") {" << std::endl;
  // Print for body
  for (const auto &stmt : body_) {
    stmt->Visit(os);
  }
  // Close for
  *os << "}" << std::endl;
}


void ForInStmt::Visit(std::ostream *os) const {
  // Print for
  *os << "for (";
  // Print type
  type_->Visit(os);
  // Print variable
  *os << " " << var_.ident_ << " : ";
  // Print iterator
  iter_->Visit(os);
  *os << ") {" << std::endl;
  // Print body
  for (const auto &stmt : body_) {
    stmt->Visit(os);
  }
  // Close for
  *os << "}" << std::endl;
}


void ReturnStmt::Visit(std::ostream *os) const {
  // Print return
  *os << "return ";
  // Print expression
  if (ret_ != nullptr) ret_->Visit(os);
  *os << ";" << std::endl;
}


void JumpStmt::Visit(std::ostream *os) const {
  switch (jump_type_) {
    case JumpType::Break:*os << "break;" << std::endl;
      break;
    case JumpType::Continue:*os << "continue;" << std::endl;
      break;
  }
}

}