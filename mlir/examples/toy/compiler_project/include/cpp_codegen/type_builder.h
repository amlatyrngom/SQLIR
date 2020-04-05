#pragma once
#include "ast.h"

namespace gen {

// Helper to build types
class TypeBuilder {
 public:
  // Constructor
  explicit TypeBuilder(CodegenContext *cg) : cg_(cg) {}

  // Gen: sym
  const Type *GetType(Symbol sym) {
    return cg_->AddType(std::make_unique<SimpleType>(sym));
  }

  // Return the associated type
  const Type* GetType(gen::BuiltinType type) {
    return cg_->GetType(type);
  }

  // Return the associated type
  const Type* GetType(gen::PrimType type) {
    return cg_->GetType(type);
  }

  // Gen: const type
  const Type *Const(const Type *type) {
    return cg_->AddType(std::make_unique<ConstType>(type));
  }

  // Gen: type&
  const Type *Ref(const Type *type) {
    return cg_->AddType(std::make_unique<RefType>(type));
  }

  // Gen: type*
  const Type *Pointer(const Type *type) {
    return cg_->AddType(std::make_unique<PointerType>(type));
  }

  // Gen: root<specials...>
  const Type *Template(const Type *root, std::vector<const Type *> &&specials) {
    return cg_->AddType(std::make_unique<TemplateType>(root, std::move(specials)));
  }

  // Gen: root<special>
  const Type *Template(const Type* root, const Type* special) {
    return Template(root, std::vector<const Type *>{special});
  }

  // Gen: const type&
  const Type *ConstRef(const Type *type) {
    return Const(Ref(type));
  }

  // Gen: const type*
  const Type *ConstPointer(const Type *type) {
    return Const(Pointer(type));
  }

  // Gen: auto
  const Type* Auto() {
    return cg_->GetType(gen::PrimType::Auto);
  }

  // Gen: const auto&
  const Type *ConstAutoRef() {
    return ConstRef(Auto());
  }

  // Gen: const auto*
  const Type *ConstAutoPointer() {
    return ConstPointer(Auto());
  }

  // Gen: auto&
  const Type *AutoRef() {
    return Ref(Auto());
  }


 private:
  CodegenContext *cg_;
};
}