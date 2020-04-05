#pragma once

#include "cpp_codegen/node.h"
#include "Dialect.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"


#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace mlirgen {

/// Implementation of a simple MLIR emission from the sqlir AST.
///
/// This will emit operations that are specific to the sqlir language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGen {
public:
  MLIRGen(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a sqlir module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(const gen::Node* node) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    
    node->Visit(this);

    // Verify
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    // Return the module
    return theModule;
  }

  mlir::ModuleOp* Module() {
    return &theModule;
  }

  mlir::OpBuilder* Builder() {
    return &builder;
  }

  llvm::ScopedHashTable<StringRef, mlir::Value>* SymTable() {
    return &symbolTable;
  }

  // Return dummy location
  mlir::Location Loc() {
    return builder.getFileLineColLoc(builder.getIdentifier("input_file.sql"), line++,
                                     0);
  }

  mlir::LogicalResult Declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

private:
  uint32_t line{0};

  /// A "module" matches a sqlir source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
};
}
