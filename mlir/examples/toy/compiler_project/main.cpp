// CppCodegen.cpp : Defines the entry point for the application.
//

#include "cpp_codegen/ast.h"
#include "cpp_codegen/node_builders.h"
#include "cpp_codegen/expr_builder.h"
#include "cpp_codegen/type_builder.h"
#include "cpp_codegen/compile.h"
#include <memory>
#include <iostream>
#include <cassert>

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Dialect.h"
#include "mlir/MLIRGen.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"


using namespace std;

void DumpMLIR(const gen::Node* node) {
  // TODO(Create Dialect);
  mlir::registerDialect<mlir::sqlir::SqlIRDialect>();
  mlir::MLIRContext context;

  mlirgen::MLIRGen mlir_gen(context);
  mlir::OwningModuleRef module = mlir_gen.mlirGen(node);
  if (!module) std::cout << "ERRRRRRR" << std::endl;

  module->dump();
}


int main() {
  gen::CodegenContext cg;
  gen::NodeFactory N(&cg);
  gen::ExprBuilder E(&cg);
  gen::TypeBuilder T(&cg);
  gen::Compiler comp;

  // Needed expressions.
  auto main_arg_ident = cg.GetSymbol("main_arg");
  auto main_arg = E.MakeExpr(main_arg_ident);

  // Make the file
  gen::FileBuilder file_builder(&cg);
  file_builder.Include(cg.GetSymbol("iostream"), true);

  // Start the main function
  gen::FunctionBuilder main_fn(&cg);
  main_fn.SetName(cg.GetSymbol("Main"));
  main_fn.AddField({T.GetType(gen::PrimType::I64), main_arg_ident});
  main_fn.SetRetType(T.GetType(gen::PrimType::I64));
  // Get Rate
  auto rate_sym = cg.GetSymbol("rate");
  auto rate = E.MakeExpr(rate_sym);
  main_fn.Declare(rate_sym, E.FloatLiteral(37.73));
  main_fn.Return(E.IMul(rate, main_arg));

  auto main_node = main_fn.Finish();

  DumpMLIR(main_node);

  // Finish main
  file_builder.Add(main_fn.Finish());

  // Finish file
  auto file = file_builder.Finish();
  file->Visit(&std::cout);
  auto module = comp.Compile(file);
  auto compiled_fn = module->GetFn();
  assert(compiled_fn(73) == 37*73);
  assert(compiled_fn(37) == 37*37);
  assert(compiled_fn(1024) == 37*1024);
  return 0;
}
