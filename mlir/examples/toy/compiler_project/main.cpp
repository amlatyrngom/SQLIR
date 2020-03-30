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

using namespace std;

void DumpMLIR(gen::Node* node) {
  // TODO(Create Dialect);
  mlir::registerDialect<mlir::sqlir::SQLIRDialect>();
  mlir::MLIRContext context;


  node.Visit(context);

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
  main_fn.AddField({T.GetType(gen::PrimType::Int), main_arg_ident});
  main_fn.SetRetType(T.GetType(gen::PrimType::Int));
  // Get Rate
  auto rate_sym = cg.GetSymbol("rate");
  auto rate = E.MakeExpr(rate_sym);
  main_fn.Declare(rate_sym, E.IntLiteral(37));
  main_fn.Return(E.Mul(rate, main_arg));

  auto main_node = main_fn.Finish();

  DumpMLIR(main_node);
  auto mlir_module = main_node.Visit();



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
