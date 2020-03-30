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

int main() {
  gen::CodegenContext cg;
  gen::NodeFactory N(&cg);
  gen::ExprBuilder E(&cg);
  gen::TypeBuilder T(&cg);
  gen::Compiler comp;

  // Needed expressions.
  auto cout_fn = E.MakeExpr(gen::BuiltinSymbol::Cout);
  auto endl_expr = E.MakeExpr(gen::BuiltinSymbol::Endl);
  auto hello_world = E.StringLiteral("Hello, World!");
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
  // Call cout
  auto cout_call = E.Shl(E.Shl(E.Shl(cout_fn, hello_world), main_arg), endl_expr);
  main_fn.Add(cout_call);
  // return the input argument
  main_fn.Return(main_arg);

  // Finish main
  file_builder.Add(main_fn.Finish());

  // Finish file
  auto file = file_builder.Finish();
  file->Visit(&std::cout);
  auto module = comp.Compile(file);
  auto compiled_fn = module->GetFn();
  assert(compiled_fn(73) == 73);
  assert(compiled_fn(37) == 37);
  assert(compiled_fn(1024) == 1024);
  return 0;
}
