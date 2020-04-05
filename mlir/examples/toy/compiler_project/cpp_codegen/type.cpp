#include "cpp_codegen/type.h"
#include "mlir/MLIRGen.h"
#include <iostream>

namespace gen {

mlir::Type Type::Visit(mlirgen::MLIRGen* mlir_gen) const {
  return mlir_gen->Builder()->getNoneType();
}

mlir::Type PrimitiveType::Visit(mlirgen::MLIRGen* mlir_gen) const {
  switch (prim_types_) {
    case PrimType::I64: {
      return mlir_gen->Builder()->getIntegerType(64, true);
    }
    case PrimType::F64: {
      return mlir_gen->Builder()->getF64Type();
    }
    default: {
      std::cout << "Unsupported Type!!!" << std::endl;
      abort();
    }
  }
}

}
