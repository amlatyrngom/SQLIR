#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffects.h"


namespace mlir {
  namespace sqlir {
    class SqlIRDialect : public mlir::Dialect {
    public:
      explicit SqlIRDialect(mlir::MLIRContext *ctx);

      /// Provide a utility accessor to the dialect namespace. This is used by
      /// several utilities for casting between dialects.
      static llvm::StringRef getDialectNamespace() { return "sqlir"; }
    };
  }
}
