#pragma once

#include <cstdint>
#include <exception>
#include <vector>
#include <memory>
#include <string>
#include <string_view>

namespace gen {

// A symbol in the program
struct Symbol {
  Symbol() = default;
  Symbol(std::string_view ident) : ident_(ident) {}
  std::string_view ident_;
};

// List of builtin symbol for convenient access
#define BUILTIN_SYMBOL(F) \
F(Size, "size") \
F(EmplaceBack, "emplace_back") \
F(Cout, "std::cout") \
F(Endl, "std::endl") \
F(Nullptr, "nullptr")

// Make enum of builtin symbols
#define BUILTIN_SYMBOL_ENTRY(type, ident) type,
enum class BuiltinSymbol : uint32_t {
  BUILTIN_SYMBOL(BUILTIN_SYMBOL_ENTRY)
};
#undef BUILTIN_SYMBOL_ENTRY
}
