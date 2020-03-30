#pragma once
#include <fstream>
#include <unistd.h>
#include <sstream>
#include <dlfcn.h>
#include <functional>
#include <memory>
#include "node.h"
#define GXX "/usr/bin/g++-9"

using RawMainFn = int(int);
using MainFn = std::function<int(int)>;

namespace gen {

// A compiled module
class Module {
 public:
  // Contructor
  Module(const Node* node, uint64_t id);

  // Deconstructor
  ~Module();

  // Explicit destroy
  void Destroy();

  // Compile the given node
  void Compile();

  // Get the compiled function.
  MainFn GetFn() {
    return fn_;
  }

 private:
  // Remove temporary files.
  void RemoveTempFiles();

  // File to compile
  const Node* node_;
  // Unique identifier
  uint64_t id_;
  // Compiled function
  MainFn fn_;
  // Prefix of the file
  std::string file_prefix_;
  // .cpp file name
  std::string cpp_filename_;
  // .o filename
  std::string o_filename_;
  // .so filename
  std::string lib_filename_;
  // Handle of the opened .so file
  void* lib_handle_{nullptr};
  // Whether this object has been explicitly destroyed.
  bool destroyed_{false};
};

class Compiler {
 public:
  Compiler() = default;

  std::unique_ptr<Module> Compile(const gen::Node* node) {
    auto module = std::make_unique<Module>(node, file_id_++);
    module->Compile();
    return module;
  }

 private:
  uint64_t file_id_{0};
};
}