#include "cpp_codegen/compile.h"

namespace gen {
Module::Module(const Node* node, uint64_t id)
    : node_{node}
    , id_{id}
    , file_prefix_(std::to_string(id_))
    , cpp_filename_(file_prefix_ + ".cpp")
    , o_filename_(file_prefix_ + ".o")
    , lib_filename_(("lib" + file_prefix_) + ".so") {}

Module::~Module() {
  if (!destroyed_) {
    Destroy();
  }
}

void Module::Destroy() {
  destroyed_ = true;
  dlclose(lib_handle_);
  std::remove(lib_filename_.data());
}

void Module::RemoveTempFiles() {
  std::remove(cpp_filename_.data());
  std::remove(o_filename_.data());
}

void Module::Compile() {
  // Make the cpp file
  {
    std::ofstream cpp_file(cpp_filename_);
    node_->Visit(&cpp_file);
    cpp_file.close();
  }
  // Make .o file
  {
    // TODO(Amadou): Pass in right optimization flags
    std::stringstream ss;
    ss << GXX << " -g -c -fpic " << cpp_filename_;
    system(ss.str().data());
  }
  // Make .so file
  {
    std::stringstream ss;
    ss << GXX << " -shared -o " << lib_filename_ << " " << o_filename_;
    system(ss.str().data());
  }

  RemoveTempFiles();
  // Now load the file
  {
    std::stringstream ss;
    ss << "./" << lib_filename_;
    lib_handle_ = dlopen(ss.str().data(), RTLD_LAZY);
    if (!lib_handle_) {
      perror("dlopen");
      abort();
    }
    dlerror();
    fn_ = reinterpret_cast<RawMainFn *>(dlsym(lib_handle_, "Main"));
    auto dlsym_error = dlerror();
    if (dlsym_error) {
      perror(dlsym_error);
      abort();
    }
  }
}

}
