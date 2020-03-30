set(CMAKE_CXX_STANDARD 20)

message("I AM BEING RUN!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# Codegen
file(GLOB_RECURSE ENTIER_SOURCES cpp_codegen/*)

message("FOUND SOURCES ${ENTIER_SOURCES}")

set(LLVM_LINK_COMPONENTS Support)

add_toy_chapter(compiler_project
  main.cpp
  ${ENTIER_SOURCES}
  )
include_directories(cpp_codegen/)
target_link_libraries(compiler_project
  PRIVATE
    MLIRSupport)
