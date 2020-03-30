set(CMAKE_CXX_STANDARD 17)
#add_subdirectory(include)

message("I AM BEING RUN!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# Codegen
file(GLOB_RECURSE ENTIER_SOURCES cpp_codegen/* mlir/*)

message("FOUND SOURCES ${ENTIER_SOURCES}")

set(LLVM_LINK_COMPONENTS Support)

add_toy_chapter(compiler_project
  main.cpp
  ${ENTIER_SOURCES}

  #DEPENDS
  #CompilerProjectOpsIncGen
  )


include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(compiler_project
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffects
    MLIRTransforms)
