cmake_minimum_required(VERSION 3.18)
project(minfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Prebuilt llama.cpp (headers + shared libs already on disk)
set(LLAMA_DIR /home/jose/llama.cpp)
set(LLAMA_BIN ${LLAMA_DIR}/build/bin)

# --- CPU reference engine (M1) ---
find_package(OpenMP)
add_executable(cpu_infer src/cpu_infer.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cpu_infer PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- naive CUDA engine (M2) ---
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_executable(cuda_infer src/cuda_infer.cu)
  set_target_properties(cuda_infer PROPERTIES CUDA_ARCHITECTURES 87)
endif()

# --- reference oracle (links prebuilt libllama) ---
add_executable(ref_oracle tools/ref_oracle.cpp)
target_include_directories(ref_oracle PRIVATE ${LLAMA_DIR}/include ${LLAMA_DIR}/ggml/include)
target_link_libraries(ref_oracle PRIVATE ${LLAMA_BIN}/libllama.so)
set_target_properties(ref_oracle PROPERTIES BUILD_RPATH ${LLAMA_BIN} INSTALL_RPATH ${LLAMA_BIN})
