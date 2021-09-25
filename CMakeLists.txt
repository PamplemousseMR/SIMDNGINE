#=======================================
#                PROJECT
#=======================================

cmake_minimum_required(VERSION 3.0)
project("SIMDNGINE" CXX)

#=======================================
#               EXECUTABLE
#=======================================

set(simdngine_src
    src/main.cpp
    src/MMXMmintrin.cpp
    src/SSE2Emmintrin.cpp
    src/SSE3Pmmintrin.cpp
    src/SSE41Smmintrin.cpp
    src/SSEXmmintrin.cpp
    src/SSSE3Tmmintrin.cpp)

set(simdngine_header
    include/simd.hpp)

add_executable(SIMDNGINE ${simdngine_src} ${simdngine_header})
target_include_directories(SIMDNGINE PRIVATE "include")
set_target_properties(SIMDNGINE PROPERTIES CXX_STANDARD 14)
set_target_properties(SIMDNGINE PROPERTIES COMPILE_FLAGS "-msse -msse2 -msse3 -msse4")
