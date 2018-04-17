cmake_minimum_required(VERSION 3.10)
project(simd_fun)

# Add CMake modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

add_executable(simd_fun main.cpp Benchmark.cpp Benchmark.h minmax.cpp minmax.h)

# https://cmake.org/cmake/help/v3.8/prop_gbl/CMAKE_CXX_KNOWN_FEATURES.html#prop_gbl:CMAKE_CXX_KNOWN_FEATURES
target_compile_features(simd_fun
        PUBLIC
        cxx_std_17 cxx_auto_type cxx_constexpr cxx_defaulted_functions cxx_deleted_functions
        cxx_override cxx_final cxx_range_for cxx_lambdas)

# Compiler optimizations
# TODO: Make cross-platform "best performance" compiler option selection.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")

# Link-Time optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT output)
if(ipo_supported)
    message("-- IPO is supported!")
    set_property(TARGET simd_fun PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO is not supported: ${output}!")
endif()

# Enable AVX for the project
include(FindAVX)
check_for_avx()
if (HAVE_AVX2_EXTENSIONS)
    message(STATUS "Enabling AVX2 support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX_FLAGS}")
elseif (HAVE_AVX_EXTENSIONS)
    message(STATUS "Enabling AVX support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX_FLAGS}")
else()
    message(FATAL_ERROR "AVX or AVX2 support is required.")
endif()
