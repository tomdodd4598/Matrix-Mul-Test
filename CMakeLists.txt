cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES 90)
project(MyProject LANGUAGES CUDA CXX)

list(APPEND CXX_COMPILE_OPTIONS "-Wall" "-Wextra" "-O3")
foreach(CXX_COMPILE_OPTION IN LISTS CXX_COMPILE_OPTIONS)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=${CXX_COMPILE_OPTION}>")
endforeach()

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-arch=compute_${CMAKE_CUDA_ARCHITECTURES}>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-code=sm_${CMAKE_CUDA_ARCHITECTURES}>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-O3>")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(IncludeDir ${MyProject_SOURCE_DIR}/include)
set(TestDir ${MyProject_SOURCE_DIR}/test)

set(BinDir ${CMAKE_BINARY_DIR}/bin)
set(LibDir ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LibDir})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LibDir})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BinDir})

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(${MyProject_SOURCE_DIR}/eigen3)

add_subdirectory(src)
