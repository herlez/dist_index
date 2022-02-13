cmake_minimum_required(VERSION 3.15)
project(index_dist)
set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)

#include libsais
add_library(libsais64
  extlib/libsais/src/libsais64.c
  extlib/libsais/src/libsais.c 
)
target_include_directories(libsais64 SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extlib/libsais/src)

#include alx_bwt
add_library(alx_bwt INTERFACE)
target_include_directories(alx_bwt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(alx_bwt INTERFACE libsais64)

#include tdc
add_library(tdc
  extlib/tdc/src/vec/int_vector.cpp
  extlib/tdc/src/vec/allocate.cpp
)
target_include_directories(tdc SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extlib/tdc/include)

#include pasta_wt
add_library(pasta_wt INTERFACE)
target_include_directories(pasta_wt 
SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extlib/wavelet_tree/
SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extlib/wavelet_tree/extlib/utils
SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extlib/wavelet_tree/extlib/bit_vector
SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extlib/tlx)

add_library(alx_rindex INTERFACE)
target_link_libraries(alx_rindex INTERFACE alx_bwt tdc pasta_wt malloc_count )

#include malloc_count
add_library(malloc_count ${CMAKE_CURRENT_SOURCE_DIR}/extlib/malloc_count/malloc_count.c)
target_include_directories(malloc_count PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extlib/malloc_count)
target_link_libraries(malloc_count -ldl)

#include tlx
add_library(tlx
  extlib/tlx/tlx/cmdline_parser.cpp
  extlib/tlx/tlx/string/parse_si_iec_units.cpp
  extlib/tlx/tlx/die/core.cpp
)
target_include_directories(tlx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extlib/tlx/tlx
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extlib/tlx)

#EXECUTABLES
add_executable(bench benchmark.cpp)
target_compile_options(bench PRIVATE -Wall -Wextra -pedantic -O3 -DNDEBUG)
target_include_directories(bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench PRIVATE alx_rindex tlx MPI::MPI_CXX malloc_count)

add_executable(gen_patterns genpatterns.c)
target_compile_options(gen_patterns PRIVATE -Wall -Wextra -pedantic -O3 -DNDEBUG)



