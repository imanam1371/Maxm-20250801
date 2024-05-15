# CMakeLists.txt for MAXM module. It creates a library with dictionary and a main program
cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
project(MAXM LANGUAGES CXX)
set(PACKAGE MAXM)

list(APPEND CMAKE_PREFIX_PATH)
message(STATUS "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})

#find_package(Boost)# REQUIRED program_options)
#include_directories(${Boost_INCLUDE_DIRS})

#--- Define the headers list
#include_directories(${PACKAGE} src dict ${CMAKE_SOURCE_DIR})
#file(GLOB_RECURSE headers ${PACKAGE}/*.h)

#--- Define the source list
#file(GLOB_RECURSE sources src/*.cpp)

#---Create a shared library with geneated dictionary
#add_library(${PACKAGE} SHARED ${sources} G__${PACKAGE}.cxx)
#target_link_libraries(${PACKAGE})