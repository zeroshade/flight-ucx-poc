# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

cmake_minimum_required(VERSION 3.16)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")
project(arrow-flight-ucx)

include(CMakeParseArguments)

# https://www.cmake.org/cmake/help/latest/policy/CMP0025.html
#
# Compiler id for Apple Clang is now AppleClang.
cmake_policy(SET CMP0025 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0042.html
#
# Enable MACOSX_RPATH by default. @rpath in a target's install name is
# a more flexible and powerful mechanism than @executable_path or
# @loader_path for locating shared libraries.
cmake_policy(SET CMP0042 NEW)

# https://www.cmake.org/cmake/help/latest/policy/CMP0054.html
#
# Only interpret if() arguments as variables or keywords when unquoted.
cmake_policy(SET CMP0054 NEW)

# https://www.cmake.org/cmake/help/latest/policy/CMP0057.html
#
# Support new if() IN_LIST operator.
cmake_policy(SET CMP0057 NEW)

# https://www.cmake.org/cmake/help/latest/policy/CMP0063.html
#
# Adapted from Apache Kudu: https://github.com/apache/kudu/commit/bd549e13743a51013585
# Honor visibility properties for all target types.
cmake_policy(SET CMP0063 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0068.html
#
# RPATH settings on macOS do not affect install_name.
cmake_policy(SET CMP0068 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0074.html
#
# find_package() uses <PackageName>_ROOT variables.
cmake_policy(SET CMP0074 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0091.html
#
# MSVC runtime library flags are selected by an abstraction.
cmake_policy(SET CMP0091 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0135.html
#
# CMP0135 is for solving re-building and re-downloading.
# We don't have a real problem with the OLD behavior for now
# but we use the NEW behavior explicitly to suppress CMP0135
# warnings.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

find_package(ArrowFlight REQUIRED)
find_package(ArrowCUDA REQUIRED)
find_package(ArrowTesting REQUIRED)
message(STATUS "Found Arrow: ${ARROW_VERSION}")

add_library(ucx::ucx INTERFACE IMPORTED)
target_include_directories(ucx::ucx INTERFACE "${UCX_INCLUDE_DIRS}")
target_link_libraries(ucx::ucx INTERFACE ucx::ucp ucx::uct ucx::ucs)

add_executable(flight-ucx-poc flight_ucx_poc.cc
                              flight_ucx_conn.cc
                              flight_ucx_utils.cc
                              flight_ucx_server.cc
                              flight_ucx_client.cc
                              ucx_mmap_alloc.cc
                              random_generation.cc)
target_link_libraries(flight-ucx-poc arrow_shared arrow_testing_shared
                                     arrow_cuda_shared
                                     arrow_flight_shared ucx::ucx)
