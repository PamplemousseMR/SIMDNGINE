#========================================
#	project
#========================================
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT("SIMDNGINE" CXX)

#================================================================================
#================================================================================

#========================================
#	variables
#========================================
SET(cppngine_src_dir "src")
SET(cppngine_header_dir "include")

#========================================
#	compiler specifications
#========================================
SET(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -msse4")

#================================================================================
#       create project
#================================================================================

#====================
#	find files
#====================
FILE(GLOB_RECURSE cppngine_src_files ${cppngine_src_dir}/*)
FILE(GLOB_RECURSE cppngine_header_files ${cppngine_header_dir}/*)

#====================
#	create
#====================
ADD_EXECUTABLE(${CMAKE_PROJECT_NAME} ${cppngine_src_files} ${cppngine_header_files})
TARGET_INCLUDE_DIRECTORIES(${CMAKE_PROJECT_NAME} PRIVATE ${cppngine_header_dir})
