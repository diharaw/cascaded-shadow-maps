cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project("CascadedShadowMaps")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(external/dwSampleFramework)

include_directories("${DW_SAMPLE_FRAMEWORK_INCLUDES}")

add_subdirectory(src)