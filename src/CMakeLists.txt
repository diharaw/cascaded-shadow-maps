cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

add_executable(CascadedShadowMaps "main.cpp" "csm.h" "csm.cpp") 

target_link_libraries(CascadedShadowMaps dwSampleFramework)