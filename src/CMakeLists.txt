cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(APPLE)
    add_executable(CascadedShadowMaps MACOSX_BUNDLE "main.cpp" "csm.h" "csm.cpp")
    set(MACOSX_BUNDLE_BUNDLE_NAME "com.dihara.csm") 
else()
    add_executable(CascadedShadowMaps "main.cpp" "csm.h" "csm.cpp") 
endif()

target_link_libraries(CascadedShadowMaps dwSampleFramework)