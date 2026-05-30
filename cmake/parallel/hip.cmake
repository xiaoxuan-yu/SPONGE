enable_language(HIP)
set(CPP_DIALECT "HIP")
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)
find_package(rocthrust REQUIRED)
find_package(hipcub REQUIRED)
get_property(rocthrust_include TARGET roc::rocthrust
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
get_property(hipcub_include TARGET hip::hipcub
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
get_property(rocprim_include TARGET roc::rocprim
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(common_libraries INTERFACE ${rocthrust_include}
                                                     ${hipcub_include}
                                                     ${rocprim_include})

include("${PROJECT_ROOT_DIR}/cmake/math/hip.cmake")

add_definitions(-DUSE_GPU)
add_definitions(-DUSE_HIP)
