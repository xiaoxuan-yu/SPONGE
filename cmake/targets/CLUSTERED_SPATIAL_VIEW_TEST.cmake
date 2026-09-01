set(SOURCES
    ${PROJECT_ROOT_DIR}/tools/clustered_spatial_view_test/clustered_spatial_view_test.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/contract/view.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/provider/provider.cpp)

add_executable(${CURRENT_TARGET} ${SOURCES})
set_target_properties(${CURRENT_TARGET}
                      PROPERTIES CXX_STANDARD 20
                                 CXX_STANDARD_REQUIRED ON
                                 CUDA_STANDARD 20
                                 CUDA_STANDARD_REQUIRED ON
                                 HIP_STANDARD 20
                                 HIP_STANDARD_REQUIRED ON)
target_include_directories(
  ${CURRENT_TARGET}
  PRIVATE ${PROJECT_ROOT_DIR}/SPONGE
          ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include)
target_compile_options(
  ${CURRENT_TARGET}
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections>)

include(CTest)
add_test(NAME ClusteredSpatialViewContract COMMAND ${CURRENT_TARGET})
