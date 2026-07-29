set(SOURCES
    ${PROJECT_ROOT_DIR}/tools/manybody_clustered_oracle_test/manybody_clustered_oracle_test.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/common.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/clustered_spatial_view.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/edip.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/tersoff.cpp)
set(TARGET_LINKER_LANGUAGE "${CPP_DIALECT}")

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
target_link_libraries(${CURRENT_TARGET} PRIVATE sponge_toml)

include(CTest)
add_test(NAME ManybodyClusteredOracle COMMAND ${CURRENT_TARGET})
