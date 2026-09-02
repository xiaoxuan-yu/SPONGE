set(SOURCES
    ${PROJECT_ROOT_DIR}/tools/manybody_clustered_oracle_test/manybody_clustered_oracle_test.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/common.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/contract/view.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/edip.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/hydrogen_bond.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/tersoff.cpp)
set(TARGET_LINKER_LANGUAGE "${CPP_DIALECT}")

find_package(HighFive CONFIG REQUIRED)
find_package(HDF5 1.10.7 REQUIRED COMPONENTS C)

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
if(TARGET HighFive::HighFive)
  target_link_libraries(${CURRENT_TARGET} PRIVATE HighFive::HighFive)
elseif(TARGET HighFive)
  target_link_libraries(${CURRENT_TARGET} PRIVATE HighFive)
else()
  message(
    FATAL_ERROR "HighFive target was not provided by find_package(HighFive)")
endif()
if(TARGET HDF5::HDF5)
  target_link_libraries(${CURRENT_TARGET} PRIVATE HDF5::HDF5)
else()
  target_include_directories(${CURRENT_TARGET} PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${CURRENT_TARGET} PRIVATE ${HDF5_LIBRARIES})
endif()

include(CTest)
add_test(NAME ManybodyClusteredOracle COMMAND ${CURRENT_TARGET})
