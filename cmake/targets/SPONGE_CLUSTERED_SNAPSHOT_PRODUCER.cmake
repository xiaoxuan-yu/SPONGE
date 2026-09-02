if(NOT PARALLEL_BACKEND STREQUAL "cuda" AND
   NOT PARALLEL_BACKEND STREQUAL "hip")
  message(FATAL_ERROR
          "SPONGE_CLUSTERED_SNAPSHOT_PRODUCER requires a GPU backend")
endif()

find_package(tomlplusplus CONFIG REQUIRED)
find_package(HighFive CONFIG REQUIRED)
find_package(HDF5 1.10.7 REQUIRED COMPONENTS C)

include(${PROJECT_ROOT_DIR}/cmake/targets/SPONGE_runtime_sources.cmake)

if(NOT TARGET sponge_toml)
  add_library(
    sponge_toml STATIC
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml_decode.cpp)
  set_source_files_properties(
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml.cpp PROPERTIES LANGUAGE CXX)
  set_source_files_properties(
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml_decode.cpp
    PROPERTIES LANGUAGE CXX)
  target_include_directories(sponge_toml PUBLIC ${PROJECT_ROOT_DIR}/SPONGE)
  target_link_libraries(sponge_toml PUBLIC tomlplusplus::tomlplusplus)
endif()

if(NOT TARGET sponge_jit_header)
  add_library(
    sponge_jit_header STATIC
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/jit/embedded_common_header.cpp)
  set_source_files_properties(
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/jit/embedded_common_header.cpp
    PROPERTIES LANGUAGE CXX)
  target_include_directories(sponge_jit_header PUBLIC ${PROJECT_ROOT_DIR}/SPONGE)
endif()

set(SOURCES
    ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/clustered_lj_snapshot_producer_main.cpp
    ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/clustered_lj_snapshot_producer.cpp
    ${SPONGE_RUNTIME_SOURCES})
set(TARGET_LINKER_LANGUAGE "${CPP_DIALECT}")

add_executable(${CURRENT_TARGET} ${SOURCES})
target_compile_definitions(${CURRENT_TARGET}
                           PRIVATE SPONGE_EMBEDDED_RUNTIME=1)
target_link_libraries(${CURRENT_TARGET} PRIVATE sponge_jit_header sponge_toml)
set_target_properties(
  ${CURRENT_TARGET}
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             HIP_STANDARD 20
             HIP_STANDARD_REQUIRED ON)
target_include_directories(
  ${CURRENT_TARGET}
  PRIVATE ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench
          ${PROJECT_ROOT_DIR}/SPONGE
          ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include)
if(TARGET HighFive::HighFive)
  target_link_libraries(${CURRENT_TARGET} PRIVATE HighFive::HighFive)
elseif(TARGET HighFive)
  target_link_libraries(${CURRENT_TARGET} PRIVATE HighFive)
endif()
if(TARGET HDF5::HDF5)
  target_link_libraries(${CURRENT_TARGET} PRIVATE HDF5::HDF5)
else()
  target_include_directories(${CURRENT_TARGET} PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${CURRENT_TARGET} PRIVATE ${HDF5_LIBRARIES})
endif()
