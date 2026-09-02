if(NOT TARGET sponge_clustered_neighbor_candidate_builder)
  add_library(
    sponge_clustered_neighbor_candidate_builder OBJECT
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/candidate_builder.cpp)
  set_target_properties(
    sponge_clustered_neighbor_candidate_builder
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               HIP_STANDARD 20
               HIP_STANDARD_REQUIRED ON)
  set_source_files_properties(
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/candidate_builder.cpp
    PROPERTIES LANGUAGE ${CPP_DIALECT}
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               HIP_STANDARD 20
               HIP_STANDARD_REQUIRED ON)
  target_include_directories(
    sponge_clustered_neighbor_candidate_builder
    PRIVATE ${PROJECT_ROOT_DIR}/SPONGE
            ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include)
  target_link_libraries(sponge_clustered_neighbor_candidate_builder
                        PUBLIC common_libraries)
endif()

add_library(
  sponge_clustered_lj_candidate_leaf_probe OBJECT
  ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/candidate_leaf_probe.cu)
set_target_properties(
  sponge_clustered_lj_candidate_leaf_probe
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             HIP_STANDARD 20
             HIP_STANDARD_REQUIRED ON)
target_include_directories(
  sponge_clustered_lj_candidate_leaf_probe
  PRIVATE ${PROJECT_ROOT_DIR}/SPONGE
          ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include)
target_link_libraries(sponge_clustered_lj_candidate_leaf_probe
                      PUBLIC common_libraries)

set(SOURCES ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/nbnxm_microbench.cu
            ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/canonical_pair_oracle.cpp
            ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/gather_experiment.cu
            ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/gromacs_forceonly_replay.cu
            ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/record_stream_probe.cu
            $<TARGET_OBJECTS:sponge_clustered_neighbor_candidate_builder>
            $<TARGET_OBJECTS:sponge_clustered_lj_candidate_leaf_probe>)
set(TARGET_LINKER_LANGUAGE "CXX")

set(NBNXM_GROMACS_SOURCE_DIR "" CACHE PATH
    "GROMACS source tree used by NBNXM_MICROBENCH")
set(NBNXM_GROMACS_BUILD_DIR "" CACHE PATH
    "Matching configured GROMACS build tree used by NBNXM_MICROBENCH")
if(NOT IS_DIRECTORY "${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs")
  message(FATAL_ERROR
          "NBNXM_MICROBENCH requires -DNBNXM_GROMACS_SOURCE_DIR=<gromacs-source>")
endif()
if(NOT IS_DIRECTORY "${NBNXM_GROMACS_BUILD_DIR}/src/include")
  message(FATAL_ERROR
          "NBNXM_MICROBENCH requires -DNBNXM_GROMACS_BUILD_DIR=<gromacs-build>")
endif()

set(GROMACS_SOURCE_DIR ${NBNXM_GROMACS_SOURCE_DIR}/src)
set(GROMACS_SOURCE_INCLUDE_DIR ${NBNXM_GROMACS_SOURCE_DIR}/src/include)
set(GROMACS_BUILD_INCLUDE_DIR ${NBNXM_GROMACS_BUILD_DIR}/src/include)
set(GROMACS_LEGACY_INCLUDE_DIR ${NBNXM_GROMACS_SOURCE_DIR}/api/legacy/include)
set(GROMACS_BUILD_LEGACY_INCLUDE_DIR ${NBNXM_GROMACS_BUILD_DIR}/api/legacy/include)
set(GROMACS_TNG_INCLUDE_DIR ${NBNXM_GROMACS_SOURCE_DIR}/src/external/tng_io/include)
set(GROMACS_BUILD_TNG_INCLUDE_DIR ${NBNXM_GROMACS_BUILD_DIR}/tng/include)
set(GROMACS_MODULE_INCLUDE_DIRS
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/math/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/timing/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/utility/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/pbcutil/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/pulling/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/topology/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/serialization/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/linearalgebra/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/simd/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/gromacs/taskassignment/include)
set(GROMACS_EXTERNAL_INCLUDE_DIRS
    ${NBNXM_GROMACS_SOURCE_DIR}/src/external/thread_mpi/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/external
    ${NBNXM_GROMACS_SOURCE_DIR}/src/external/rpc_xdr
    ${NBNXM_GROMACS_SOURCE_DIR}/src/external/muparser/include
    ${NBNXM_GROMACS_SOURCE_DIR}/src/external/lmfit
    ${NBNXM_GROMACS_SOURCE_DIR}/src/external/colvars
    ${NBNXM_GROMACS_SOURCE_DIR}/src/external/plumed)

add_executable(${CURRENT_TARGET} ${SOURCES})
set_target_properties(${CURRENT_TARGET}
                      PROPERTIES CXX_STANDARD 20
                                 CXX_STANDARD_REQUIRED ON
                                 CUDA_STANDARD 20
                                 CUDA_STANDARD_REQUIRED ON
                                 HIP_STANDARD 20
                                 HIP_STANDARD_REQUIRED ON)
target_include_directories(${CURRENT_TARGET}
                           PRIVATE ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench
                                   ${PROJECT_ROOT_DIR}/SPONGE
                                   ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include
                                   ${GROMACS_SOURCE_DIR}
                                   ${GROMACS_SOURCE_INCLUDE_DIR}
                                   ${GROMACS_BUILD_INCLUDE_DIR}
                                   ${GROMACS_LEGACY_INCLUDE_DIR}
                                   ${GROMACS_BUILD_LEGACY_INCLUDE_DIR}
                                   ${GROMACS_TNG_INCLUDE_DIR}
                                   ${GROMACS_BUILD_TNG_INCLUDE_DIR}
                                   ${GROMACS_MODULE_INCLUDE_DIRS})
target_include_directories(${CURRENT_TARGET}
                           SYSTEM PRIVATE ${GROMACS_EXTERNAL_INCLUDE_DIRS})
target_compile_definitions(${CURRENT_TARGET}
                           PRIVATE HAVE_CONFIG_H
                                   GMX_DOUBLE=0
                                   USE_STD_INTTYPES_H)
target_compile_options(${CURRENT_TARGET}
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                               $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(${CURRENT_TARGET} PRIVATE common_libraries)
install(TARGETS ${CURRENT_TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
