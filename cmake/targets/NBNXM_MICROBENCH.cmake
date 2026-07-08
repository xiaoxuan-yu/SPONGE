if(NOT TARGET sponge_clustered_lj_count_experiments)
  add_library(
    sponge_clustered_lj_count_experiments OBJECT
    ${PROJECT_ROOT_DIR}/SPONGE/Lennard_Jones_force/clustered_lj_count_experiments.cpp)
  set_target_properties(
    sponge_clustered_lj_count_experiments
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               HIP_STANDARD 20
               HIP_STANDARD_REQUIRED ON)
  set_source_files_properties(
    ${PROJECT_ROOT_DIR}/SPONGE/Lennard_Jones_force/clustered_lj_count_experiments.cpp
    PROPERTIES LANGUAGE ${CPP_DIALECT}
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               HIP_STANDARD 20
               HIP_STANDARD_REQUIRED ON)
  target_include_directories(
    sponge_clustered_lj_count_experiments
    PRIVATE ${PROJECT_ROOT_DIR}/SPONGE
            ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include)
  target_link_libraries(sponge_clustered_lj_count_experiments
                        PUBLIC common_libraries)
endif()

set(SOURCES ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/nbnxm_microbench.cu
            ${PROJECT_ROOT_DIR}/tools/nbnxm_microbench/gromacs_forceonly_replay.cu
            $<TARGET_OBJECTS:sponge_clustered_lj_count_experiments>)
set(TARGET_LINKER_LANGUAGE "CXX")

set(GROMACS_PROJECT_DIR /home/youmans/sidereus/gromacs-2026.2)
set(GROMACS_SOURCE_DIR ${GROMACS_PROJECT_DIR}/src)
set(GROMACS_SOURCE_INCLUDE_DIR ${GROMACS_PROJECT_DIR}/src/include)
set(GROMACS_BUILD_INCLUDE_DIR ${GROMACS_PROJECT_DIR}/build-cuda/src/include)
set(GROMACS_LEGACY_INCLUDE_DIR ${GROMACS_PROJECT_DIR}/api/legacy/include)
set(GROMACS_BUILD_LEGACY_INCLUDE_DIR ${GROMACS_PROJECT_DIR}/build-cuda/api/legacy/include)
set(GROMACS_TNG_INCLUDE_DIR ${GROMACS_PROJECT_DIR}/src/external/tng_io/include)
set(GROMACS_BUILD_TNG_INCLUDE_DIR ${GROMACS_PROJECT_DIR}/build-cuda/tng/include)
set(GROMACS_MODULE_INCLUDE_DIRS
    ${GROMACS_PROJECT_DIR}/src/gromacs/math/include
    ${GROMACS_PROJECT_DIR}/src/gromacs/timing/include
    ${GROMACS_PROJECT_DIR}/src/gromacs/utility/include
    ${GROMACS_PROJECT_DIR}/src/gromacs/pbcutil/include
    ${GROMACS_PROJECT_DIR}/src/gromacs/pulling/include
    ${GROMACS_PROJECT_DIR}/src/gromacs/topology/include
    ${GROMACS_PROJECT_DIR}/src/gromacs/serialization/include
    ${GROMACS_PROJECT_DIR}/src/gromacs/linearalgebra/include
    ${GROMACS_PROJECT_DIR}/src/gromacs/simd/include
    ${GROMACS_PROJECT_DIR}/src/gromacs/taskassignment/include)
set(GROMACS_EXTERNAL_INCLUDE_DIRS
    ${GROMACS_PROJECT_DIR}/src/external/thread_mpi/include
    ${GROMACS_PROJECT_DIR}/src/external
    ${GROMACS_PROJECT_DIR}/src/external/rpc_xdr
    ${GROMACS_PROJECT_DIR}/src/external/muparser/include
    ${GROMACS_PROJECT_DIR}/src/external/lmfit
    ${GROMACS_PROJECT_DIR}/src/external/colvars
    ${GROMACS_PROJECT_DIR}/src/external/plumed)

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
