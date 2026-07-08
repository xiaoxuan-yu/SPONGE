set(SPONGE_NEIGHBOR_API_PROBE_CSTONE_SOURCES)
if(PARALLEL_BACKEND STREQUAL "cuda" OR PARALLEL_BACKEND STREQUAL "hip")
  list(APPEND SPONGE_NEIGHBOR_API_PROBE_CSTONE_SOURCES
       ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include/cstone/cuda/device_vector.cu
       ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include/cstone/primitives/primitives_gpu.cu
       ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include/cstone/tree/csarray_gpu.cu
       ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include/cstone/tree/octree_gpu.cu)
endif()

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

set(SOURCES
    ${PROJECT_ROOT_DIR}/tools/neighbor_api_probe/neighbor_api_probe.cu
    $<TARGET_OBJECTS:sponge_clustered_lj_count_experiments>
    ${SPONGE_NEIGHBOR_API_PROBE_CSTONE_SOURCES})
set(TARGET_LINKER_LANGUAGE "CXX")

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
target_link_libraries(${CURRENT_TARGET} PRIVATE common_libraries)
target_compile_options(${CURRENT_TARGET}
                       PRIVATE
                         $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                         $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

if(SPONGE_NEIGHBOR_API_PROBE_CSTONE_SOURCES)
  set_source_files_properties(
    ${SPONGE_NEIGHBOR_API_PROBE_CSTONE_SOURCES}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               HIP_STANDARD 20
               HIP_STANDARD_REQUIRED ON)
endif()

install(TARGETS ${CURRENT_TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
