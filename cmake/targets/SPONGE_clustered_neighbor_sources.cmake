include_guard(GLOBAL)

set(SPONGE_CLUSTERED_NEIGHBOR_SOURCES
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/contract/view.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/provider/provider.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/provider/config.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/builder.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/provider/endpoint_incidence.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/provider/pair_shift.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/provider/lifecycle.cpp)

if(PARALLEL_BACKEND STREQUAL "cuda" OR PARALLEL_BACKEND STREQUAL "hip")
  list(
    APPEND
    SPONGE_CLUSTERED_NEIGHBOR_SOURCES
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include/cstone/cuda/device_vector.cu
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include/cstone/primitives/primitives_gpu.cu
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include/cstone/tree/csarray_gpu.cu
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include/cstone/tree/octree_gpu.cu
  )
else()
  list(APPEND SPONGE_CLUSTERED_NEIGHBOR_SOURCES
       ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/cpu_builder.cpp
       ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/payload_builder.cpp)
endif()

function(sponge_add_clustered_neighbor_object target_name source_file)
  if(TARGET ${target_name})
    return()
  endif()

  add_library(${target_name} OBJECT ${source_file})
  set_target_properties(
    ${target_name}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               HIP_STANDARD 20
               HIP_STANDARD_REQUIRED ON)
  set_source_files_properties(
    ${source_file}
    PROPERTIES LANGUAGE ${CPP_DIALECT}
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               HIP_STANDARD 20
               HIP_STANDARD_REQUIRED ON)
  target_include_directories(
    ${target_name}
    PRIVATE ${PROJECT_ROOT_DIR}/SPONGE
            ${PROJECT_ROOT_DIR}/SPONGE/third_party/cornerstone_octree/include)
  target_link_libraries(${target_name} PUBLIC common_libraries)
  if(TARGET HighFive::HighFive)
    target_link_libraries(${target_name} PRIVATE HighFive::HighFive)
  elseif(TARGET HighFive)
    target_link_libraries(${target_name} PRIVATE HighFive)
  else()
    message(
      FATAL_ERROR "HighFive target was not provided by find_package(HighFive)")
  endif()
  if(TARGET HDF5::HDF5)
    target_link_libraries(${target_name} PRIVATE HDF5::HDF5)
  else()
    target_include_directories(${target_name} PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${HDF5_LIBRARIES})
  endif()
endfunction()

if(PARALLEL_BACKEND STREQUAL "cuda" OR PARALLEL_BACKEND STREQUAL "hip")
  sponge_add_clustered_neighbor_object(
    sponge_clustered_neighbor_payload_builder
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/payload_builder.cpp)
  list(APPEND SPONGE_CLUSTERED_NEIGHBOR_SOURCES
       $<TARGET_OBJECTS:sponge_clustered_neighbor_payload_builder>)
endif()

set(SPONGE_CLUSTERED_NEIGHBOR_OBJECT_SOURCES
    sponge_clustered_neighbor_record_stream_builder
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/record_stream_builder.cpp
    sponge_clustered_neighbor_spatial_builder
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/spatial_builder.cpp
    sponge_clustered_neighbor_active_refresh
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/active_refresh.cpp
    sponge_clustered_neighbor_candidate_stage
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/candidate_stage.cpp
    sponge_clustered_neighbor_candidate_builder
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/builder/candidate_builder.cpp)

while(SPONGE_CLUSTERED_NEIGHBOR_OBJECT_SOURCES)
  list(POP_FRONT SPONGE_CLUSTERED_NEIGHBOR_OBJECT_SOURCES target_name
       source_file)
  sponge_add_clustered_neighbor_object(${target_name} ${source_file})
  list(APPEND SPONGE_CLUSTERED_NEIGHBOR_SOURCES
       $<TARGET_OBJECTS:${target_name}>)
endwhile()

unset(SPONGE_CLUSTERED_NEIGHBOR_OBJECT_SOURCES)
