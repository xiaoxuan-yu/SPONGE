include(${PROJECT_ROOT_DIR}/cmake/targets/SPONGE_runtime_sources.cmake)

set(SOURCES
    ${PROJECT_ROOT_DIR}/SPONGE/cli_main.cpp
    ${SPONGE_RUNTIME_SOURCES}
    $<TARGET_OBJECTS:sponge_clustered_lj_count_experiments>)
set(TARGET_LINKER_LANGUAGE "${CPP_DIALECT}")

if(NOT TARGET sponge_toml)
  find_package(tomlplusplus CONFIG REQUIRED)
  add_library(
    sponge_toml STATIC
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml_decode.cpp)
  set_source_files_properties(
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml.cpp PROPERTIES LANGUAGE
                                                                    CXX)
  set_source_files_properties(
    ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml_decode.cpp
    PROPERTIES LANGUAGE CXX)
  target_include_directories(sponge_toml PUBLIC ${PROJECT_ROOT_DIR}/SPONGE)
  target_link_libraries(sponge_toml PUBLIC tomlplusplus::tomlplusplus)
endif()

add_executable(${CURRENT_TARGET} ${SOURCES})
target_link_libraries(${CURRENT_TARGET} PRIVATE sponge_toml)
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

set(SPONGE_CSTONE_STANDARD_SOURCES
    ${PROJECT_ROOT_DIR}/SPONGE/Lennard_Jones_force/clustered_lj.cpp
    ${SPONGE_CSTONE_SOURCES})
if(SPONGE_CSTONE_STANDARD_SOURCES)
  set_source_files_properties(
    ${SPONGE_CSTONE_STANDARD_SOURCES}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               HIP_STANDARD 20
               HIP_STANDARD_REQUIRED ON)
endif()
if(WIN32)
  target_link_libraries(${CURRENT_TARGET} PRIVATE ws2_32)
endif()
install(TARGETS ${CURRENT_TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
