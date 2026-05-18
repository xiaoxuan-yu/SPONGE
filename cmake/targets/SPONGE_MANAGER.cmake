include(${PROJECT_ROOT_DIR}/cmake/targets/SPONGE_runtime_sources.cmake)

set(SOURCES
    ${PROJECT_ROOT_DIR}/SPONGE/manager/config.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manager/core/manager.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manager/algorithms/remd/hamiltonian_remd.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manager/algorithms/remd/temperature_remd.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manager/algorithms/remd/temperature_hamiltonian_remd.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/worker_protocol/child_process_worker.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manager/manager_main.cpp
    ${SPONGE_RUNTIME_SOURCES})
set(TARGET_SOURCE_LANGUAGE "${CPP_DIALECT}")
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
if(WIN32)
  target_link_libraries(${CURRENT_TARGET} PRIVATE ws2_32)
endif()
install(TARGETS ${CURRENT_TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
