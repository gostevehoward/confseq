cmake_minimum_required(VERSION 3.15...3.19)

project(confseq VERSION "0.0.9")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Boost 1.70.0)
include_directories(${Boost_INCLUDE_DIRS})

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path automatically,
  # so we need to add it _or_ the pybind11 specific directory here.
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(boundaries MODULE src/confseq/boundaries.cpp)
pybind11_add_module(quantiles MODULE src/confseq/quantiles.cpp)

target_compile_definitions(boundaries PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_definitions(quantiles PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS boundaries DESTINATION .)
install(TARGETS quantiles DESTINATION .)
