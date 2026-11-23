cmake_minimum_required(VERSION 3.15...3.31)

project(confseq VERSION "0.0.10")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Boost: we only use header-only parts of Boost.Math. Use an installed Boost
# (>= 1.70) if one is available; otherwise (or if CONFSEQ_FETCH_BOOST is ON)
# download pinned Boost.Math + Boost.Config headers and use Boost.Math in
# standalone mode. This lets `pip install confseq` build from source without a
# system Boost, and makes the release wheels independent of the build image.
option(CONFSEQ_FETCH_BOOST
  "Download Boost.Math headers instead of using an installed Boost" OFF)
if(DEFINED ENV{CONFSEQ_FETCH_BOOST})
  set(CONFSEQ_FETCH_BOOST "$ENV{CONFSEQ_FETCH_BOOST}")
endif()

if(NOT CONFSEQ_FETCH_BOOST)
  find_package(Boost 1.70.0 CONFIG QUIET)
endif()

add_library(confseq_boost INTERFACE)
if(Boost_FOUND)
  message(STATUS "Using installed Boost ${Boost_VERSION} from ${Boost_DIR}")
  target_link_libraries(confseq_boost INTERFACE Boost::headers)
else()
  message(STATUS "Installed Boost not used; downloading Boost.Math headers")
  include(FetchContent)
  set(_confseq_boost_tag "boost-1.92.0")
  # SOURCE_SUBDIR points at a directory without a CMakeLists.txt so that
  # FetchContent_MakeAvailable only downloads the headers; we don't want
  # Boost's own CMake project (and its dependency tree).
  FetchContent_Declare(confseq_boost_math
    URL "https://github.com/boostorg/math/archive/refs/tags/${_confseq_boost_tag}.tar.gz"
    URL_HASH SHA256=aa84eec67402a51cb98172ece53b82f0500e95a80656de109074a71920cd2c48
    SOURCE_SUBDIR include)
  FetchContent_Declare(confseq_boost_config
    URL "https://github.com/boostorg/config/archive/refs/tags/${_confseq_boost_tag}.tar.gz"
    URL_HASH SHA256=b4171037f13373203ba79cbc141d612982052283e696a315185ab5bea46102a0
    SOURCE_SUBDIR include)
  FetchContent_MakeAvailable(confseq_boost_math confseq_boost_config)
  target_include_directories(confseq_boost SYSTEM INTERFACE
    "${confseq_boost_math_SOURCE_DIR}/include"
    "${confseq_boost_config_SOURCE_DIR}/include")
  # Boost.Config is only needed for <boost/cstdint.hpp>; Boost.Math itself
  # runs without the rest of Boost in standalone mode.
  target_compile_definitions(confseq_boost INTERFACE BOOST_MATH_STANDALONE)
endif()

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

target_link_libraries(boundaries PRIVATE confseq_boost)
target_link_libraries(quantiles PRIVATE confseq_boost)

target_compile_definitions(boundaries PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_definitions(quantiles PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS boundaries DESTINATION .)
install(TARGETS quantiles DESTINATION .)
