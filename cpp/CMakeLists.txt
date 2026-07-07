cmake_minimum_required(VERSION 3.18)
project(monocon_metis VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "──────────────────────────────────────────")
message(STATUS "  MonoCon-Metis ${PROJECT_VERSION}")
message(STATUS "──────────────────────────────────────────")

# ──────────────────────────────────────────────
# Voyager SDK root
# Priority: -DVOYAGER_SDK_ROOT > env var VOYAGER_SDK_ROOT
# ──────────────────────────────────────────────
if (NOT VOYAGER_SDK_ROOT)
    if (DEFINED ENV{VOYAGER_SDK_ROOT})
        set(VOYAGER_SDK_ROOT $ENV{VOYAGER_SDK_ROOT})
    endif ()
endif ()
if (VOYAGER_SDK_ROOT)
    message(STATUS "  Voyager SDK : ${VOYAGER_SDK_ROOT}")
else ()
    message(WARNING "VOYAGER_SDK_ROOT not set — ORT may not be found.\n"
            "Pass -DVOYAGER_SDK_ROOT=/path/to/voyager-sdk or export it.")
endif ()

# ──────────────────────────────────────────────
# Detect architecture → pick correct ORT subdir
# ──────────────────────────────────────────────
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ORT_ARCH_DIR "onnxruntime-linux-aarch64-1.17.1")
else ()
    set(ORT_ARCH_DIR "onnxruntime-linux-x64-1.17.1")
endif ()
set(ORT_SDK_ROOT "${VOYAGER_SDK_ROOT}/operators/onnxruntime/${ORT_ARCH_DIR}")
message(STATUS "  ORT subdir  : ${ORT_SDK_ROOT}")

# ──────────────────────────────────────────────
# AxRuntime root — default /opt/axelera
# Override with -DAXRUNTIME_ROOT=...
# ──────────────────────────────────────────────
if (NOT AXRUNTIME_ROOT)
    set(AXRUNTIME_ROOT "/opt/axelera")
endif ()
message(STATUS "  AxRuntime   : ${AXRUNTIME_ROOT}")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
# ──────────────────────────────────────────────
# ONNXRuntime (from Voyager SDK bundle)
# ──────────────────────────────────────────────
find_library(ORT_LIB
        NAMES onnxruntime
        HINTS ${ORT_SDK_ROOT}/lib
        /usr/lib
        /usr/local/lib)
find_path(ORT_INCLUDE
        NAMES onnxruntime_cxx_api.h
        HINTS ${ORT_SDK_ROOT}/include
        /usr/include/onnxruntime
        /usr/local/include/onnxruntime)

if (NOT ORT_LIB OR NOT ORT_INCLUDE)
    message(FATAL_ERROR
            "ONNXRuntime not found.\n"
            "Expected under: ${ORT_SDK_ROOT}\n"
            "Set -DVOYAGER_SDK_ROOT= correctly.")
endif ()
message(STATUS "  ORT lib     : ${ORT_LIB}")
message(STATUS "  ORT include : ${ORT_INCLUDE}")

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}")

# ──────────────────────────────────────────────
# AxRuntime
# ──────────────────────────────────────────────
file(GLOB AXRUNTIME_VERSION_DIRS "${AXRUNTIME_ROOT}/runtime-*/include")
find_path(AXR_INCLUDE
        NAMES axruntime/axruntime.hpp
        HINTS ${AXRUNTIME_VERSION_DIRS}
        ${AXRUNTIME_ROOT}/include)

find_library(AXR_LIB
        NAMES axruntime
        HINTS ${AXRUNTIME_VERSION_DIRS}/../lib
        ${AXRUNTIME_ROOT}/lib)

if (NOT AXR_LIB OR NOT AXR_INCLUDE)
    message(FATAL_ERROR
            "AxRuntime not found.\n"
            "Expected under: ${AXRUNTIME_ROOT}\n"
            "Override with -DAXRUNTIME_ROOT=...")
endif ()
message(STATUS "  AXR lib     : ${AXR_LIB}")
message(STATUS "  AXR include : ${AXR_INCLUDE}")

add_library(axruntime::axruntime UNKNOWN IMPORTED)
set_target_properties(axruntime::axruntime PROPERTIES
        IMPORTED_LOCATION "${AXR_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${AXR_INCLUDE}")

# ──────────────────────────────────────────────
# monocon_cpp static library
# ──────────────────────────────────────────────
set(MONOCON_SOURCES
        # preprocess
        src/preprocess/opencv_preprocess.cpp
        src/preprocess/quantize.cpp

        # inference
        src/inference/aipu_inference.cpp
        src/inference/onnx_inference.cpp
        src/inference/backbone_neck.cpp
        src/inference/head.cpp

        # postprocess
        src/postprocess/dequantize.cpp
        src/postprocess/decode.cpp

        # utils
        src/utils/visualize.cpp
        src/utils/geometry_ops.cpp
        src/utils/calib.cpp

        # top-level orchestrator
        src/monocon.cpp
)

add_library(monocon_cpp STATIC ${MONOCON_SOURCES})

target_include_directories(monocon_cpp
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(monocon_cpp
        PUBLIC
        ${OpenCV_LIBS}
        onnxruntime::onnxruntime
        axruntime::axruntime
)
target_link_libraries(monocon_cpp PUBLIC OpenMP::OpenMP_CXX)
# ──────────────────────────────────────────────
# Executable
# ──────────────────────────────────────────────
add_executable(monocon_infer main.cpp)
target_link_libraries(monocon_infer PRIVATE monocon_cpp)