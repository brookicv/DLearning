cmake_minimum_required(VERSION 2.6)

project(ImageRetrievalService)

FIND_PACKAGE(OpenCV REQUIRED)

# set(OpenCV_DIR "/usr/local/opencv3.4/share/OpenCV")
# debug
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -pthread ")
#set(cmake_cxx_flags_release "$ENV{CXXFLAGS} -O3 -Wall") 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_VERBOSE_MAKEFILE on) # Output compile commandline, VERBOSE=1

set(CMAKE_CXX_COMPILE "/usr/local/bin/g++")

FIND_PACKAGE( OpenMP REQUIRED)


if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# include directory
include_directories(/usr/include)
include_directories(/home/liqiang/GIT/ncnn/build/install/include)
include_directories(/usr/local/include)
include_directories(/usr/opencv3.4.4/include)
include_directories(/usr/local/curl/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# lib directory
link_directories(/usr/opencv3.4.4/lib)
link_directories(/usr/lib)
link_directories(/home/liqiang/GIT/ncnn/build/install/lib)
link_directories(/usr/local/curl/lib)
link_directories(/usr/local/lib)


aux_source_directory(. DIR_SRCS)

add_executable(main main.cc Yolov4TinyDetector.h Yolov4TinyDetector.cc AgeGenderRecongnizer.h AgeGenderRecongnizer.cc)
add_executable(yolov4 yolov4.cc)
add_executable(yolov3-mnn yolov3-mnn.cc)

target_link_libraries(main ${OpenCV_LIBS}  -lpthread ncnn libMNN.so)
target_link_libraries(yolov4 ${OpenCV_LIBS}  -lpthread ncnn)
target_link_libraries(yolov3-mnn ${OpenCV_LIBS}  -lpthread libMNN.so)

message(STATUS "cmake_build_type = " ${CMAKE_BUILD_TYPE})
message(STATUS "cmake_cxx_flags_debug=" ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "project_dir = " ${PROJECT_SOURCE_DIR})
message(STATUS "opencv_dir=" ${OpenCV_INCLUDE_DIRS})