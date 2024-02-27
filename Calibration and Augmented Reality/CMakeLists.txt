cmake_minimum_required(VERSION 3.1)
project(main.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(aruco REQUIRED)
include_directories(${aruco_INCLUDE_DIRS})
add_executable(main.cpp main.cpp operations.cpp operations.h)
add_executable(vir_reality.cpp vir_reality.cpp operations.cpp operations.h extensions.cpp extensions.h)
add_executable(Aruco.cpp Aruco.cpp operations.cpp operations.h extensions.cpp extensions.h)
target_link_libraries(main.cpp ${OpenCV_LIBS})
target_link_libraries(vir_reality.cpp ${OpenCV_LIBS})
target_link_libraries(Aruco.cpp ${OpenCV_LIBS} ${aruco_LIBS})




