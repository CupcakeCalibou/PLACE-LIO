cmake_minimum_required(VERSION 2.8.3)
project(place_lio)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.2.0 REQUIRED)
find_package(PCL 1.10 REQUIRED)
find_package(TBB NO_MODULE)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include
)

add_message_files(
  FILES
  Pose6D.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_executable(data_converter src/dataConverter.cpp)
target_link_libraries(data_converter ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(data_converter PRIVATE ${PYTHON_INCLUDE_DIRS})

add_executable(laser_mapping src/laserMapping.cpp)
target_link_libraries(laser_mapping ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} TBB::tbb)
target_include_directories(laser_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})