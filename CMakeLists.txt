cmake_minimum_required(VERSION 2.8.3)
project(laser_slam)

find_package(Eigen3 REQUIRED)

find_package(PCL 1.7 REQUIRED)
if(PCL_FOUND)
    #set(PCL_INCLUDE_DIRS "/usr/local/include/pcl-1.8")
    #set(PCL_LIBRARY_DIRS "/usr/local/lib")
    message(STATUS "Using PCL PATH:" "\n" 
    		${PCL_INCLUDE_DIRS} "\n" 
    		${PCL_LIBRARY_DIRS}"\n")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
endif(PCL_FOUND)


# Regular catkin package follows.
find_package(catkin REQUIRED COMPONENTS 
			roscpp 
			sensor_msgs
			cv_bridge
			pcl_conversions
            point_cloud_filter
            point_cloud_odometry
			pcl_ros
            ndt_omp
            geometry_msgs
)
catkin_package(CATKIN_DEPENDS roscpp)

find_package(OpenCV REQUIRED)
message(STATUS " version:${OpenCV_VERSION}")

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(slam_front_end_ros_node src/slam_front_end_ros.cpp)
target_link_libraries(slam_front_end_ros_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(laser_correction src/laser_correction.cpp)
target_link_libraries(laser_correction ${catkin_LIBRARIES} ${PCL_LIBRARIES})

set(CMAKE_CXX_FLAGS "-fpermissive -std=c++11")
