cmake_minimum_required(VERSION 2.8.3)
project(orbbec_astra_ip)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  tf
	genmsg
	pcl_conversions
	pcl_ros
	roscpp
	rospy
	sensor_msgs
	std_msgs
	tf_conversions
  visualization_msgs
)

catkin_package(
)

include_directories(
	${catkin_INCLUDE_DIRS}
)

add_executable(extract_floor src/extract_floor.cpp)
target_link_libraries(extract_floor ${catkin_LIBRARIES})

add_executable(cylinder_segment src/cylinder_segment.cpp)
target_link_libraries(cylinder_segment ${catkin_LIBRARIES})

add_executable(handle_segments src/handle_cylinder_segments.cpp)
target_link_libraries(handle_segments ${catkin_LIBRARIES})
