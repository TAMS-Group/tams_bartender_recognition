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

add_executable(pcl_bottle_recognition src/bottle_recognition.cpp)
target_link_libraries(pcl_bottle_recognition ${catkin_LIBRARIES})
