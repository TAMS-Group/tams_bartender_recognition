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
  geometry_msgs
  sensor_msgs
  std_msgs
  tf_conversions
  visualization_msgs
)

add_message_files(
	FILES
	SegmentedBottle.msg
	SegmentedBottleArray.msg
	)

generate_messages(
	DEPENDENCIES
	std_msgs
	geometry_msgs
	sensor_msgs
	)

catkin_package(
	CATKIN_DEPENDS message_runtime
)



include_directories(
	${catkin_INCLUDE_DIRS}
)

add_executable(pcl_bottle_recognition src/bottle_recognition.cpp)
target_link_libraries(pcl_bottle_recognition ${catkin_LIBRARIES})
add_dependencies(pcl_bottle_recognition ${${PROJECT_NAME}_EXPORTED_TARGETS})
