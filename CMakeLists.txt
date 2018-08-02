cmake_minimum_required(VERSION 2.8.3)
project(pcl_object_recognition)

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
	SegmentedObject.msg
	SegmentedObjectArray.msg
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

add_executable(object_segmentation src/object_segmentation.cpp)
target_link_libraries(object_segmentation ${catkin_LIBRARIES})
add_dependencies(object_segmentation ${${PROJECT_NAME}_EXPORTED_TARGETS})
