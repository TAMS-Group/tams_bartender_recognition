cmake_minimum_required(VERSION 2.8.3)
project(pcl_object_recognition)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  tf
  apriltags2_ros
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
  tiago_bartender_msgs
  moveit_ros_planning_interface
)

add_message_files(
	FILES
	SegmentedObject.msg
	SegmentedObjectArray.msg
        RecognizedObject.msg
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

add_executable(detect_bottles_action src/detect_bottles_action.cpp)
target_link_libraries(detect_bottles_action ${catkin_LIBRARIES})
add_dependencies(detect_bottles_action ${${PROJECT_NAME}_EXPORTED_TARGETS})

add_executable(detect_glass_action src/detect_glass_action.cpp)
target_link_libraries(detect_glass_action ${catkin_LIBRARIES})
add_dependencies(detect_glass_action ${${PROJECT_NAME}_EXPORTED_TARGETS})
