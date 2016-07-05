cmake_minimum_required(VERSION 2.8.3)
project(hello_real_sense)

## Find catkin and any catkin packages
# find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_generation
  message_runtime
  messages
  # roscpp
  rospy
  std_msgs
)

## Declare ROS messages and services

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package()

## Build talker and listener
# include_directories(include ${catkin_INCLUDE_DIRS})

# catkin_package(
#   INCLUDE_DIRS include
#   CATKIN_DEPENDS message_generation message_runtime messages roscpp rospy std_msgs image_transport cv_bridge
# )

install(PROGRAMS
  scripts/video_processing_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# %EndTag(FULLTEXT)%