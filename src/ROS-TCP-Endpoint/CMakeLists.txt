cmake_minimum_required(VERSION 2.8.3)
project(ros_tcp_endpoint)

find_package(catkin REQUIRED COMPONENTS
  rosgraph_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_msgs
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS message_runtime rosgraph_msgs rospy sensor_msgs std_msgs tf2_msgs)
