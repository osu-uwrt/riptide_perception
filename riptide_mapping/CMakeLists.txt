cmake_minimum_required(VERSION 3.8)
project(riptide_mapping2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

#add c++ files
add_executable(dependentFramePublisher ${PROJECT_NAME}/dependentFramePublisher.cpp)

ament_target_dependencies(
    dependentFramePublisher
    rclcpp
    tf2
    tf2_ros
    geometry_msgs
    tf2_geometry_msgs
)

#install c++
install(
    TARGETS dependentFramePublisher
    DESTINATION lib/${PROJECT_NAME}
)

#install programs
install(
    PROGRAMS ${PROJECT_NAME}/mapping.py ${PROJECT_NAME}/calibrate.py ${PROJECT_NAME}/estimate.py
    DESTINATION lib/${PROJECT_NAME}
)

#install launch
install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
