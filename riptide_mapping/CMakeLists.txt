cmake_minimum_required(VERSION 3.8)
project(riptide_mapping)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(GTSAM REQUIRED)

set(deps
 ament_cmake
 rclcpp
 vision_msgs
 geometry_msgs
 GTSAM
)

add_library(mapping SHARED
 src/mapping.cpp
 src/location.cpp
)

target_include_directories(mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)


ament_target_dependencies(mapping ${deps})

add_executable(mapping_node src/mapping_node.cpp)

target_link_libraries(mapping_node mapping)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY include/
  DESTINATION include/
)

install(TARGETS mapping_node
    DESTINATION lib/${PROJECT_NAME})

ament_package()
