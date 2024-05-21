cmake_minimum_required(VERSION 3.8)
project(tensor_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclpy REQUIRED)
find_package(OpenCV 4 REQUIRED)

# setup CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/utils/cuda" )
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

	if(CUDA_VERSION_MAJOR GREATER 9)
		message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
	endif()

	if(CUDA_VERSION_MAJOR GREATER 10)
		message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
	endif()
endif()

# Install the Python node
install(PROGRAMS
  src/yolo_orientation.py 
  DESTINATION lib/${PROJECT_NAME}  
)

#install launch
install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

#install config
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

#install weights
install(
  DIRECTORY weights
  DESTINATION share/${PROJECT_NAME}
)

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

ament_package()
