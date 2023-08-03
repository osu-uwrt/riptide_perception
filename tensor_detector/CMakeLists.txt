cmake_minimum_required(VERSION 3.8)
project(tensor_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# get opencv
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

# make the library first
cuda_add_library(yolov5_tensorrt SHARED
  src/tensorrt_detector/yolov5_detection.cpp
  src/tensorrt_detector/yolov5_detector.cpp
  src/tensorrt_detector/yolov5_detector_internal.cpp
  src/tensorrt_detector/yolov5_common.cpp
  src/tensorrt_detector/yolov5_logging.cpp
)
target_include_directories(yolov5_tensorrt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(yolov5_tensorrt PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(yolov5_tensorrt nvinfer nvinfer_plugin)
ament_target_dependencies(yolov5_tensorrt OpenCV)


# now make our executable
add_executable(tensorrtWrapper src/tensorrtWrapper.cpp)
target_include_directories(tensorrtWrapper PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(tensorrtWrapper PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(tensorrtWrapper yolov5_tensorrt)
ament_target_dependencies(tensorrtWrapper OpenCV rclcpp geometry_msgs vision_msgs sensor_msgs cv_bridge)



install(TARGETS tensorrtWrapper yolov5_tensorrt
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

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
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
