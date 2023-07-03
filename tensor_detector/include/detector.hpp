#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>
#include <chrono>

class Logger : public nvinfer1::ILogger;

size_t getSizeByDim(const nvinfer1::Dims &dims);

struct YoloDetect
{
    cv::Rect bounds;
    int class_id;
    float conf;
};

// comparison operator for sort
bool operator<(const YoloDetect &a, const YoloDetect &b);

// comparison operator for sort
bool operator>(const YoloDetect &a, const YoloDetect &b);

class YoloInfer
{
private:
    // TRT runtime system data
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context_ptr;

    // I/O bindings for the network
    std::unique_ptr<nvinfer1::Dims> input_dims;  // we expect only one input
    std::unique_ptr<nvinfer1::Dims> output_dims; // and one output

    void *input_buffer = nullptr;        // only one input tensor
    void *output_buffer = nullptr;       // one output tensor
    std::vector<void *> ordered_buffers; // all the buffers in order that they need to be passed in

    // post processor info
    int num_classes = 0;
    bool mutli_label = false;
    float iou_thresh = 0.4; // upper bound for iou
    float min_conf = 0.015;

public:
    /**
     * Constructor for the inference engine. Loads the model from the compiled binary engine and builds the
     * runtime needed for TRT to inference on the target
     *
     * @param engine_file_path, a string locating the binary engine file. can be absolute or relative.
     */
    YoloInfer(const std::string &engine_file_path);

    void loadNextImage(const cv::cuda::GpuMat &gpu_frame);

    void inferLoadedImg();

    /**
     * Function assumes detections is empty
     */
    void nonMaximumSuppression(const cv::Mat &out_tensor, std::vector<std::vector<YoloDetect>> &classed_detections);

    void postProcessResults(std::vector<YoloDetect> &detections);
};