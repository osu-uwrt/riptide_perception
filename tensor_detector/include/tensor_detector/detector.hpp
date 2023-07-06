#pragma once

#include <NvInfer.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <memory>

namespace tensor_detector
{

    class Logger : public nvinfer1::ILogger
    {
        void log(Severity sev, const char *msg) noexcept override;
    };

    /**
     * Contaier for detection results
     *
     * @param bounds, The image space bounds of the detection
     * @param class_id, The integer class id of the detection as the model was trained
     * @param conf, The confidence of the class of the detection
     */
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
        Logger logger;

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

        /**
         * Function assumes detections is empty
         */
        void nonMaximumSuppression(const cv::Mat &out_tensor, std::vector<std::vector<YoloDetect>> &classed_detections);

        size_t getSizeByDim(const nvinfer1::Dims &dims);

    public:
        /**
         * Constructor for the inference engine. Loads the model from the compiled binary engine and builds the
         * runtime needed for TRT to inference on the target. This should be constructed outside of the inference
         * runtime, and only loaded once per intended model.
         *
         * @param engine_file_path, a string locating the binary engine file. can be absolute or relative.
         */
        YoloInfer(const std::string &engine_file_path);

        /**
         * Ingest the gpu transferred image, preprocess it and copy it into the pipeline.
         * Call this function before the infer step with each new frame you intend to operate on
         *
         * @param gpu_frame, the input frame already uploaded to the GPU
         *
         */
        void loadNextImage(const cv::cuda::GpuMat &gpu_frame);

        /**
         * Take the loaded and pre-processed image and run it through the tensorrt system synchronously.
         * Should be called after an image has been loaded previously.
         */
        void inferLoadedImg();

        /**
         * Take the results copied from the inference buffer back to the CPU, and determine all detections.
         * Also applies IOU and NMS algorithms to clean up overlapped boxes
         *
         * @param detections, the vector to put the resulting YoloDetect detections in
         */
        void postProcessResults(std::vector<YoloDetect> &detections);

        ~YoloInfer();
    };

} // namespace tensor_detector
