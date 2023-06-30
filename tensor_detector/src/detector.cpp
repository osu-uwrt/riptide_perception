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

class Logger : public nvinfer1::ILogger
{
    void log(Severity sev, const char *msg) noexcept override
    {
        if (sev <= Severity::kINFO)
        {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} logger;

size_t getSizeByDim(const nvinfer1::Dims &dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

class YoloInfer
{
private:
    // TRT runtime system data
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context_ptr;

    // I/O bindings for the network
    std::unique_ptr<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims;    // and four outputs

    void *input_buffer = nullptr;        // should only be one input buffer too
    std::vector<void *> output_buffers;  // should be four output buffers too
    std::vector<void *> ordered_buffers; // all the buffers in order that they need to be passed in

public:
    /**
     * Constructor for the inference engine. Loads the model from the compiled binary engine and builds the
     * runtime needed for TRT to inference on the target
     *
     * @param engine_file_path, a string locating the binary engine file. can be absolute or relative.
     */
    YoloInfer(const std::string &engine_file_path)
    {
        std::ifstream engineFile(engine_file_path, std::ios::binary);

        // make sure we can open it
        if (engineFile.fail())
            throw std::runtime_error("Failed to find engine file: " + engine_file_path);

        // Read the binary file
        engineFile.seekg(0, std::ifstream::end);
        auto engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ifstream::beg);

        std::vector<char> engineData(engineSize);
        engineFile.read(engineData.data(), engineSize);

        // Try create the TRT Runtime and load the engine
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
        engine_ptr.reset(runtime->deserializeCudaEngine(engineData.data(), engineSize));
        assert(engine_ptr.get() != nullptr);

        // load the execution context
        context_ptr.reset(engine_ptr->createExecutionContext());
        assert(context_ptr.get() != nullptr);

        // prepare I/O Bindings
        for (int i = 0; i < engine_ptr->getNbBindings(); i++)
        {
            // reserve GPU memory for the input and mark it
            void *buffer_ptr = nullptr;
            auto binding_size = getSizeByDim(engine_ptr->getBindingDimensions(i)) * sizeof(float);
            auto cuda_err = cudaMalloc(&buffer_ptr, binding_size);
            if (cuda_err != cudaSuccess)
                throw std::runtime_error(std::string("GPU malloc failed while reserving memeory buffers for NN I/O ") + cudaGetErrorString(cuda_err));

            printf("%s nominal binding size %li\n", engine_ptr->getBindingName(i), binding_size);

            // check to see if we have an input buffer
            if (engine_ptr->bindingIsInput(i))
            {
                input_buffer = buffer_ptr;
                input_dims = std::make_unique<nvinfer1::Dims>(engine_ptr->getBindingDimensions(i));
            }
            else
            {
                output_buffers.push_back(buffer_ptr);
                output_dims.push_back(engine_ptr->getBindingDimensions(i));
            }

            ordered_buffers.push_back(buffer_ptr);
        }

        // Verify we have at least 1 input and 1 output otherwise we have an issue
        if (!input_dims)
            throw std::runtime_error("Model did not contain any inputs when loaded");
        else if (output_dims.empty())
            throw std::runtime_error("Model did not contain any outputs when loaded");
    }

    void loadNextImage(const cv::Mat &input_image)
    {
        // Take the cv image and CUDA load it to GPU
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(input_image);

        printf("Input frame copied to GPU memory\n");

        // dims is ordered # channels, width, height
        // should only be one input at index 0
        auto final_size = cv::Size(input_dims->d[2], input_dims->d[1]);

        // resize the image to match the nn input tensor size
        cv::cuda::GpuMat resized;
        cv::cuda::resize(gpu_frame, resized, final_size, 0, 0, cv::INTER_NEAREST);

        printf("Input image underwent resize\n");

        // pre-normalize the input image
        cv::cuda::GpuMat flt_image;
        resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
        cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
        cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

        printf("Input image normalized\n");

        // sync form of the copy operation in a sneaky manner
        std::vector<cv::cuda::GpuMat> chw;

        // setup a list of gpu mats pointing into our input address space
        for (size_t i = 0; i < flt_image.channels(); ++i)
        {
            chw.emplace_back(cv::cuda::GpuMat(final_size, CV_32FC1, input_buffer + i * flt_image.rows * flt_image.step));
        }

        // copy the data from the normalized image to the input tensors
        cv::cuda::split(flt_image, chw);

    }

    void inferLoadedImg()
    {
        if (!context_ptr->executeV2(ordered_buffers.data()))
        {
            throw std::runtime_error("NN inference failed :(");
        }
    }

    ~YoloInfer()
    {
    }
};

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    printf("hello world tensor_detector package\n");

    // model info
    std::string input_file = "durr.png";
    std::string engine_file = "yolo.engine";

    // Load the inference engine and context
    YoloInfer infer = YoloInfer(engine_file);

    printf("Model loading complete, preparing to infer\n");

    cv::Mat frame = cv::imread(input_file);

    printf("Input image opened\n");

    infer.loadNextImage(frame);

    printf("Input image loaded\n");

    infer.inferLoadedImg();

    printf("Input image inferred\n");

    return 0;
}
