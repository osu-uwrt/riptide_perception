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

struct InferDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

class YoloInfer
{
private:
    // TRT runtime system data
    UniquePtr<nvinfer1::ICudaEngine> engine_ptr;
    UniquePtr<nvinfer1::IExecutionContext> context_ptr;

    // I/O bindings for the network
    std::vector<nvinfer1::Dims> input_dims;  // we expect only one input
    std::vector<void *> input_buffers;       // should only be one input buffer too
    std::vector<nvinfer1::Dims> output_dims; // and four outputs
    std::vector<void *> output_buffers;      // should be four output buffers too

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
        UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
        engine_ptr.reset(runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr));
        assert(engine_ptr.get() != nullptr);

        // load the execution context
        context_ptr.reset(engine_ptr->createExecutionContext());
        assert(context_ptr.get() != nullptr);

        // prepare I/O Bindings
        for (int i = 0; i < engine_ptr->getNbBindings(); ++i)
        {
            // reserve GPU memory for the input and mark it
            void** buffer_ptr;
            auto binding_size = getSizeByDim(engine_ptr->getBindingDimensions(i)) * sizeof(float);
            if(cudaMalloc(buffer_ptr, binding_size) != cudaSuccess)
                throw std::runtime_error("GPU malloc failed while reserving memeory buffers for NN I/O");

            // put the buffer ptr onto the right vector
            (engine_ptr->bindingIsInput(i) ? input_buffers : output_buffers).push_back(buffer_ptr);

            // load the binding into the corresponding list
            (engine_ptr->bindingIsInput(i) ? input_dims : output_dims).push_back(engine_ptr->getBindingDimensions(i));
        }

        // Verify we have at least 1 input and 1 output otherwise we have an issue
        if (input_dims.empty())
            throw std::runtime_error("Model did not contain any inputs when loaded");
        else if (output_dims.empty())
            throw std::runtime_error("Model did not contain any outputs when loaded");
    }

    void loadNextImage(const cv::Mat &input_image)
    {
        // Take the cv image and CUDA load it to GPU
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(input_image);

        // dims is ordered # channels, width, height
        // should only be one input at index 0
        auto final_size = cv::Size(input_dims.at(0).d[2], input_dims.at(0).d[1]);

        // resize the image to match the nn input tensor size
        cv::cuda::GpuMat resized;
        cv::cuda::resize(gpu_frame, resized, final_size, 0, 0, cv::INTER_NEAREST);

        // pre-normalize the input image
        cv::cuda::GpuMat flt_image;
        resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
        cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
        cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

        // prepare the memcpy on the gpu so it doesnt need to go back to the CPU
        cudaMemcpyAsync(input_buffers.at(0), flt_image.ptr<uint8_t>(), flt_image.channels() * flt_image.rows * flt_image.step, cudaMemcpyDeviceToDevice);
    }

    void inferLoadedImg(){

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

    return 0;
}
