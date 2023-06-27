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
    UniquePtr<nvinfer1::ICudaEngine> enginePtr;
    UniquePtr<nvinfer1::IExecutionContext> contextPtr;

    // I/O bindings for the network
    std::vector<nvinfer1::Dims> input_dims;  // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and four outputs
    std::vector<void *> buffers;             // buffers for input and output data

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
        enginePtr.reset(runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr));
        assert(enginePtr.get() != nullptr);

        // load the execution context
        contextPtr.reset(enginePtr->createExecutionContext());
        assert(contextPtr.get() != nullptr);

        // prepare I/O Bindings
        buffers.reserve(enginePtr->getNbBindings());
        for (int i = 0; i < enginePtr->getNbBindings(); ++i)
        {
            // reserve GPU memory for the input and mark it
            auto binding_size = getSizeByDim(enginePtr->getBindingDimensions(i)) * sizeof(float);
            cudaMalloc(&buffers[i], binding_size);

            // load the binding into the corresponding list
            (enginePtr->bindingIsInput(i) ? input_dims : output_dims).emplace_back(enginePtr->getBindingDimensions(i));
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

        // run a resize to get down to network input size
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
