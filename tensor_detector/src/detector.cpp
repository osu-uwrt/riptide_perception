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

public:
    YoloInfer(const std::string &engine_file_path)
    {
        std::ifstream engineFile(engine_file_path, std::ios::binary);

        // make sure we can open it
        if (engineFile.fail())
        {
            return;
        }

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

    return 0;
}
