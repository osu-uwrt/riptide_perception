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

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger
{
    void log(Severity sev, const char *msg) noexcept override
    {
        if (sev <= Severity::kINFO)
        {
            std::cout << msg << std::endl;
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

void preprocessImage(const std::string &image_path, float *gpu_input, const nvinfer1::Dims &dims)
{
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }

    // upload image to GPU
    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(frame);

    // resize the image to match the nn input tensor size
    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

    // pre-normalize
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // send it into a tensor now
    std::vector<cv::cuda::GpuMat> chw;
    for (int i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}

void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims)
{
    // get class names
    std::vector<std::string> classes = {"class_a"};

    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims));
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val)
                   { return std::exp(val); });
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(dims));
    // generate sequence 0, 1, 2, 3, ..., 999
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2)
              { return cpu_output[i1] > cpu_output[i2]; });
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "n";
        ++i;
    }
}

// size_t loadEngine(std::string engine_file, void* serialized_engine){
//     std::ifstream file(engine_file.c_str(), std::ios::binary);
//     if (!file.good())
//     {
//         throw std::runtime_error("Failed to open file: " + engine_file);
//     }

//     // get the size of the file and seek back
//     size_t size = 0;
//     file.seekg(0, file.end);
//     size = file.tellg();
//     file.seekg(0, file.beg);

//     serialized_engine = new char[size];
//     file.read(serialized_engine, size);
//     file.close();

//     return size;
// }

size_t createEngine(std::string onnx_file, void* serialized_engine){
    //define trt flags for network
    uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    IBuilder * builder = createInferBuilder(logger);
    INetworkDefinition * network = builder->createNetworkV2(flags); 
    IParser * parser = createParser(*network, logger);

    // load a model
    parser->parseFromFile(onnx_file.c_str(), static_cast<uint32_t>(ILogger::Severity::kINFO));
    
    // check for errors
    for(size_t i = 0; i < parser->getNbErrors(); i++){
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    printf("onnx file loaded\n");

    // configure the model builder
    IBuilderConfig * builderConfig = builder->createBuilderConfig();

    // serialize the network
    IHostMemory * serializedModel = builder->buildSerializedNetwork(*network, *builderConfig);

    // memory leak w/o this :/
    delete builder;
    delete network;
    delete builderConfig;
    delete builder;

    serialized_engine = serializedModel->data();

    return serializedModel->size();  
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    printf("hello world tensor_detector package\n");

    // model info
    std::string input_file = "durr.png";
    // std::string engine_file = "yolo.engine";
    std::vector<std::string> input_blobs = {};
    std::vector<std::string> output_blobs = {};

    // open the file
    // char *engine_data;
    // size_t engine_size = loadEngine(engine_file, engine_data);

    std::string modelFile = "best.onnx";

    // open the file
    void *engine_data;
    size_t engine_size = createEngine(modelFile, engine_data);


    // make the inference runtime
    std::unique_ptr<IRuntime> runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));

    // load the model and context
    printf("Loading engine\n");
    std::unique_ptr<ICudaEngine> engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(engine_data, engine_size));
    std::unique_ptr<IExecutionContext> context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());

    // configure the context
    context->setDebugSync(true);

    // try to get the io dims
    printf("Binding I/O\n");
    std::vector<nvinfer1::Dims> input_dims;               // we expect only one input
    std::vector<nvinfer1::Dims> output_dims;              // and one output
    std::vector<void *> buffers(engine->getNbBindings()); // buffers for input and output data
    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for networkn";
        return -1;
    }

    // preprocess input data
    preprocessImage(input_file, (float *)buffers[0], input_dims[0]);
    // inference
    context->enqueue(1, buffers.data(), 0, nullptr);
    // post-process results
    postprocessResults((float *)buffers[1], output_dims[0]);

    for (void *buf : buffers)
    {
        cudaFree(buf);
    }

    return 0;
}
