#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>

#include <cstdio>
#include <iostream>
#include <fstream>

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger
{
    void log(Severity sev, const char * msg) noexcept override{
        if(sev <= Severity::kWARNING){
            std::cout << msg << std::endl;
        }
    }
} logger;

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    printf("hello world tensor_detector package\n");

    //define trt flags for network
    uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    IBuilder * builder = createInferBuilder(logger);
    INetworkDefinition * network = builder->createNetworkV2(flags); 
    IParser * parser = createParser(*network, logger);

    std::string modelFile = "/home/coalman321/colcon_deploy/src/riptide_yolo/weights/best.onnx";

    // load a model
    parser->parseFromFile(modelFile.c_str(), static_cast<uint32_t>(ILogger::Severity::kINFO));
    
    // check for errors
    for(size_t i = 0; i < parser->getNbErrors(); i++){
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // configure the model builder
    IBuilderConfig * buildercfg = builder->createBuilderConfig();
    buildercfg->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 25);

    // serialize the network
    IHostMemory * serializedModel = builder->buildSerializedNetwork(*network, *buildercfg);

    // can remove all of the configuration stuff
    delete parser; 
    delete network;
    delete buildercfg;
    delete builder;

    std::ofstream ofs("serialized_engine.trt", std::ios::out | std::ios::binary);
    ofs.write((char*)(serializedModel->data()), serializedModel->size());
    ofs.close();

    // this model should now be saved and be re-opened for further use
    // the serialization process can take a hot minute

    delete serializedModel;

    return 0;
}
