#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <iostream>

using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity sev, const char * msg) noexcept override{
        if(sev <= Severity::kINFO){
            std::cout << msg << std::endl;
        }
    }
} logger;

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    printf("hello world tensor_detector package\n");

    IBuilder * builder = createInferBuilder(logger);

    return 0;
}
