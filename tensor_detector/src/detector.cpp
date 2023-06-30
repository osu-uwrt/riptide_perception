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

struct YoloDetect
{
    cv::Rect bounds;
    int class_id;
    float conf;
};

// comparison operator for sort
bool operator<(const YoloDetect & a, const YoloDetect & b){
    return a.conf < b.conf;
}

// comparison operator for sort
bool operator>(const YoloDetect & a, const YoloDetect & b){
    return a.conf > b.conf;
}

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
    float iou_min_thresh = 0.4;
    float min_conf = 0.3;

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
            {
                for (auto ptr : ordered_buffers)
                    cudaFree(ptr);
                ordered_buffers.clear();
                throw std::runtime_error(std::string("GPU malloc failed while reserving memeory buffers for NN I/O ") +
                                         cudaGetErrorString(cuda_err));
            }

            // printf("%s nominal binding size %li\n", engine_ptr->getBindingName(i), binding_size);

            // check to see if we have an input buffer and update the corresponding tensor info
            (engine_ptr->bindingIsInput(i) ? input_buffer : output_buffer) = buffer_ptr;
            (engine_ptr->bindingIsInput(i) ? input_dims : output_dims) = std::make_unique<nvinfer1::Dims>(engine_ptr->getBindingDimensions(i));

            ordered_buffers.push_back(buffer_ptr);
        }

        // Verify we have at least 1 input and 1 output otherwise we have an issue
        if (input_dims == nullptr)
        {
            for (auto ptr : ordered_buffers)
                cudaFree(ptr);
            ordered_buffers.clear();
            throw std::runtime_error("Model did not contain any inputs when loaded");
        }

        else if (output_dims == nullptr)
        {
            for (auto ptr : ordered_buffers)
                cudaFree(ptr);
            ordered_buffers.clear();
            throw std::runtime_error("Model did not contain any outputs when loaded");
        }

        // compute the number of classes and if multi-box must be enabled
        // this is a bit of pre-processing for the post processing step
        num_classes = output_dims->d[2] - 5;
        mutli_label = num_classes > 1;
    }

    void loadNextImage(const cv::cuda::GpuMat &gpu_frame)
    {
        // dims is ordered # channels, width, height
        // should only be one input at index 0
        auto final_size = cv::Size(input_dims->d[2], input_dims->d[1]);

        // resize the image to match the nn input tensor size
        cv::cuda::GpuMat resized;
        // takes about 67 ms
        cv::cuda::resize(gpu_frame, resized, final_size, 0, 0, cv::INTER_NEAREST);

        // printf("Input image underwent resize\n");

        // pre-normalize the input image
        resized.convertTo(resized, CV_32FC3, 1.f / 255.f);

        // expensive but idk why -- about 231 ms right now
        cv::cuda::subtract(resized, cv::Scalar(0.485f, 0.456f, 0.406f), resized, cv::noArray(), -1);

        // takes about 3 ms
        cv::cuda::divide(resized, cv::Scalar(0.229f, 0.224f, 0.225f), resized, 1, -1);

        // printf("Input image normalized\n");

        // sync form of the copy operation in a sneaky manner
        std::vector<cv::cuda::GpuMat> chw;

        // setup a list of gpu mats pointing into our input address space
        for (int i = 0; i < resized.channels(); ++i)
        {
            chw.emplace_back(cv::cuda::GpuMat(final_size, CV_32FC1, input_buffer + i * resized.rows * resized.step));
        }

        // copy the data from the normalized image to the input tensors
        cv::cuda::split(resized, chw);
    }

    void inferLoadedImg()
    {
        if (!context_ptr->executeV2(ordered_buffers.data()))
        {
            for (auto ptr : ordered_buffers)
                cudaFree(ptr);
            ordered_buffers.clear();
            throw std::runtime_error("NN inference failed :(");
        }
    }

    void nonMaximumSuppression(const cv::Mat &out_tensor, std::vector<YoloDetect> &detections)
    {
        // create a vector for the hypotheses
        std::vector<YoloDetect> raw_detections;
        raw_detections.reserve(out_tensor.rows);

        // create a variable used for finding the class
        int class_id;
        bool found = false;

        // work each hypothesis
        auto cols = out_tensor.cols;
        for (int row = 0; row < out_tensor.rows; row++)
        {
            // order in the tensor is center_x, center_y, width, height, conf, class 1, class 2, ..., class n
            // the type casting here is intentional to narrow back to an int
            const int center_x = out_tensor.at<float>(row, 0);
            const int center_y = out_tensor.at<float>(row, 1);
            const int height = out_tensor.at<float>(row, 2);
            const int width = out_tensor.at<float>(row, 3);
            const float conf = out_tensor.at<float>(row, 4);
            if (conf > min_conf)
            {
                assert(conf <= 1.0f);

                // find the argmax of this detection
                cv::Mat class_hyp = out_tensor(cv::Range(row, row + 1), cv::Range(5, cols));
                cv::minMaxIdx(class_hyp, NULL, NULL, NULL, &class_id);

                // rescale the coords back to the og image


                // build the detection
                YoloDetect detection = {
                    cv::Rect(),
                    class_id,
                    conf
                };
                
                raw_detections.emplace_back(detection);

                if (!found)
                {

                    printf("hyp mat size -> rows: %i, cols: %i\ndetection -> cx: %i, cy: %i h: %i, w: %i\n", 
                        class_hyp.rows, class_hyp.cols, center_x, center_y, height, width);

                    found = true;
                }
            }
        }

        // sort the raw detections vector by confidence
        std::sort(raw_detections.begin(), raw_detections.end(), std::greater<YoloDetect>());


        // clear bboxes to prep for injection
        detections.clear();

        // for (int c = 0; c < ObjectClass::NUM_CLASS; ++c)
        // {

        //     std::sort(bboxes[c].begin(), bboxes[c].end(), BoundingBox::sortComparisonFunction);
        //     const size_t bboxes_size = bboxes[c].size();
        //     size_t valid_count = 0;

        //     for (size_t i = 0; i < bboxes_size && valid_count < MAX_OUTPUT_BBOX_COUNT; ++i)
        //     {
        //         if (!bboxes[c][i].valid_)
        //         {
        //             continue;
        //         }

        //         for (size_t j = i + 1; j < bboxes_size; ++j)
        //         {
        //             bboxes[c][i].compareWith(bboxes[c][j], NMS_THRESH);
        //         }

        //         ++valid_count;
        //     }
        // }
    }

    void postProcessResults(std::vector<YoloDetect> &detections)
    {

        auto out_size = cv::Size(output_dims->d[2], output_dims->d[1]);

        // create a gpu mat pointing to the output buffer
        cv::cuda::GpuMat gpu_tensor = cv::cuda::GpuMat(out_size, CV_32FC1, output_buffer);

        // download the gpu mat back to cpu
        cv::Mat cpu_tensor;
        gpu_tensor.download(cpu_tensor);

        // run NMS to get the true bboxes
        nonMaximumSuppression(cpu_tensor, detections);
    }

    ~YoloInfer()
    {
        for (auto ptr : ordered_buffers)
            cudaFree(ptr);
        ordered_buffers.clear();
    }
};

int main(int argc, char **argv)
{
    // model info
    std::string input_file = "durr.png";
    std::string engine_file = "yolo.engine";

    // Load the inference engine and context
    YoloInfer infer = YoloInfer(engine_file);

    printf("Model loading complete, preparing to infer\n");

    cv::Mat frame = cv::imread(input_file);

    // pre-load and modify the image to fit in the input tensor
    for (int i = 0; i < 50; i++)
    {
        auto init_time = std::chrono::steady_clock::now();

        // Take the cv image and CUDA load it to GPU
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);

        infer.loadNextImage(gpu_frame);

        // run the inference cycle on the new input
        infer.inferLoadedImg();

        // get the results
        std::vector<YoloDetect> detections;
        infer.postProcessResults(detections);

        auto diff = std::chrono::steady_clock::now() - init_time;
        printf("Input image inferred in %li us\n", std::chrono::duration_cast<std::chrono::microseconds>(diff).count());
    }

    return 0;
}

/*

auto init_time = std::chrono::steady_clock::now();

auto diff = std::chrono::steady_clock::now() - init_time;
printf("Filter processed in %li us\n", std::chrono::duration_cast<std::chrono::microseconds>(diff).count());

*/
