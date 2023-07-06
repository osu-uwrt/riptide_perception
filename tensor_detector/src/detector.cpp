
#include "tensor_detector/detector.hpp"

#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>

#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>

namespace tensor_detector
{

    void Logger::log(Severity sev, const char *msg) noexcept
    {
        if (sev <= Severity::kINFO)
        {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }

    // comparison operator for sort
    bool operator<(const YoloDetect &a, const YoloDetect &b)
    {
        return a.conf < b.conf;
    }

    // comparison operator for sort
    bool operator>(const YoloDetect &a, const YoloDetect &b)
    {
        return a.conf > b.conf;
    }

    /**
     * Constructor for the inference engine. Loads the model from the compiled binary engine and builds the
     * runtime needed for TRT to inference on the target
     *
     * @param engine_file_path, a string locating the binary engine file. can be absolute or relative.
     */
    YoloInfer::YoloInfer(const std::string &engine_file_path)
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

    void YoloInfer::loadNextImage(const cv::cuda::GpuMat &gpu_frame)
    {
        // dims is ordered # channels, width, height
        // should only be one input at index 0
        auto final_size = cv::Size(input_dims->d[2], input_dims->d[1]);

        // resize the image to match the nn input tensor size
        cv::cuda::GpuMat resized;
        cv::cuda::resize(gpu_frame, resized, final_size, 0, 0, cv::INTER_NEAREST);

        // printf("Input image underwent resize\n");

        // pre-normalize the input image
        resized.convertTo(resized, CV_32FC3, 1.f / 255.f);
        cv::cuda::subtract(resized, cv::Scalar(0.485f, 0.456f, 0.406f), resized, cv::noArray(), -1);
        cv::cuda::divide(resized, cv::Scalar(0.229f, 0.224f, 0.225f), resized, 1, -1);

        // printf("Input image normalized\n");

        // right now i think this is actually correct, im wondering if its other pre-processing steps that are missing

        // sync form of the copy operation in a sneaky manner
        std::vector<cv::cuda::GpuMat> chw;

        // setup a list of gpu mats pointing into our input address space
        for (int i = 0; i < resized.channels(); ++i)
            chw.emplace_back(cv::cuda::GpuMat(final_size, CV_32FC1, input_buffer + i * resized.rows * resized.step));

        // copy the data from the normalized image to the input tensors
        cv::cuda::split(resized, chw);
    }

    void YoloInfer::inferLoadedImg()
    {
        if (!context_ptr->executeV2(ordered_buffers.data()))
        {
            for (auto ptr : ordered_buffers)
                cudaFree(ptr);
            ordered_buffers.clear();
            throw std::runtime_error("NN inference failed :(");
        }
    }

    /**
     * Function assumes detections is empty
     */
    void YoloInfer::nonMaximumSuppression(const cv::Mat &out_tensor, std::vector<std::vector<YoloDetect>> &classed_detections)
    {
        // create a vector for the hypotheses
        std::vector<std::vector<YoloDetect>> raw_detections;

        // create a variable used for finding the class
        int class_id[2];

        const int cols = out_tensor.cols;

        raw_detections.reserve(cols - 5);
        for (int i = 0; i < cols - 5; i++)
        {
            raw_detections.emplace_back(std::vector<YoloDetect>());

            // WARNING THIS IS AN ARBITRARY ASSUMPTION!!!!!!
            // to save on memory, I assume that at most 50% of the detections availaible will be a single class
            // raw_detections.at(i).reserve(out_tensor.rows * 0.5);
        }

        // work each hypothesis
        for (int row_idx = 0; row_idx < out_tensor.rows; row_idx++)
        {
            // order in the tensor is center_x, center_y, width, height, conf, class 1, class 2, ..., class n
            const float conf = out_tensor.at<float>(row_idx, 4);
            if (conf > min_conf)
            {

                // make sure that the conf never exceeds 1.0
                assert(conf <= 1.0f);

                // the type casting here is intentional to narrow back to an int
                const int center_x = out_tensor.at<float>(row_idx, 0);
                const int center_y = out_tensor.at<float>(row_idx, 1);
                const int height = out_tensor.at<float>(row_idx, 2);
                const int width = out_tensor.at<float>(row_idx, 3);

                // find the argmax of this detection
                // watch the pointer arith with minMaxIdx
                cv::Mat class_hyps = out_tensor(cv::Range(row_idx, row_idx + 1), cv::Range(5, cols));

                // should always be fixed size
                assert(class_hyps.rows == 1 && class_hyps.cols == cols - 5 && class_hyps.channels() == 1);

                cv::minMaxIdx(class_hyps, NULL, NULL, NULL, (int *)class_id);

                // TODO rescale the coords back to the og image

                // build the detection
                YoloDetect detection = {
                    cv::Rect(center_x - width / 2, center_y + height / 2, width, height),
                    class_id[1],
                    conf};

                // printf("0: %i, 1: %i\n", class_id[0], class_id[1]);

                raw_detections.at(class_id[1]).emplace_back(detection);
            }
        }

        classed_detections.reserve(cols - 5);
        for (int i = 0; i < cols - 5; i++)
        {
            classed_detections.emplace_back(std::vector<YoloDetect>());
        }

        // run IOU top down on the raw detections and preserve those that dont overlap by a certain threshold
        for (int class_idx = 0; class_idx < raw_detections.size(); class_idx++)
        {

            auto iou_class_vector = raw_detections.at(class_idx);

            if (iou_class_vector.size() > 0)
            {
                // sort the raw detections vector by confidence descending
                std::sort(iou_class_vector.begin(), iou_class_vector.end(), std::greater<YoloDetect>());

                // automatically take the first detection
                classed_detections.at(class_idx).emplace_back(iou_class_vector.at(0));

                for (int det_idx = 1; det_idx < iou_class_vector.size(); det_idx++)
                {
                    bool low_overlap = true;

                    for (auto const passed_detections : classed_detections.at(class_idx))
                    {
                        cv::Rect intersection = iou_class_vector.at(det_idx).bounds & iou_class_vector.at(det_idx).bounds;
                        cv::Rect union_rect = iou_class_vector.at(det_idx).bounds | iou_class_vector.at(det_idx).bounds;

                        // want to keep low options and get rid of high IOU
                        if (intersection.area() / union_rect.area() > iou_thresh)
                        {
                            low_overlap &= false;
                        }
                    }

                    if (low_overlap)
                    {
                        classed_detections.at(class_idx).emplace_back(iou_class_vector.at(det_idx));
                    }
                }
            }
        }
    }

    void YoloInfer::postProcessResults(std::vector<YoloDetect> &detections)
    {

        auto out_size = cv::Size(output_dims->d[2], output_dims->d[1]);

        // create a gpu mat pointing to the output buffer
        cv::cuda::GpuMat gpu_tensor = cv::cuda::GpuMat(out_size, CV_32FC1, output_buffer);

        // download the gpu mat back to cpu
        cv::Mat cpu_tensor;
        gpu_tensor.download(cpu_tensor);

        std::vector<std::vector<YoloDetect>> raw_detections;

        // run NMS to get the true bboxes
        nonMaximumSuppression(cpu_tensor, raw_detections);

        // need to empty the results
        detections.clear();

        // flatten detections with low copy
        for (auto const &class_vector : raw_detections)
        {
            detections.insert(detections.end(), class_vector.begin(), class_vector.end());
        }
    }

    size_t YoloInfer::getSizeByDim(const nvinfer1::Dims &dims)
    {
        int size = 1;
        for (int i = 0; i < dims.nbDims; ++i)
        {
            size *= dims.d[i];
        }
        return size;
    }

    YoloInfer::~YoloInfer()
    {
        for (auto ptr : ordered_buffers)
            cudaFree(ptr);
        ordered_buffers.clear();
    }

} // namespace tensor_detector

int main(int argc, char **argv)
{
    // model info
    std::string input_file = "durr_3.png";
    std::string engine_file = "yolo.engine";

    // Load the inference engine and context
    tensor_detector::YoloInfer infer = tensor_detector::YoloInfer(engine_file);

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
        std::vector<tensor_detector::YoloDetect> detections;
        infer.postProcessResults(detections);

        auto diff = std::chrono::steady_clock::now() - init_time;
        printf("Input image inferred in %li us\n", std::chrono::duration_cast<std::chrono::microseconds>(diff).count());
        printf("Input image has %li detections\n", detections.size());

        for (int i = 0; i < detections.size(); i++)
        {
            printf("\t det %i -> class %i, conf %f, xy (%i, %i), wh(%i, %i)\n", i, detections.at(i).class_id, detections.at(i).conf, detections.at(i).bounds.x, detections.at(i).bounds.y, detections.at(i).bounds.width, detections.at(i).bounds.height);
        }
    }

    return 0;
}

/*

auto init_time = std::chrono::steady_clock::now();

auto diff = std::chrono::steady_clock::now() - init_time;
printf("Filter processed in %li us\n", std::chrono::duration_cast<std::chrono::microseconds>(diff).count());

*/
