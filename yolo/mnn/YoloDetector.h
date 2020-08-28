
#ifndef YOLO_DETECTION_H
#define YOLO_DETECTION_H

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class MNN_YoloDetector
{

public:
    MNN_YoloDetector(const std::string &modelPath, int targetWidth = 608, int targetHeight = 608);
    ~MNN_YoloDetector() {}

public:
    inline void setNumThread(int num) { mNumThreads = num; }
    inline void setConfidenceThreshold(float threshold) { mConfidenceThreshold = threshold; }
    inline void setNmsThreshold(float threshold) { mNmsThreshold = threshold;}
    inline void setForwardType(MNNForwardType type) { mForwardType = type; }

    void detect(cv::Mat &img, std::vector<Object> &objces);

private:
    void bboxesNms(std::vector<Object> &objects, std::vector<int> &picked);
    float intersection_area(const Object &a, const Object &b);
    cv::Mat letterbox_resize(const cv::Mat &src);

    inline float sigmoid(float x)
    {
        return (1 / (1 + exp(-x)));
    }

private:
    int mTargetWidth;
    int mTargetHeight;

    float mConfidenceThreshold;
    float mNmsThreshold;
    float mDetectionThreshold;

    int mNumThreads;

    std::string mModelPath;
    MNNForwardType mForwardType;

    std::shared_ptr<MNN::Interpreter> mNet;
    MNN::Session* mSession;
};

#endif