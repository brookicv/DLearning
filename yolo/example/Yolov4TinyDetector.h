#ifndef YOLOV4_TINY_DETECTOR_H
#define YOLOV4_TINY_DETECTOR_H

#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <memory>

struct Yolov4Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class YoloTinyDetector
{
public:
    YoloTinyDetector(const std::string &paramPath,const std::string &binPath, int targetSize = 416, int numThread = 4);
    ~YoloTinyDetector(){}

    void detect(cv::Mat &img, std::vector<Yolov4Object> &objects);

private:
    cv::Mat letterbox_resize(const cv::Mat &src);

private:
    ncnn::Net mYolov4Net;
    int mTargetSize;
    int mNumThreads;
};

#endif // YOLOV4_TINY_DETECTOR_H