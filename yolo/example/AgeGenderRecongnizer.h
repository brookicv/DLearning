#ifndef AGE_GENDER_RECONGNITION_H_
#define AGE_GENDER_RECONGNITION_H_

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class A
{
    int a;
};

class AgeGenderRecongnizer
{

public:
    AgeGenderRecongnizer(const std::string &modelPath, int targetWidth = 192, int targetHeight = 256);
    ~AgeGenderRecongnizer() {}

public:
    void recongnize(cv::Mat &img, int &age, int &gender);

public:
    inline void setNumThread(int num) { mNumThreads = num; }
    inline void setForwardType(MNNForwardType type) { mForwardType = type; }

private:
    inline float sigmoid(float x)
    {
        return (1 / (1 + exp(-x)));
    }
    cv::Mat letterbox_resize(const cv::Mat &src);

private:
    int mTargetWidth;
    int mTargetHeight;

    int mNumThreads;

    std::string mModelPath;
    MNNForwardType mForwardType;

    std::shared_ptr<MNN::Interpreter> mNet;
    MNN::Session *mSession;

    std::shared_ptr<MNN::CV::ImageProcess> mPreProcess;
};

#endif 