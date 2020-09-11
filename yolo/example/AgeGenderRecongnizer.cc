#include "AgeGenderRecongnizer.h"

using namespace std;
using namespace MNN;

AgeGenderRecongnizer::AgeGenderRecongnizer(const std::string &modelPath, int targetWidth, int targetHeight) : 
    mModelPath(modelPath), mTargetWidth(targetWidth), mTargetHeight(targetHeight), mNumThreads(4), mForwardType(MNN_FORWARD_CPU)
{
    mNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mModelPath.c_str()));
    MNN::ScheduleConfig netConfig;
    netConfig.type = mForwardType;
    netConfig.numThread = mNumThreads;

    mSession = mNet->createSession(netConfig);

    float means[3] = {103.94f, 116.78f, 123.68f};
    float norms[3] = {0.017f, 0.017f, 0.017f};

    CV::ImageProcess::Config preProcessConfig;
    ::memcpy(preProcessConfig.mean, means, sizeof(means));
    ::memcpy(preProcessConfig.normal, norms, sizeof(norms));
    preProcessConfig.sourceFormat = CV::RGB;
    preProcessConfig.destFormat = CV::RGB;
    preProcessConfig.filterType = CV::BILINEAR;

   mPreProcess = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(preProcessConfig));
}

cv::Mat AgeGenderRecongnizer::letterbox_resize(const cv::Mat &src)
{
    auto src_width = static_cast<float>(src.cols);
    auto src_height = static_cast<float>(src.rows);

    auto ewidth_f = static_cast<float>(mTargetWidth);
    auto eheight_f = static_cast<float>(mTargetHeight);

    auto scale = ewidth_f / src_width > eheight_f / src_height ? eheight_f / src_height : ewidth_f / src_width;

    auto width = static_cast<int>(scale * src_width);
    auto height = static_cast<int>(scale * src_height);

    cv::Mat dst;
    cv::resize(src, dst, cv::Size(width, height));

    auto top = (mTargetHeight - height) / 2;
    auto bottom = mTargetHeight - top - height;
    auto left = (mTargetWidth - width) / 2;
    auto right = mTargetWidth - left - width;

    cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT);
    return dst;
}

void AgeGenderRecongnizer::recongnize(cv::Mat &img, int &age, int &gender)
{
    int originalWidth = img.cols;
    int originalHeight = img.rows;

    auto input = mNet->getSessionInput(mSession, nullptr);

    CV::Matrix trans;
    // Dst -> [0, 1]
    trans.postScale(1.0 / mTargetHeight, 1.0 / mTargetWidth);
    //[0, 1] -> Src
    trans.postScale(originalWidth, originalHeight);
    // mPreProcess->setMatrix(trans);

    mPreProcess->convert(reinterpret_cast<uint8_t *>(img.data), originalWidth, originalHeight, 0, input);
    input->printShape();

    // run interfence
    mNet->runSession(mSession);
    auto output = mNet->getSessionOutput(mSession, "output1");

    auto size = output->shape()[1];
    auto values = output->host<float>();

    vector<float> ageProbs;
    for (int i = 30; i < size-1; i++)
    {
        auto prob = sigmoid(values[i]);
        ageProbs.push_back(prob);
        cout << " " << prob;
    }

    auto maxCls = std::max_element(ageProbs.begin(), ageProbs.end());
    auto clsIdx = maxCls - ageProbs.begin();

    age = clsIdx; // age index

    auto male = sigmoid(values[size-1]);
    cout << " " << male << endl;

    gender = male >= 0.5 ? 1 : 0;
}