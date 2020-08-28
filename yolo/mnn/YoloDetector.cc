
#include "YoloDetector.h"

using namespace std;

MNN_YoloDetector::MNN_YoloDetector(const std::string &modelPath, int targetWidth, int targetHeight) : 
    mModelPath(modelPath), mTargetWidth(targetWidth), mTargetHeight(targetHeight), mConfidenceThreshold(0.2),
    mNmsThreshold(0.4), mDetectionThreshold(0.5), mForwardType(MNN_FORWARD_CPU), mNumThreads(4)
{
    mNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mModelPath.c_str()));
    MNN::ScheduleConfig netConfig;
    netConfig.type = mForwardType;
    netConfig.numThread = mNumThreads;

    mSession = mNet->createSession(netConfig);
}

void MNN_YoloDetector::detect(cv::Mat &img, std::vector<Object> &objects)
{
    int originalWidth = img.cols;
    int originalHeight = img.rows;

    auto letter_image = letterbox_resize(img);
    cv::Mat targetImage;
    cv::cvtColor(letter_image, targetImage, cv::COLOR_BGR2RGB);
    targetImage.convertTo(targetImage, CV_32FC3);
    targetImage = targetImage / 255.0;

    // copy data to mnn tensor from opencv mat
    vector<int> dim{1, mTargetHeight, mTargetWidth, 3};
    auto nhwc_tensor = MNN::Tensor::create<float>(dim, nullptr, MNN::Tensor::TENSORFLOW);
    auto nhwc_data = nhwc_tensor->host<float>();
    auto nhwc_size = nhwc_tensor->size();
    ::memcpy(nhwc_data, targetImage.data, nhwc_size);

    auto input = mNet->getSessionInput(mSession, nullptr);
    input->copyFromHostTensor(nhwc_tensor);

    // run session forward
    mNet->runSession(mSession);

    // get bboxes output
    auto output = mNet->getSessionOutput(mSession, "bboxes");
    auto size = output->elementSize();
    vector<float> boxes(size);
    auto tmpOutput = output->host<float>();
    for (int i = 0; i < size; i++)
    {
        boxes[i] = tmpOutput[i];
    }

    // get classes output
    auto classes = mNet->getSessionOutput(mSession, "classes");
    auto probs = classes->host<float>();
    size = classes->elementSize();
    vector<float> tempValues(size);
    for (int i = 0; i < size; i++)
    {
        tempValues[i] = probs[i];
    }

    // parse the output from yolo
    vector<Object> allObjects;
    auto scale = std::min(static_cast<float>(mTargetWidth) / static_cast<float>(originalWidth),
                          static_cast<float>(mTargetHeight) / static_cast<float>(originalHeight));
    
    for (int i = 0; i < 22743; i++)
    {
        auto maxCls = std::max_element(tempValues.begin() + i * 80, tempValues.begin() + (i + 1) * 80);
        auto clsIdx = maxCls - (tempValues.begin() + i * 80);

        auto confidence = boxes[i * 5 + 4];
        auto score = confidence * (*maxCls);

        if (score < 0.2)
            continue;

        if (clsIdx != 0)
            continue; // person id = 0

        auto width = boxes[i * 5 + 2];
        auto height = boxes[i * 5 + 3];
        auto xmin = boxes[i * 5 + 0] - width / 2;
        auto ymin = boxes[i * 5 + 1] - height / 2;

        xmin -= (mTargetWidth - scale * originalWidth) / 2;
        ymin -= (mTargetHeight - scale * originalHeight) / 2;

        xmin /= scale;
        ymin /= scale;
        width /= scale;
        height /= scale;

        Object obj;
        obj.rect = cv::Rect_<float>(xmin, ymin, width, height);
        obj.label = clsIdx;
        obj.prob = score;
        allObjects.push_back(obj);
    }

    std::sort(allObjects.begin(), allObjects.end(), [](const Object &a, const Object &b) { return a.prob > b.prob; });

    vector<int> picked;
    bboxesNms(allObjects, picked);

    // return the finale result
    for(int i = 0; i < picked.size();i ++)
    {
        objects.push_back(allObjects[picked[i]]);
    }
}

void MNN_YoloDetector::bboxesNms(std::vector<Object> &objects, std::vector<int> &picked)
{
    picked.clear();

    const int n = objects.size();

    vector<float> areas(n);
    for (int i = 0; i < n; ++i)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; ++i)
    {
        const Object &a = objects[i];

        int keep = 1;
        for (int j = 0; j < picked.size(); j++)
        {
            const Object &b = objects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;

            if (inter_area / union_area > mNmsThreshold)
                keep = 0;
        }
        if (keep)
        {
            picked.push_back(i);
        }
    }
}
float MNN_YoloDetector::intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
cv::Mat MNN_YoloDetector::letterbox_resize(const cv::Mat &src)
{
    auto src_width = static_cast<float>(src.cols);
    auto src_height = static_cast<float>(src.rows);

    auto ewidth_f = static_cast<float>(mTargetWidth);
    auto eheight_f = static_cast<float>(mTargetHeight);

    auto scale = ewidth_f / src_width > eheight_f / src_height ? eheight_f / src_height : ewidth_f / src_width;

    auto width = static_cast<int>(scale * src_width);
    auto height = static_cast<int>(scale * src_height);

    cv::Mat dst;
    resize(src, dst, cv::Size(width, height));

    auto top = (mTargetHeight - height) / 2;
    auto bottom = mTargetHeight - top - height;
    auto left = (mTargetWidth - width) / 2;
    auto right = mTargetWidth - left - width;

    copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT);
    return dst;
}