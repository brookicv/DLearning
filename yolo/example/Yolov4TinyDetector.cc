
#include "Yolov4TinyDetector.h"

YoloTinyDetector::YoloTinyDetector(const std::string &paramPath, const std::string &binPath, int targetSize, int numThread)
    : mTargetSize(targetSize), mNumThreads(numThread)
{

    mYolov4Net.opt.use_vulkan_compute = false;

    mYolov4Net.load_param(paramPath.c_str());
    mYolov4Net.load_model(binPath.c_str());

    mExtractor = std::make_shared<ncnn::Extractor>(mYolov4Net.create_extractor());
    mExtractor->set_light_mode(true);
    mExtractor->set_num_threads(mNumThreads);
}

cv::Mat YoloTinyDetector::letterbox_resize(const cv::Mat &src)
{
    auto src_width = static_cast<float>(src.cols);
    auto src_height = static_cast<float>(src.rows);

    auto ewidth_f = static_cast<float>(mTargetSize);
    auto eheight_f = static_cast<float>(mTargetSize);

    auto scale = ewidth_f / src_width > eheight_f / src_height ? eheight_f / src_height : ewidth_f / src_width;

    auto width = static_cast<int>(scale * src_width);
    auto height = static_cast<int>(scale * src_height);

    cv::Mat dst;
    cv::resize(src, dst, cv::Size(width, height));

    auto top = (mTargetSize - height) / 2;
    auto bottom = mTargetSize - top - height;
    auto left = (mTargetSize - width) / 2;
    auto right = mTargetSize - left - width;

    cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT);
    return dst;
}

void YoloTinyDetector::detect(cv::Mat &img, std::vector<Yolov4Object> &objects)
{
    int img_w = img.cols;
    int img_h = img.rows;

    auto letterboxImage = letterbox_resize(img);
    auto in = ncnn::Mat::from_pixels(letterboxImage.data, ncnn::Mat::PIXEL_BGR, letterboxImage.cols, letterboxImage.rows);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    mExtractor->input("data", in);

    ncnn::Mat out;
    mExtractor->extract("output", out);

    auto scale = std::min(static_cast<float>(mTargetSize) / static_cast<float>(img_w),
                          static_cast<float>(mTargetSize) / static_cast<float>(img_h));

    objects.clear();
    for (int i = 0; i < out.h; i++)
    {   
        const float *values = out.row(i);
        if(int(values[0]) != 1) continue;

        Yolov4Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * letterboxImage.cols;
        object.rect.y = values[3] * letterboxImage.rows;
        object.rect.width = values[4] * letterboxImage.cols - object.rect.x;
        object.rect.height = values[5] * letterboxImage.rows - object.rect.y;

        std::cout << object.rect.x << " " << object.rect.y << " " << object.rect.width << " " << object.rect.height << std::endl;

        object.rect.x -= (mTargetSize - scale * img_w) / 2;
        object.rect.y -= (mTargetSize - scale * img_h) / 2;

        object.rect.x /= scale;
        object.rect.y /= scale;
        object.rect.width /= scale;
        object.rect.height /= scale;

        objects.push_back(object);
    }
}

