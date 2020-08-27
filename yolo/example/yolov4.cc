#include <ncnn/net.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

#define YOLOV4_TINY 1 //0 or undef for yolov4

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

Mat letterbox_resize(const Mat &src, int expected_width, int expected_height)
{
    auto src_width = static_cast<float>(src.cols);
    auto src_height = static_cast<float>(src.rows);

    auto ewidth_f = static_cast<float>(expected_width);
    auto eheight_f = static_cast<float>(expected_height);

    auto scale = ewidth_f / src_width > eheight_f / src_height ? eheight_f / src_height : ewidth_f / src_width;

    auto width = static_cast<int>(scale * src_width);
    auto height = static_cast<int>(scale * src_height);

    Mat dst;
    resize(src, dst, Size(width, height));

    auto top = (expected_height - height) / 2;
    auto bottom = expected_height - top - height;
    auto left = (expected_width - width) / 2;
    auto right = expected_width - left - width;

    copyMakeBorder(dst, dst, top, bottom, left, right, BORDER_CONSTANT);
    return dst;
}

static int detect_yolov4(const cv::Mat &bgr, std::vector<Object> &objects)
{
    ncnn::Net yolov4;

    yolov4.opt.use_vulkan_compute = false;

    // original pretrained model from https://github.com/AlexeyAB/darknet
    // the ncnn model https://drive.google.com/drive/folders/1YzILvh0SKQPS_lrb33dmGNq7aVTKPWS0?usp=sharing
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
#if YOLOV4_TINY
    yolov4.load_param("../../models/yolov4-tiny-opt.param");
    yolov4.load_model("../../models/yolov4-tiny-opt.bin");
    const int target_size = 416;
#else
    yolov4.load_param("../../models/yolov4-opt.param");
    yolov4.load_model("../../models/yolov4-opt.bin");
    const int target_size = 608;
#endif
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    auto t1 = getTickCount();
    ncnn::Extractor ex = yolov4.create_extractor();
    ex.set_num_threads(8);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);
    auto t2 = static_cast<float>(getTickCount() - t1) / static_cast<float>(getTickFrequency());
    cout << "consumed: " << t2 << endl;

    printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float *values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}



static void draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects)
{
    static const char *class_names[] = {"background", "person", "bicycle",
                                        "car", "motorbike", "aeroplane", "bus", "train", "truck",
                                        "boat", "traffic light", "fire hydrant", "stop sign",
                                        "parking meter", "bench", "bird", "cat", "dog", "horse",
                                        "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase",
                                        "frisbee", "skis", "snowboard", "sports ball", "kite",
                                        "baseball bat", "baseball glove", "skateboard", "surfboard",
                                        "tennis racket", "bottle", "wine glass", "cup", "fork",
                                        "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                                        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                                        "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
                                        "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                                        "cell phone", "microwave", "oven", "toaster", "sink",
                                        "refrigerator", "book", "clock", "vase", "scissors",
                                        "teddy bear", "hair drier", "toothbrush"};

    cv::Mat image = bgr.clone();
    auto src_width = static_cast<float>(image.cols);
    auto src_height = static_cast<float>(image.rows);

    auto scale = 416.0 / src_width > 416.0 / src_height ? 416.0 / src_height : 416.0 / src_width;

    for (size_t i = 0; i < objects.size(); i++)
    {
        Object obj = objects[i];
        if(obj.label == 1)
        {

            fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            float x1 = obj.rect.x - ((416 - scale * src_width)) / 2;
            float y1 = obj.rect.y - ((416 - scale * src_height)) / 2;
            float x2 = obj.rect.br().x - ((416 - scale * src_width)) / 2;
            float y2 = obj.rect.br().y - ((416 - scale * src_height)) / 2;

            x1 /= scale;
            x2 /= scale;
            y1 /= scale;
            y2 /= scale;

            Rect2f box(Point2f(x1, y1), Point2f(x2, y2));

            cv::rectangle(image,box, cv::Scalar(255, 0, 0));

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = box.x;
            int y = box.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            /*cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                        cv::Scalar(255, 255, 255), -1);*/

            /*cv::putText(image, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));*/
        }
    }

    cv::imshow("image", image);
    
}

int main(int argc, char **argv)
{
    vector<cv::String> fileNames;
    cv::String folder = "../../imgs/*.jpg";
    cv::glob(folder,fileNames);

    float times = 0.f;
    for (int i = 0; i < fileNames.size(); i++)
    {   
        cout << fileNames[i] << endl;
        cv::Mat m = cv::imread(fileNames[i], 1);
        auto letter_img = letterbox_resize(m,416,416);
        std::vector<Object> objects;
        detect_yolov4(letter_img, objects);

        draw_objects(m, objects);
        if(cv::waitKey(0) == 27)
            break;
    }

    return 0;
}