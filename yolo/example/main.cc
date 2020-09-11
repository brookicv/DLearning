#include <stdio.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Yolov4TinyDetector.h"
#include "AgeGenderRecongnizer.h"

#define CAMERA 0

using namespace std;
using namespace cv;

float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}

float get_iou(Rect rect1,Rect rect2)
{
    int xx1, yy1, xx2, yy2;
    xx1 = max(rect1.x, rect2.x);
    yy1 = max(rect1.y, rect2.y);
    xx2 = min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    yy2 = min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

    int insection_width, insection_height;
    insection_width = max(0, xx2 - xx1 + 1);
    insection_height = max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = float(insection_width) * insection_height;
    union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);

    iou = insection_area / union_area;
    return iou;
}

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


int main(int argc, char* argv[])
{   
    //string path = string(argv[1]);
    // auto img = imread(path);

    auto detector = YoloTinyDetector("../../models/yolov4-tiny-opt.param", "../../models/yolov4-tiny-opt.bin");
    AgeGenderRecongnizer ageGenderRecongnizer("../../models/ped_attr.mnn");

    #if CAMERA
    VideoCapture capture(0);
    if(!capture.isOpened()){
        cout << "Could not open camera" << endl;
        return -1;
    }
    Mat img;
    while (capture.read(img))
    {
        vector<Yolov4Object> objects;
        detector.detect(img , objects);

        stringstream ss;
        for (auto object : objects)
        {
            int x1 = std::max(object.rect.x, float(0));
            int y1 = std::max(object.rect.y, float(0));
            int x2 = std::min(object.rect.x + object.rect.width, float(img.cols));
            int y2 = std::min(object.rect.y + object.rect.height, float(img.rows));
            Rect r(cv::Point(x1, y1), cv::Point(x2, y2));
            auto person = img(r);
            int age, gender;
            if (person.isContinuous())
            {
                ageGenderRecongnizer.recongnize(person, age, gender);
            }
            else
            {
                uchar *colorImgData = new uchar[person.total() * 3];
                // 新建同等大小的Mat,通道为8UC3
                Mat MatTemp(person.size(), CV_8UC3, colorImgData);
                cv::cvtColor(person, MatTemp, CV_BGR2RGB);

                ageGenderRecongnizer.recongnize(MatTemp, age, gender);
                delete[] colorImgData;
            }
            rectangle(img, object.rect, Scalar(0, 255, 0));

            ss.str("");
            ss << "age:" << age;
            putText(img, ss.str(), Point(object.rect.x + 20, object.rect.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            ss.str("");
            ss << "gender:" << gender;
            putText(img, ss.str(), Point(object.rect.x + 20, object.rect.y + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255));
        }

        cout << "count: " << objects.size() << endl;
        imshow("person", img);
        if(waitKey(25) == 27)
            break;
    }
    #else

    vector<cv::String> fileNames;
    cv::glob("../../imgs", fileNames);
    for(size_t i = 0; i < fileNames.size(); i ++)
    {
        Mat img = cv::imread(fileNames[i]);
        std::cout << fileNames[i] << std::endl;

        vector<Yolov4Object> objects;
        detector.detect(img , objects);

        stringstream ss;
        for (auto object : objects)
        {
            auto person = img(object.rect);
            int age, gender;
            if (person.isContinuous())
            {
                ageGenderRecongnizer.recongnize(person, age, gender);
            }
            else
            {
                uchar *colorImgData = new uchar[person.total() * 3];
                // 新建同等大小的Mat,通道为8UC3
                Mat MatTemp(person.size(), CV_8UC3, colorImgData);
                cv::cvtColor(person, MatTemp, CV_BGR2RGB);

                ageGenderRecongnizer.recongnize(MatTemp, age, gender);
                delete[] colorImgData;
            }
            rectangle(img, object.rect, Scalar(0, 255, 0));

            ss.str("");
            ss << "age:" << age;
            putText(img, ss.str(), Point(object.rect.x + 20, object.rect.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            ss.str("");
            ss << "gender:" << gender;
            putText(img, ss.str(), Point(object.rect.x + 20, object.rect.y + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255));
        }
        cout << "count: " << objects.size() << endl;
        imshow("person", img);
        if (waitKey() == 27)
            break;
    }
    #endif

    return 0;
}
