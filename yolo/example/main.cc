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

#include <ncnn/net.h>
using namespace std;
using namespace cv;

float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}

struct BBox {
    Rect2f box;  // bbox
    float confidence; // 置信度
    int index; // 边框的index
    int cls_index; // 所属类别的index
    float cls_prob; //类别概率
};

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

int main(int argc, char **argv)
{
    ncnn::Net model;
    model.load_param("../../models/yolov3.param");
    model.load_model("../../models/yolov3.bin");

    const char *imagePath = argv[1];
    cv::Mat cv_img = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
    if (cv_img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagePath);
        return -1;
    }

    auto letterbox_img = letterbox_resize(cv_img, 608, 608);

    ncnn::Mat in = ncnn::Mat::from_pixels(letterbox_img.data, ncnn::Mat::PIXEL_BGR2RGB, letterbox_img.cols, letterbox_img.rows);
    //const float mean_vals[3] = {127.5, 127.5, 127.5};
    //const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
    const float mean_vals[3] = {0.485, 0.456, 0.406};
    const float norm_vals[3] = {0.229, 0.224, 0.225};

    in.substract_mean_normalize(mean_vals, norm_vals);

    double t1 = (double)getTickCount();
    ncnn::Extractor ex = model.create_extractor();

    ex.set_num_threads(16);
    ex.set_light_mode(true);
    ex.input("input1", in);

    ncnn::Mat out;
    ncnn::Mat out2;
    ex.extract("bboxes", out);
    ex.extract("classes", out2);

    double t = double(getTickCount() - t1) / double(getTickFrequency());
    cout << t << endl;

    int strides[3] = {32, 16, 8};
    for (int i = 0; i < out.h; i++)
    {
        float *row = out.row(i);
        if (i < 1083){
            row[0] *= 32;
            row[1] *= 32;
            row[2] *= 32;
            row[3] *= 32;
        }
    }

        auto rows = out.h;
    vector<Point> maxLocs;
    for (int i = 0; i < rows; i++)
    {
        float *row = out2.row(i);
        cv::Mat prob(1, out2.w, CV_32FC1, out2.row(i));
        Point maxLoc;
        cv::minMaxLoc(prob,nullptr,nullptr,nullptr,&maxLoc);
        maxLocs.push_back(maxLoc);
    }

    float confidence_threshold = 0.5;
    float iou_threshold = 0.4;
    vector<BBox> bboxes;
    for (int i = 0; i < rows; i++)
    {
        float *row = out.row(i);
        float *row2 = out2.row(i);
        cout << row[0] << " " << row[1] << " " << row[2]
             << " " << row[3] << " " << row[4] << " " << endl;

        if (row[4] > confidence_threshold)
        {
            if (row[2] <= 0 || row[3] <= 0)
                continue;
            float x1 = row[0] - row[2] / 2;
            float y1 = row[1] - row[3] / 2;
            float x2 = row[0] + row[2] / 2;
            float y2 = row[1] + row[3] / 2;

            Rect2f box(Point2f(x1, y1), Point2f(x2, y2));
            BBox bbox;
            bbox.box = box;
            bbox.confidence = row[4];
            bbox.cls_index = maxLocs[i].x;
            bbox.index = i;
            bbox.cls_prob = row2[maxLocs[i].x];

            bboxes.push_back(bbox);
        }
    }

    sort(bboxes.begin(), bboxes.end(),[](BBox &a, BBox &b){ return a.confidence > b.confidence; });

    vector<int> indices;
    int box_size = bboxes.size();
    for (int i = 0; i < box_size; i++)
    {
        auto bbox = bboxes[i];
        if(bbox.cls_index != 0)  // 只处理person
            continue;

        indices.push_back(i);
        for (int j = i + 1; j < box_size; j++)
        {
            float iou = get_iou(bbox.box,bboxes[j].box);
            if(iou > iou_threshold)
            {
                bboxes.erase(bboxes.begin() + j); // 移除该边框
                box_size = bboxes.size();
            }
        }
    }

    cout << indices.size() <<endl;

    for(int i = 0; i < indices.size();i ++)
    {
        auto bbox = bboxes[indices[i]];


        if(bbox.confidence * bbox.cls_prob > 0.5)
        {
            cout << bbox.confidence << endl;
            cout << bbox.cls_prob << endl;
            cout << bbox.box.x << " " << bbox.box.y << endl;
            cout << bbox.box.width << " " << bbox.box.height << endl;

            if (bbox.box.width > 608 || bbox.box.height > 608)
                cout << indices[i] << endl;
        }


    }

    return 0;

    /*
    for (int i = 0; i < out.h; i++)
    {
        float *row = out.row(i);
        for (int j = 0; j < out.w; j++)
        {
            row[j] = sigmoid(row[j]);
            cout << row[j] << " ";
        }
        cout << endl;
    }*/
}