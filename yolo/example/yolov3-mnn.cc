#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace MNN;
using namespace std;
using namespace cv;

float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
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

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

float intersection_area(const Object &a, const Object &b) {
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void bboxes_nms(vector<Object> &objects,vector<int> &picked,float nms_threshold) 
{
    picked.clear();

    const int n = objects.size();
    
    vector<float> areas(n);
    for(int i = 0; i < n; ++i){ 
        areas[i] = objects[i].rect.area();
    }

    for(int i = 0; i < n; ++i){
        const Object &a = objects[i];

        int keep = 1;
        for(int j = 0; j <picked.size(); j ++)
        { 
            const Object &b = objects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;

            if(inter_area  / union_area > nms_threshold)
                keep = 0;
        }
        if(keep){
            picked.push_back(i);
        }
    }
}


int main(int argc, char* argv[])
{

    const auto poseModel           = argv[1];
    const auto inputImageFileName  = argv[2];

    const int targetWidth = 608;
    const int targetHeight = 608;
    auto raw_image = cv::imread(inputImageFileName);
    int originalWidth = raw_image.cols;
    int originalHeight = raw_image.rows;

    auto letter_image = letterbox_resize(raw_image, targetWidth, targetHeight);
    Mat targetImage;
    cv::cvtColor(letter_image, targetImage, cv::COLOR_BGR2RGB);
    targetImage.convertTo(targetImage, CV_32FC3);
    targetImage = targetImage / 255.0;

    // copy data to mnn tensor from opencv mat
    vector<int> dim{ 1, targetHeight, targetWidth, 3 };
    auto nhwc_tensor = MNN::Tensor::create<float>(dim, nullptr, MNN::Tensor::TENSORFLOW);
    auto nhwc_data = nhwc_tensor->host<float>();
    auto nhwc_size = nhwc_tensor->size();

    ::memcpy(nhwc_data, targetImage.data, nhwc_size);

    // create net and session
    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(poseModel));
    MNN::ScheduleConfig netConfig;
    netConfig.type      = MNN_FORWARD_CPU;
    netConfig.numThread = 4;
    auto session        = mnnNet->createSession(netConfig);

    auto input = mnnNet->getSessionInput(session, nullptr);
    input->copyFromHostTensor(nhwc_tensor);

    // run interfence
    mnnNet->runSession(session);

    auto output = mnnNet->getSessionOutput(session,"bboxes");

    auto size = output->elementSize();
    vector<float> boxes(size);
    auto tmpOutput = output->host<float>();
    for (int i = 0; i < size; i++) {
        boxes[i] = tmpOutput[i];
    }
    

    auto classes = mnnNet->getSessionOutput(session,"classes") ;
    classes->printShape();

    vector<Object> objects;

    auto probs = classes->host<float>();
    size = classes->elementSize();
    vector<float> tempValues(size);
    for(int i=0 ; i < size ; i++){
        tempValues[i] = probs[i];
    }

    auto scale = std::min(static_cast<float>(targetWidth) / static_cast<float>(originalWidth), 
        static_cast<float>(targetHeight) / static_cast<float>(originalHeight));

    for(int i=0 ; i < 22743; i++){
        auto maxCls = std::max_element(tempValues.begin()+ i * 80,tempValues.begin() + (i + 1) * 80);
        auto clsIdx = maxCls - (tempValues.begin() + i * 80);
        
        auto confidence = boxes[i * 5 + 4];
        auto score = confidence * (*maxCls);
        
        if(score < 0.2) continue;
        cout << "clsIdx: " << clsIdx << " score: " << score << endl;

        if(clsIdx != 0) continue; // person id = 0

        auto width = boxes[i* 5 + 2];
        auto height = boxes[i* 5 + 3];
        auto xmin = boxes[i * 5 + 0] - width / 2;
        auto ymin = boxes[i * 5 + 1] - height /2;

        xmin -= (608 - scale * originalWidth) / 2;
        ymin -= (608 - scale * originalHeight) /2;

        xmin /= scale;
        ymin /= scale;
        width /= scale;
        height /= scale;

        cout << "xmin: " << xmin << " ymin: " << ymin << " width: " << width << " height: " << height << endl;
        
        Object obj;
        obj.rect = cv::Rect_<float>(xmin,ymin,width,height);
        obj.label = clsIdx;
        obj.prob = score;
        objects.push_back(obj);
    } 

    std::sort(objects.begin(),objects.end(),[](const Object& a, const Object& b){ return a.prob  > b.prob; });

    vector<int> picked;
    bboxes_nms(objects, picked, 0.4);

    for(int i = 0; i < picked.size(); i++){
        auto obj = objects[picked[i]];

        cv::rectangle(raw_image, obj.rect, (255,255,255), 1);
    }

    imshow("person", raw_image);
    cv::waitKey();

    return 0;
}