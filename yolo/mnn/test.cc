
#include "YoloDetector.h"
#include "AgeGenderRecongnizer.h"

using namespace std;
using namespace cv;

int main()
{
    MNN_YoloDetector detector("../yolov3_608.mnn");

    AgeGenderRecongnizer ageGenderRecongnizer("../ped_attr.mnn");

    auto img = imread("../imgs/camera-11.jpg");

    vector<Object> objects;
    detector.detect(img,objects);


    for (auto obj : objects)
    {
        int age, gender;
        auto person = img(obj.rect);

        std::vector<unsigned char> buff;
        imencode(".jpg", person, buff);

        person = imdecode(buff, 1);

        ageGenderRecongnizer.recongnize(person, age, gender);

        rectangle(img, obj.rect, cv::Scalar(0, 255, 0));
        string text = gender == 1 ? "male" : "female";
        cv::putText(img, text, cv::Point(obj.rect.x + 20, obj.rect.y + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }

    imshow("person",img);
    waitKey();

    return 0;
}