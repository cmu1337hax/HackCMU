#include <string>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>

#define CONTOUR_AREA 75
#define CONTOUR_AREA_MAX 25000
#define DEBUG FALSE
#define nl "\n"

using namespace std;
using namespace cv;

int img = 0;

int main(){

    //open camera
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH,1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,720);

    if(!cap.isOpened())
        return -1;

    Mat edges;
    namedWindow("Debug View",1);

    while(true){
        Mat frame;
        if(DEBUG)
            cap >> frame;
        else
            Mat frame= imread("test.jpg");
        cout<<"Loaded Frame!"<<nl;

        /*Text Detection*/
        //greyscale
        Mat grey;
        cvtColor(frame,grey,CV_BGR2GRAY);
        grey.convertTo(grey, CV_8U);

        cout<<"Greyscaled!"<<nl;

        //blur
        Mat blur;
        GaussianBlur(grey,blur,Size(3,3),1.5); //kernel size, sigma y

        cout<<"Blurred!"<<nl;

        //open
        Mat const structure_elem = cv::getStructuringElement(MORPH_RECT, Size(4, 2)); //rect for morphologyEx
        Mat open;
        morphologyEx(blur, open, MORPH_OPEN, structure_elem);

        cout<<"Opened!"<<nl;

        //Threshold
        Mat threshold;
        adaptiveThreshold(open, threshold, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 51, 10);

        cout<<"Thresholded!"<<nl;

        //close
        Mat const structure_elem1 = cv::getStructuringElement(MORPH_RECT, Size(17, 5)); //change size for word detection
        Mat close;
        morphologyEx(threshold, close, MORPH_CLOSE, structure_elem1);

        cout<<"Closed!"<<nl;

        //contours
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        cv::findContours(close, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        Mat drawing = Mat::zeros(close.size(), CV_8UC1);
        vector<vector<Point> > contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());

        Mat word;

        for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]){
            // calculate parameters for filter the letters
            approxPolyDP(Mat(contours[idx]), contours_poly[idx], 3, true);
            boundRect[idx] = boundingRect(Mat(contours_poly[idx]));
            float occupyrate;
            occupyrate = (contourArea(contours[idx]) / (boundRect[idx].width * boundRect[idx].height));
            float aspectratio;
            aspectratio = max(boundRect[idx].height, boundRect[idx].width) / min(boundRect[idx].height, boundRect[idx].width);
            float perimeter;
            perimeter = arcLength(contours[idx], true);
            float compactness;
            compactness = contourArea(contours[idx]) / (perimeter * perimeter);


            // filter contours by region areas and parameters and draw
            RNG rng(12345);
            {
                if ((contourArea(contours[idx]) > CONTOUR_AREA) & (contourArea(contours[idx]) <= CONTOUR_AREA_MAX)){
                    if ((occupyrate >= 0.03) & (occupyrate <= 0.95)){
                        if (aspectratio <= 6){
                            if ((compactness > 0.003) & (compactness <= 0.95)){
                                if(!DEBUG){
                                    Rect crop(boundRect[idx].tl(), boundRect[idx].br());
                                    word = frame(crop);
                                    string filepath ="/output/word";
                                    filepath += idx + ".jpg";
                                    imwrite(filepath, word);
                                    cout<<"Saved: "<<filepath<<nl;
                                }
                                else{
                                    Scalar color(rand() & 255, rand() & 255, rand() & 255);
                                    drawContours(drawing, contours, idx, color, CV_FILLED, 8, hierarchy);
                                    rectangle( frame, boundRect[idx].tl(), boundRect[idx].br(), color, 2, 8, 0 );
                                }
                            }
                        }
                    }
                }
            }
        }

        imshow("Debug View", frame);

        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;

}
