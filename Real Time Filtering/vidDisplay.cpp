//
//  main.cpp
//  Test
//
//  Created by Vidya Ganesh on 1/25/23.
//

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "filter.h"


using namespace std;
using namespace cv;


int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        //printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;
        cv::Mat out;
        cv::Mat outScaled;
        cv::Mat out1;
        cv::Mat out2;
        bool grey = false;
        bool blur = false;
        bool sobelX = false;
        bool sobelY = false;
        bool opencvgrey = false;
        bool gradMagnitude = false;
        bool quantize = false;
        bool cartoonize = false;
        bool neg = false;
        bool cap = false;
        string tagline;
        double angle;
        bool rot = false;
    
        for(;;) {
            *capdev >> frame; // get a new frame from the camera, treat as a stream
            if( frame.empty() ) {
              printf("frame is empty\n");
              break;
            }
            if( opencvgrey==true) {
                cv::namedWindow("Grey Video", 1);
                cv::Mat gray_image;
                cv::cvtColor( frame, gray_image, COLOR_BGR2GRAY );
                cv::imshow("Grey Video", gray_image);
            }
            if(grey==true) {
                cv::namedWindow("Grey Custom Video", 1);
                greyscale(frame,out);
                cv::convertScaleAbs(out, outScaled);
                cv::imshow("Grey Custom Video", outScaled);
            }
            if(blur==true) {
                cv::namedWindow("Blur Video", 1);
                blur5X5(frame, out);
                cv::convertScaleAbs(out, outScaled);
                cv::imshow("Blur Video", outScaled);
            }
            if(sobelX==true) {
                cv::namedWindow("Sobel-X Video", 1);
                sobelX3x3(frame, out);
                cv::convertScaleAbs(out, outScaled, 2);
                cv::imshow("Sobel-X Video", outScaled);
            }
            if(sobelY==true) {
                cv::namedWindow("Sobel-Y Video", 1);
                sobelY3x3(frame, out);
                cv::convertScaleAbs(out, outScaled, 2);
                cv::imshow("Sobel-Y Video", outScaled);
            }
            if(gradMagnitude==true) {
                cv::namedWindow("Magnitude Video", 1);
                sobelX3x3(frame, out1);
                sobelY3x3(frame, out2);
                magnitude(out1, out2, out);
                cv::convertScaleAbs(out, outScaled);
                cv::imshow("Magnitude Video", outScaled);
            }
            if(quantize==true) {
                cv::namedWindow("BlurQuantize Video", 1);
                blurQuantize(frame, out, 15);
                cv::convertScaleAbs(out, outScaled);
                cv::imshow("BlurQuantize Video", outScaled);
            }
            if(cartoonize==true) {
                cv::namedWindow("Cartoon Video", 1);
                cartoon(frame, out, 15, 15);
                cv::convertScaleAbs(out, outScaled);
                cv::imshow("Cartoon Video", outScaled);
            }
            if(neg==true) {
                cv::namedWindow("Negative Video", 1);
                negative(frame, out);
                cv::convertScaleAbs(out, outScaled);
                cv::imshow("Negative Video", outScaled);
            }
            if(cap==true) {
                cv::namedWindow("Add Caption", 1);
                caption(frame, out, tagline);
                cv::convertScaleAbs(out, outScaled);
                cv::imshow("Add Caption", outScaled);
            }
            if(rot==true) {
                cv::namedWindow("Rotate Video", 1);
                rotate(frame, out, angle);
                cv::convertScaleAbs(out, outScaled);
                cv::imshow("Rotate Video", outScaled);
            }
            cv::imshow("Video", frame);

            // see if there is a waiting keystroke
            char key = cv::waitKey(10);
            if( key == 'q') {
                break;
            }
            if( key == 's') {
                imwrite("/Users/vidyaganesh/Desktop/cv_project/project1/frame.png", frame);
            }
            if( key == 'g') {
                opencvgrey = true;
            }
            if( key == 'h') {
                grey = true;
            }
            if( key == 'b') {
                blur = true;
            }
            if( key == 'x') {
                sobelX = true;
            }
            if( key == 'y') {
                sobelY = true;
            }
            if( key == 'm') {
                gradMagnitude = true;
            }
            if( key == 'l') {
                quantize = true;
            }
            if( key == 'c') {
                cartoonize = true;
            }
            if( key == 'n') {
                neg = true;
            }
            if( key == 't') {
                cap = true;
                cout << "Enter the caption: ";
                cin>> tagline;
            }
            if( key == 'r') {
                cout << "Enter the angle: ";
                cin>> angle;
                rot = true;
            }
        }

        delete capdev;
        return(0);
}
