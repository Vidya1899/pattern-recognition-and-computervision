//
//  main.cpp
//  Test
//
//  Created by Vidya Ganesh on 1/25/23.
//

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "filter.h"


using namespace std;
using namespace cv;

int main() {
    std::string image_path = "/Users/vidyaganesh/Desktop/Test/Test/sample_img.jpeg";
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    imshow("Display window", img);
    while(true){
        int k = waitKey(0); // Wait for a keystroke in the window
        if(k=='q'){
            break;
        }
    }
    return 0;
    
}
