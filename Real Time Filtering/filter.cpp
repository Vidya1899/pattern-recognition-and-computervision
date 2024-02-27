#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "filter.h"
#include <cmath>

using namespace std;

int greyscale(cv::Mat &src, cv::Mat &dst){
    dst =  cv::Mat::zeros(src.size(), CV_16SC3);
    
    for(int i=0;i<src.rows;i++) {
        // get the row pointer for row i
        cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
        // loop over the columns
        for(int j=0;j<src.cols;j++) {
          // modify epach color channel
          // using vec3b row pointers  (much faster)
          dptr[j][0] = (rptr[j][0] + rptr[j][1] + rptr[j][2])/3;
          dptr[j][1] = (rptr[j][0] + rptr[j][1] + rptr[j][2])/3;
          dptr[j][2] = (rptr[j][0] + rptr[j][1] + rptr[j][2])/3;
        }
      }
      return(0);
}


int blur5X5(cv::Mat &src, cv::Mat &dst){
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    src.copyTo(dst);
    dst.convertTo(dst, CV_16SC3);
    
    //[1 2 4 2 1] 1X5
    for(int i=2;i<src.rows-2;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                temp.at<cv::Vec3s>(i,j)[c] = (1*src.at<cv::Vec3b>(i,j-2)[c] + 2*src.at<cv::Vec3b>(i,j-1)[c] + 4*src.at<cv::Vec3b>(i, j)[c] + 2*src.at<cv::Vec3b>(i, j+1)[c] + 1*src.at<cv::Vec3b>(i, j+2)[c])/10;
            }
        }
    }
    //[ 1 2 4 2 1] 5x1
    for(int i=2;i<src.rows-2;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                dst.at<cv::Vec3s>(i,j)[c] = (1*temp.at<cv::Vec3s>(i-2,j)[c] + 2*temp.at<cv::Vec3s>(i-1,j)[c] + 4*temp.at<cv::Vec3s>(i, j)[c] + 2*temp.at<cv::Vec3s>(i+1, j)[c] + 1*temp.at<cv::Vec3s>(i+2, j)[c])/10;
            }
        }
    }
    return 0;
}


int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    src.copyTo(dst);
    dst.convertTo(dst, CV_16SC3);
    
    //[-1 0 1]
    for(int i=2;i<src.rows-2;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                temp.at<cv::Vec3s>(i,j)[c] =  -1*src.at<cv::Vec3b>(i,j-1)[c] + 0*src.at<cv::Vec3b>(i, j)[c] + 1*src.at<cv::Vec3b>(i, j+1)[c];
            }
        }
    }
    //[1 2 1]
    for(int i=2;i<src.rows-2;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                dst.at<cv::Vec3s>(i,j)[c] = (1*temp.at<cv::Vec3s>(i-1,j)[c] + 2*temp.at<cv::Vec3s>(i, j)[c] + 1*temp.at<cv::Vec3s>(i+1, j)[c])/4;
            }
        }
    }
    return 0;
}


int sobelY3x3( cv::Mat &src, cv::Mat &dst ) {
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    src.copyTo(dst);
    dst.convertTo(dst, CV_16SC3);
    
    //[-1 0 1]
    for(int i=2;i<src.rows-2;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                temp.at<cv::Vec3s>(i,j)[c] = (-1*src.at<cv::Vec3b>(i-1,j)[c] + 0*src.at<cv::Vec3b>(i, j)[c] + 1*src.at<cv::Vec3b>(i+1, j)[c]);
            }
        }
    }
    //[1 2 1]
    for(int i=2;i<src.rows-2;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                dst.at<cv::Vec3s>(i,j)[c] =  (1*temp.at<cv::Vec3s>(i,j-1)[c] + 2*temp.at<cv::Vec3s>(i, j)[c] + 1*temp.at<cv::Vec3s>(i, j+1)[c])/4;
            }
        }
    }
    return 0;
}


int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ) {
    //generates a gradient magnitude image using Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
    dst = cv::Mat::zeros(sx.size(), CV_16SC3);
    for(int i=2;i<sx.rows-2;i++){
        for(int j=2;j<sx.cols-2;j++){
            for(int c=0;c<3;c++){
                dst.at<cv::Vec3s>(i,j)[c] = sqrt(sx.at<cv::Vec3s>(i,j)[c]*sx.at<cv::Vec3s>(i,j)[c] + sy.at<cv::Vec3s>(i,j)[c]*sy.at<cv::Vec3s>(i,j)[c]);
            }
        }
    }
    return 0;
}


int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ) {
    //the size of a bucket using
    int b = 255/levels;
    //Take a color channel value x and first execute xt = x / b, then execute xf = xt * b. After executing this for
    //each pixel and each color channel, the image will have only levels**3 possible color values.
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                int xt = src.at<cv::Vec3b>(i,j)[c]/b;
                int xf = xt*b;
                dst.at<cv::Vec3s>(i,j)[c] = xf;
            }
        }
    }
    return 0;
}


int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold ) {
    //First calculating the gradient magnitude.
    //Then blur and quantize the image.
    //Finally, modify the blurred and quantized image by setting to black any
    //pixels with a gradient magnitude larger than a threshold.
    
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat sobelX = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat sobelY = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat gradMag = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat blurQuant = cv::Mat::zeros(src.size(), CV_16SC3);
    
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);
    magnitude(sobelX, sobelY, gradMag);
    blurQuantize(src, blurQuant, 15);
    
    for(int i=0;i<blurQuant.rows;i++){
        for(int j=0;j<blurQuant.cols;j++){
            for(int c=0;c<3;c++){
                if(gradMag.at<cv::Vec3s>(i,j)[c] <= magThreshold) {
                    dst.at<cv::Vec3s>(i,j)[c] = blurQuant.at<cv::Vec3s>(i,j)[c];}
                else{
                    dst.at<cv::Vec3s>(i,j)[c] = 0;}
            }
        }
    }
    return 0;
}
    

// 10. Change around the color palette or make the image a negative of itself.
// Implemented negative of video
int negative( cv::Mat &src, cv::Mat &dst) {
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                dst.at<cv::Vec3s>(i,j)[c] = 255 - src.at<cv::Vec3b>(i,j)[c];
            }
        }
    }
    return 0;
}


// Extension: 1
// Add caption to video live stream
int caption( cv::Mat &src, cv::Mat &dst, string tagline) {
    src.copyTo(dst);
    dst.convertTo(dst, CV_16SC3);
    cv::putText(dst, //target image
                tagline, //text
                cv::Point(10, dst.rows / 2), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(118, 185, 0), //font color
                2);
    return 0;
}

// Extension: 2
// Add caption to video live stream
int rotate( cv::Mat &src, cv::Mat &dst, double angle) {
     
    // get the center coordinates of the image to create the 2D rotation matrix
    //cv::Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
    // using getRotationMatrix2D() to get the rotation matrix
    //cv::Mat rotation_matix = getRotationMatrix2D(center, angle, 1.0);
 
    // rotate the image using warpAffine
    //warpAffine(src, dst, rotation_matix, src.size());
    
    
    // get rotation matrix for rotating the image around its center in pixel coordinates
    cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

    cv::warpAffine(src, dst, rot, bbox.size());

    return 0;
}

//    cv:: Mat temp;
//    dst =  cv::Mat::zeros(src.size(), CV_8UC3);
//    temp =  cv::Mat::zeros(src.size(), CV_8UC3);


//    for(int i=2;i<src.rows-2;i++) {
//        // get the row pointer for row i
//        cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
//        cv::Vec3b *dptr = temp.ptr<cv::Vec3b>(i);
//
//        // loop over the columns
//        for(int j=2;j<src.cols-2;j++) {
//            // modify each color channel
//            for(int c=0;c<3;c++) {
//                dptr[j][c] = rptr[j-2][c] * 0.1 + rptr[j-1][c]*0.2 + rptr[j][c] *0.4 + rptr[j+1][c]*0.2 + rptr[j+2][c] * 0.1;
//            }
//        }
//    }
//    for(int i=2; i<src.rows-2; i++) {
//        /*cv::Vec3b *rptrm2 = temp.ptr<cv::Vec3b>(i-2);
//        cv::Vec3b *rptrm1 = temp.ptr<cv::Vec3b>(i-1);
//        cv::Vec3b *rptr = temp.ptr<cv::Vec3b>(i);
//        cv::Vec3b *rptrp1 = temp.ptr<cv::Vec3b>(i+1);
//        cv::Vec3b *rptrp2 = temp.ptr<cv::Vec3b>(i+2);
//        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);*/
//        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
//
//        for(int j=2; i<src.cols-2; j++) {
//            for(int c =0;c<3; c++) {
//                //dptr[j][c] = rptrm2[j][c]*0.1 + rptrm1[j][c]*0.2 + rptr[j][c]*0.4 +rptrp1[j][c]*0.2 + rptrp2[j][c];
//
//            }
//        }
//    }
//    return 0;
/*
int gradX( cv::Mat &src, cv::Mat &dst ) {
  // allocate dst image
  dst = cv::Mat::zeros( src.size(), CV_16SC3 ); // signed short data type
  // loop over src and apply a 3x3 filter
  for(int i=1;i<src.rows-1;i++) {
    // src pointer
    cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i+1);
    // destination pointer
    cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<src.cols-1;j++) {
      // for each color channel
      for(int c=0;c<3;c++) {
dptr[j][c] = (-1 * rptrm1[j-1][c] + 1 * rptrm1[j+1][c] +
      -2*rptr[j-1][c] + 2*rptr[j+1][c] +
      -1 * rptrp1[j-1][c] + 1*rptrp1[j+1][c]) / 4;
      }
    }
  }
  // return
  return(0);
}*/
