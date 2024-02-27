/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */
#include<iostream>
#include<opencv2/opencv.hpp>
#include "extensions.h"
#include "operations.h"

int main() {
  // open the default camera.
  cv::VideoCapture cap(1);

  // if not success exit the program.
  if (!cap.isOpened()) {
	std::cout << "Cannot open the video camera" << std::endl;
	std::cin.get(); // wait for a key press.
	return -1;
  }

  cv::Mat cameraMatrix, distCoeffs; // Mats to store intrinsic parametres.
  float markerLength = 0.05;
  // Open the xml file for reading.
  cv::FileStorage
	  fs("/Users/jyothivishnuvardhankolla/Desktop/Project-4-Calibration-Augmented-Reality/camera_params.xml",
		 cv::FileStorage::READ);

  // Read camera Matrix from the xml file.
  fs["camera_matrix"] >> cameraMatrix;
  // Read the distortion Matrix from the xml file.
  fs["dist_coeffs"] >> distCoeffs;
  fs.release();

  // Set coordinate system
  cv::Mat objPoints(4, 1, CV_32FC3);
  objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
  objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
  objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
  objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

  int flag = 0;
  while (true) {
	cv::Mat frame;
	bool bsuccess = cap.read(frame);

	cv::Mat gray_frame;
	cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
	// break from the loop if frames cannot be captured properly.
	if (!bsuccess) {
	  std::cout << "Video camera is disconnected" << std::endl;
	  std::cin.get();
	  break;
	}

	cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
	cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary((cv::aruco::DICT_6X6_250));
	cv::aruco::ArucoDetector detector(dictionary, detectorParams);

	std::vector<int> markerIds;
	std::vector<std::vector<cv::Point2f>> markerCorners;
	detector.detectMarkers(frame, markerCorners, markerIds);

	cv::Mat imagecopy;
	frame.copyTo(imagecopy);
	detect_markers(imagecopy, markerCorners, markerIds);
	draw_3d_axes(markerCorners, markerIds, imagecopy, objPoints, cameraMatrix, distCoeffs);
	/*if (!markerIds.empty() && flag==1)
	  perform_homography(imagecopy, markerCorners, markerIds);*/

	cv::imshow("Aruco Detection", imagecopy);
	int k = cv::waitKey(5);
	if (k=='q') {
	  break;
	} else if (k=='v') {
	  flag = !flag;
	} else if (k=='h') {
	  detect_harris_corners(gray_frame, frame);
	  cv::imshow("harris_corners", frame);
	}
  }
  return 0;
}