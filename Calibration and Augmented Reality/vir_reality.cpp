/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */

#include<iostream>
#include<opencv2/opencv.hpp>
#include "operations.h"
#include "extensions.h"

int main() {
  // open the default camera.
  cv::VideoCapture cap(1);

  // if not success, exit the program.
  if (!cap.isOpened()) {
	std::cout << "Cannot open the video camera" << std::endl;
	std::cin.get(); // wait for a key press.
	return -1;
  }

  std::string window_name = "Real-Time-Video"; // window to display real-time video.
  cv::namedWindow(window_name);

  cv::Mat cameraMatrix; // object to store camera matrix.
  cv::Mat distortion_coefficient; // object to store distortion-coeffs.
  cv::Mat rotation_vector, translation_vector; // object to store rot, trans matrices.

  // Open the xml file for reading.
  cv::FileStorage
	  fs("/Users/jyothivishnuvardhankolla/Desktop/Project-4-Calibration-Augmented-Reality/camera_params.xml",
		 cv::FileStorage::READ);

  // Read camera Matrix from the xml file.
  fs["camera_matrix"] >> cameraMatrix;
  // Read the distortion Matrix from the xml file.
  fs["dist_coeffs"] >> distortion_coefficient;
  fs.release();

  int darken = 0;

  while (true) {
	cv::Mat frame;
	cv::Mat gray_frame;
	bool bsuccess = cap.read(frame);

	// break from the loop if frames cannot be captured properly.
	if (!bsuccess) {
	  std::cout << "Video camera is disconnected" << std::endl;
	  std::cin.get();
	  break;
	}

	std::vector<cv::Point2f> corners; // Vector to store the corners of the chess board.
	std::vector<cv::Vec3f> point_set; // Vector to store world coordinates.
	cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
	bool found = cv::findChessboardCorners(frame, cv::Size(9, 6), corners);

	if (found) { // checking whether bound is found or not.
	  //std::cout << corners.size() << " " << "first corner is" << corners[0].x << "," << corners[0].y << std::endl;
	  cv::cornerSubPix(gray_frame,
					   corners,
					   cv::Size(11, 11),
					   cv::Size(-1, -1),
					   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
	  get_world_coordinates(point_set, 6, 9, 23.5f);
	  // get rotation, translation matrix of the camera.
	  cv::solvePnP(point_set, corners, cameraMatrix, distortion_coefficient, rotation_vector, translation_vector);
	  display_rot_trans(cameraMatrix, distortion_coefficient, frame); // display extrinsic param in real time.
	  std::vector<cv::Point2f> imagePoints;
	  cv::projectPoints(point_set,
						rotation_vector,
						translation_vector,
						cameraMatrix,
						distortion_coefficient,
						imagePoints);
	  // draw circles for corners and display a 3D axes through origin.
	  cv::circle(frame, imagePoints[0], 15, cv::Scalar(0, 0, 255), 7);
	  cv::circle(frame, imagePoints[8], 15, cv::Scalar(0, 0, 255), 7);
	  cv::circle(frame, imagePoints[45], 15, cv::Scalar(0, 0, 255), 7);
	  cv::circle(frame, imagePoints[53], 15, cv::Scalar(0, 0, 255), 7);
	  cv::drawFrameAxes(frame, cameraMatrix, distortion_coefficient, rotation_vector, translation_vector, 1);
	  draw_house(rotation_vector,
				 translation_vector,
				 cameraMatrix,
				 distortion_coefficient,
				 frame,
				 darken); // Draw a house to the real-world
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
	cv::imshow(window_name, imagecopy);

	int k = cv::waitKey(5);
	if (k=='q') {
	  cv::destroyWindow(window_name);
	  break;
	} else if (k=='r') {
	  darken = !darken;
	}
  }
  return 0;
}