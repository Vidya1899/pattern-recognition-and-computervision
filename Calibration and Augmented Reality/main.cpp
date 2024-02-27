/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "operations.h"
#include <opencv2/core/version.hpp>

int main() {
  // open the default camera.
  cv::VideoCapture cap(1);

  // if not success, exit program.
  if (!cap.isOpened()) {
	std::cout << "Cannot open the video camera" << std::endl;
	std::cin.get(); // Wait for a key press.
	return -1;
  }

  std::string window_name = "Real-Time-Video";
  cv::namedWindow(window_name);

  std::vector<std::vector<cv::Vec3f>> point_list; // 2D vector to store world coordinates.
  std::vector<std::vector<cv::Point2f>> corners_list; // 2D vector to store corner points.

  int frame_captured = 0;
  int frames_captured_till_now = 0;// keeps the count of no.of frames captured.
  cv::Mat cameraMatrix, distortion_coefficients;
  cv::Mat first_frame;
  // Initialize camera and distortion matrices.
  initialize_camera_distortion_mats(cameraMatrix, distortion_coefficients, first_frame);
  cap.read(first_frame);
  std::vector<cv::Mat> rotation_vector, translation_vector;

  while (true) {
	cv::Mat frame; // Mat object to capture each frame of video.
	cv::Mat gray_frame; // container to store greyscaled frame;
	cv::Mat valid_frame; // container to store valid frame containing chessboard.
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
	  valid_frame = gray_frame.clone();
	}

	// draw the corners in the chessboard.
	cv::drawChessboardCorners(frame, cv::Size(9, 6), corners, found);

	// if the required number of images for calibration are obtained.
	if (frame_captured >= 5 && (frames_captured_till_now!=frame_captured)) {
	  perform_calibration(point_list,
						  corners_list,
						  cameraMatrix,
						  distortion_coefficients,
						  rotation_vector,
						  translation_vector,
						  frame);
	  frames_captured_till_now++;
	}

	cv::imshow(window_name, frame);
	int k = cv::waitKey(5);
	if (k=='q') {
	  std::cout << cameraMatrix << " " << std::endl;
	  std::cout << distortion_coefficients << " " << std::endl;
	  save_calibration(cameraMatrix, distortion_coefficients);
	  cv::destroyWindow(window_name);
	  break;
	} else if (k=='s') { // Save the corner pixels and its respective world coordinates.
	  if (!valid_frame.empty()) {
		corners_list.push_back(corners);
		get_world_coordinates(point_set, 6, 9, 23.5f);
		point_list.push_back(point_set);
		save_frame(frame);
		frame_captured++;
		//std::cout<<check_validity(point_list, corners_list)<<std::endl;
	  }
	}
  }
  return 0;
}
