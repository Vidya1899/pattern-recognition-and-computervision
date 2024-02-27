//
// Created by Jyothi vishnu vardhan Kolla on 2/4/23.
// CS-5330 Spring semester.
//

#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <vector>
#include "csv_util.h"
#include "featureVectors.h"
#include "distanceMetrics.h"
#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace std;

/*
 * Given target Image, feature set, feature vectors file computes the
 feature set of target image, reads feature vector file and indentifies top N images.
 */
int main(int argc, char *argv[]) {
  if (argc!=3) {
	cout << "Invalid number of command line arguements";
	exit(-1);
  }

  char target_image_path[256]; // store the path of target image.
  string featureset; // get the feature set to use.
  char feature_vector_file[256]; // store the path of feature vector file.
  int no_of_matches = atoi(argv[2]); // top N matches to find.

  strcpy(target_image_path, argv[1]);
  //cout << target_image_path << endl;
  //featureset = argv[2];

  // create a window to create buttons.
  string ui_component = "GUI Interface";
  cv::namedWindow(ui_component, 1);

  cvui::init(ui_component);
  cv::Mat frame = cv::Mat(400, 300, CV_8UC3);
  int flag = 0;
  while (true) {
	frame = cv::Scalar(37, 32, 45);
	// multiHistLeftRight button.
	if (cvui::button(frame, 100, 40, "multiHistLeftRight")) {
	  if (flag==0) {
		featureset = "multiHistLeftRight";
		::strcpy(feature_vector_file,
				 "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/multiHistLeftRight.csv");
		flag = 1;
		break;
	  }
	}

	// multiHist Button.
	if (cvui::button(frame, 100, 70, "multiHist")) {
	  if (flag==0) {
		featureset = "multiHist";
		::strcpy(feature_vector_file,
				 "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/multiHistFeaturevectors.csv");
		flag = 1;
		break;
	  }
	}

	// textureHistogram Button.
	if (cvui::button(frame, 100, 100, "textureHist")) {
	  if (flag==0) {
		featureset = "textureHist";
		::strcpy(feature_vector_file,
				 "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/textureHistogram.csv");
		flag = 1;
		break;
	  }
	}

	// squareFeatureVector Button.
	if (cvui::button(frame, 100, 130, "9X9Square")) {
	  if (flag==0) {
		featureset = "square";
		::strcpy(feature_vector_file,
				 "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/featureVectors.csv");
		flag = 1;
		break;
	  }
	}

	// Hist2d button.
	if (cvui::button(frame, 100, 160, "2DHistogram")) {
	  if (flag==0) {
		featureset = "hist2D";
		::strcpy(feature_vector_file,
				 "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/Hist2DfeatureVector.csv");
		flag = 1;
		break;
	  }
	}

	// Hist3d button.
	if (cvui::button(frame, 100, 190, "3DHistogram")) {
	  if (flag==0) {
		featureset = "hist3D";
		::strcpy(feature_vector_file,
				 "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/Hist3DfeatureVector.csv");
		flag = 1;
		break;
	  }
	}

	// Laplacian button.
	if (cvui::button(frame, 100, 220, "LaplacianHistogram")) {
	  if (flag==0) {
		featureset = "LaplacianHist";
		::strcpy(feature_vector_file,
				 "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/LaplacianHistFeatures.csv");
		flag = 1;
		break;
	  }
	}

	// Banana Button
	if (cvui::button(frame, 100, 250, "Get Bananas")) {
	  if (flag==0) {
		featureset = "getbanana";
		::strcpy(feature_vector_file,
				 "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/YellowFeatureVector.csv");
		flag = 1;
		break;
	  }
	}

	// Thrashcan Button
	if (cvui::button(frame, 100, 280, "Get Bluecans")) {
	  if (flag==0) {
		featureset = "getbluecans";
		::strcpy(feature_vector_file,
				 "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/BlueFeatureVector.csv");
		flag = 1;
		break;
	  }
	}
	cvui::update(ui_component);
	cvui::imshow(ui_component, frame);
	if (cv::waitKey(20)==27) {
	  cv::destroyWindow(ui_component);
	}
  }

  vector<char *> filenames; // vector to store filenames.
  vector<vector<float>> data; // vectors to data of feature sets.

  // compute the feature vector of target Image.
  cv::Mat targetImage = cv::imread(target_image_path);
  vector<float> targetImageFeatureVector;

  if (featureset=="square") {
	targetImageFeatureVector = nineXnineSquare(targetImage);
  } else if (featureset=="hist2D") {
	//cout << "computing in hist2D" << endl;
	targetImageFeatureVector = twodHistogram(targetImage);
	cout << targetImageFeatureVector.size();
  } else if (featureset=="hist3D") {
	targetImageFeatureVector = ThreedHistogram(targetImage);
  } else if (featureset=="multiHist") {
	targetImageFeatureVector = multiHistogram(targetImage);
  } else if (featureset=="textureHist") {
	targetImageFeatureVector = colorTexture(targetImage);
  } else if (featureset=="multiHistLeftRight") {
	targetImageFeatureVector = multiHistogramLeftRight(targetImage);
  } else if (featureset=="LaplacianHist") {
	targetImageFeatureVector = LaplaciancolorTexture(targetImage);
  } else if (featureset=="getbanana") {
	targetImageFeatureVector = yellowThresholding(targetImage);
  } else if (featureset=="getbluecans") {
	targetImageFeatureVector = blueThresholding(targetImage);
  }

  // call the function to read and store data from csv_file.
  read_image_data_csv(feature_vector_file, filenames, data);


  /*
   * Caluculate the distance between target and image databases using given distance metric.
   * sort them based on distances and return the top N images.
   */
  vector<pair<string, int>> results;
  vector<pair<string, float>> results2;
  if (featureset=="square") {
	cout << "Matching aquare" << endl;
	results = sum_of_squared_difference(targetImageFeatureVector, data, filenames);
	for (int i = 0; i <= no_of_matches; i++) {
	  cout << results[i].first << ":" << results[i].second << endl;
	  cv::imshow(results[i].first, cv::imread(results[i].first));
	}
  } else if (featureset=="hist2D") {
	cout << "Matching hist2d";
	results2 = histogram_intersection(targetImageFeatureVector, data, filenames);
	cout << results2.size();
	for (int i = 0; i < 4; i++) {
	  cout << results2[i].first << ":" << results2[i].second << endl;
	  cv::imshow(results2[i].first, cv::imread(results2[i].first));
	}
  } else if (featureset=="hist3D") {
	cout << "Matching hist3d";
	results2 = histogram_intersection(targetImageFeatureVector, data, filenames);
	cout << results2.size();
	for (int i = 0; i < 4; i++) {
	  cout << results2[i].first << ":" << results2[i].second << endl;
	  cv::imshow(results2[i].first, cv::imread(results2[i].first));
	}
  } else if (featureset=="multiHist") {
	cout << "Matching top bottom hist";
	results2 = histogram_intersection_for_2histograms(targetImageFeatureVector, data, filenames);
	cout << results2.size();
	for (int i = 0; i < 4; i++) {
	  cout << results2[i].first << ":" << results2[i].second << endl;
	  cv::imshow(results2[i].first, cv::imread(results2[i].first));
	}
  } else if (featureset=="textureHist") {
	cout << "Matching textureHist";
	results2 = histogram_intersection_for_2histograms(targetImageFeatureVector, data, filenames);
	cout << results2.size();
	for (int i = 0; i < 4; i++) {
	  cout << results2[i].first << ":" << results2[i].second << endl;
	  cv::imshow(results2[i].first, cv::imread(results2[i].first));
	}
  } else if (featureset=="multiHistLeftRight") {
	cout << "Matching left right";
	results2 = histogram_intersection_for_2histograms(targetImageFeatureVector, data, filenames);
	cout << results2.size();
	for (int i = 0; i < 10; i++) {
	  cout << results2[i].first << ":" << results2[i].second << endl;
	  cv::imshow(results2[i].first, cv::imread(results2[i].first));
	}
  } else if (featureset=="LaplacianHist") {
	cout << "Laplacian filter" << endl;
	results2 = entopyDistance(targetImageFeatureVector, data, filenames);
	cout << results2.size();
	for (int i = 0; i < 5; i++) {
	  cout << results2[i].first << ":" << results2[i].second << endl;
	  cv::imshow(results2[i].first, cv::imread(results2[i].first));
	}
  } else if (featureset=="getbanana") {
	cout << "getbanana" << endl;
	results2 = histogram_intersection(targetImageFeatureVector, data, filenames);
	cout << results2.size();
	for (int i = 0; i < 50; i++) {
	  cout << results2[i].first << ":" << results2[i].second << endl;
	  cv::imshow(results2[i].first, cv::imread(results2[i].first));
	}
  } else if (featureset=="getbluecans") {
	results = sum_of_squared_difference(targetImageFeatureVector, data, filenames);
	for (int i = 0; i < 10; i++) {
	  cout << results[i].first << ":" << results[i].second << endl;
	  cv::imshow(results[i].first, cv::imread(results[i].first));
	}
  }

  if (cv::waitKey(0)=='q') {
	for (int i = 0; i < 4; i++) {
	  cv::destroyWindow(results2[i].first);
	}
  };
  return (0);
}
