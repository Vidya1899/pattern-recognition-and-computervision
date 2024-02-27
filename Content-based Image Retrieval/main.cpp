//
// Created by Jyothi vishnu vardhan Kolla on 2/4/23.
// CS-5330 Spring semester.
//
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <vector>
#include "featureVectors.h"
#include "csv_util.h"
#include "utils.h"
using namespace std;

/*
  Given a directory, feature set on the command line, computes the
  feature vector of each image in the file using given feature set and
  stores them in a CSV file.
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  string featureType = argv[2];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // check for sufficient arguments
  if (argc < 3) {
	printf("usage: %s <directory path>\n", argv[0]);
	exit(-1);
  }

  //printf("%s\n", argv[2]);

  // get the directory path
  strcpy(dirname, argv[1]);
  //printf("Processing directory %s\n", dirname );


  // open the directory
  dirp = opendir(dirname);
  if (dirp==NULL) {
	printf("Cannot open directory %s\n", dirname);
	exit(-1);
  }

  // loop over all the files in the image file listing
  while ((dp = readdir(dirp))!=NULL) {

	// check if the file is an image
	if (strstr(dp->d_name, ".jpg") ||
		strstr(dp->d_name, ".png") ||
		strstr(dp->d_name, ".ppm") ||
		strstr(dp->d_name, ".tif")) {

	  //printf("processing image file: %s\n", dp->d_name);

	  // build the overall filename
	  strcpy(buffer, dirname);
	  strcat(buffer, "/");
	  strcat(buffer, dp->d_name);

	  //printf("full path name: %s\n", buffer);
	  vector<float> featureVector;

	  // compute the feature vector of each image.
	  cv::Mat src = cv::imread(buffer); // Read the image.
	  // stroring feature vectors in csv file.
	  char filename_square_filter[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/featureVectors.csv";
	  char filename_Hist2D[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/Hist2DfeatureVector.csv";
	  char filename_Hist3D[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/Hist3DfeatureVector.csv";
	  char filename_multihistogram[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/multiHistFeaturevectors.csv";
	  char filename_texturehistogram[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/textureHistogram.csv";
	  char filename_multihistLeftRight[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/multiHistLeftRight.csv";
	  char filename_multihistLaplacian[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/LaplacianHistFeatures.csv";
	  char filename_yellowThresholding[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/YellowFeatureVector.csv";
	  char filename_blueThresholding[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-2:Content-BasedImageRetrieval/content-based-image-retrieval/BlueFeatureVector.csv";

	  if (featureType=="square") {
		featureVector = nineXnineSquare(src); // compute the feature vector.
		append_image_data_csv(filename_square_filter, buffer, featureVector, 0);
	  } else if (featureType=="hist2D") {
		//cout << "computing in hist2D" << endl;
		featureVector = twodHistogram(src);
		append_image_data_csv(filename_Hist2D, buffer, featureVector, 0);
	  } else if (featureType=="hist3D") {
		featureVector = ThreedHistogram(src);
		append_image_data_csv(filename_Hist3D, buffer, featureVector, 0);
	  } else if (featureType=="multiHist") {
		featureVector = multiHistogram(src);
		append_image_data_csv(filename_multihistogram, buffer, featureVector, 0);
	  } else if (featureType=="multiHistLeftRight") {
		featureVector = multiHistogramLeftRight(src);
		append_image_data_csv(filename_multihistLeftRight, buffer, featureVector, 0);
	  } else if (featureType=="textureHist") {
		featureVector = colorTexture(src);
		append_image_data_csv(filename_texturehistogram, buffer, featureVector, 0);
	  } else if (featureType=="LaplacianHist") {
		featureVector = LaplaciancolorTexture(src);
		append_image_data_csv(filename_multihistLaplacian, buffer, featureVector, 0);
	  } else if (featureType=="yellowThresholding") {
		featureVector = yellowThresholding(src);
		cout << "appending data";
		append_image_data_csv(filename_yellowThresholding, buffer, featureVector, 0);
	  } else if (featureType=="blueThresholding") {
		featureVector = blueThresholding(src);
		//cout << "appending data";
		append_image_data_csv(filename_blueThresholding, buffer, featureVector, 0);
	  }
	}
  }

  printf("Terminating\n");

  return (0);
}


