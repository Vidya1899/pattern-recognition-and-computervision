1.Operating System used: Macos
2.Ide Used: Clion


Instructions for Running Executables:

* All the Configurations related to Project are setup in the File CMakeLists.txt, inorder to exegute the code The project
must be opened in the Clion Ide which is already configured with an c++ 14 compiler and add all the .cpp,.h files
into this project and then create and build a new project.

Once this process is completed just run the main.cpp file by passing the path of the image database and type of feature vector to use
as command line arguements and exegute the file, this populates the feature vector and imagepaths into its related csv_file as given in the program.

**** Type of feature vector to pass in command line to compute and store resultant feature vectors in csv file:
1. square: To store the feature vector in csv file for baseline matching using 9X9 square in middle of image.
2. hist2D: Compute 2d-histogram to get feature vector and store in related csv file.
3. hist3D: Compute 3d-histogram to get feature vector and store in related csv file.
4. multiHist: Compute two histograms one being tophalf and other being bottom half of the images and store the
              feature vector in related csv file.
5. multiHistLeftRight: Compute two histograms one being lefthalf and other being righthalf of the images and store the
                                     feature vector in related csv file.
6. textureHist: compute two histograms, one computes a 3d histogram on given image, and compute another histogram by
applying gradient magnitude on given image and then compute 3d histogram and store them as feature vectors in related
csv file.
7. LaplacianHist: compute two histograms, one computes a 3d histogram on given image, and compute another histogram by
                  applying Laplacian Image on given image and then compute 3d histogram and store them as feature vectors in related
                  csv file.

8. yellowThresholding: computes and stores the feature vector related to yellow spatial frequency and stores it in
related csv which can be used to identify bananas.

9. blueThresholding: computes and stores the feature vector related to Blue spatial frequency and stores it in
                     related csv which can be used to identify bananas.

Exeguting Matching.cpp:

The goal of this code file is to perform Image Matching using a targrt image and retrive the top matches for this image
from the database, the command line to pass for this file is the path of the target image and no of top N matches
to show.

The Gui designed for this allows the user to select the type of matching they want to perform, the instructions
for testing the GUI is as shown in this video.

https://drive.google.com/file/d/1UuEBb3m2AwyCe4LkpZjt6xZBvkaIij3f/view?usp=share_link

*** CMAKE configuration for the project.
cmake_minimum_required(VERSION 2.8)
project(main.cpp)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(main.cpp main.cpp featureVectors.cpp csv_util.cpp utils.cpp utils.h)
add_executable(Matching.cpp Matching.cpp featureVectors.cpp csv_util.cpp distanceMetrics.cpp utils.cpp utils.h)
target_link_libraries(main.cpp ${OpenCV_LIBS})
target_link_libraries(Matching.cpp ${OpenCV_LIBS})

Structure:

vidDisplay.cpp      --> is the main program of the project.
filters.cpp         --> Containts the functions for implementing filters for the first 9 tasks(except gaussian blur).
effects.cpp         --> contains the function to create negative Image of given rgb color Image.
BlurFilters.cpp     --> contains the code all the BLurFilters implemented as part of this project including
                        5X5 gaussian filter.