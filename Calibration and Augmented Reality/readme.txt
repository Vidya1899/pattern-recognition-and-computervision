Project Members:
1. Jyothi Vishnu Vardhan Kolla-> Neuid:002752854
2. Vidya Ganesh->NUID:002766414

* Number of time-travel days used -> 2 days.
1.Operating System used: Mac Os
2.Ide Used: CLion


Instructions for Running Executables:

* All the Configurations related to the Project are setup in the File CMakeLists.txt, inorder to execute the code The project
must be opened in the Clion Ide which is already configured with a C++ 17 compiler and add all the .cpp,.h files
into this project and then create and build a new project.

Structure of the Project:
1. main.cpp -> File that detects the chessboard in real time and saves the calibration images on pressing 's' key
and once count reaches 5 or more we perform camera calibration and store in XML file, press 'q' to quit the window.
2. operations.cpp  -> Contains all the helper functions required for calibration and projecting virtual image on to the target.
3. vir_reality.cpp   -> reads the camera calibration parametres from xml file and projects virtual object in real time video
   press 'q to quit the window'.
4. Aruco.cpp  -> contains the code that creates aruco markers and detects the aruco markers.

   * For demonstration we also added the feature where this takes a path to single image and displays the classification result
   in a window.
5. extensions.cpp -> contains the helper functions used for extension tasks.

Extension-1: Integrated aruco module with our opencv system.
* run aruco.cpp file to test this feature and show it few aruco markers and it will detect them.

Extension-2: From the previous setup we got our system running with multiple targets.

* Run vir_reality.cpp and test it by showing camera and aruco markers.

Extension-3: Not only add a virtual object but something to make it not like a target anymore.
https://drive.google.com/file/d/1Eho-cn3SE0oc_cz02fhy-aB39SMrVdOD/view?usp=sharing

Extension-4: Testing calibration with different cameras.

Extension-5: used opencv aruco markers to overlay an virtual image on to the marker.
Run arucoModule.py
https://drive.google.com/file/d/1PYm8uKytTqb3aaOXf6DTIgMAF56eoddn/view?usp=sharing


cmake_minimum_required(VERSION 3.1)
project(main.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(aruco REQUIRED)
include_directories(${aruco_INCLUDE_DIRS})
add_executable(main.cpp main.cpp operations.cpp operations.h)
add_executable(vir_reality.cpp vir_reality.cpp operations.cpp operations.h extensions.cpp extensions.h)
add_executable(Aruco.cpp Aruco.cpp operations.cpp operations.h extensions.cpp extensions.h)
target_link_libraries(main.cpp ${OpenCV_LIBS})
target_link_libraries(vir_reality.cpp ${OpenCV_LIBS})
target_link_libraries(Aruco.cpp ${OpenCV_LIBS} ${aruco_LIBS})





