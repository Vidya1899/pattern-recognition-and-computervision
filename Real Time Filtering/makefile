# MacOS compiler
CC = /Applications/Xcode.app/Contents/Developer/usr/bin/g++

# Make it the C++ compiler too
CXX = $(CC)

# OSX include paths for MacPorts
CFLAGS = -Wc++11-extensions -std=c++11 -I/opt/homebrew/Cellar/opencv/4.7.0_1/include/opencv4 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF

# Make these the C++ flags too
CXXFLAGS = $(CFLAGS)

# Library paths, update to wwhere your openCV libraries are stored
# these settings work for macports
#LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.7.0/include/opencv4/ -L/opt/homebrew/Cellar/opencv/4.7.0/lib/
LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.7.0_1/lib/# opencv libraries are here

# opencv libraries and MacOS frameworks
# these settings work for macOS and macports
LDLIBS = -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect

#LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -lIlmImf -lgs -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lIlmImf

# Location to write the executables
BINDIR = ./bin

# rules for making various executables
all : img vid

img : imgDisplay.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	
vid: vidDisplay.o filter.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

#
#face: faceDetect.o filters.o
#	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
#
#imod: imgMod.o
#	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
#
#cartoon: cartoonize.o filters.o
#	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
#
#macbeth: macbeth.o
#	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~



