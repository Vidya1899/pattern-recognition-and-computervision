# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
import cv2.aruco as aruco
import numpy as np
import os

def findArucoMarkers(img, markerSize =6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    # arucoDict = aruco.Dictionary_get(key)
    arucoDict = cv2.aruco.getPredefinedDictionary(key)
    # arucoParam = aruco.DetectorParameters_create()
    # API update
    arucoParam = cv2.aruco.DetectorParameters()
    bbox, ids , rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bbox)
    return [bbox, ids]

def augmentAruco(bbox, id, img, imgAug, drawId=True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0],  [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgOut = img + imgOut
    return imgOut

def main():
    cap = cv2.VideoCapture(1)
    imgAug = cv2.imread("Marker/red.jpg")

    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)

        # Looping through all the markers and augment each one
        if len(arucoFound[0])!=0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                img = augmentAruco(bbox, id, img, imgAug)
                #print(bbox)
                print(id)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ =="__main__":
    main()

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
