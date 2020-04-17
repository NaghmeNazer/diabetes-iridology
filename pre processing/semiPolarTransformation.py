#imports
import cv2
import numpy as np


def iris_ring_to_rect(pupilCenter, pupilContour, irisContour, Image):
    normalHeight = 201
    normalWidth = 720
    polCol = 0
    theta = 0
    polarImg = np.zeros((normalHeight, normalWidth, Image.shape[2]), dtype=np.uint8)
    while(theta<=359):
        Point_X = pupilCenter[0]
        Point_Y = pupilCenter[1]
        while(cv2.pointPolygonTest(pupilContour, (np.int16(Point_X), np.int16(Point_Y)),False) ==1):
            Point_X += np.cos(theta * np.pi / 180.0)
            Point_Y += np.sin(theta * np.pi / 180.0)

        xptet = np.int16(Point_X - np.cos(theta * np.pi / 180.0))
        yptet = np.int16(Point_Y - np.sin(theta * np.pi / 180.0))

        while (cv2.pointPolygonTest(irisContour, (np.int16(Point_X), np.int16(Point_Y)), False) == 1):
            Point_X += np.cos(theta * np.pi / 180.0)
            Point_Y += np.sin(theta * np.pi / 180.0)


        xitet = np.int16(Point_X - np.cos(theta * np.pi / 180.0))
        yitet = np.int16(Point_Y - np.sin(theta * np.pi / 180.0))

        r = 0.0
        polRow = 0
        while r <= 1:
            polarImg[polRow, polCol,:] = Image[np.int16((1.0-r)*yptet+r*yitet), np.int16((1.0-r)*xptet+r*xitet), :]
            r += 1.0/(normalHeight-1.0)
            polRow += 1
        theta += 360.0/normalWidth
        polCol += 1
    return polarImg