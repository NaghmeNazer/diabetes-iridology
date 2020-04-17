import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import os
from scipy.spatial import distance
from semiPolarTransformation import iris_ring_to_rect
import argparse

def my_circle(myc_x, myc_y, myc_radius):
    s = np.linspace(0, 2 * np.pi, 400)
    x = myc_x + myc_radius * np.cos(s)
    y = myc_y + myc_radius * np.sin(s)
    my_circle_out = np.array([x, y]).T
    return my_circle_out

def get_area(ga_snake):
    Maar = np.int32(ga_snake)
    Maar = np.reshape(Maar, (Maar.shape[0], 1, Maar.shape[1]))
    contours = []
    contours.append(Maar)
    ga_cnt = contours[0]
    ga_out = cv2.contourArea(ga_cnt)
    return ga_out, ga_cnt


def draw_last_circle(dlc_cnt):
    lastcontour = np.reshape(dlc_cnt, (dlc_cnt.shape[0], dlc_cnt.shape[2]))
    dlc_lp_center, dlc_lp_radius = cv2.minEnclosingCircle(lastcontour)
    dlc_lp_center = np.int32(dlc_lp_center)
    dlc_lp_radius = np.int32(dlc_lp_radius)

    s = np.linspace(0, 2 * np.pi, 400)
    dlc_x = dlc_lp_center[0] + dlc_lp_radius * np.cos(s)
    dlc_y = dlc_lp_center[1] + dlc_lp_radius * np.sin(s)
    return dlc_lp_center, dlc_lp_radius, dlc_x, dlc_y


def detect_pupil_and_iris(rout_address, is_plot):
    for ii in os.listdir(rout_address):
        print (rout_address+ii)
        if os.path.isdir(rout_address+ii):
            if not os.path.exists(rout_address+ii+"/ans"):
                os.makedirs(rout_address+ii+"/ans")
            if not os.path.exists(rout_address+ii+"/ans_polar"):
                os.makedirs(rout_address+ii+"/ans_polar")
            for jj in os.listdir(rout_address+ii):
                chck = jj.lower()
                if chck.endswith(".jpg") and not(chck.startswith(".")):

                    II = cv2.imread(rout_address+ii+"/"+jj)
                    orginalImage = cv2.imread(rout_address+ii+"/"+jj)

                    unchanged_img = cv2.imread(rout_address+ii+"/"+jj)

                    II = cv2.resize(II, (1280, 853))
                    orginalImage = cv2.resize(orginalImage, (1280, 853))
                    unchanged_img = cv2.resize(unchanged_img, (1280, 853))
                    unchanged_img = cv2.pyrDown(unchanged_img)
                    unchanged_img = cv2.pyrDown(unchanged_img)
                    orginalImage = cv2.GaussianBlur(orginalImage, (5, 5), 0)
                    orginalImage = cv2.pyrDown(orginalImage)
                    orginalImage = cv2.pyrDown(orginalImage)
                    orginalImage = cv2.erode(orginalImage, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
                    orginalImage = cv2.dilate(orginalImage, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

                    HSVimage = cv2.cvtColor(orginalImage, cv2.COLOR_BGR2HSV)
                    value = HSVimage[:, :, 2]

                    treshcons = 40
                    if is_plot:
                        plt.figure(1)
                        plt.imshow(value, cmap='gray')

                    _, threshIris = cv2.threshold(value, treshcons, 255, cv2.THRESH_BINARY)

                    if is_plot:
                        plt.figure(2)
                        plt.imshow(threshIris)

                    II = threshIris
                    II = rgb2gray(II)

                    init = my_circle(orginalImage.shape[1] / 2, orginalImage.shape[0] / 2, 100)
                    snake = active_contour(gaussian(II, 3), init, alpha=0.2, beta=100)

                    if is_plot:
                        fig = plt.figure()
                        ax = fig.add_subplot(121)
                        plt.gray()
                        ax.imshow(II)
                        ax.plot(snake[:, 0], snake[:, 1], '--r', lw=3)
                        ax.plot(init[:, 0], init[:, 1], '--g', lw=3)

                    area, cnt = get_area(snake)


                    while area < 2500:

                        treshcons = treshcons + 10
                        if is_plot:
                            plt.figure(1)
                            plt.imshow(value, cmap='gray')

                        _, threshIris = cv2.threshold(value, treshcons, 255, cv2.THRESH_BINARY_INV)

                        if is_plot:
                            plt.figure(2)
                            plt.imshow(threshIris)

                        II = threshIris
                        II = rgb2gray(II)

                        snake = active_contour(gaussian(II, 3), init, alpha=0.2, beta=100)
                        if is_plot:
                            fig = plt.figure()
                            ax = fig.add_subplot(121)
                            plt.gray()
                            ax.imshow(II)
                            ax.plot(snake[:, 0], snake[:, 1], '--r', lw=3)
                            ax.plot(init[:, 0], init[:, 1], '--g', lw=3)
                            plt.ion()
                            plt.show()
                            plt.pause(1)

                        area, cnt = get_area(snake)

                    # 2nd active contour for S Channel

                    value = HSVimage[:, :, 1]
                    snake2 = active_contour(gaussian(value, 3), snake, alpha=0.2, beta=100)

                    value = HSVimage[:, :, 2]

                    area2, cnt2 = get_area(snake2)

                    if not area > (5 * area2):
                        snake = snake2
                        cnt = cnt2

                    (lp_center, lp_radius, x, y) = draw_last_circle(cnt)

                    value = cv2.medianBlur(value, 5)

                    M = cv2.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    pupil_radius = np.int32(np.sqrt(area / np.pi))

                    if pupil_radius < 40:
                        init2 = my_circle(cx, cy, (2 * pupil_radius))
                    else:
                        init2 = my_circle(cx, cy, (1.4 * pupil_radius))

                    CNV = 0.35
                    area2 = 10**20

                    while area2 > 10 * area:
                        OuterSnake = active_contour(gaussian(value, 3), init2, alpha=-0.1, beta=100, gamma=0.001, convergence=CNV)
                        area2, cnt = get_area(OuterSnake)
                        CNV = CNV * 1.2

                    if is_plot:
                        fig = plt.figure()
                        ax = fig.add_subplot(121)
                        plt.gray()
                        ax.imshow(value)
                        ax.plot(OuterSnake[:, 0], OuterSnake[:, 1], '--r', lw=3)
                        ax.plot(snake2[:, 0], snake2[:, 1], '--g', lw=3)
                        plt.ion()
                        plt.show()
                        plt.pause(1)

                    (li_center, li_radius, x, y) = draw_last_circle(cnt)
                    OuterSnake_circle = np.copy(OuterSnake)
                    diff = li_center[1]+li_radius - value.shape[0]
                    if diff > 0:
                        asd, cnt = get_area(OuterSnake)
                        M = cv2.moments(cnt)
                        badContourCenterX = int(M['m10'] / M['m00'])
                        badContourCenterY = int(M['m01'] / M['m00'])
                        badContourCenter = (badContourCenterX, badContourCenterY)
                        Dist = np.zeros([len(OuterSnake), 1])
                        for d in range(len(OuterSnake)):
                            Dist[d] = distance.euclidean(badContourCenter, (OuterSnake[d, 0], OuterSnake[d, 1]))
                        li_radius = np.min(Dist)

                        s = np.linspace(0, 2 * np.pi, 400)
                        x = badContourCenterX + li_radius * np.cos(s)
                        y = badContourCenterY + li_radius * np.sin(s)
                        OuterSnake_circle[:, 0] = x
                        OuterSnake_circle[:, 1] = y

                        diff = li_center[1] + li_radius - value.shape[0]

                    if diff > 0:
                        diff = diff + 1
                        li_radius = li_radius - diff
                        x = badContourCenterX + li_radius * np.cos(s)
                        y = badContourCenterY + li_radius * np.sin(s)
                        OuterSnake_circle[:, 0] = x
                        OuterSnake_circle[:, 1] = y


                    print ('Iris center', li_center)
                    print ('Iris radious', li_radius)

                    unchanged_img = cv2.cvtColor(unchanged_img, cv2.COLOR_BGR2RGB)
                    if is_plot:
                        ax = fig.add_subplot(122)
                        ax.imshow(unchanged_img)
                        ax.plot(OuterSnake_circle[:, 0], OuterSnake_circle[:, 1], 'r', lw=1)
                        ax.plot(snake2[:, 0], snake2[:, 1], 'g', lw=1)
                        plt.ion()
                        plt.show()
                        plt.pause(1)
                        plt.savefig(rout_address+ii+"/ans/"+jj+"ans.jpg")
                        plt.close()

                    # Ring to Rect Transformation
                    grayImage = value
                    pupilCenter = (lp_center[0], lp_center[1])
                    pupilRadius = np.int(lp_radius)
                    irisCenter = (li_center[0], li_center[1])
                    irisRadius = np.int(li_radius)

                    pupilContour = []
                    pupilContour.append(snake2)
                    pupilContour = np.array(pupilContour, dtype=np.int64)
                    pupilContour = np.reshape(pupilContour, (1, snake2.shape[0], snake2.shape[1]))

                    c, r = cv2.minEnclosingCircle(pupilContour)
                    pupilContours = list(pupilContour)

                    irisContour = []
                    irisContour.append(OuterSnake_circle)
                    irisContour = np.array(irisContour, dtype=np.int64)
                    irisContour = np.reshape(irisContour, (1, OuterSnake_circle.shape[0], OuterSnake_circle.shape[1]))
                    irisContours = list(irisContour)

                    polarImage = iris_ring_to_rect(c, pupilContour, irisContour, orginalImage)

                    if is_plot:
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.imshow(polarImage)
                        plt.ion()
                        plt.show()
                        plt.pause(1)
                    filename = rout_address+ii+'/ans_polar/'+jj.split('.')[0] + '_polarImage.bmp'
                    cv2.imwrite(filename, polarImage)
                    if is_plot:
                        plt.close('all')
                        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parameters for iris and pupil detection')
    parser.add_argument("--visualize", required=True, action='store_true', help='visualize detections')
    parser.add_argument("--root_folder", required=True, help='path to folder of image folders')
    args = parser.parse_args()
    detect_pupil_and_iris(args.root_folder, args.visualize)
    # rout_address = "/Volumes/ELEMENTS/Projects/Iridology/iridology_new/Paper_Publish/Iris_Database_FARABI/Selected_Images_rewised/Control/"
    # is_plot      = False