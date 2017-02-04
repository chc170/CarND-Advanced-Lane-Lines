import argparse
import cv2
import os
import numpy as np


class EdgeFinder:
    def __init__(self, image):
        self.image = image
        self._channel1 = 0
        self._channel2 = 0
        self._channel3 = 0
        self._channel4 = 0
        self._channel5 = 0
        self._channel6 = 0
        self._channel7 = 0
        self._channel8 = 0
        self._channel9 = 0
        self._channel10 = 0
        self._channel11 = 0
        self._channel12 = 0
        self._channel13 = 0
        self._channel14 = 0
        self._filtered_img = None

        def onchangeChannel1(pos):
            self._channel1 = pos
            self._render()

        def onchangeChannel2(pos):
            self._channel2 = pos
            self._render()

        def onchangeChannel3(pos):
            self._channel3 = pos
            self._render()

        def onchangeChannel4(pos):
            self._channel4 = pos
            self._render()

        def onchangeChannel5(pos):
            self._channel5 = pos
            self._render()

        def onchangeChannel6(pos):
            self._channel6 = pos
            self._render()

        def onchangeChannel7(pos):
            self._channel7 = pos
            self._render()

        def onchangeChannel8(pos):
            self._channel8 = pos
            self._render()

        def onchangeChannel9(pos):
            self._channel9 = pos
            self._render()

        def onchangeChannel10(pos):
            self._channel10 = pos
            self._render()

        def onchangeChannel11(pos):
            self._channel11 = pos
            self._render()

        def onchangeChannel12(pos):
            self._channel12 = pos
            self._render()

        def onchangeChannel13(pos):
            self._channel13 = pos
            self._render()

        def onchangeChannel14(pos):
            self._channel14 = pos
            self._render()

        cv2.namedWindow('yellow')
        cv2.namedWindow('white')
        cv2.namedWindow('sobel')

        cv2.createTrackbar('ch1 lower', 'yellow', self._channel1, 255, onchangeChannel1)
        cv2.createTrackbar('ch2 lower', 'yellow', self._channel2, 255, onchangeChannel2)
        cv2.createTrackbar('ch3 lower', 'yellow', self._channel3, 255, onchangeChannel3)
        cv2.createTrackbar('ch1 upper', 'yellow', self._channel4, 255, onchangeChannel4)
        cv2.createTrackbar('ch2 upper', 'yellow', self._channel5, 255, onchangeChannel5)
        cv2.createTrackbar('ch3 upper', 'yellow', self._channel6, 255, onchangeChannel6)
        cv2.createTrackbar('ch4 lower', 'white', self._channel7, 255, onchangeChannel7)
        cv2.createTrackbar('ch5 lower', 'white', self._channel8, 255, onchangeChannel8)
        cv2.createTrackbar('ch6 lower', 'white', self._channel9, 255, onchangeChannel9)
        cv2.createTrackbar('ch4 upper', 'white', self._channel10, 255, onchangeChannel10)
        cv2.createTrackbar('ch5 upper', 'white', self._channel11, 255, onchangeChannel11)
        cv2.createTrackbar('ch6 upper', 'white', self._channel12, 255, onchangeChannel12)
        cv2.createTrackbar('sobel lower', 'sobel', self._channel13, 255, onchangeChannel13)
        cv2.createTrackbar('sobel upper', 'sobel', self._channel14, 255, onchangeChannel14)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('edges')

    def channel1(self):
        return self._channel1

    def channel2(self):
        return self._channel2

    def channel3(self):
        return self._channel3

    def channel4(self):
        return self._channel4

    def channel5(self):
        return self._channel5

    def channel6(self):
        return self._channel6

    def channel5(self):
        return self._channel7

    def channel6(self):
        return self._channel8

    def channel3(self):
        return self._channel9

    def channel4(self):
        return self._channel10

    def channel5(self):
        return self._channel11

    def channel6(self):
        return self._channel12

    def channel5(self):
        return self._channel13

    def channel6(self):
        return self._channel14

    def filteredImage(self):
        return self._filtered_img

    def _render(self):
        #img_yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        #image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        image = self.image

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
        y_threshed = cv2.inRange(
            hls, 
            np.array([self._channel1, self._channel2, self._channel3]), 
            np.array([self._channel4, self._channel5, self._channel6]))
        
        w_threshed = cv2.inRange(
            hls, 
            np.array([self._channel7, self._channel8, self._channel9]), 
            np.array([self._channel10, self._channel11, self._channel12]))

        sobel_threshed = cv2.inRange(
            hls, np.array([self._channel13]), np.array([self._channel14]))

        binary = np.zeros_like(hls[:,:,0])
        binary[(sobel_threshed > 0) | (y_threshed > 0) | (w_threshed > 0)] = 1
        cv2.imshow('filtered', binary)

def main():
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('filename')

    args = parser.parse_args()

    img = cv2.imread(args.filename)

    cv2.imshow('input', img)

    edge_finder = EdgeFinder(img)

    print('({}, {}, {})'.format(edge_finder._channel1, edge_finder._channel2, edge_finder._channel3))
    print('({}, {}, {})'.format(edge_finder._channel4, edge_finder._channel5, edge_finder._channel6))
    print('({}, {}, {})'.format(edge_finder._channel7, edge_finder._channel8, edge_finder._channel9))
    print('({}, {}, {})'.format(edge_finder._channel10, edge_finder._channel11, edge_finder._channel12))
    print('({}, {})'.format(edge_finder._channel13, edge_finder._channel14))



    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()