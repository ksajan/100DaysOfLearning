import cv2 as cv
from cv2 import Laplacian
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('th.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)
lines = cv.HoughLines(edges, 1, np.pi/180, 200)
line2 = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
Laplacian = cv.Laplacian(gray, cv.CV_64FC4)
for line in line2:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv.imshow('img', Laplacian)
cv.waitKey(0)
cv.destroyAllWindows()