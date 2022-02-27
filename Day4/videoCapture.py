import cv2 as cv

url = 'rtsp://admin:admin%40123@223.255.244.81:554/cam/realmonitor?channel=18&subtype=1'

try:
    cap = cv.VideoCapture(url)
except:
    print('Error: Cannot open the video')
