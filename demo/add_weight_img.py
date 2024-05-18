import os 
import cv2
import numpy
from os import listdir
from os.path import isfile, isdir, join

img_path = r'output/test_img/_2024_05_07_15_49_38'

files = listdir(img_path)
img_canvas = cv2.imread(img_path + "/"+ files[0])

for files in os.listdir(img_path):
    img = cv2.imread(img_path + "/"+ files)
    # print(img.shape[1], img.shape[0])
    img = cv2.resize(img,(img_canvas.shape[1], img_canvas.shape[0]))
    img_canvas = cv2.add(img_canvas, img)

cv2.imwrite(img_path+ "overlapeed"+ ".png",img_canvas)