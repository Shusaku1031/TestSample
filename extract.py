import wx
import cv2
import random
import numpy as np
import pandas as pd
import copy
pos = []
img_adr = 'crack.jpg'

img_filt = cv2.imread(img_adr)
#img_filt = cv2.resize(img_filt,dsize=None, fx=0.5, fy=0.5)
src = copy.copy(img_filt)
print(img_adr)

min_table = 10
max_table = 150
diff_table = max_table - min_table
lut = np.arange(256,dtype = 'uint8')

for i in range(0,min_table):
    lut[i] = 0
for i in range(min_table,max_table):
    lut[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    lut[i] = 255

img_median = cv2.medianBlur(img_filt,25)

cv2.imshow('',img_filt)
cv2.waitKey(0)
cv2.imshow('',img_median)
cv2.waitKey(0)

#cv2.imwrite('free-median.jpg',img_median)

img2gray = cv2.cvtColor(img_filt,cv2.COLOR_BGR2GRAY)
img_median2gray = cv2.cvtColor(img_median,cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(img2gray,img_median2gray)
diff = cv2.LUT(diff,lut)

cv2.imshow('',diff)
cv2.waitKey(0)
cv2.imwrite('free-diff.jpg',diff)

ret,thresh = cv2.threshold(diff,40,255,cv2.THRESH_BINARY)
dilate = cv2.dilate(thresh,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations = 1)
nlabels,labelimg,contours,gosc = cv2.connectedComponentsWithStats(dilate)

            #image_area = contours[0][2] * contours[0][3]

print(nlabels)
for d in range(1,nlabels,1):
    x,y,w,h,size = contours[d]
    image_area = w * h
    _x = x #- 100
    _y = y #- 100
    _w = w #+ 100
    _h = h #+ 100

    if _x >= 0:
        x = _x
                
    if _y >= 0:
        y = _y

    if (x+_w) <= img_filt.shape[1]:
        w = _w
                
    if (y+_h) <= img_filt.shape[0]:
        h = _h
                #size = w*h

    if size > 200 and image_area > size:
        #print(image_area)
        #print(size)
        #print("n:%d,x:%d,y:%d,w:%d,h:%d"%(d,contours[d][0],contours[d][1],contours[d][2],contours[d][3]))
        pos.append([x,y,w,h])

        
number = len(pos)

for i in range(number):
    drew = cv2.rectangle(img_filt,(pos[i][0],pos[i][1]),(pos[i][0]+pos[i][2],pos[i][1]+pos[i][3]),(0,0,255),2)

cv2.imshow('',drew)
cv2.waitKey(0)
#cv2.imwrite('free-detect.jpg',drew)

color = []
nlabels, labelImage = cv2.connectedComponents(dilate)

for i in range(1,nlabels + 1):
    color.append(np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)]))

height, width = src.shape[:2]

for y in range(0,height):
    for x in range(0,width):
        if labelImage[y, x] > 0:
            src[y, x] = color[labelImage[y, x]]
        else:
            src[y, x] = [0, 0, 0]

cv2.imshow('',src)
cv2.waitKey(0)
#cv2.imwrite('free-colored.jpg', src)