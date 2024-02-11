################## library installation #################

# pip install opencv-python
# pip install numpy 


# media_dir = D:\Seivan's Workshop\Opencv Tutorial\tutorial_media

#################################################################### Chapter 2
################## show image preview ###################

import cv2 as cv 


image = cv.imread("D:/Seivan's Workshop/Opencv Tutorial/tutorial_media/lenna.png")

cv.imshow("image", image)
cv.waitKey(0)



################## show video preview ###################

cap = cv.VideoCapture(1)

while True:
    
    success, frame = cap.read()
    
    cv.imshow("video_1", frame)
    
    if cv.waitKey(30) & 0xFF == ord("q"):
        break
    



#################################################################### Chapter 3
################### Create blank image ##################

import cv2 as cv 
import numpy as np


blank_image_black = np.zeros((500,500,3), dtype="uint8")  # 0.0 float  ,  0 uint8

blank_image_white = np.ones((500,500,3), dtype="uint8")




cv.imshow("blank_image_black", blank_image_black)
cv.imshow("blank_image_white", blank_image_white)
cv.waitKey(0)




################# Line, Rectangle, Circle ###############

cv.line(blank_image_black, (100,100), (250,250), (255,0,0), 3)   # BGR 

cv.rectangle(blank_image_black, (300,350), (450,500), (0,255,0), 1)

cv.circle(blank_image_black, (100,100), 50, (0,0,255), 3)


cv.imshow("rectangle", blank_image_black)
cv.waitKey(0)



############### Writing a text on images  ###############

text = "welcome home"

cv.putText(blank_image_white, text, (100,100), cv.FONT_HERSHEY_COMPLEX, 1, (100,100,255))

cv.imshow("text_image", blank_image_white)
cv.waitKey(0)




#################################################################### Chapter 4
################# Resizing and Cropping #################

import cv2 as cv 
import numpy as np 


img = cv.imread("D:/Seivan's Workshop/Opencv Tutorial/tutorial_media/lenna.png")

img_resized = cv.resize(img, (256,256))

img_resized1 = cv.resize(img_resized, (720,720))


cv.imshow("image", img)
# cv.imshow("img_resized", img_resized)
cv.imshow("img_resized1", img_resized1)


cv.waitKey(0)



img_cropped = img[100:400,150:300]  # [y:y+h , x:x+w]

cv.imshow("copped", img_cropped)
cv.imshow("image",img)
cv.waitKey(0)



#################################################################### Chapter 5
################# Change image channels #################

# BGR

rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

ycbcr = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)


cv.imshow("original_img",img)
# cv.imshow("gray",gray)
cv.imshow("hsv",hsv)
cv.imshow("ycbcr",ycbcr)
cv.waitKey(0)



#################### Bluring images #####################

img_blur2 = cv.GaussianBlur(img, (7,7), 2)
img_blur4 = cv.GaussianBlur(img, (17,17), 4)
img_blur8 = cv.GaussianBlur(img, (27,27), 8)

cv.imshow("original_img",img)
cv.imshow("img_blur2",img_blur2)
cv.imshow("img_blur4",img_blur4)
cv.imshow("img_blur8",img_blur8)
cv.waitKey(0)



################# Canny edge detection ##################


img_edges = cv.Canny(img, 200, 200)


cv.imshow("original_img",img)
cv.imshow("img_edges",img_edges)
cv.waitKey(0)





#################################################################### Chapter 6
################# Erosion and Dilation ##################


kernel = np.ones((3,3),dtype="uint8")

img_dilated = cv.dilate(img_edges, kernel, iterations=3)

img_eroded = cv.erode(img_dilated, kernel , iterations=3)


cv.imshow("img_edges",img_edges)
cv.imshow("img_dilated",img_dilated)
cv.imshow("img_eroded",img_eroded)
cv.waitKey(0)




#################################################################### Chapter 7
#################### Image rotating #####################


import cv2 as cv 
import numpy as np 


img = cv.imread("D:/Seivan's Workshop/Opencv Tutorial/tutorial_media/lenna.png")

img_rotated_90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

img_rotated_180 = cv.rotate(img, cv.ROTATE_180)

img_rotated_270 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)



cv.imshow("lenna", img)
# cv.imshow("lenna_90", img_rotated_90)
# cv.imshow("lenna_180", img_rotated_180)
cv.imshow("lenna_270", img_rotated_270)

cv.waitKey(0)






#################################################################### Chapter 8
#################### Image Stacking #####################


img1 = cv.imread("D:/Seivan's Workshop/Opencv Tutorial/tutorial_media/lenna.png")

img2 = cv.imread("D:/Seivan's Workshop/Opencv Tutorial/tutorial_media/bird_resized.jpg")

vstack = cv.vconcat([img1,img2])

hstack = cv.hconcat([img1,img2])


np_vstack = np.vstack((img1,img2))

np_hstack = np.hstack((img1,img2))


cv.imshow("vstack", vstack)
cv.imshow("hstack", hstack)

cv.waitKey(0)



#################################################################### Chapter 9
#################### Image contours #####################

img_shapes = cv.imread("D:/Seivan's Workshop/Opencv Tutorial/tutorial_media/shapes.png")

img_shapes_gray = cv.cvtColor(img_shapes, cv.COLOR_BGR2GRAY)

val , img_threshold = cv.threshold(img_shapes_gray, 150, 255, cv.THRESH_BINARY)

cont , hier = cv.findContours(img_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

draw_cont_shapes = cv.drawContours(img_shapes, cont, -1, (255,0,0), thickness=4)

x,y,w,h =cv.boundingRect(cont[2])

cv.rectangle(img_shapes,(x,y),(x+w,y+h), (255,255,0), thickness=4)


for i in range(len(cont)):
    x,y,w,h =cv.boundingRect(cont[i])
    
    shape = img_shapes[y:y+h, x:x+w]
    
    cv.imshow("shape", shape)
    cv.waitKey(0)



#################################################################### Chapter 10
#################### Face Detection #####################


faceCascade = cv.CascadeClassifier("D:/Seivan's Workshop/Opencv Tutorial/tutorial_media/haarcascade_frontalface_default.xml")

lenna = cv.imread("D:/Seivan's Workshop/Opencv Tutorial/tutorial_media/lenna.png")

face = faceCascade.detectMultiScale(lenna, 1.2, 7)

for (x,y,w,h) in face :
    cv.rectangle(lenna,(x,y),(x+w,y+h), (0,255,0), thickness=2)
    
cv.imshow("face detection", lenna)
cv.waitKey(0)



cap = cv.VideoCapture(1)

while True:
    _,frame = cap.read()
    
    frame_resized = cv.resize(frame, (600,400))
    
    face = faceCascade.detectMultiScale(frame_resized, 1.2, 7)
    
    for (x,y,w,h) in face :
        cv.rectangle(frame_resized,(x,y),(x+w,y+h), (0,255,0), thickness=2)
        
    
    cv.imshow("face detection", frame_resized)
    cv.waitKey(1)
        
    
























