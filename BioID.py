import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt



def get_eye_pos(eye_file):
	with open(eye_file,"r") as f:
		eye = f.readline()
		eye = f.readline()
		pos_list = eye.split('\t')
	
		LX = int(pos_list[0])
		LY = int(pos_list[1])
		RX = int(pos_list[2])
		RY = int(pos_list[3])
		return [[LX,LY],[RX,RY]]


path_root = "E:\tf_learn\BioID_Face\data\BioID-FaceDatabase-V1.2"


img_file = os.path.join(path_root,"BioID_0000.pgm")
eye_file = os.path.join(path_root,"BioID_0000.eye")

img = cv2.imread(img_file)

eye = get_eye_pos(eye_file)

cv2.circle(img,eye[0],2,(0,0,255))
cv2.circle(img,eye[1],2,(0,0,255))

cv2.imshow("a",img)

cv2.waitKey()





