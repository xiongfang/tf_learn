import cv2
import os
import pathlib
import random


path_root = "E:\tf_learn\BioID_Face\data\BioID-FaceDatabase-V1.2"


data_root = pathlib.Path(path_root)

print(data_root)

all_image_paths = list(data_root.glob('*.pgm'))


for path in all_image_paths:
	filename = path.with_suffix(".jpg")
	print(filename)
	img = cv2.imread(str(path))
	cv2.imwrite(str(filename),img)
