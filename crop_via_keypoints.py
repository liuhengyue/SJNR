from PIL import Image
import matplotlib.pyplot as plt
import json
from pprint import pprint
import os
from test.py import getTestFiles
image_name = '/home/herny/Research/SJNR/3217_1.png'
json_file = os.path.splitext(image_name)[0] + '_keypoints.json'
parsed_json = json.load(open(json_file))

im = plt.imread(image_name)
implot = plt.imshow(im)
img = Image.open(image_name)
num_people = len(parsed_json['people'])
box = {}
for i in range(num_people):
	# 18 keypoints boundingbox by 2, 5, 8, 11 2-Rshoulder
	keypoints_x = []
	keypoints_y = []
	for j in [2, 5, 11, 8, 2]:
		keypoints_x.append(parsed_json['people'][i]['pose_keypoints_2d'][j*3])
		keypoints_y.append(parsed_json['people'][i]['pose_keypoints_2d'][j*3 + 1])
	img_cropped = img.crop((min(keypoints_x), min(keypoints_y), max(keypoints_x), max(keypoints_y)))
	print os.path.splitext(image_name)[0] + '_cropped.png'
	img_cropped.save(os.path.splitext(image_name)[0] + '_cropped.png')
	plt.scatter(keypoints_x, keypoints_y, c='r')
	plt.plot(keypoints_x, keypoints_y, c='r')
	# break


plt.show()

def crop_images():
	pass