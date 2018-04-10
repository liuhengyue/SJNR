import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pprint import pprint
import os, errno, json
from test import getTestFiles
tf.app.flags.DEFINE_string('image_path', None, 'Path to image path / image test folder')
tf.app.flags.DEFINE_string('crop_path', '/home/henry/Research/cropped/', 'Path to cropped image path')
FLAGS = tf.app.flags.FLAGS
def main(_):
	image_path = FLAGS.image_path
	crop_path = FLAGS.crop_path
	# if the dir does not exist, create one
	try:
	    os.makedirs(crop_path)
	except OSError as e:
	    if e.errno != errno.EEXIST:
	        raise
	# crop images
	if os.path.isdir(image_path):
		image_paths = getTestFiles(image_path)
		pprint(image_paths)
		for image_path in image_paths['test']:
			crop_images(image_path, crop_path)
	else:	
		crop_images(image_path, crop_path)

def crop_images(image_path, crop_path):
	# get the keypoints from json
	json_file = os.path.splitext(image_path)[0] + '_keypoints.json'
	parsed_json = json.load(open(json_file))

	im = plt.imread(image_path)
	implot = plt.imshow(im)
	img = Image.open(image_path)
	num_people = len(parsed_json['people'])
	box = {}
	for i in range(num_people):
		# 18 keypoints boundingbox by 2, 5, 8, 11 2-Rshoulder
		keypoints_x = []
		keypoints_y = []
		for j in [2, 5, 11, 8, 2]:
			keypoints_x.append(parsed_json['people'][i]['pose_keypoints_2d'][j*3])
			keypoints_y.append(parsed_json['people'][i]['pose_keypoints_2d'][j*3 + 1])
		# pprint(keypoints_x)
		# pprint(keypoints_y)
		# missing body parts - continue looking for the next person
		if 0 in keypoints_x or 0 in keypoints_y:
			continue
		img_cropped = img.crop((0.9*min(keypoints_x), min(keypoints_y), 1.1*max(keypoints_x), 0.9*max(keypoints_y)))
		print crop_path + os.path.splitext(os.path.basename(image_path))[0] + '_' + str(i) + '_cropped.png'
		img_cropped.save(crop_path + os.path.splitext(os.path.basename(image_path))[0] + '_' + str(i) + '_cropped.png')
		plt.scatter(keypoints_x, keypoints_y, c='r')
		plt.plot(keypoints_x, keypoints_y, c='r')
		break


	# plt.show()
	plt.savefig(crop_path + os.path.splitext(os.path.basename(image_path))[0] + '_boxed.png')

if __name__ == '__main__':
	tf.app.run(main=main)