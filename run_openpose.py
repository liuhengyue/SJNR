# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:55:44 2018

@author: henry
"""

from subprocess import call
from glob import glob
data_dir = '/media/henry/data/Research/player_jersey_number/'
sub_dirs = glob(data_dir + "*/")
sub_dirs = sub_dirs[:-1] # remove negative
for sub_dir in sub_dirs:
    call(["openpose", "--image_dir", sub_dir, "--write_json", sub_dir])