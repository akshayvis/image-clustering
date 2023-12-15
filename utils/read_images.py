#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:37:20 2023

@author: aviswanathan
"""

import cv2
from PIL import Image

def is_image_file(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except (IOError, SyntaxError):
        return False

def read_image(img_path):
    if is_image_file(img_path):
        img_content = cv2.imread(img_path)
        img_content = cv2.cvtColor(img_content, cv2.COLOR_BGR2RGB)
    else:
        return []
    return img_content