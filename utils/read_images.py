#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:37:20 2023

@author: aviswanathan
"""

import cv2

def read_image(img_path):
    
    img_content = cv2.imread(img_path)
    img_content = cv2.cvtColor(img_content, cv2.COLOR_BGR2RGB)

    return img_content