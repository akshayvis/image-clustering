#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:42:22 2023

@author: aviswanathan
"""

from utils.read_images import read_image
from utils.face_operations import detect_faces
from utils.face_operations import cluster_faces_kmeans
import os
import pandas as pd


def main():
    
    #Images are ignored in github using .gitignore
    img_dir = 'imgs'

    df_columns=['img_name','img_path','img_content','img_face','face_vector','cluster']
    
    img_data = []
    
    print('Started: Image Read and Face Detection')
    
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img_content = read_image(img_path)
        if len(img_content)>0:
            img_face, face_vector = detect_faces(img_content)
            img_data.append([img_name,img_path,img_content,img_face,face_vector,None])
    
    print('Completed: Image Read and Face Detection')
    
    df = pd.DataFrame(img_data, columns=df_columns)
    
    print('Started: Image Clustering')
    
    img_vectors, clusters = cluster_faces_kmeans(df['face_vector'].tolist(),6,20)
    
    print('Completed: Image Clustering')
    
    df['cluster'] = clusters
    
    result_images = [[] for i in range(6)]
    
    for i, row in df.iterrows():
        result_images[row['cluster']].append(row['img_name'])
        
    print('Clustering Results:')

    for image_cluster in  result_images:
        print(image_cluster)


if __name__ == "__main__":
    main()