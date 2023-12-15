#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:01:19 2023

@author: aviswanathan
"""

import face_recognition
import numpy as np
from sklearn.metrics import silhouette_score

def detect_faces(img_content):
    face_coords = face_recognition.face_locations(img_content)
    
    img_face_vector = face_recognition.face_encodings(img_content, face_coords)
    
    return face_coords, img_face_vector


def cluster_faces_kmeans(img_face_vectors,k, n_iter):

    def find_cluster(points, centroids):
        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        clust_dist = np.argmin(distances, axis=0)
        return clust_dist


    def find_new_centroids(img_vector, closest, centroids):
        closest = find_cluster(img_vector, centroids)
        new_centroid = []
        for k in range(centroids.shape[0]):
            new_centroid.append(img_vector[closest==k].mean(axis=0))
        return np.array(new_centroid)
    
    img_vectors = np.zeros(128)
    max_silhouette_score = 0

    for i in range(len(img_face_vectors)):
        img_vectors = np.vstack((img_vectors,img_face_vectors[i][0]))
            
    img_vectors = np.delete(img_vectors, 0, 0) 
    
    init_centers_list = []
    
    for _ in range(n_iter):
        while True:
            random_rows_indices = sorted(np.random.choice(img_vectors.shape[0], size=k, replace=False))
            if random_rows_indices not in init_centers_list:
                init_centers_list.append(random_rows_indices)
                break
        
        # Use np.vstack to stack the selected rows vertically
        c = np.vstack(img_vectors[random_rows_indices])
    
        # c = np.vstack((img_vectors[0],img_vectors[5],img_vectors[11],img_vectors[16],img_vectors[21],img_vectors[27]))
    
        c_extended = c[: , np.newaxis, :]
    
    
        new_c_extended = c_extended.copy()    
    
        i=0
        while(True):
            i+=1
            c_extended = new_c_extended.copy()        
            c = find_new_centroids(img_vectors, find_cluster(img_vectors, c), c)
            new_c_extended = c[: , np.newaxis, :]
            new_clusters = np.argmin(np.sqrt(((img_vectors - c_extended)**2).sum(axis=2)), axis=0)
            if((c_extended == new_c_extended).all()):
                break
            
            
        score = silhouette_score(img_vectors, new_clusters)
        if score > max_silhouette_score:
            max_silhouette_score = score
            final_clusters = new_clusters
    
    return img_vectors, final_clusters