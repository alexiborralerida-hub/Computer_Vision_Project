import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
def produce_outputs(img_train_vector, img_query_vector, MIN_MATCH_COUNT = 20):
    for i, path in enumerate(img_train_vector):
        img_train = cv2.imread('scenes/'+path,0)
        sift = cv2.SIFT_create()
        kp_train = sift.detect(img_train)
        kp_train, des_train = sift.compute(img_train, kp_train)
        print('\nScene image {}: {}'.format(i+1,path))
        produce_query(img_query_vector, sift, kp_train, des_train, img_train, MIN_MATCH_COUNT)

        

def produce_query(img_query_vector, sift, kp_train, des_train, img_train, MIN_MATCH_COUNT):
    for i, path in enumerate(img_query_vector):
        img_query = cv2.imread('models/'+path,0)
        kp_query = sift.detect(img_query)
        kp_query, des_query = sift.compute(img_query, kp_query)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_query,des_train,k=2)
        good = []

        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
            h,w = img_query.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
    
            #Left
            if dst[0,0,0] < dst[1,0,0]:
                dst[1,0,0] = dst[0,0,0]
            else: 
                dst[0,0,0] = dst[1,0,0]

            #Right
            if dst[2,0,0] > dst[3,0,0]:
                dst[3,0,0] = dst[2,0,0]
            else: 
                dst[2,0,0] = dst[3,0,0]    

            #Up
            if dst[0,0,1] < dst[3,0,1]:
                dst[3,0,1] = dst[0,0,1]
            else: 
                dst[0,0,1] = dst[3,0,1]

            #Down
            if dst[2,0,1] > dst[1,0,1]:
                dst[1,0,1] = dst[2,0,1]
            else: 
                dst[2,0,1] = dst[1,0,1] 

            height_bb = round(np.linalg.norm(dst[0]-dst[1]))
            width_bb = round(np.linalg.norm(dst[0]-dst[3]))
            x_center = round(dst[0,0,0] + width_bb/2)
            y_center = round(dst[0,0,1] + height_bb/2)
            # Drawing the bounding box
            img_train = cv2.polylines(img_train,[np.int32(dst)],True,(0, 0, 255),3, cv2.LINE_AA)
            
            print('Product {}: 1 instance found: \n  Instance 1 position: ({},{}), width: {} px, height: {} px'.format(i+1,x_center,y_center,width_bb,height_bb))
            print('------------------------------------------------')
        else:
            print("Product {}: 0 instance found".format(i+1) )
            print('------------------------------------------------')
           

MIN_MATCH_COUNT = 110
img_train_vector = ['e1.png', 'e2.png', 'e3.png', 'e4.png', 'e5.png']

img_query_vector = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']
produce_outputs(img_train_vector, img_query_vector, MIN_MATCH_COUNT)