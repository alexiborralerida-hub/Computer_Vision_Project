from IPython import display
from IPython import display
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
scenes_dir = os.path.join(script_dir, 'scenes')
models_dir = os.path.join(script_dir, 'models')

model_name_array = ['Regular Nesquick', 'Orange Krave', 'Blue Krave', 'Jordans', 'Fitness', 'Nesquick Pink', 'Coco Pops']

def produce_outputs(img_train_vector, img_query_vector, MIN_MATCH_COUNT = 20):
    for i, path in enumerate(img_train_vector):
        scene_path = os.path.join(scenes_dir, path)
        img_train = cv2.imread(scene_path)
        b_train, g_train, r_train = cv2.split(img_train)
        img_train_gray = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY)
        img_train_hsv = cv2.cvtColor(img_train, cv2.COLOR_BGR2HSV)
        sift = cv2.SIFT_create()
        kp_train = sift.detect(img_train_gray)
        kp_train, des_train = sift.compute(img_train_gray, kp_train)
        
        des_train_final = []
        patch_size = 2
        for k, keypoint in enumerate(kp_train):

            # keypoint coordinates
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])

            # extract local patch
            x1 = max(0, x - patch_size // 2)
            x2 = min(img_train.shape[1], x + patch_size // 2)

            y1 = max(0, y - patch_size // 2)
            y2 = min(img_train.shape[0], y + patch_size // 2)

            patch = img_train_hsv[y1:y2, x1:x2]

            # compute HSV histogram
            hist = cv2.calcHist(
                [patch],
                [0, 1],          # H and S channels
                None,
                [8, 8],          # bins
                [0, 180, 0, 256]
            )

            # normalize histogram
            hist = cv2.normalize(hist, hist).flatten()

            # concatenate SIFT + color histogram
            combined = np.concatenate([des_train[k], hist])

            des_train_final.append(combined)

        des_train_final = np.array(des_train_final)

        print('\nScene image {}: {}'.format(i+1,path))
        print('------------------------------------------------')
        produce_query(img_query_vector, sift, kp_train, des_train_final, img_train, MIN_MATCH_COUNT)

        

def produce_query(img_query_vector, sift, kp_train, des_train_final, img_train, MIN_MATCH_COUNT):
    for i, path in enumerate(img_query_vector):
        model_path = os.path.join(models_dir, path)
        img_query = cv2.imread(model_path)
        b_query, g_query, r_query = cv2.split(img_query)
        img_query_gray = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
        img_query_hsv = cv2.cvtColor(img_query, cv2.COLOR_BGR2HSV)
        kp_query = sift.detect(img_query_gray)
        kp_query, des_query = sift.compute(img_query_gray, kp_query)
        
        des_query_final = []
        patch_size = 2
        for k, keypoint in enumerate(kp_query):

            # keypoint coordinates
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])

            # extract local patch
            x1 = max(0, x - patch_size // 2)
            x2 = min(img_query.shape[1], x + patch_size // 2)

            y1 = max(0, y - patch_size // 2)
            y2 = min(img_query.shape[0], y + patch_size // 2)

            patch = img_query_hsv[y1:y2, x1:x2]

            # compute HSV histogram
            hist = cv2.calcHist(
                [patch],
                [0, 1],          # H and S channels
                None,
                [8, 8],          # bins
                [0, 180, 0, 256]
            )

            # normalize histogram
            hist = cv2.normalize(hist, hist).flatten()

            # concatenate SIFT + color histogram
            combined = np.concatenate([des_query[k], hist])

            des_query_final.append(combined)

        des_query_final = np.array(des_query_final)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 300)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_query_final,des_train_final,k=2)
        good = []

        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append(m)

        if len(good)>=MIN_MATCH_COUNT:
            src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
            h,w = img_query_gray.shape
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
            #img_train = cv2.polylines(img_train,[np.int32(dst)],True,(0, 0, 255),3, cv2.LINE_AA)
            
            print('Product {} ({}): 1 instance found: \n  Instance 1 position: ({},{}), width: {} px, height: {} px with {} matches'.format(i+1,model_name_array[i],x_center,y_center,width_bb,height_bb, len(good)))
            print('------------------------------------------------')
        else:
            print('Product {} ({}): 0 instance found, rejected with {} matches'.format(i+1, model_name_array[i], len(good)))
            print('------------------------------------------------')
           

MIN_MATCH_COUNT = 70
img_train_vector = ['e1.png', 'e2.png', 'e3.png', 'e4.png', 'e5.png']

img_query_vector = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']
produce_outputs(img_train_vector, img_query_vector, MIN_MATCH_COUNT)