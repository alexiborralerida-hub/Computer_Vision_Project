import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
scenes_dir = os.path.join(script_dir, 'scenes')
models_dir = os.path.join(script_dir, 'models')

# ── Mahalanobis colour similarity helper ──────────────────────────────────────
def mahalanobis_color_distance(img_color_query, img_color_scene_crop):
    """
    Compare the colour of a scene crop against a query model using Mahalanobis
    distance on mean BGR pixel values.

    Steps:
      1. Reshape both images to (N_pixels, 3) arrays of BGR values.
      2. From the QUERY: compute mean vector µ (shape 3,) and
         covariance matrix Σ (shape 3×3) across all pixels.
      3. From the SCENE CROP: compute mean vector x (shape 3,).
      4. Mahalanobis distance = sqrt( (x−µ)ᵀ · Σ⁻¹ · (x−µ) )

    A small distance → similar average colour.
    A large distance → different colour (box variant to reject).

    Returns:
        distance (float) – lower is more similar.
    """
    # Flatten to (N, 3) float arrays
    pixels_query = img_color_query.reshape(-1, 3).astype(np.float64)
    pixels_crop  = img_color_scene_crop.reshape(-1, 3).astype(np.float64)

    # Mean and covariance from the query (reference) image
    mu    = pixels_query.mean(axis=0)                   # shape (3,)
    sigma = np.cov(pixels_query, rowvar=False)           # shape (3,3)

    # Regularise covariance to avoid singular matrix
    sigma += np.eye(3) * 1e-6

    sigma_inv = np.linalg.inv(sigma)

    # Mean colour of the scene crop
    x = pixels_crop.mean(axis=0)                        # shape (3,)

    diff = x - mu
    dist = np.sqrt(diff @ sigma_inv @ diff)
    return dist


def produce_outputs(img_train_vector, img_query_vector,
                    MIN_MATCH_COUNT=20, COLOR_THRESH=15.0):
    for i, path in enumerate(img_train_vector):
        # Load scene in colour; keep a greyscale copy for SIFT
        scene_path = os.path.join(scenes_dir, path)
        img_train_color = cv2.imread(scene_path)
        
        if img_train_color is None:
            print(f"Error: Could not load scene image at {scene_path}")
            continue
            
        img_train = cv2.cvtColor(img_train_color, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp_train = sift.detect(img_train)
        kp_train, des_train = sift.compute(img_train, kp_train)

        print('\nScene image {}: {}'.format(i + 1, path))
        produce_query(img_query_vector, sift,
                      kp_train, des_train,
                      img_train, img_train_color,
                      MIN_MATCH_COUNT, COLOR_THRESH)


def produce_query(img_query_vector, sift,
                  kp_train, des_train,
                  img_train, img_train_color,
                  MIN_MATCH_COUNT, COLOR_THRESH):
    for i, path in enumerate(img_query_vector):
        # Load query in colour; keep a greyscale copy for SIFT
        query_path = os.path.join(models_dir, path)
        img_query_color = cv2.imread(query_path)
        
        if img_query_color is None:
            print(f"Error: Could not load model image at {query_path}")
            continue
            
        img_query = cv2.cvtColor(img_query_color, cv2.COLOR_BGR2GRAY)

        kp_query = sift.detect(img_query)
        kp_query, des_query = sift.compute(img_query, kp_query)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_query, des_train, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_train[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = img_query.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # ── Straighten bounding box corners ───────────────────────────────
            # Left
            if dst[0, 0, 0] < dst[1, 0, 0]:
                dst[1, 0, 0] = dst[0, 0, 0]
            else:
                dst[0, 0, 0] = dst[1, 0, 0]

            # Right
            if dst[2, 0, 0] > dst[3, 0, 0]:
                dst[3, 0, 0] = dst[2, 0, 0]
            else:
                dst[2, 0, 0] = dst[3, 0, 0]

            # Up
            if dst[0, 0, 1] < dst[3, 0, 1]:
                dst[3, 0, 1] = dst[0, 0, 1]
            else:
                dst[0, 0, 1] = dst[3, 0, 1]

            # Down
            if dst[2, 0, 1] > dst[1, 0, 1]:
                dst[1, 0, 1] = dst[2, 0, 1]
            else:
                dst[2, 0, 1] = dst[1, 0, 1]

            height_bb = round(np.linalg.norm(dst[0] - dst[1]))
            width_bb  = round(np.linalg.norm(dst[0] - dst[3]))
            x_center  = round(dst[0, 0, 0] + width_bb  / 2)
            y_center  = round(dst[0, 0, 1] + height_bb / 2)

            # ── Mahalanobis colour verification ───────────────────────────────
            # Crop the matched bounding-box region from the colour scene image
            x1 = max(0, int(dst[0, 0, 0]))
            y1 = max(0, int(dst[0, 0, 1]))
            x2 = min(img_train_color.shape[1], int(dst[3, 0, 0]))
            y2 = min(img_train_color.shape[0], int(dst[1, 0, 1]))

            colour_ok = False
            if x2 > x1 and y2 > y1:
                scene_crop = img_train_color[y1:y2, x1:x2]
                dist = mahalanobis_color_distance(img_query_color, scene_crop)
                colour_ok = dist <= COLOR_THRESH
                colour_info = 'Mahalanobis dist = {:.2f}'.format(dist)
            else:
                colour_ok = True   # crop failed, skip colour check
                colour_info = 'crop failed – colour check skipped'

            if colour_ok:
                img_train_color = cv2.polylines(
                    img_train_color, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
                print('Product {} ({}): 1 instance found [{}]: \n  Instance 1 position: ({},{}), '
                      'width: {} px, height: {} px'.format(
                          i + 1, path, colour_info, x_center, y_center, width_bb, height_bb))
            else:
                print('Product {} ({}): 0 instance found '
                      '(geometry matched but colour rejected – {})'.format(i + 1, path, colour_info))
            print('------------------------------------------------')

        else:
            print("Product {} ({}): 0 instance found".format(i + 1, path))
            print('------------------------------------------------')


# ── Entry point ───────────────────────────────────────────────────────────────
MIN_MATCH_COUNT = 100
# Mahalanobis distance threshold (in units of std-devs of the query colour).
# Lower = stricter colour match.  Start around 10–20 and tune.
COLOR_THRESH = 1

img_train_vector = ['e1.png', 'e2.png', 'e3.png', 'e4.png', 'e5.png']
img_query_vector = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']

produce_outputs(img_train_vector, img_query_vector, MIN_MATCH_COUNT, COLOR_THRESH)