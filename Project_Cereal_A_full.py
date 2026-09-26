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

    Both images are brightness-normalised (divided by their own global mean)
    before comparison, making the distance invariant to per-scene lighting
    differences while preserving hue/saturation ratios between channels.

    Returns:
        distance (float) – lower is more similar.
    """
    # Flatten to (N, 3) float arrays
    pixels_query = img_color_query.reshape(-1, 3).astype(np.float64)
    pixels_crop  = img_color_scene_crop.reshape(-1, 3).astype(np.float64)

    # ── Brightness normalisation ───────────────────────────────────────────────
    # Divide each image by its own overall mean pixel value so that both are
    # mapped to ≈1.0 average brightness, cancelling per-scene lighting shifts
    # without discarding colour ratios (hue/saturation information is kept).
    pixels_query /= (pixels_query.mean() + 1e-6)
    pixels_crop  /= (pixels_crop.mean()  + 1e-6)

    # Mean and covariance from the brightness-normalised query image
    mu    = pixels_query.mean(axis=0)          # shape (3,)
    sigma = np.cov(pixels_query, rowvar=False)  # shape (3,3)

    # Regularise covariance to avoid singular matrix
    sigma += np.eye(3) * 1e-6

    sigma_inv = np.linalg.inv(sigma)

    # Mean colour of the brightness-normalised scene crop
    x = pixels_crop.mean(axis=0)               # shape (3,)

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


def _filter_keypoints_outside_poly(kp, des, polygon):
    """
    Return (kp_kept, des_kept) — keypoints whose position lies OUTSIDE `polygon`.
    `polygon` is a (4,1,2) float32 array as returned by perspectiveTransform.
    """
    poly = np.int32(polygon)  # (4,1,2)
    kept_kp  = []
    kept_des = []
    for k, d in zip(kp, des):
        pt = (int(k.pt[0]), int(k.pt[1]))
        # pointPolygonTest returns > 0 if inside, 0 on edge, < 0 outside
        if cv2.pointPolygonTest(poly, pt, False) < 0:
            kept_kp.append(k)
            kept_des.append(d)
    if kept_des:
        return kept_kp, np.array(kept_des, dtype=np.float32)
    return [], None


def _straighten_bb(dst):
    """Straighten bounding-box corners in-place and return dst."""
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
    return dst


def produce_query(img_query_vector, sift,
                  kp_train, des_train,
                  img_train, img_train_color,
                  MIN_MATCH_COUNT, COLOR_THRESH,
                  MAX_RETRIES=3):
    """
    For each query model:
      1. Run SIFT matching against the full scene keypoint pool.
      2. If a geometric match is found but Mahalanobis rejects it (wrong colour),
         remove the scene keypoints inside that bounding box and retry (up to
         MAX_RETRIES times) so the correct colour variant can be found.
    """
    # COLOR_THRESH can be a scalar or a per-query list
    thresh_vector = COLOR_THRESH if isinstance(COLOR_THRESH, (list, tuple, np.ndarray)) \
                    else [COLOR_THRESH] * len(img_query_vector)

    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    for i, path in enumerate(img_query_vector):
        query_thresh = thresh_vector[i]
        # Load query in colour; keep a greyscale copy for SIFT
        query_path = os.path.join(models_dir, path)
        img_query_color = cv2.imread(query_path)

        if img_query_color is None:
            print(f"Error: Could not load model image at {query_path}")
            continue

        img_query = cv2.cvtColor(img_query_color, cv2.COLOR_BGR2GRAY)
        kp_query = sift.detect(img_query)
        kp_query, des_query = sift.compute(img_query, kp_query)

        h, w = img_query.shape
        query_corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

        # Working copies of the scene keypoint pool — shrunk on each colour rejection
        kp_scene  = list(kp_train)
        des_scene = des_train.copy()

        found      = False
        attempt    = 0

        while attempt <= MAX_RETRIES and not found:
            attempt += 1
            if des_scene is None or len(kp_scene) < MIN_MATCH_COUNT:
                break   # not enough scene keypoints left to match

            flann   = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des_query, des_scene, k=2)

            good = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good) <= MIN_MATCH_COUNT:
                break   # not enough inliers even before colour check

            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                break

            dst = cv2.perspectiveTransform(query_corners, M)
            dst = _straighten_bb(dst)

            height_bb = round(np.linalg.norm(dst[0] - dst[1]))
            width_bb  = round(np.linalg.norm(dst[0] - dst[3]))
            x_center  = round(dst[0,0,0] + width_bb  / 2)
            y_center  = round(dst[0,0,1] + height_bb / 2)

            # ── Sanity check: reject degenerate bounding boxes ─────────────────
            # A valid box must be at least 50×50 px; anything smaller is a bad
            # homography and should be discarded as a false match.
            MIN_BB_SIZE = 50
            if width_bb < MIN_BB_SIZE or height_bb < MIN_BB_SIZE:
                print('  [skip] Degenerate bbox ({}x{} px) – discarding match'.format(
                      width_bb, height_bb))
                break   # no point retrying; homography is fundamentally broken

            # ── Mahalanobis colour verification ───────────────────────────────
            x1 = max(0, int(dst[0,0,0]))
            y1 = max(0, int(dst[0,0,1]))
            x2 = min(img_train_color.shape[1], int(dst[3,0,0]))
            y2 = min(img_train_color.shape[0], int(dst[1,0,1]))

            if x2 > x1 and y2 > y1:
                scene_crop = img_train_color[y1:y2, x1:x2]
                dist       = mahalanobis_color_distance(img_query_color, scene_crop)
                colour_ok  = dist <= query_thresh
                colour_info = 'Mahalanobis dist = {:.4f} (thresh={})'.format(dist, query_thresh)
            else:
                # Crop region is outside the image bounds — treat as a rejection
                # so the retry loop can try a different scene region.
                colour_ok   = False
                colour_info = 'crop out-of-bounds – treated as rejection'

            if colour_ok:
                found = True
                img_train_color = cv2.polylines(
                    img_train_color, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)
                print('Product {} ({}): 1 instance found [{}]: \n'
                      '  Instance 1 position: ({},{}), width: {} px, height: {} px'.format(
                          i+1, path, colour_info, x_center, y_center, width_bb, height_bb))
            else:
                # ── Colour rejected: remove matched region from the scene pool ──
                # This forces the next attempt to find a different region
                attempt_tag = '' if attempt == 1 else f' (attempt {attempt})'
                print('  [retry{}] Colour rejected at ({},{}) [{}] – removing region and retrying…'.format(
                      attempt_tag, x_center, y_center, colour_info))
                kp_scene, des_scene = _filter_keypoints_outside_poly(
                    kp_scene, des_scene, dst)

        if not found:
            print('Product {} ({}): 0 instance found'.format(i+1, path))
        print('------------------------------------------------')


# ── Entry point ───────────────────────────────────────────────────────────────
MIN_MATCH_COUNT = 73

img_train_vector = ['e1.png', 'e2.png', 'e3.png', 'e4.png', 'e5.png']
img_query_vector = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']

# Per-query Mahalanobis distance threshold — one value per entry in img_query_vector.
# Index:         0      1      2      3      4      5      6
#             0.jpg  1.jpg 11.jpg 19.jpg 24.jpg 26.jpg 25.jpg
# Look at the printed distances when running and tune each value to sit just
# above the true-positive distances and below the false-positive distances.
COLOR_THRESH = [0.45,   0.92,   1.0,   1.0,   1.0,   0.28,   1.0]

produce_outputs(img_train_vector, img_query_vector, MIN_MATCH_COUNT, COLOR_THRESH)