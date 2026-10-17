import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os

# Get directories
script_dir = os.path.dirname(os.path.abspath(__file__))
scenes_dir = os.path.join(script_dir, 'scenes')
models_dir = os.path.join(script_dir, 'models')

# Brand names corresponding to img_query_vector
model_name_vector = [
    'Regular Nesquick',  # 0.jpg  (Product 1)
    'Blue Krave',      # 1.jpg  (Product 2)
    'Orange Krave',        # 11.jpg (Product 3)
    'Jordans',           # 19.jpg (Product 4)
    'Fitness',           # 24.jpg (Product 5)
    'Nesquick Pink',     # 26.jpg (Product 6)
    'Red Coco Pops'          # 25.jpg (Product 7)
]

# Mahalanobis colour similarity helper
def mahalanobis_color_distance(img_color_query, img_color_scene_crop):
    """
    Compare the colour of a scene crop against a query model using Mahalanobis
    distance on mean BGR pixel values.
    Both images are brightness-normalised (divided by their own global mean)
    before comparison to be invariant to per-scene lighting shifts.
    """
    pixels_query = img_color_query.reshape(-1, 3).astype(np.float64)
    pixels_crop  = img_color_scene_crop.reshape(-1, 3).astype(np.float64)

    # Brightness normalisation
    pixels_query /= (pixels_query.mean() + 1e-6)
    pixels_crop  /= (pixels_crop.mean()  + 1e-6)

    mu    = pixels_query.mean(axis=0)
    sigma = np.cov(pixels_query, rowvar=False) + np.eye(3) * 1e-6
    sigma_inv = np.linalg.inv(sigma)

    x = pixels_crop.mean(axis=0)
    diff = x - mu
    dist = np.sqrt(diff@sigma_inv@diff)
    return dist


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


def _compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) of two axis-aligned bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# Difference of parameters between Step A and GHT scripts:
# Step A: MIN_MATCH_COUNT=20 -> changed to 73, COLOR_THRESH=15.0 -> changed to 1.0 with Mahalanobis normalization
# Step B GHT: MIN_MATCH_COUNT=25, MIN_PEAK_VOTES=20, MIN_INLIER_COUNT=20, BIN_SIZE=40, COLOR_THRESH=1.0
# Step B Crop: MIN_MATCH_COUNT=30, MIN_INLIER_COUNT=25, COLOR_THRESH=1.0, MAX_RETRIES=3
def produce_outputs(img_train_vector, img_query_vector,
                    MIN_MATCH_COUNT=25, MIN_PEAK_VOTES=20,
                    MIN_INLIER_COUNT=20, BIN_SIZE=40,
                    COLOR_THRESH=1.0,
                    model_names=None):
    if model_names is None:
        model_names = model_name_vector

    output_dir = os.path.join(script_dir, 'output_scenes')
    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(img_train_vector):
        scene_path = os.path.join(scenes_dir, path)
        img_train_color = cv2.imread(scene_path)
        
        if img_train_color is None:
            print(f"Error: Could not load scene image at {scene_path}")
            continue
            
        img_train = cv2.cvtColor(img_train_color, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp_train, des_train = sift.detectAndCompute(img_train, None)

        img_output = img_train_color.copy()

        print(f"Scene image {i + 1}: {path} ({len(kp_train)} keypoints)")

        produce_query(img_query_vector, sift,
                      kp_train, des_train,
                      img_train_color, img_output,
                      MIN_MATCH_COUNT=MIN_MATCH_COUNT,
                      MIN_PEAK_VOTES=MIN_PEAK_VOTES,
                      MIN_INLIER_COUNT=MIN_INLIER_COUNT,
                      BIN_SIZE=BIN_SIZE,
                      COLOR_THRESH=COLOR_THRESH,
                      model_names=model_names)

        output_filename = os.path.splitext(path)[0] + '_detected_GHT.png'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img_output)
        print(f"\n[Saved annotated scene to: {output_path}]")


def produce_query(img_query_vector, sift,
                  kp_train, des_train,
                  img_train_color, img_output,
                  MIN_MATCH_COUNT=25, MIN_PEAK_VOTES=20,
                  MIN_INLIER_COUNT=20, BIN_SIZE=40,
                  COLOR_THRESH=1.0,
                  model_names=None):
    """
    Step B - Generalized Hough Transform (GHT) with Local Invariant Features:
    1. Model Training: Compute displacement vectors joining model keypoints to their barycenter.
    2. Runtime Voting: Matched scene features cast votes for the barycenter in the scene,
       scaled by the ratio of sizes and rotated by orientation difference.
    3. Peak Extraction & Verification: Peaks in the accumulator represent candidate instances;
       each peak is verified using local RANSAC and Mahalanobis color verification.
    """
    thresh_vector = COLOR_THRESH if isinstance(COLOR_THRESH, (list, tuple, np.ndarray)) \
                    else [COLOR_THRESH] * len(img_query_vector)

    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    h_s, w_s = img_train_color.shape[:2]

    for i, path in enumerate(img_query_vector):
        query_thresh = thresh_vector[i]
        brand_name   = model_names[i] if model_names and i < len(model_names) else path

        query_path = os.path.join(models_dir, path)
        img_query_color = cv2.imread(query_path)

        if img_query_color is None:
            print(f"Error: Could not load model image at {query_path}")
            continue

        img_query = cv2.cvtColor(img_query_color, cv2.COLOR_BGR2GRAY)
        kp_query, des_query = sift.detectAndCompute(img_query, None)

        h_m, w_m = img_query.shape
        # 1. Model Barycenter
        xc_m = w_m / 2.0
        yc_m = h_m / 2.0
        query_corners = np.float32([[0,0],[0,h_m-1],[w_m-1,h_m-1],[w_m-1,0]]).reshape(-1,1,2)

        # 2. Feature Matching
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_query, des_train, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) < MIN_MATCH_COUNT:
            print(f"Product {i+1} ({brand_name}, {path}): 0 instance found ({len(good)} matches).")
            continue

        # 3. GHT Voting
        # Accumulator grid definition
        grid_h = int(np.ceil(h_s / float(BIN_SIZE)))
        grid_w = int(np.ceil(w_s / float(BIN_SIZE)))
        acc = np.zeros((grid_h, grid_w), dtype=np.int32)
        bin_matches = {}

        for m in good:
            km = kp_query[m.queryIdx]
            ks = kp_train[m.trainIdx]

            # Scale and rotation ratios
            scale_ratio = ks.size / (km.size + 1e-6)
            d_theta     = np.radians(ks.angle - km.angle)

            # Vector from model keypoint to barycenter
            dx = xc_m - km.pt[0]
            dy = yc_m - km.pt[1]

            # Rotate and scale vector into scene coordinate space
            cos_t = np.cos(d_theta)
            sin_t = np.sin(d_theta)
            vx = ks.pt[0] + scale_ratio * (dx * cos_t - dy * sin_t)
            vy = ks.pt[1] + scale_ratio * (dx * sin_t + dy * cos_t)

            gx = int(vx // BIN_SIZE)
            gy = int(vy // BIN_SIZE)

            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                acc[gy, gx] += 1
                bin_matches.setdefault((gy, gx), []).append(m)

        # 4. Accumulator Peak Finding & Candidate Pooling
        # Find local peaks: pool matches in 3x3 neighborhood around candidate cells
        candidate_peaks = []
        for gy in range(grid_h):
            for gx in range(grid_w):
                # Count total votes in 3x3 window around (gy, gx)
                window_votes = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = gy + dy, gx + dx
                        if (ny, nx) in bin_matches:
                            window_votes += len(bin_matches[(ny, nx)])

                if window_votes >= MIN_PEAK_VOTES:
                    candidate_peaks.append((window_votes, gy, gx))

        # Sort candidate peaks by vote count descending
        candidate_peaks.sort(key=lambda x: x[0], reverse=True)

        found_instances = []
        detected_boxes  = []

        for window_votes, gy, gx in candidate_peaks:
            # Gather all unique matches in the 3x3 neighborhood
            pooled = {}
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = gy + dy, gx + dx
                    if (ny, nx) in bin_matches:
                        for m in bin_matches[(ny, nx)]:
                            pooled[(m.queryIdx, m.trainIdx)] = m

            peak_matches = list(pooled.values())
            if len(peak_matches) < MIN_PEAK_VOTES:
                continue

            # 5. Local RANSAC on Peak Inliers
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in peak_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_train[m.trainIdx].pt for m in peak_matches]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                continue

            inlier_count = mask.ravel().tolist().count(1)
            if inlier_count < MIN_INLIER_COUNT:
                continue

            dst = cv2.perspectiveTransform(query_corners, M)
            dst = _straighten_bb(dst)

            height_bb = round(np.linalg.norm(dst[0] - dst[1]))
            width_bb  = round(np.linalg.norm(dst[0] - dst[3]))
            x_center  = round(dst[0,0,0] + width_bb  / 2)
            y_center  = round(dst[0,0,1] + height_bb / 2)

            # Sanity check: bounding box dimensions
            MIN_BB_SIZE = 40
            MAX_BB_SIZE = 1200
            if width_bb < MIN_BB_SIZE or height_bb < MIN_BB_SIZE or width_bb > MAX_BB_SIZE or height_bb > MAX_BB_SIZE:
                continue

            x1 = max(0, int(dst[0,0,0]))
            y1 = max(0, int(dst[0,0,1]))
            x2 = min(img_train_color.shape[1], int(dst[3,0,0]))
            y2 = min(img_train_color.shape[0], int(dst[1,0,1]))
            if x2 <= x1 or y2 <= y1:
                continue

            current_box = [x1, y1, x2, y2]

            # 6. Non-Maximum Suppression (avoid duplicate detections)
            is_duplicate = False
            for prev_box in detected_boxes:
                if _compute_iou(current_box, prev_box) > 0.35:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            # 7. Mahalanobis Colour Verification
            scene_crop  = img_train_color[y1:y2, x1:x2]
            dist        = mahalanobis_color_distance(img_query_color, scene_crop)
            colour_ok   = dist <= query_thresh
            colour_info = f"Mahalanobis dist = {dist:.4f} (thresh={query_thresh})"

            if colour_ok:
                found_instances.append(dst)
                detected_boxes.append(current_box)
                instance_id = len(found_instances)

                # Draw bounding box & label on img_output
                img_output = cv2.polylines(
                    img_output, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(
                    img_output, f"P{i+1}: {brand_name} # {instance_id}",
                    (int(dst[0,0,0]), max(20, int(dst[0,0,1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                print(f"  -> Instance {instance_id} found via GHT [{colour_info}]:\n"
                      f"     Position: ({x_center}, {y_center}), Width: {width_bb} px, Height: {height_bb} px, Peak Votes: {window_votes}, Inliers: {inlier_count}")

        count = len(found_instances)
        if count > 0:
            print(f"Product {i+1} ({brand_name}, {path}): {count} instance(s) detected via GHT.")
        else:
            print(f"Product {i+1} ({brand_name}, {path}): 0 instance found.")


# Entry point
if __name__ == '__main__':
    # Settings for GHT Multiple Instance Detection on m1-m5 scenes
    MIN_MATCH_COUNT  = 25
    MIN_PEAK_VOTES   = 15
    MIN_INLIER_COUNT = 15
    BIN_SIZE         = 40

    img_train_vector = ['m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png']
    img_query_vector = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']

    # Per-query Mahalanobis distance thresholds
    COLOR_THRESH = [0.45,   0.92,   1.0,   1.0,   1.0,   0.35,   1.0]

    produce_outputs(img_train_vector, img_query_vector,
                    MIN_MATCH_COUNT=MIN_MATCH_COUNT,
                    MIN_PEAK_VOTES=MIN_PEAK_VOTES,
                    MIN_INLIER_COUNT=MIN_INLIER_COUNT,
                    BIN_SIZE=BIN_SIZE,
                    COLOR_THRESH=COLOR_THRESH,
                    model_names=model_name_vector)
