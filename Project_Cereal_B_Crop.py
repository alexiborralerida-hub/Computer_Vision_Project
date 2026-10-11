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

def _filter_keypoints_outside_poly(kp, des, polygon):
    """
    Keeps only keypoints and descriptors whose (x, y) coordinates lie OUTSIDE polygon.
    Cropping out the detected instance from the scene pool.
    """
    poly = np.int32(polygon)
    kept_kp  = []
    kept_des = []
    for k, d in zip(kp, des):
        pt = (int(k.pt[0]), int(k.pt[1]))
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
# Difference of parameters between Step A and GHT scripts:
# Step A: MIN_MATCH_COUNT=20 -> changed to 73, COLOR_THRESH=15.0 -> changed to 1.0 with Mahalanobis normalization
# Step B GHT: MIN_MATCH_COUNT=25, MIN_PEAK_VOTES=20, MIN_INLIER_COUNT=20, BIN_SIZE=40, COLOR_THRESH=1.0
# Step B Crop: MIN_MATCH_COUNT=30, MIN_INLIER_COUNT=25, COLOR_THRESH=1.0, MAX_RETRIES=3
def produce_outputs(img_train_vector, img_query_vector,
                    MIN_MATCH_COUNT=30, MIN_INLIER_COUNT=15,
                    COLOR_THRESH=1.0, MAX_RETRIES=3,
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
                      MIN_INLIER_COUNT=MIN_INLIER_COUNT,
                      COLOR_THRESH=COLOR_THRESH,
                      MAX_RETRIES=MAX_RETRIES,
                      model_names=model_names)

        output_filename = os.path.splitext(path)[0] + '_detected_crop.png'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img_output)
        print(f"\n[Saved annotated scene to: {output_path}]")


def produce_query(img_query_vector, sift,
                  kp_train, des_train,
                  img_train_color, img_output,
                  MIN_MATCH_COUNT=30, MIN_INLIER_COUNT=25,
                  COLOR_THRESH=1.0, MAX_RETRIES=3,
                  model_names=None):
    """
    Step B - Sequential RANSAC with Keypoint Suppression:
    For each query model, repeatedly detect instances and 'crop' (suppress) their
    keypoints from the scene so subsequent instances of the SAME product can be matched.
    """
    thresh_vector = COLOR_THRESH if isinstance(COLOR_THRESH, (list, tuple, np.ndarray)) \
                    else [COLOR_THRESH] * len(img_query_vector)

    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

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

        h, w = img_query.shape
        query_corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

        # Working copy of scene keypoints , suppressed iteratively as instances are found
        kp_scene  = list(kp_train)
        des_scene = des_train.copy()

        found_instances = []
        retries = 0

        while des_scene is not None and len(kp_scene) >= MIN_MATCH_COUNT:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des_query, des_scene, k=2)

            good = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good) < MIN_MATCH_COUNT:
                break  # No more matching clusters for this product

            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                break

            inlier_count = mask.ravel().tolist().count(1)
            if inlier_count < MIN_INLIER_COUNT:
                break

            dst = cv2.perspectiveTransform(query_corners, M)
            dst = _straighten_bb(dst)

            height_bb = round(np.linalg.norm(dst[0] - dst[1]))
            width_bb  = round(np.linalg.norm(dst[0] - dst[3]))
            x_center  = round(dst[0,0,0] + width_bb  / 2)
            y_center  = round(dst[0,0,1] + height_bb / 2)

            # Sanity check: reject degenerate bounding boxes
            MIN_BB_SIZE = 40
            MAX_BB_SIZE = 1200
            if width_bb < MIN_BB_SIZE or height_bb < MIN_BB_SIZE or width_bb > MAX_BB_SIZE or height_bb > MAX_BB_SIZE:
                # Suppress degenerate keypoints and retry
                kp_scene, des_scene = _filter_keypoints_outside_poly(kp_scene, des_scene, dst)
                retries += 1
                if retries > MAX_RETRIES:
                    break
                continue

            # Mahalanobis colour verification
            x1 = max(0, int(dst[0,0,0]))
            y1 = max(0, int(dst[0,0,1]))
            x2 = min(img_train_color.shape[1], int(dst[3,0,0]))
            y2 = min(img_train_color.shape[0], int(dst[1,0,1]))

            if x2 > x1 and y2 > y1:
                scene_crop = img_train_color[y1:y2, x1:x2]
                dist       = mahalanobis_color_distance(img_query_color, scene_crop)
                colour_ok  = dist <= query_thresh
                colour_info = f"Mahalanobis dist = {dist:.4f} (thresh={query_thresh})"
            else:
                colour_ok   = False
                colour_info = "crop out of bounds, treated as rejection"

            if colour_ok:
                found_instances.append(dst)
                instance_id = len(found_instances)

                # Draw bounding box & label on img_output
                img_output = cv2.polylines(
                    img_output, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(
                    img_output, f"P{i+1}: {brand_name} # {instance_id}",
                    (int(dst[0,0,0]), max(20, int(dst[0,0,1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                print(f"  -> Instance {instance_id} found [{colour_info}]:\n"
                      f"     Position: ({x_center}, {y_center}), Width: {width_bb} px, Height: {height_bb} px, Inliers: {inlier_count}")

                # Suppress (crop out) detected keypoints from scene to search for the next instance
                kp_scene, des_scene = _filter_keypoints_outside_poly(kp_scene, des_scene, dst)
                retries = 0  # Reset retries upon success
            else:
                # Wrong color: suppress false region and retry
                kp_scene, des_scene = _filter_keypoints_outside_poly(kp_scene, des_scene, dst)
                retries += 1
                if retries > MAX_RETRIES:
                    break

        count = len(found_instances)
        if count > 0:
            print(f"Product {i+1} ({brand_name}, {path}): {count} instance(s) detected.")
        else:
            print(f"Product {i+1} ({brand_name}, {path}): 0 instance found.")


# Entry point
if __name__ == '__main__':
    # Settings for Multiple Instance Detection on m1-m5 scenes
    MIN_MATCH_COUNT  = 30
    MIN_INLIER_COUNT = 25
    MAX_RETRIES      = 3

    img_train_vector = ['m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png']
    img_query_vector = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']

    # Per-query Mahalanobis distance thresholds
    COLOR_THRESH = [0.41,   0.75,   1.0,   1.0,   1.0,   0.35,   1.0]

    produce_outputs(img_train_vector, img_query_vector,
                    MIN_MATCH_COUNT=MIN_MATCH_COUNT,
                    MIN_INLIER_COUNT=MIN_INLIER_COUNT,
                    COLOR_THRESH=COLOR_THRESH,
                    MAX_RETRIES=MAX_RETRIES,
                    model_names=model_name_vector)
