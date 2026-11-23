import numpy as np
import cv2
import os

# Get directories
script_dir = os.path.dirname(os.path.abspath(__file__))
scenes_dir = os.path.join(script_dir, 'scenes')
models_dir = os.path.join(script_dir, 'models')

# Brand names corresponding to img_query_vector
model_name_vector = [
    'Regular Nesquick',  # 0.jpg  (Product 1)
    'Blue Krave',        # 1.jpg  (Product 2)
    'Orange Krave',      # 11.jpg (Product 3)
    'Jordans',           # 19.jpg (Product 4)
    'Fitness',           # 24.jpg (Product 5)
    'Nesquick Pink',     # 26.jpg (Product 6)
    'Red Coco Pops'      # 25.jpg (Product 7)
]


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
    dist = np.sqrt(diff @ sigma_inv @ diff)
    return dist


def _filter_keypoints(kp_list, des_array, polygon=None, remove_indices=None):
    """
    Suppresses keypoints by explicit scene index and/or spatial polygon containment.
    Guarantees monotonic forward progress on retries.
    """
    remove_set = set(remove_indices) if remove_indices is not None else set()
    poly = np.int32(polygon) if polygon is not None else None

    kept_kp = []
    kept_des = []

    for idx, (k, d) in enumerate(zip(kp_list, des_array)):
        if idx in remove_set:
            continue
        if poly is not None:
            pt = (int(k.pt[0]), int(k.pt[1]))
            if cv2.pointPolygonTest(poly, pt, False) >= 0:
                continue
        kept_kp.append(k)
        kept_des.append(d)

    if kept_kp and kept_des:
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


def _find_image_path(base_dir, filename):
    """Support both .png and .jpg seamlessly."""
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        return path
    root, _ = os.path.splitext(filename)
    for alt_ext in ['.png', '.jpg', '.jpeg']:
        alt_path = os.path.join(base_dir, root + alt_ext)
        if os.path.exists(alt_path):
            return alt_path
    return path


def produce_outputs(img_train_vector, img_query_vector,
                    MIN_MATCH_COUNT=25, MIN_INLIER_COUNT=15,
                    COLOR_THRESH=1.0, MAX_RETRIES=4,
                    model_names=None):
    if model_names is None:
        model_names = model_name_vector

    thresh_vector = COLOR_THRESH if isinstance(COLOR_THRESH, (list, tuple, np.ndarray)) \
                    else [COLOR_THRESH] * len(img_query_vector)

    output_dir = os.path.join(script_dir, 'output_scenes')
    os.makedirs(output_dir, exist_ok=True)

    sift = cv2.SIFT_create()

    # Precompute model descriptors once
    print("Precomputing SIFT descriptors for Step B models...")
    models_data = []
    for i, path in enumerate(img_query_vector):
        query_path = _find_image_path(models_dir, path)
        img_query_color = cv2.imread(query_path)
        if img_query_color is None:
            print(f"Error: Could not load model image at {query_path}")
            continue

        img_query_gray = cv2.cvtColor(img_query_color, cv2.COLOR_BGR2GRAY)
        kp_query, des_query = sift.detectAndCompute(img_query_gray, None)
        h_m, w_m = img_query_gray.shape

        brand_name = model_names[i] if i < len(model_names) else path
        query_thresh = thresh_vector[i] if i < len(thresh_vector) else 1.0

        models_data.append({
            'id': i,
            'path': path,
            'name': brand_name,
            'img_color': img_query_color,
            'kp': kp_query,
            'des': des_query,
            'w': w_m,
            'h': h_m,
            'thresh': query_thresh
        })

    print(f"Loaded {len(models_data)} models successfully.\n" + "=" * 70)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    for s_idx, scene_filename in enumerate(img_train_vector):
        scene_path = _find_image_path(scenes_dir, scene_filename)
        img_train_color = cv2.imread(scene_path)
        if img_train_color is None:
            print(f"Error: Could not load scene image at {scene_path}")
            continue

        h_s, w_s = img_train_color.shape[:2]
        img_train_gray = cv2.cvtColor(img_train_color, cv2.COLOR_BGR2GRAY)
        kp_train, des_train = sift.detectAndCompute(img_train_gray, None)

        img_output = img_train_color.copy()
        print(f"\nScene image {s_idx + 1}: {scene_filename} ({len(kp_train)} keypoints)")

        if des_train is None or len(kp_train) < 10:
            print("Not enough scene keypoints detected.")
            continue

        scene_candidates = []

        for m_dict in models_data:
            mi = m_dict['id']
            brand_name = m_dict['name']
            path = m_dict['path']
            img_query_color = m_dict['img_color']
            kp_query = m_dict['kp']
            des_query = m_dict['des']
            w_m, h_m = m_dict['w'], m_dict['h']
            query_thresh = m_dict['thresh']

            if des_query is None or len(kp_query) < 10:
                continue

            query_corners = np.float32([[0, 0], [0, h_m - 1], [w_m - 1, h_m - 1], [w_m - 1, 0]]).reshape(-1, 1, 2)
            model_ar = float(h_m) / float(w_m)

            # Working copy of scene keypoints, suppressed iteratively
            kp_scene = list(kp_train)
            des_scene = des_train.copy()

            retries = 0

            while des_scene is not None and len(kp_scene) >= MIN_MATCH_COUNT:
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                # Scene as query, Model as train
                matches = flann.knnMatch(des_scene, des_query, k=2)
                good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.75 * n.distance]

                if len(good) < MIN_MATCH_COUNT:
                    break

                # m.queryIdx -> Scene, m.trainIdx -> Model
                src_pts = np.float32([kp_query[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_scene[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is None:
                    break

                inliers_mask = mask.ravel()
                inlier_count = int(np.sum(inliers_mask))
                matched_scene_indices = [m.queryIdx for idx, m in enumerate(good) if inliers_mask[idx] == 1]

                if inlier_count < MIN_INLIER_COUNT:
                    # Suppress inlier keypoints that failed threshold to allow forward progress
                    kp_scene, des_scene = _filter_keypoints(kp_scene, des_scene, remove_indices=matched_scene_indices)
                    retries += 1
                    if retries > MAX_RETRIES:
                        break
                    continue

                # Inlier spread check on model coordinates
                inlier_model_pts = src_pts[inliers_mask == 1, 0, :]
                span_x = np.ptp(inlier_model_pts[:, 0]) / float(w_m)
                span_y = np.ptp(inlier_model_pts[:, 1]) / float(h_m)
                if span_x < 0.12 or span_y < 0.12:
                    kp_scene, des_scene = _filter_keypoints(kp_scene, des_scene, remove_indices=matched_scene_indices)
                    retries += 1
                    if retries > MAX_RETRIES:
                        break
                    continue

                # Perspective transform & polygon convexity validation
                dst_raw = cv2.perspectiveTransform(query_corners, M)
                if not cv2.isContourConvex(np.int32(dst_raw)):
                    kp_scene, des_scene = _filter_keypoints(kp_scene, des_scene, remove_indices=matched_scene_indices)
                    retries += 1
                    if retries > MAX_RETRIES:
                        break
                    continue

                h_bb_raw = np.linalg.norm(dst_raw[0] - dst_raw[1])
                w_bb_raw = np.linalg.norm(dst_raw[0] - dst_raw[3])
                box_ar = h_bb_raw / (w_bb_raw + 1e-6)
                if not (0.60 * model_ar <= box_ar <= 1.60 * model_ar):
                    kp_scene, des_scene = _filter_keypoints(kp_scene, des_scene, remove_indices=matched_scene_indices)
                    retries += 1
                    if retries > MAX_RETRIES:
                        break
                    continue

                # Straighten bounding box corners
                dst = _straighten_bb(dst_raw)

                height_bb = round(np.linalg.norm(dst[0] - dst[1]))
                width_bb  = round(np.linalg.norm(dst[0] - dst[3]))
                x_center  = round(dst[0, 0, 0] + width_bb  / 2.0)
                y_center  = round(dst[0, 0, 1] + height_bb / 2.0)

                # Size bounds relative to scene
                if height_bb < 0.15 * h_s or height_bb > 0.95 * h_s:
                    kp_scene, des_scene = _filter_keypoints(kp_scene, des_scene, polygon=dst, remove_indices=matched_scene_indices)
                    retries += 1
                    if retries > MAX_RETRIES:
                        break
                    continue

                x1 = max(0, int(dst[0, 0, 0]))
                y1 = max(0, int(dst[0, 0, 1]))
                x2 = min(w_s, int(dst[2, 0, 0]))
                y2 = min(h_s, int(dst[2, 0, 1]))

                if x2 <= x1 + 20 or y2 <= y1 + 30:
                    kp_scene, des_scene = _filter_keypoints(kp_scene, des_scene, polygon=dst, remove_indices=matched_scene_indices)
                    retries += 1
                    if retries > MAX_RETRIES:
                        break
                    continue

                # Mahalanobis colour verification
                scene_crop = img_train_color[y1:y2, x1:x2]
                dist = mahalanobis_color_distance(img_query_color, scene_crop)
                colour_ok = dist <= query_thresh

                if colour_ok:
                    current_box = [x1, y1, x2, y2]
                    score = inlier_count / (dist + 0.1)

                    scene_candidates.append({
                        'model_id': mi,
                        'brand_name': brand_name,
                        'path': path,
                        'box': current_box,
                        'cx': x_center,
                        'cy': y_center,
                        'width': width_bb,
                        'height': height_bb,
                        'inliers': inlier_count,
                        'dist': dist,
                        'score': score,
                        'dst': dst
                    })

                    # Suppress detected region from scene to search for subsequent instances
                    kp_scene, des_scene = _filter_keypoints(kp_scene, des_scene, polygon=dst, remove_indices=matched_scene_indices)
                    retries = 0
                else:
                    # Color mismatch: suppress keypoints and retry
                    kp_scene, des_scene = _filter_keypoints(kp_scene, des_scene, polygon=dst, remove_indices=matched_scene_indices)
                    retries += 1
                    if retries > MAX_RETRIES:
                        break

        # Cross-model overlap removal across all 7 models
        scene_candidates.sort(key=lambda d: d['score'], reverse=True)
        final_detections = []
        for cand in scene_candidates:
            boxA = cand['box']
            suppressed = False
            for f in final_detections:
                boxB = f['box']
                if _compute_iou(boxA, boxB) > 0.35:
                    suppressed = True
                    break
            if not suppressed:
                final_detections.append(cand)

        # Sort final detections top-to-bottom, left-to-right
        final_detections.sort(key=lambda d: (d['cy'] // max(20, int(round(h_s * 0.18))), d['cx']))

        # Draw and report
        model_instance_counter = {}
        for det in final_detections:
            mid = det['model_id']
            model_instance_counter[mid] = model_instance_counter.get(mid, 0) + 1
            instance_num = model_instance_counter[mid]

            dst = det['dst']
            b_name = det['brand_name']
            x_c, y_c = det['cx'], det['cy']
            w_b, h_b = det['width'], det['height']
            inls = det['inliers']
            m_dist = det['dist']

            img_output = cv2.polylines(
                img_output, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            label = f"P{mid + 1}: {b_name} #{instance_num}"
            cv2.putText(
                img_output, label, (int(dst[0, 0, 0]), max(20, int(dst[0, 0, 1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            colour_info = f"Mahalanobis dist = {m_dist:.4f} (thresh={models_data[mid]['thresh']})"
            print(f"  -> Instance {instance_num} found via Crop [{colour_info}]:\n"
                  f"     Position: ({x_c}, {y_c}), Width: {w_b} px, Height: {h_b} px, Inliers: {inls}")

        # Summary per product
        print("\n--- Product Detection Summary ---")
        for i, m_dict in enumerate(models_data):
            mid = m_dict['id']
            cnt = model_instance_counter.get(mid, 0)
            if cnt > 0:
                print(f"Product {mid + 1} ({m_dict['name']}, {m_dict['path']}): {cnt} instance(s) detected via Crop.")
            else:
                print(f"Product {mid + 1} ({m_dict['name']}, {m_dict['path']}): 0 instance found.")

        total_instances = len(final_detections)
        print(f"\nTotal products detected in {scene_filename}: {total_instances} instances")

        root_name = os.path.splitext(scene_filename)[0]
        output_filename = f"{root_name}_detected_crop.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img_output)
        print(f"[Saved annotated scene to: {output_path}]\n" + "-" * 70)


# Entry point
if __name__ == '__main__':
    MIN_MATCH_COUNT  = 25
    MIN_INLIER_COUNT = 15
    MAX_RETRIES      = 4

    img_train_vector = ['m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png']
    img_query_vector = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '26.jpg', '25.jpg']

    # Per-query Mahalanobis distance thresholds
    COLOR_THRESH = [0.45, 0.85, 1.0, 1.0, 1.0, 0.35, 1.0]

    produce_outputs(img_train_vector, img_query_vector,
                    MIN_MATCH_COUNT=MIN_MATCH_COUNT,
                    MIN_INLIER_COUNT=MIN_INLIER_COUNT,
                    COLOR_THRESH=COLOR_THRESH,
                    MAX_RETRIES=MAX_RETRIES,
                    model_names=model_name_vector)
