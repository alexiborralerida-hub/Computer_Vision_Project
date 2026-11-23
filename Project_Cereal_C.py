import numpy as np
import cv2
import os

# Get directories
script_dir = os.path.dirname(os.path.abspath(__file__))
scenes_dir = os.path.join(script_dir, 'scenes')
models_dir = os.path.join(script_dir, 'models')

# Brand names corresponding to models 0.jpg to 23.jpg
model_name_vector = [
    'Regular Nesquik',                # 0.jpg  (Product 1)
    'Blue Krave',                     # 1.jpg  (Product 2)
    'Nesquik Duo',                    # 2.jpg  (Product 3)
    'Chocapic',                       # 3.jpg  (Product 4)
    'Fitness Original',               # 4.jpg  (Product 5)
    'Cheerios Multigrain',            # 5.jpg  (Product 6)
    'Cheerios Honey',                 # 6.jpg  (Product 7)
    'Fitness Fruits',                 # 7.jpg  (Product 8)
    'Lion',                           # 8.jpg  (Product 9)
    'Special K Classic',              # 9.jpg  (Product 10)
    'Special K Red Berries',          # 10.jpg (Product 11)
    'Orange Krave',                   # 11.jpg (Product 12)
    'Cookie Crisp',                   # 12.jpg (Product 13)
    'Crunchy Nut',                    # 13.jpg (Product 14)
    'Coco Pops Chocos',               # 14.jpg (Product 15)
    'Corn Flakes',                    # 15.jpg (Product 16)
    'Frosties',                       # 16.jpg (Product 17)
    'Rice Krispies',                  # 17.jpg (Product 18)
    'Jordans Country Crisp Berries',   # 18.jpg (Product 19)
    'Jordans Country Crisp Chocolate', # 19.jpg (Product 20)
    'Special K Dark Chocolate',       # 20.jpg (Product 21)
    'Crunchy Nut Clusters',           # 21.jpg (Product 22)
    'Jordans Country Crisp Nuts',      # 22.jpg (Product 23)
    'Special K Classic'               # 23.jpg (Product 24, duplicate of 9)
]


def to_rootsift(des):
    """
    Apply RootSIFT transform (Arandjelovic & Zisserman, 2012):
    L1 normalize descriptors then take square root.
    Euclidean distance between RootSIFT descriptors is equivalent to Hellinger distance.
    """
    if des is None:
        return None
    des = des.astype(np.float32)
    l1 = np.linalg.norm(des, ord=1, axis=1, keepdims=True)
    return np.sqrt(des / (l1 + 1e-7))


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
    root, ext = os.path.splitext(filename)
    for alt_ext in ['.png', '.jpg', '.jpeg']:
        alt_path = os.path.join(base_dir, root + alt_ext)
        if os.path.exists(alt_path):
            return alt_path
    return path


def produce_outputs(img_train_vector, img_query_vector,
                    scale=2.0,
                    MIN_INLIER_COUNT=5,
                    BIN_SIZE_BASE=26,
                    COLOR_THRESH=1.25,
                    model_names=None):
    """
    Step C - Whole Shelve Challenge Pipeline:
    1. Precomputes RootSIFT descriptors for all 24 reference product models.
    2. Upscales low-resolution scene image by 2x (bicubic) to extract 4x more features.
    3. Runs hybrid Generalized Hough Transform (GHT) with scale regularization and spatial binning.
    4. Performs local RANSAC homography estimation, straightening, and geometric box sanity checks.
    5. Filters distractors (price tags, shelf edges) via aspect ratio and minimum height bounds.
    6. Verifies candidate crops using brightness-normalized Mahalanobis color distance.
    7. Resolves cross-model overlaps and look-alike variants via Cross-Model Non-Maximum Suppression.
    8. Reports detailed instance dimensions, positions, and prints summary counts.
    """
    if model_names is None:
        model_names = model_name_vector

    output_dir = os.path.join(script_dir, 'output_scenes')
    os.makedirs(output_dir, exist_ok=True)

    # SIFT detector with optimized contrast threshold for subtle cereal packaging textures
    sift = cv2.SIFT_create(contrastThreshold=0.014)

    # FLANN Matcher configuration
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Precompute model descriptors
    print("Precomputing RootSIFT descriptors for reference product models...")
    models_data = []
    for i, path in enumerate(img_query_vector):
        query_path = _find_image_path(models_dir, path)
        img_query_color = cv2.imread(query_path)
        if img_query_color is None:
            print(f"Error: Could not load model image at {query_path}")
            continue

        img_query_gray = cv2.cvtColor(img_query_color, cv2.COLOR_BGR2GRAY)
        kp_query, des_query = sift.detectAndCompute(img_query_gray, None)
        des_query_root = to_rootsift(des_query)
        h_m, w_m = img_query_gray.shape

        brand_name = model_names[i] if i < len(model_names) else path
        models_data.append({
            'id': i,
            'path': path,
            'name': brand_name,
            'img_color': img_query_color,
            'kp': kp_query,
            'des': des_query_root,
            'w': w_m,
            'h': h_m
        })

    print(f"Loaded {len(models_data)} models successfully.\n" + "="*70)

    # Process each scene image
    for s_idx, scene_filename in enumerate(img_train_vector):
        scene_path = _find_image_path(scenes_dir, scene_filename)
        img_train_color = cv2.imread(scene_path)

        if img_train_color is None:
            print(f"Error: Could not load scene image at {scene_path}")
            continue

        h_s, w_s = img_train_color.shape[:2]

        # 2x Upscaling for low-resolution store shelf feature recovery
        img_scaled = cv2.resize(img_train_color, (int(w_s * scale), int(h_s * scale)),
                                interpolation=cv2.INTER_CUBIC)
        img_train_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

        kp_train, des_train = sift.detectAndCompute(img_train_gray, None)
        des_train_root = to_rootsift(des_train)

        img_output = img_train_color.copy()

        print(f"\nScene image {s_idx + 1}: {scene_filename} ({len(kp_train)} keypoints at {scale}x scale)")

        scene_candidates = []

        # Process each product model
        for m_dict in models_data:
            mi = m_dict['id']
            brand_name = m_dict['name']
            path = m_dict['path']
            img_m = m_dict['img_color']
            kp_m = m_dict['kp']
            des_m = m_dict['des']
            w_m, h_m = m_dict['w'], m_dict['h']

            if des_m is None or len(kp_m) < 5:
                continue

            matches = flann.knnMatch(des_m, des_train_root, k=2)
            good = [m for m, n in matches if m.distance < 0.78 * n.distance]

            if len(good) < 5:
                continue

            # Expected scale factor for typical shelf box height (~150px in 2x space)
            exp_scale = 150.0 / float(h_m)
            xc_m, yc_m = w_m / 2.0, h_m / 2.0

            BIN_SIZE = BIN_SIZE_BASE * scale
            grid_h = int(np.ceil(h_s * scale / BIN_SIZE))
            grid_w = int(np.ceil(w_s * scale / BIN_SIZE))

            bin_matches = {}

            # Hybrid Voting: GHT barycenter vote + spatial position pooling
            for m in good:
                km = kp_m[m.queryIdx]
                ks = kp_train[m.trainIdx]

                # Constrain SIFT scale ratio around physical expected box scale
                s_raw = ks.size / (km.size + 1e-6)
                s = np.clip(s_raw, 0.4 * exp_scale, 2.0 * exp_scale)

                # Damped orientation difference (supermarket cereal boxes are upright)
                d_theta = np.radians(ks.angle - km.angle)
                if abs(np.degrees(d_theta)) > 25:
                    cos_t, sin_t = 1.0, 0.0
                else:
                    cos_t, sin_t = np.cos(d_theta), np.sin(d_theta)

                dx = xc_m - km.pt[0]
                dy = yc_m - km.pt[1]
                vx = ks.pt[0] + s * (dx * cos_t - dy * sin_t)
                vy = ks.pt[1] + s * (dx * sin_t + dy * cos_t)

                gx = int(vx // BIN_SIZE)
                gy = int(vy // BIN_SIZE)
                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    bin_matches.setdefault((gy, gx), []).append(m)

                # Direct spatial location pooling
                sgx = int(ks.pt[0] // BIN_SIZE)
                sgy = int(ks.pt[1] // BIN_SIZE)
                if 0 <= sgx < grid_w and 0 <= sgy < grid_h:
                    bin_matches.setdefault((sgy, sgx), []).append(m)

            # Accumulator peak extraction (3x3 neighborhood pooling)
            candidate_peaks = []
            for gy in range(grid_h):
                for gx in range(grid_w):
                    unique_m = {}
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            for m in bin_matches.get((gy + dy, gx + dx), []):
                                unique_m[(m.queryIdx, m.trainIdx)] = m
                    if len(unique_m) >= 5:
                        candidate_peaks.append((len(unique_m), gy, gx, list(unique_m.values())))

            candidate_peaks.sort(key=lambda x: x[0], reverse=True)

            query_corners = np.float32([[0,0],[0,h_m-1],[w_m-1,h_m-1],[w_m-1,0]]).reshape(-1,1,2)

            for count, gy, gx, peak_matches in candidate_peaks:
                src_pts = np.float32([kp_m[m.queryIdx].pt for m in peak_matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp_train[m.trainIdx].pt for m in peak_matches]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0 * scale)
                if M is None:
                    continue

                inlier_count = int(mask.sum())
                if inlier_count < MIN_INLIER_COUNT:
                    continue

                # Project corners to scene and scale back to 1x original resolution
                dst_scaled = cv2.perspectiveTransform(query_corners, M)
                dst = dst_scaled / scale
                dst = _straighten_bb(dst)

                height_bb = round(np.linalg.norm(dst[0] - dst[1]))
                width_bb  = round(np.linalg.norm(dst[0] - dst[3]))
                x_center  = round(dst[0,0,0] + width_bb  / 2.0)
                y_center  = round(dst[0,0,1] + height_bb / 2.0)

                # Distractor rejection & bounding box sanity checks:
                # Discard price tags (height < 45px), extreme outliers, or degenerate shapes
                if not (24 <= width_bb <= 115 and 45 <= height_bb <= 145):
                    continue

                aspect_ratio = height_bb / (width_bb + 1e-5)
                if not (0.75 <= aspect_ratio <= 2.5):
                    continue

                # Center must reside inside the visible image frame
                if not (10 <= x_center <= w_s - 10 and 15 <= y_center <= h_s - 15):
                    continue

                x1 = max(0, int(dst[0,0,0]))
                y1 = max(0, int(dst[0,0,1]))
                x2 = min(w_s, int(dst[2,0,0]))
                y2 = min(h_s, int(dst[2,0,1]))
                if x2 <= x1 + 18 or y2 <= y1 + 30:
                    continue

                # Mahalanobis colour verification on brightness-normalised crop
                scene_crop = img_train_color[y1:y2, x1:x2]
                dist = mahalanobis_color_distance(img_m, scene_crop)

                # Tiered colour threshold: low-inlier detections require stricter colour match
                if inlier_count < 7 and dist > 0.65:
                    continue
                if inlier_count < 10 and dist > 0.95:
                    continue
                if dist > COLOR_THRESH:
                    continue

                box = [x1, y1, x2, y2]
                score = inlier_count / (1.0 + dist * 0.4)

                scene_candidates.append({
                    'model_id': mi,
                    'brand_name': brand_name,
                    'path': path,
                    'box': box,
                    'cx': x_center,
                    'cy': y_center,
                    'width': width_bb,
                    'height': height_bb,
                    'peak_votes': count,
                    'inliers': inlier_count,
                    'dist': dist,
                    'score': score,
                    'dst': dst
                })

        # Cross-Model Non-Maximum Suppression (Inter-Class & Intra-Class Conflict Resolution)
        scene_candidates.sort(key=lambda d: d['score'], reverse=True)
        final_detections = []

        for cand in scene_candidates:
            boxA = cand['box']
            suppressed = False
            for f in final_detections:
                boxB = f['box']
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])

                interArea = max(0, xB - xA) * max(0, yB - yA)
                areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

                iou = interArea / float(areaA + areaB - interArea + 1e-6)
                io_min = interArea / float(min(areaA, areaB) + 1e-6)

                if iou > 0.35 or io_min > 0.55:
                    suppressed = True
                    break

            if not suppressed:
                final_detections.append(cand)

        # Sort detections by shelf (vertical) then horizontally for clean output reporting
        final_detections.sort(key=lambda d: (d['cy'] // 90, d['cx']))

        # Group by model for structured output reporting
        model_instance_counter = {}
        for det in final_detections:
            mid = det['model_id']
            model_instance_counter[mid] = model_instance_counter.get(mid, 0) + 1
            instance_num = model_instance_counter[mid]

            dst = det['dst']
            b_name = det['brand_name']
            x_c, y_c = det['cx'], det['cy']
            w_b, h_b = det['width'], det['height']
            votes = det['peak_votes']
            inls = det['inliers']
            m_dist = det['dist']

            # Draw bounding box and label
            img_output = cv2.polylines(
                img_output, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)
            label = f"P{mid+1} #{instance_num}"
            cv2.putText(
                img_output, label, (int(dst[0,0,0]), max(15, int(dst[0,0,1]) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

            colour_info = f"Mahalanobis dist = {m_dist:.4f}"
            print(f"  -> Instance {instance_num} found for P{mid+1} ({b_name}) [{colour_info}]:\n"
                  f"     Position: ({x_c}, {y_c}), Width: {w_b} px, Height: {h_b} px, Peak Votes: {votes}, Inliers: {inls}")

        # Summary per product
        print("\n--- Product Detection Summary ---")
        for i, m_dict in enumerate(models_data):
            mid = m_dict['id']
            cnt = model_instance_counter.get(mid, 0)
            if cnt > 0:
                print(f"Product {mid+1:2d} ({m_dict['name']}, {m_dict['path']}): {cnt} instance(s) detected.")
            else:
                print(f"Product {mid+1:2d} ({m_dict['name']}, {m_dict['path']}): 0 instance found.")

        total_scene_instances = len(final_detections)
        print(f"\nTotal products detected in {scene_filename}: {total_scene_instances} instances")

        # Save annotated images (both StepC and generic detected name)
        root_name = os.path.splitext(scene_filename)[0]
        output_filename = f"{root_name}_detected_StepC.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img_output)
        # Also save as {root_name}_detected.png and {root_name}_detected_GHT.png for compatibility
        cv2.imwrite(os.path.join(output_dir, f"{root_name}_detected.png"), img_output)
        print(f"[Saved annotated scene to: {output_path} and {root_name}_detected.png]\n" + "-"*70)


# Entry point
if __name__ == '__main__':
    # Test scenes: {h1.png, h2.png, h3.png, h4.png, h5.png} (or .jpg fallback)
    img_train_vector = ['h1.png', 'h2.png', 'h3.png', 'h4.png', 'h5.png']

    # Reference product images: {from 0.jpg to 23.jpg}
    img_query_vector = [f"{i}.jpg" for i in range(24)]

    produce_outputs(img_train_vector, img_query_vector,
                    scale=2.0,
                    MIN_INLIER_COUNT=5,
                    BIN_SIZE_BASE=26,
                    COLOR_THRESH=1.25,
                    model_names=model_name_vector)
