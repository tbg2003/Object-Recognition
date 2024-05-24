import numpy as np
import cv2
import os
import sys
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='dart boartd detection')
parser.add_argument('-name', '-n', type=str, default='Dartboard/dart0.jpg')
args = parser.parse_args()

cascade_name = "Dartboardcascade/cascade.xml"

# get viola jones detection boxes and draw
def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
    boxes = np.zeros((len(faces), 4))
    for i in range(0, len(faces)):
        start_point = (faces[i][0], faces[i][1])
        end_point = (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
        boxes[i]=([start_point[0], start_point[1], end_point[0], end_point[1]])
    return boxes

# read ground truth boxes and draw 
def readGroundtruth(img_name, frame=None, filename='groundtruth.txt'):
    img_name = img_name.split("/")[1]
    dim = 0
    boxes = []
    with open(filename) as f:
        for line in f.readlines():
            name = line.split(",")[0]
            if img_name == name:
                dim += 1
                x = float(line.split(",")[1])
                y = float(line.split(",")[2])
                width = float(line.split(",")[3])
                height = float(line.split(",")[4])
                start = np.rint((x,y)).astype(int)
                end = np.rint((x + width, y + height)).astype(int)
                colour = (0,0,255)
                thickness = 2
                frame = cv2.rectangle(frame, start, end, colour, thickness)
                boxes.append([start[0], start[1], end[0], end[1]])
    return np.array(boxes)

# calculate iou between two boxes
def IOU(boxA, boxB):
   
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])


    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# apply sobel filter to input image 
def apply_sobel_filter(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient = np.arctan2(sobely, sobelx)

    return magnitude, gradient

# apply hough transfrom to input img magnitude
def hct(imgMag, imgGradDir):
  
    rangeH, rangeW = imgMag.shape
    maxR = 150  
    minR = 5

    HS = np.zeros((rangeH, rangeW, maxR))
    for j in range(rangeH):
        for i in range(rangeW):
            if imgMag[j,i] == 255:
                for r in range(minR, maxR):
       
                    a = int(i - r * np.cos(imgGradDir[j,i]))
                    b = int(j - r * np.sin(imgGradDir[j,i]))
                    
                    if 0 <= a < rangeW and 0 <= b < rangeH:
                        HS[b,a,r] += imgGradDir[j,i]
                    
                    a = int(i + r * np.cos(imgGradDir[j,i]))
                    b = int(j + r * np.sin(imgGradDir[j,i]))
                    
                    if 0 <= a < rangeW and 0 <= b < rangeH:
                        HS[b,a,r] += imgGradDir[j,i]
    centre_radius = {}
    for y in range(HS.shape[0]):
        for x in range(HS.shape[1]):
            radius_votes = HS[y,x,:]
            best_radius_idx = np.argmax(radius_votes)
            best_radius_votes = radius_votes[best_radius_idx]
            if best_radius_votes > 0:
                centre_radius[(x,y)] = best_radius_idx
    
    
    threshold_vote = np.max(HS) * 0.2
    HS[HS < threshold_vote] = 0
    summed_HS = np.sum(HS, axis=2)

    return summed_HS, centre_radius

# apply hough line transfrom 
def hlt(imgMag, imgGradDir):
    line_info = {}
    # Hough Line Transform: rho = x * cos(theta) + y * sin(theta)
    height, width = imgMag.shape
    max_distance = int(np.hypot(height, width))
    thetas = np.deg2rad(np.arange(-90, 90))
    rhos = np.linspace(-max_distance, max_distance, 2*max_distance)
    
    HS = np.zeros((2*max_distance, len(thetas)), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            if imgMag[y, x] == 255:
                for idx, theta in enumerate(thetas):
                    rho = int(x * np.cos(theta) + y * np.sin(theta)) + max_distance
                    HS[rho, idx] += 1
                    line = (rho, idx)
                    if line not in line_info:
                        line_info[line] = []
                    line_info[line].append((x,y))
    
    threshold_vote = np.max(HS) * 0.3
    HS[HS < threshold_vote] = 0
    return HS, rhos, thetas, line_info 

# threshold all values over sepcified th 
def threshold(img, th=45):
    th_img = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y,x] > th:
                th_img[y,x] = 255
            else: 
                th_img[y,x] = 0
    return th_img

# find local maxima within a window area of a hough space
def non_maximum_suppression(hough_space, window_size=120, threshold=100):
    centers = []
    values = []
    h, w = hough_space.shape
    half_window = window_size // 2

    for y in range(half_window, h - half_window):
        for x in range(half_window, w - half_window):
            local_max = np.max(hough_space[y-half_window:y+half_window+1, x-half_window:x+half_window+1])
            if hough_space[y, x] == local_max and local_max > threshold:
                if len(centers) >= 1:
                    for i in range(len(centers)):
                        if hough_space[y, x] >= values[i]:
                            centers.insert(i, (x,y))
                            values.insert(i, (hough_space[y,x]))
                            break
                        else:
                            centers.append((x, y))
                            values.append(hough_space[y,x])
                            break
                else:
                    centers.append((x, y))
                    values.append(hough_space[y,x])
    return centers

# filter out horizontal and vertical angles, cluster lines with close angles 
def filter_angle(line_info, thetas, cluster_tol=2, angle_tol=30):
    new_info = {}
    angle_tolerance_rad = np.deg2rad(angle_tol)
    horizontal_angles = [0, np.pi] 
    vertical_angle = np.pi / 2
    for line_id, points in line_info.items(): 
        _, theta_idx = line_id
        theta = thetas[theta_idx]
        if any(abs(theta - ha) <= angle_tolerance_rad for ha in horizontal_angles):
            continue
        if abs(theta - vertical_angle) <= angle_tolerance_rad:
            continue
        
        theta_cluster = round(theta/cluster_tol)
        if theta_cluster not in new_info:
            new_info[theta_cluster] = []
        new_info[theta_cluster].append(line_id)
    return new_info

# filter lines by angle, length and breaks    
def filter_lines(line_info, thetas):
    seg_lines = {}
    for line_id, points in line_info.items():    
        segments, lengths = segment_line(points, 10)
        for i, length in enumerate(lengths):
            pt1, pt2 = segments[i]
            if 30 < length < 280 :
                if line_id not in seg_lines:  
                    seg_lines[line_id] = []
                seg_lines[line_id].append((pt1, pt2)) 

    
    updt_lines = filter_angle(seg_lines, thetas)
    return updt_lines, seg_lines

# function to calculate distance

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# draw centres of circles with specified radius stored in dict 
def draw_centers(image, centers, centre_radius, color=(0, 255, 0)):
    img_with_centers = image.copy()
    for center in centers:
        start, end = int(center[0]), int(center[1])
        if centre_radius is None:
            radius = 10
        else:     
            radius = centre_radius[center]
        cv2.circle(img_with_centers, (start,end), radius, color, thickness=3)

    return img_with_centers

# draw lines on image 
def draw_lines(image, lines, color=(255,0,255)):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input 'image' must be a NumPy array.")
    image_with_lines = image.copy()
    for line in lines:
        start_point, end_point = line
        if not all(isinstance(pt, tuple) and len(pt) == 2 for pt in line):
            raise ValueError("Line endpoints must be tuples of two integers.")
        cv2.line(image_with_lines, start_point, end_point, color, thickness=1)
    return image_with_lines

# normalize score, allowing you to compar them 
def normalize_scores(scores):
    scores = [np.array([score]) if np.isscalar(score) else score for score in scores]
    all_values = np.concatenate([score for score in scores if score.size > 0])

    if all_values.size == 0:
        return [np.array([]) for score in scores]

    min_score = np.min(all_values)
    max_score = np.max(all_values)

    if max_score == min_score:
        return [np.zeros_like(score) for score in scores]  

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores

# combine detections by indicators 
def combine_detections(centres, centre_radius, lines, vj_boxes, radius = 60):
    line_count_circle = []
    ious = []
    for centre in centres:
        center_x, center_y = centre
        radius = centre_radius[centre]
        if radius < 6:
            continue
        if radius < 12:
            radius *= 2 
        cand_box = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        line_count = count_lines_in_circle(lines, (center_x, center_y), radius)
        for vj_box in vj_boxes:
            iou = IOU(vj_box, cand_box)
            line_count_circle.append(line_count) 
            ious.append(iou)

        box_lines = find_boxes_with_lines(vj_boxes, lines)
    return normalize_scores(line_count_circle), normalize_scores(ious), normalize_scores(box_lines)

# return boolean for line detectionin detected box 
def is_line_in_box(line, box, padding=20):
    (x1, y1), (x2, y2) = line[0], line[1]
    bx, by, bwidth, bheight = box
    return (bx-padding <= x1 <= bx + bwidth+padding and by-padding <= y1 <= by + bheight+padding) and \
           (bx-padding <= x2 <= bx + bwidth+padding and by-padding <= y2 <= by + bheight+padding)
# function to return no. lines in box 
def find_boxes_with_lines(boxes, lines, min_lines=0):
    box_line_counts = {i: 0 for i in range(len(boxes))}
    
    for i, box in enumerate(boxes):
        for line in lines:
            if is_line_in_box(line, box):
                box_line_counts[i] += 1
    return [boxes[i] if count >= min_lines else 0 for i, count in box_line_counts.items()]

# breaks up line by specified gap th, if gap over break line into two
def segment_line(points, gap_threshold):
    x_coords, y_coords = zip(*points)
    range_x = max(x_coords) - min(x_coords)
    range_y = max(y_coords) - min(y_coords)
    if range_x > range_y:
        sorted_points = sorted(points, key=lambda p: p[0])
    else:
        sorted_points = sorted(points, key=lambda p: p[1])

    line_segments = []
    lengths = []
    segment_start = sorted_points[0]

    for i in range(1, len(sorted_points)):
        if euclidean_distance(sorted_points[i - 1], sorted_points[i]) > gap_threshold:
            line_segments.append((segment_start, sorted_points[i - 1]))
            lengths.append(euclidean_distance(segment_start, sorted_points[i -1]))
            segment_start = sorted_points[i]
    line_segments.append((segment_start, sorted_points[-1]))
    lengths.append(euclidean_distance(segment_start, sorted_points[-1]))
    return line_segments, lengths  

# define if line intersects circle
def is_line_in_circle(line, center, radius, padding=1):
    (x1, y1), (x2, y2) = line[0], line[1]
    return (np.linalg.norm(np.array([x1, y1]) - np.array(center)) <= radius+padding or
            np.linalg.norm(np.array([x2, y2]) - np.array(center)) <= radius+padding)
# count no. lines in circle 
def count_lines_in_circle(lines, circle_center, radius):
    count = 0
    for line in lines:
        if is_line_in_circle(line, circle_center, radius):
            count += 1
    return count/radius 
# calculate F1 score
def fOneScore(tps, fps, fns):
    precision = tps / (tps + fps) if tps + fps > 0 else 0
    recall = tps / (tps + fns) if tps + fns > 0 else 0
    if precision + recall == 0:
        return 0
    else:
        F1 = 2 * ((precision * recall) / (precision + recall))
    return F1
# Calculate score used for report 
def calculate_metrics(detected_boxes, ground_truth_boxes, iou_threshold=0.35):
    tps, fps, fns = 0, 0, 0
    matched = set()

    for det_box in detected_boxes:
        best_iou = 0
        best_gt_idx = -1
        for idx, gt_box in enumerate(ground_truth_boxes):
            iou = IOU(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold:
            if best_gt_idx not in matched:
                tps += 1
                matched.add(best_gt_idx)
        else:
            fps += 1
    fns = len(ground_truth_boxes) - len(matched)

    f1_score = fOneScore(tps, fps, fns)
    return tps, fps, fns, f1_score



# ==== MAIN ==============================================


imageName = args.name

# ignore if no such file is present.
if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print('No such file')
    sys.exit(1)

# 1. Read Input Image
frame = cv2.imread(imageName, 1)
frame2 = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(frame) is np.ndarray):
    print('Not image data')
    sys.exit(1)

# 2. Load the Strong Classifier in a structure called `Cascade'
model = cv2.CascadeClassifier(cascade_name)
if not model.load(cascade_name): # if got error, you might need `if not model.load(cv2.samples.findFile(cascade_name)):' instead
    print('--(!)Error loading cascade model')
    exit(0)




frame = cv2.imread(imageName, 1)
frame2 = cv2.imread(imageName, 1)
# results = cv2.imread(imageName, 1)
# results2 = cv2.imread(imageName, 1)

detected = detectAndDisplay( frame )
groundTruth = readGroundtruth(imageName, frame)

img = cv2.imread(imageName, 1)
gray_image = cv2.imread(imageName,1)
if gray_image.shape[2] >= 3:
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.float32)
else:
    gray_image = gray_image.astype(np.float32)
if gray_image is None:
    print("Error loading image!")
    exit()

TH = 140

while True:
    mag, imgGradDir_out = apply_sobel_filter(cv2.blur((gray_image), (5,5)))
    imgMag_out_th = threshold(mag, th=TH)
    HSC, centre_radius = hct(imgMag_out_th, imgGradDir_out)
    refined_HSC = non_maximum_suppression(HSC, 200, 60)
    if len(refined_HSC) >= 1:
        break
    TH -= 10
name = f"imageMag{TH}.jpg"


HSL, rhos, thetas, line_info = hlt(imgMag_out_th, imgGradDir_out)
cv2.imwrite("HSL.jpg", HSL)
db_lines = []
updt_lines, seg_lines = filter_lines(line_info, thetas)

   
for theta_cluster, line_ids in updt_lines.items():
        for line_id in line_ids:
            for point in seg_lines[line_id]:
                db_lines.append(point)

def standardize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    standardized_arr = (arr - mean) / std
    return standardized_arr
line_count_circle, ious, box_lines = combine_detections(refined_HSC, centre_radius, db_lines, detected)

max_box_lines = np.max(box_lines)
lcw = 0.4
iouw = 0.3
blw = 0.3
scores = []

for i in range(len(ious)):
    lc_idx = i % len(line_count_circle)
    bl_idx = i % len(box_lines)

    score = (line_count_circle[lc_idx] * lcw + ious[i] * iouw + box_lines[bl_idx] * blw)
    scores.append(score)


thresh = np.percentile([np.sum(score) for score in scores], 98)
box_idxs = []
hough_boxes = []
for x, score_array in enumerate(scores):
    if np.sum(score_array) >= thresh:
        box_idx = x % len(detected)
        box = detected[box_idx]
        box_idxs.append(box_idx)
        hough_boxes.append(box)
        startx, starty, endx, endy = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
# REPORT WRITING        
# for box in hough_boxes:
#     startx, starty, endx, endy = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#     cv2.rectangle(frame2, (startx, starty), (endx, endy), (0, 255, 0), 1)


img_path = "template.jpg"
template = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
target = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
if template is None:
    print("Error loading image!")
    exit()


sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(template,None)
kp2, des2 = sift.detectAndCompute(target,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) 
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good_matches = [m for m, n in matches if m.distance < 0.95 * n.distance]
matched_kp_target = [kp2[m.trainIdx].pt for m in good_matches]
def is_point_inside_box(point, box):
    px, py = point
    bx, by, bw, bh = box
    return bx <= px <= bx + bw and by <= py <= by + bh

match_threshold = 15  

boxes_with_matches = []
match_counts = []
best_boxes = []
max_point_count = 0

for idx, box in enumerate(detected):
    match_count = 0
    for point in matched_kp_target:
        if is_point_inside_box(point, box):
            match_count += 1
    if match_count > max_point_count:
        max_point_count = match_count

    boxes_with_matches.append(box)
    match_counts.append(match_count)

indicator_1 = [] # Blue
indicator_2 = [] # Green
indicator_3 = [] # Red

for idx, box in enumerate(detected):
    if match_counts[idx] >= max_point_count * 0.9:
        indicator_1.append(box)
        if not any(np.array_equal(box, b) for b in best_boxes):
            best_boxes.append(box)
            
    if idx in box_idxs:
        if match_counts[idx] >= max_point_count * 0.5:
                indicator_2.append(box)
                if not any(np.array_equal(box, b) for b in best_boxes):
                    best_boxes.append(box)
                    
        if match_counts[idx] >= max_point_count * 0.2:
            if any(box_lines[idx] >= max_box_lines * 0.7):
                    indicator_3.append(box)
                    if not any(np.array_equal(box, b) for b in best_boxes):
                        best_boxes.append(box)
                        

### REPORT WRITING 


# results = draw_lines(results, db_lines, (0,0,255)) 
# results = draw_centers(results, refined_HSC, centre_radius, (0,255,0))
# for pt in matched_kp_target:
#     x, y = int(pt[0]), int(pt[1])
#     cv2.circle(results, (x, y), 4, (255, 0, 0), -1)

# for box in hough_boxes:
#     startx, starty, endx, endy = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#     cv2.rectangle(frameH, (startx, starty), (endx, endy), (255, 255, 0), 2)
# for box in indicator_1:
#     startx, starty, endx, endy = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#     cv2.rectangle(results, (startx, starty), (endx, endy), (255, 0, 0), 10)
# for box in indicator_2:
#     startx, starty, endx, endy = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#     cv2.rectangle(results, (startx, starty), (endx, endy), (0, 255, 0), 4)
# for box in indicator_3:
#     startx, starty, endx, endy = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#     cv2.rectangle(results, (startx, starty), (endx, endy), (0, 0, 255), 2)

# for box in best_boxes:
#     startx, starty, endx, endy = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#     cv2.rectangle(results2, (startx, starty), (endx, endy), (0, 255, 0), 2)

# cv2.imwrite(f"HoughShape.jpg", frameH)
# cv2.imwrite(f"results.jpg", results)
# cv2.imwrite(f"results2.jpg", results2)
cv2.imwrite(f"detected.jpg", frame2)