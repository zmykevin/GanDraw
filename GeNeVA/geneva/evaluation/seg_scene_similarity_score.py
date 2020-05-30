import numpy as np
import cv2
import re
import os

from skimage import io, morphology, measure
from scipy import ndimage
from itertools import permutations
from math import sqrt

L2NORML_RAW = {
            156: {"name": "sky", "index": 0},
            110: {"name": "dirt", "index": 1},
            124: {"name": "gravel", "index": 2},
            135: {"name": "mud", "index": 3},
            14: {"name": "sand", "index": 4},
            105: {"name": "clouds", "index": 5},
            119: {"name": "fog", "index": 6},
            126: {"name": "hill", "index": 7},
            134: {"name": "mountain", "index": 8},
            147: {"name": "river", "index": 9},
            149: {"name": "rock", "index": 10},
            154: {"name": "sea", "index": 11},
            158: {"name": "snow", "index": 12},
            161: {"name": "stone", "index": 13},
            177: {"name": "water", "index": 14},
            96: {"name": "bush", "index": 15},
            118: {"name": "flower", "index": 16},
            123: {"name": "grass", "index": 17},
            162: {"name": "straw", "index": 18},
            168: {"name": "tree", "index": 19},
            181: {"name": "wood", "index": 20},
            148: {"name": "road", "index": 21}
        }

NORML2COLOR_RAW = {
            0: {"name": "sky", "color": np.array([134, 193, 46])},
            1: {"name": "dirt", "color": np.array([30, 22, 100])},
            2: {"name": "gravel", "color": np.array([163, 164, 153])},
            3: {"name": "mud", "color": np.array([35, 90, 74])},
            4: {"name": "sand", "color": np.array([196, 15, 241])},
            5: {"name": "clouds", "color": np.array([198, 182, 115])},
            6: {"name": "fog", "color": np.array([76, 60, 231])},
            7: {"name": "hill", "color": np.array([190, 128, 82])},
            8: {"name": "mountain", "color": np.array([122, 101, 17])},
            9: {"name": "river", "color": np.array([97, 140, 33])},
            10: {"name": "rock", "color": np.array([90, 90, 81])},
            11: {"name": "sea", "color": np.array([255, 252, 51])},
            12: {"name": "snow", "color": np.array([51, 255, 252])},
            13: {"name": "stone", "color": np.array([106, 107, 97])},
            14: {"name": "water", "color": np.array([0, 255, 0])},
            15: {"name": "bush", "color": np.array([204, 113, 46])},
            16: {"name": "flower", "color": np.array([0, 0, 255])},
            17: {"name": "grass", "color": np.array([255, 0, 0])},
            18: {"name": "straw", "color": np.array([255, 51, 252])},
            19: {"name": "tree", "color": np.array([255, 51, 175])},
            20: {"name": "wood", "color": np.array([66, 18, 120])},
            21: {"name": "road", "color": np.array([255, 255, 0])},
        }

NORML2COLOR = {k: x["color"] for k, x in NORML2COLOR_RAW.items()}
NORML2LABEL = {x["index"]:k for k,x in L2NORML_RAW.items()}


def color2label(color):
    for k, x in NORML2COLOR.items():
        if x[0] == color[0] and x[1] == color[1] and x[2] == color[2]:
            return NORML2LABEL[k]
    return None
def segconverter(seg_im):
    """
    Convert the segmentation image to a 2D matrix with assigned label.
    Input is a numpy matrix
    """
    gen_result = np.zeros((seg_im.shape[0], seg_im.shape[1]), dtype=np.int32)
    #convert gen_raw to 2D matrix
    unique_colors = np.unique(seg_im.reshape(-1, seg_im.shape[2]), axis=0)
    #print(unique_colors)
    for c in unique_colors:
        c_index = np.where(np.all(seg_im == c, axis=-1))
        #
        c_label = color2label(c)
        assert c_label is not None, "label cannot be None"
        gen_result[c_index] = c_label
    return gen_result

def eval_label_prediction(data_gen_file, data_gt_human_file):
	precision_acc = 0
	recall_acc = 0
	acc = 0
	F1 = 0
	#count the most missed_labels
	missed_label_count = np.zeros(22)
	total_label_count = np.zeros(22)

	for gen_im_path, gt_human_im_path in zip(data_gen_file, data_gt_human_file):
	    gen_raw = cv2.imread(gen_im_path)
	    gen_raw = cv2.cvtColor(gen_raw, cv2.COLOR_BGR2RGB)
	    gen_seg = segconverter(gen_raw)
	    gen_seg_set = set(np.unique(gen_seg))
	    
	    #print(np.unique(gen_seg))
        
	    gt_human_raw = cv2.imread(gt_human_im_path)
	    gt_human_raw = cv2.cvtColor(gt_human_raw, cv2.COLOR_BGR2RGB)
	    gt_human_seg = segconverter(gt_human_raw)
	    gt_human_seg_set = set(np.unique(gt_human_seg))

	    tp = gen_seg_set.intersection(gt_human_seg_set)

	    fn_fp_tp = gen_seg_set.union(gt_human_seg_set)
	    #Update the precision_acc & recall_acc
	    current_precision = len(tp)/len(gen_seg_set)
	    current_recall = len(tp)/len(gt_human_seg_set)
	    precision_acc += current_precision
	    recall_acc += current_recall
	    acc += len(tp)/len(fn_fp_tp)
	    if current_precision > 0 or current_recall > 0:
	    	F1 += 2*(current_precision*current_recall)/(current_precision+current_recall)

	    #missed labels count
	    missed_set = gt_human_seg_set.difference(tp)
	    for l in list(missed_set):
	    	missed_label_count[L2NORML_RAW[l]['index']] += 1

	    #total label count
	    for l in list(gt_human_seg_set):
	    	total_label_count[L2NORML_RAW[l]['index']] += 1

	i = 0    
	for x,y in zip(missed_label_count, total_label_count):
		if y > 0:
			missed_label_count[i] = x/y
		i += 1 
	        
	precision_acc = precision_acc / len(data_gen_file)
	recall_acc = recall_acc / len(data_gen_file)
	acc = acc /len(data_gen_file)
	F1 = F1 / len(data_gen_file)
	return precision_acc, recall_acc, acc, F1, missed_label_count

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    #print(hist)
    return hist

#Mean_IOU is the metrics that we want to use
def mean_IoU(label_gt, label_pred, n_class):
    hist = np.zeros((n_class, n_class))
    #loop throught the matrix row by row
    for lt, lp in zip(label_gt, label_pred):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    #print(iu)
    valid = hist.sum(axis=1) > 0  # added
    #print(valid)
    mean_iu = np.nanmean(iu[valid])
    
    return mean_iu

def pixel_accuracy(label_gt, label_pred, n_class):
    hist = np.zeros((n_class, n_class))
    #loop throught the matrix row by row
    for lt, lp in zip(label_gt, label_pred):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    #The [i,j] the element in hist indicate the number of values when gt_matrix = i and pred_matrix =j 
    acc = np.diag(hist).sum() / hist.sum()
    return acc

def best_smooth_method(image,dominant_thred=1000):
    """
    image is the 3D RGB numpy matrix of the image.
    return the 2D matrix map.
    """
    #Deterine the dominant pixels

    r,g,b = cv2.split(image)


    #Find dominant labels in the image
    unique_class, unique_counts = np.unique(r, return_counts=True)
    #Need to smooth gt_labels as well
    result = np.where(unique_counts > dominant_thred)
    dominant_class = unique_class[result].tolist()
#     a,b= np.histogram(r,bins=np.unique(r))
#     result = np.where(a > dominant_thred)
#     dominant_class = []
#     for x in range(result[0].shape[0]):
#         #if b[result[0][x]] in landscape_labels_raw + [0]:
#         dominant_class.append(b[result[0][x]])

    num_cluster = len(dominant_class)

    #Reshape the image
    image_2D = image.reshape((image.shape[0]*image.shape[1],3))

    # convert to np.float32
    image_2D = np.float32(image_2D)

    #define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv2.kmeans(image_2D, num_cluster, None,criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    
    #Mapping the segmented image to labels
    drawing2landscape = [
        ([0, 0, 0],156), #sky
        ([110, 180, 232], 156),#sky
        ([60, 83, 163], 154), #sea
        ([128, 119, 97], 134), #mountain
        ([99, 95, 93], 149), #rock
        ([108, 148, 96], 126), #hill
        ([242, 218, 227], 105), #clouds
        ([214, 199, 124], 14), #sand
        ([145, 132, 145], 124), #gravel
        ([237, 237, 237], 158), #snow
        ([101, 163, 152], 147), #river
        ([70, 150, 50], 96), #bush
        ([135, 171, 111], 168), #tree
        ([65, 74, 74], 148), #road
        ([150, 126, 84], 110), #dirt 
        ([120, 75, 38], 135), #mud 
        ([141, 159, 184], 119), #fog 
        ([156, 156, 156], 161), #stone
        ([82, 107, 217], 177), #water
        ([230, 190, 48], 118), #flower
        ([113, 204, 43], 123), #grass
        ([232, 212, 35], 162), #straw
    ]

    #Find the closest labels
    label = label.reshape((image.shape[:2]))

    #map label to corresponding tag
    converstion = {}
    for i, c in enumerate(center):
        #sort the defined centers
        drawing2landscape.sort(key = lambda p: sqrt((p[0][0] - c[0])**2 + (p[0][1] - c[1])**2 + (p[0][2] - c[2])**2))
        #construct the mapping
        label[np.where(label==i)] = drawing2landscape[0][1]

    return label


def merge_noisy_pixels(sample_matrix, d_class,kernel_width=3, kernel_height=3,sliding_size=1):
    #First extract all the (x,y) positions of a certain label
    i = 0
    j = 0
    #sample_matrix_copy = sample_matrix
    converted_pixels = 0
    while i + kernel_width <= sample_matrix.shape[1]:
        while j + kernel_height <= sample_matrix.shape[0]:
            current_window = sample_matrix[j:j+kernel_width,i:i+kernel_height]
            current_window_list = current_window.flatten().tolist()
            # Check whether there is unknown labels in current_window
            current_labels = set(current_window_list)
            if not current_labels.issubset(set(d_class)):
                #replace the noisy labels with the closes 
                d_class_subset = list(current_labels.intersection(set(d_class)))
                if d_class_subset:
                    #If there is some intersection between dominant class and the unknown class
                    # replace the noisy labels with the dominant class 
                    d_class_subset.sort(key = lambda p : current_window_list.count(p), reverse=True)
                    dominant_d_class = d_class_subset[0]
                    mask = ~ np.isin(current_window, d_class)
                    #converted_pixels += np.sum(mask)
                    current_window[mask] = dominant_d_class
                    sample_matrix[j:j+kernel_width,i:i+kernel_height] = current_window
#                 else:
#                     #This will only happen if the first window contain the unknown labels. Which can be acceptable
#                     pass
                    
            j += sliding_size
        i += sliding_size
        j = 0
    return sample_matrix

def find_label_centers(image, shared_label, noisy_filter=1000):
    """
      find the centers of the labels in the image
    """
    image_shared = {l:None for l in shared_label}
    #construct the center for draw_shared
    for key in image_shared.keys():
        mask = np.int_((image == key))
        lbl = ndimage.label(mask)[0]
        unique_labels, unique_counts = np.unique(lbl,return_counts=True)
        filtered_labels = unique_labels[np.where(unique_counts > noisy_filter)]
        filtered_labels = filtered_labels[np.where(filtered_labels > 0)]
        
#         for l in filtered_labels:
#             plt.imshow(np.int_(lbl==l))
#             plt.show()
        centers = ndimage.measurements.center_of_mass(mask, lbl, filtered_labels)
        image_shared[key] = centers
    return image_shared

def pair_objects(draw_shared, gt_shared):
    """
    Pair the regions in drawer's image and the regions in groundtruth image based on the mean square distance.
    TODO:
    1. Rethink the way we pair the objects
    """
    gt_draw_shared = {}

    for key in draw_shared.keys():
        #check if the number maps
        if len(draw_shared[key]) == len(gt_shared[key]) and len(draw_shared[key]) == 1:
            gt_draw_shared[key] = {"draw_center": draw_shared[key], "gt_center": gt_shared[key], "max_num_objects": 1}
        else:
            #pair the centers
            pair_regions = []
            pair_index = []
            for i, draw_c in enumerate(draw_shared[key]):
                for j, gt_c in enumerate(gt_shared[key]):
                    pair_regions.append((draw_c, gt_c))
                    pair_index.append((i,j))
            #group all possible combinations of i,j together
            if len(draw_shared[key]) < len(gt_shared[key]):
                perm = permutations(range(len(gt_shared[key])),len(draw_shared[key]))
                pair_candidates = []
                for p in perm:
                    #form the groups
                    pair_centers = []
                    for i in range(len(draw_shared[key])):
                        current_pair = (i,p[i])
                        current_pair_index = pair_index.index(current_pair)
                        current_pair_centers = pair_regions[current_pair_index]
                        pair_centers.append(current_pair_centers)
                    pair_candidates.append(pair_centers)
            else:
                perm = permutations(range(len(draw_shared[key])),len(gt_shared[key]))
                pair_candidates = []
                for p in perm:
                    #form the groups
                    pair_centers = []
                    for i in range(len(gt_shared[key])):
                        current_pair = (p[i],i)
                        current_pair_index = pair_index.index(current_pair)
                        current_pair_centers = pair_regions[current_pair_index]
                        pair_centers.append(current_pair_centers)
                    pair_candidates.append(pair_centers)
            
            #sort pair_candidates based on their pair sum
            pair_candidates.sort(key = lambda p: sum([sqrt((c[0][0]-c[1][0])**2 + (c[0][1]-c[1][1])**2) for c in p]))
            
            optimal_pair = pair_candidates[0]
            paired_draw_centers = [x[0] for x in optimal_pair]
            paired_gt_centers = [x[1] for x in optimal_pair]

            gt_draw_shared[key] = {"draw_center": paired_draw_centers, "gt_center": paired_gt_centers, "max_num_objects": max(len(draw_shared[key]), len(gt_shared[key]))}
    return gt_draw_shared

def relevant_score(x,y):
    """
    x and y are two turples of the objects in drawed image and ground truth image.
    """
    
    score_x =  1 if (x[0][0]-y[0][0])*(x[1][0]-y[1][0]) > 0 else 0 
    score_y =  1 if (x[0][1]-y[0][1])*(x[1][1]-y[1][1]) > 0 else 0
    
    return score_x+score_y

def relevant_eval_metrics(draw_raw, gt_raw, d_smooth=False, g_smooth=False, g_smooth_thred=1000, relevant_mode="unknown_count"):
    """
    Compute the relevant_eval_metrics from draw_segments and ground_truth_segments
    """
    #compute noisy_filter
    if draw_raw.shape[0] == 64:
        noisy_filter_thred = 30
    else:
        noisy_filter_thred = 1000
    #smooth drawer_image if required
    if d_smooth:
        #replace the river and sea label into water
        draw_smooth = best_smooth_method(draw_raw)
        draw_smooth[np.where(draw_smooth == 147)] = 177
        draw_smooth[np.where(draw_smooth == 154)] = 177
    else:
        draw_smooth = draw_raw
#     plt.imshow(draw_smooth)
#     plt.show()
    
    #smooth groundtruth_image if required
    if g_smooth:
        g_unique_class, g_unique_counts = np.unique(gt_raw, return_counts=True)
        #Need to smooth gt_labels as well

        result = np.where(g_unique_counts > g_smooth_thred) #3000 is a bit too much
        dominant_class = g_unique_class[result].tolist()
        gt_smooth = merge_noisy_pixels(gt_raw,dominant_class)
        gt_smooth[np.where(gt_smooth == 147)] = 177
        gt_smooth[np.where(gt_smooth == 154)] = 177
    else:
        gt_smooth = gt_raw
    
#     plt.imshow(gt_smooth)
#     plt.show()
        
    #Now let's compute the relevant locations
    draw_set = np.unique(draw_smooth).tolist()
    #print("Unique Labels in draw_smooth: {}".format(draw_set))
    gt_set = np.unique(gt_smooth).tolist()
    #print("Unique Labels in gt_smooth: {}".format(gt_set))
    shared_labels =  set(draw_set).intersection(set(gt_set))
    #print("Shared labels between draw_smooth and gt_smooth: {}".format(shared_labels))
    
    #Find the centers of each region in shared label
    draw_shared = find_label_centers(draw_smooth, shared_labels, noisy_filter = noisy_filter_thred)
    gt_shared = find_label_centers(gt_smooth, shared_labels, noisy_filter = noisy_filter_thred)

    if relevant_mode == "unknown_count":
        #Find the centers of each region in unshared label
        draw_unshared = find_label_centers(draw_smooth, set(draw_set)-shared_labels, noisy_filter_thred)
        gt_unshared = find_label_centers(gt_smooth, set(gt_set)-shared_labels, noisy_filter_thred)
    else:
        draw_unshared = set(draw_set) - shared_labels
        gt_unshared = set(gt_set) - shared_labels
        
    #Resolve the unmatched pairs between draw_shared and gt_shared
    gt_draw_shared = pair_objects(draw_shared, gt_shared)
    if len(gt_draw_shared) > 0:
        #decouple the gt_draw_shared to a list of turples
        shared_item_list = []
        for key, value in gt_draw_shared.items():
            for d_center,gt_center in zip(value['draw_center'],value['gt_center']):
                shared_item_list.append((d_center, gt_center))
        if relevant_mode == "unknown_count":
            #decouple the unshared objects to a list of turples
            unshared_item_list = []
            for key, value in draw_unshared.items():
                unshared_item_list += value
            for key, value in gt_unshared.items():
                unshared_item_list += value
        else:
            unshared_item_list = list(draw_unshared)+list(gt_unshared)

        #compute the numerator score
        score = 0
        for x in range(len(shared_item_list)):
            for y in range(x+1, len(shared_item_list)):
                score += relevant_score(shared_item_list[x], shared_item_list[y])

        
        #compute the denomenator
        union = len(unshared_item_list)
        #union = len(draw_unshared) + len(gt_unshared)
        for key, value in gt_draw_shared.items():
            union += value['max_num_objects']
        #print("Union is: {}".format(union))
        intersection = len(shared_item_list)
        if intersection > 1:
            #print(score)
            final_score = score/(union*(intersection-1))
        else:
            final_score = score/union
    else:
        final_score = 0
    return final_score

def gaugancodraw_eval_metrics(label_d, label_gt, n_class, d_smooth=False, g_smooth=False, g_smooth_thred=1000, score_1_mode ="meanIoU", score_2_mode = "unknown_count"):
    #check if smooth is applied
    if d_smooth:
        #replace the river and sea label into water
        draw_smooth = best_smooth_method(label_d)
        draw_smooth[np.where(draw_smooth == 147)] = 177
        draw_smooth[np.where(draw_smooth == 154)] = 177
    else:
        draw_smooth = label_d
#     plt.imshow(draw_smooth)
#     plt.show()
    
    #smooth groundtruth_image if required
    if g_smooth:
        g_unique_class, g_unique_counts = np.unique(label_gt, return_counts=True)
        #Need to smooth gt_labels as well

        result = np.where(g_unique_counts > g_smooth_thred) #3000 is a bit too much
        dominant_class = g_unique_class[result].tolist()
        print(dominant_class)
        gt_smooth = merge_noisy_pixels(gt_raw,dominant_class)
        gt_smooth[np.where(gt_smooth == 147)] = 177
        gt_smooth[np.where(gt_smooth == 154)] = 177
    else:
        gt_smooth = label_gt
    
#     plt.imshow(gt_smooth)
#     plt.show()
    #compute mean_IOU
    if score_1_mode == "meanIoU":
        score_1 = mean_IoU(gt_smooth,draw_smooth, n_class)
    elif score_1_mode == "pixelAccuracy":
        score_1 = pixel_accuracy(gt_smooth, draw_smooth, n_class)
    
    #print("score_1 is: {}".format(score_1))
    #compute relevant_score
    score_2 = relevant_eval_metrics(draw_smooth, gt_smooth, relevant_mode=score_2_mode)
    #print("score_2 is: {}".format(score_2))
    
    final_score = 2*score_1+3*score_2
    return final_score

def compute_scene_sim_score_metrics(data_gen_file, data_gt_file):
    """
    compute the evaluation results of the generated files vs groundtruth files
    """
    result_mean_gt = 0
    result_mean_gt_meanIoU = 0
    result_gt = []
    result_gt_meanIoU = []
    for gen_im_path, gt_im_path in zip(data_gen_file, data_gt_file):
        gen_raw = cv2.imread(gen_im_path)
        gen_raw = cv2.cvtColor(gen_raw, cv2.COLOR_BGR2RGB)
        gen_seg = segconverter(gen_raw)

        gt_raw = cv2.imread(gt_im_path)
        gt_raw = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2RGB)
        gt_seg = segconverter(gt_raw)
        
        c_meanIoU = mean_IoU(gt_seg,gen_seg, 182)
        c_eval_score = gaugancodraw_eval_metrics(gen_seg, gt_seg, 182, score_1_mode="meanIoU", score_2_mode="unknown")
        result_gt_meanIoU.append(c_meanIoU)
        result_gt.append(c_eval_score)
        
    #print(result_gt)
    result_mean_gt = sum(result_gt)/len(result_gt)
    result_mean_gt_meanIoU = sum(result_gt_meanIoU)/len(result_gt_meanIoU)
        
    return result_mean_gt, result_mean_gt_meanIoU, result_gt, result_gt_meanIoU

def compute_label_prediction_metrics(data_gen_file, data_gt_file):
    precision_acc = 0
    recall_acc = 0
    acc = 0
    F1 = 0
    #count the most missed_labels
    missed_label_count = np.zeros(22)
    total_label_count = np.zeros(22)

    for gen_im_path, gt_im_path in zip(data_gen_file, data_gt_file):
        gen_raw = cv2.imread(gen_im_path)
        gen_raw = cv2.cvtColor(gen_raw, cv2.COLOR_BGR2RGB)
        gen_seg = segconverter(gen_raw)
        gen_seg_set = set(np.unique(gen_seg))
        
        #print(np.unique(gen_seg))
        
        gt_raw = cv2.imread(gt_im_path)
        gt_raw = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2RGB)
        gt_seg = segconverter(gt_raw)
        gt_seg_set = set(np.unique(gt_seg))

        tp = gen_seg_set.intersection(gt_seg_set)

        fn_fp_tp = gen_seg_set.union(gt_seg_set)
        #Update the precision_acc & recall_acc
        current_precision = len(tp)/len(gen_seg_set)
        current_recall = len(tp)/len(gt_seg_set)
        precision_acc += current_precision
        recall_acc += current_recall
        acc += len(tp)/len(fn_fp_tp)
        if current_precision > 0 or current_recall > 0:
            F1 += 2*(current_precision*current_recall)/(current_precision+current_recall)

        #missed labels count
        missed_set = gt_seg_set.difference(tp)
        for l in list(missed_set):
            missed_label_count[L2NORML_RAW[l]['index']] += 1

        #total label count
        for l in list(gt_seg_set):
            total_label_count[L2NORML_RAW[l]['index']] += 1

    i = 0    
    for x,y in zip(missed_label_count, total_label_count):
        if y > 0:
            missed_label_count[i] = x/y
        i += 1 
            
    precision_acc = precision_acc / len(data_gen_file)
    recall_acc = recall_acc / len(data_gen_file)
    acc = acc /len(data_gen_file)
    F1 = F1 / len(data_gen_file)
    return precision_acc, recall_acc, acc, F1, missed_label_count

def _plot_scalar_metric(visualizer, value, iteration, metric_name, split_name="val"):
    visualizer.plot(metric_name, split_name, iteration, value)

#Final Metrics to Run
def report_gandraw_eval_result(visualizer, iteration, data_path, use_test=False, use_human=False):
    #Load the generated image path and ground truth image path
    #Extract the generation folder nad gt folder
    if use_test:
        split_name = "test"
    else:
        split_name = "val"
        #split_name =  "test"

    all_dir = os.listdir(data_path)
    if ".DS_Store" in all_dir:
        all_dir.remove(".DS_Store")
    data_gt_path = sorted([x for x in all_dir if re.search("gt", x) and not re.search('DA', x) and re.search(split_name, x)], key = lambda x: int(x.split('_')[0]))
    data_gen_path = sorted([x for x in all_dir if x not in data_gt_path and not re.search('DA', x) and re.search(split_name, x)], key = lambda x: int(x.split('_')[0]))
    data_gt_path = [os.path.join(data_path,x) for x in data_gt_path]
    data_gen_path = [os.path.join(data_path,x) for x in data_gen_path] 

    #print(data_gen_path)

    folder_index = []

    #extract corresponding_generation_file
    data_gen_file = []
    for gen_path in data_gen_path:
        all_images = sorted([x for x in os.listdir(gen_path) if re.search(".png",x) and re.search("{}".format(iteration), x)], key = lambda x: int(x.split('_')[0]))        
        try:
            data_gen_file.append(os.path.join(gen_path, all_images[-1]))
            folder_index.append(gen_path.split('/')[-1])
        except:
            print(gen_path)

    #extract corresponding_gt_file
    data_gt_human_file = []
    data_gt_file = []
    for gt_path in data_gt_path:
        all_images = sorted([x for x in os.listdir(gt_path) if re.search(".png",x) and re.search("target", x) is None], key = lambda x: int(x.split('.')[0]))        
        data_gt_human_file.append(os.path.join(gt_path, all_images[-1]))
        data_gt_file.append(os.path.join(gt_path, "target.png"))
    
    #print("number of human_gt_files: {}".format(len(data_gt_human_file)))
    #print("number of gt_files: {}".format(len(data_gt_file)))

    if use_human:
        scene_similarity_score, meanIoU, _, _ = compute_scene_sim_score_metrics(data_gen_file, data_gt_human_file)
        precision, recall, acc, F1, _ = compute_label_prediction_metrics(data_gen_file, data_gt_human_file)
    else:    
        scene_similarity_score, meanIoU, _, _ = compute_scene_sim_score_metrics(data_gen_file, data_gt_file)
        precision, recall, acc, F1, _ = compute_label_prediction_metrics(data_gen_file, data_gt_file)
    
    
    if  visualizer is not None:
        _plot_scalar_metric(visualizer, scene_similarity_score, iteration, 'scene_similarity_score', split_name=split_name)
        _plot_scalar_metric(visualizer, meanIoU, iteration, 'meanIoU', split_name=split_name)
        _plot_scalar_metric(visualizer,precision, iteration, 'precision', split_name=split_name)
        _plot_scalar_metric(visualizer, recall, iteration, 'recall', split_name=split_name)
        _plot_scalar_metric(visualizer, acc, iteration, 'accuracy', split_name=split_name)
        _plot_scalar_metric(visualizer, F1, iteration, 'F1', split_name=split_name)

    return scene_similarity_score, meanIoU, precision, recall, acc, F1

