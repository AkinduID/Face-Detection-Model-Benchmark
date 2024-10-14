# This codebase is built upon the initial work from the following repository:
# https://github.com/nodefluxio/face-detector-benchmark

# Modifications were made to adapt the code for the specific needs of this project,
# including the selection of evaluated models, evaluation metrics, and output formatting.

import cv2
import numpy as np
from face_detector import *
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    boxA = np.array( [ xmin, ymin, xmax, ymax ] )
    boxB = np.array( [ xmin, ymin, xmax, ymax ] )

    Returns
    -------
    float
        in [0, 1]
    """

    bb1 = {'x1': boxA[0], 'y1': boxA[1], 'x2': boxA[2], 'y2': boxA[3]}
    bb2 = {'x1': boxB[0], 'y1': boxB[1], 'x2': boxB[2], 'y2': boxB[3]}

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['y1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return max(0.0, min(1.0, iou))


def extract_and_filter_data(splits):
    bb_gt_collection = dict()

    for split in splits:
        with open(os.path.join('dataset', 'wider_face_split', f'wider_face_{split}_bbx_gt.txt'), 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.endswith('.jpg'):
                image_path = os.path.join('dataset', f'WIDER_{split}', 'images', line)
                bb_gt_collection[image_path] = []
            line_components = line.split(' ')
            if len(line_components) > 1 and int(line_components[7]) != 1:
                x1 = int(line_components[0])
                y1 = int(line_components[1])
                w = int(line_components[2])
                h = int(line_components[3])

                if w > 15 and h > 15:
                    bb_gt_collection[image_path].append(np.array([x1, y1, x1 + w, y1 + h]))

    return bb_gt_collection

def evaluate(face_detector, bb_gt_collection, iou_threshold):
    total_data = len(bb_gt_collection.keys())
    data_total_iou = 0
    data_total_precision = 0
    data_total_inference_time = 0
    valid_image_count = 0
    tp_count = 0
    fp_count = 0
    fn_count = 0
    tn_count = 0

    for i, key in tqdm(enumerate(bb_gt_collection), total=total_data):
        image_data = cv2.imread(key)
        if image_data is None:
            print(f'Error reading image {key}')
            continue
        valid_image_count += 1 
        face_bbs_gt = np.array(bb_gt_collection[key])
        total_gt_face = len(face_bbs_gt)

        start_time = time.time()
        face_pred = face_detector.detect_face(image_data)
        inf_time = time.time() - start_time
        data_total_inference_time += inf_time

        total_iou = 0
        pred_dict = dict()
        
        # Initialize arrays to track matched predictions
        matched_predictions = [False] * len(face_pred)
        tp = 0

        for gt in face_bbs_gt:
            max_iou_per_gt = 0
            best_pred_index = -1
            cv2.rectangle(image_data, (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 2)

            for i, pred in enumerate(face_pred):
                cv2.rectangle(image_data, (pred[0], pred[1]), (pred[2], pred[3]), (0, 0, 255), 2)
                iou = get_iou(gt, pred)

                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                    best_pred_index = i
            
            # If the best prediction's IoU is above the threshold, count it as TP
            if max_iou_per_gt >= iou_threshold and best_pred_index != -1:
                if not matched_predictions[best_pred_index]:
                    tp += 1
                    matched_predictions[best_pred_index] = True  # Mark this prediction as matched
            
            total_iou += max_iou_per_gt

        fp = sum(1 for matched in matched_predictions if not matched)  # Count unmatched predictions
        fn = total_gt_face - tp  # False negatives

        # Aggregate counts for overall metrics
        tp_count += tp
        fp_count += fp
        fn_count += fn

        if total_gt_face == 0 and len(face_pred) == 0:
            tn_count += 1

        # Calculate precision and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt_face if total_gt_face > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Average IoU for this image
        image_average_iou = total_iou / total_gt_face if total_gt_face > 0 else 0
        image_average_precision = precision
        
        data_total_iou += image_average_iou
        data_total_precision += image_average_precision

    result = {
        'average_iou': data_total_iou / valid_image_count if valid_image_count > 0 else 0,
        'mean_average_precision': data_total_precision / valid_image_count if valid_image_count > 0 else 0,
        'average_inferencing_time': data_total_inference_time / valid_image_count if valid_image_count > 0 else 0,
        'TP': tp_count,
        'FP': fp_count,
        'FN': fn_count,
        'TN': tn_count,
        'F1_score': f1_score,
        'Precision': precision,
        'Recall': recall
    }

    return result



def plotter(title, x, y, listx, listy, color='#7db6d9'):
    plt.clf()
    plt.bar(listx, listy, color=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    # plt.tick_params(axis='y', format='%.6f')
    plt.savefig(os.path.join('results',f'{title}_comparison_plot.png'))

def plot_heatmap(tp, fp, fn, tn, model_name):
    heatmap_data = np.array([[tp, fn], [fp, tn]])  # Adjusted for confusion matrix format
    plt.figure(figsize=(8, 6))  # Optional: Set figure size
    plt.imshow(heatmap_data, cmap='Blues', alpha=0.8)
    plt.colorbar()
    plt.xticks([0, 1], ['Positive', 'Negative'])
    plt.yticks([0, 1], ['Positive', 'Negative'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, heatmap_data[i, j], ha='center', va='center', color='black')
    plt.title(f'Heatmap of Detection Results of {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()  # Adjust layout to fit the labels
    plt.savefig(os.path.join('results',f'detection_results_heatmap_{model_name}.png'))

def run_benchmark():
    main_start=time.time()
    splits = ['val']
    iou_threshold = 0.5
    haar_detector = OpenCVHaarFaceDetector(scaleFactor=1.3, minNeighbors=5, model_path='models/haarcascade_frontalface_default.xml')
    blazeface_detector = MediaPipeBlazeFaceDetector()
    holistic_detector = MediaPipeHolisticDetector()
    mobilenet_detector=TensorFlowMobilNetSSDFaceDetector()
    yolov8n_detector = YOLOFaceDetector()

    # Name_List = ['OpenCV Haar Cascades', 'MediaPipe Blazeface', 'MediaPipe Holistics']
    # Model_List = [haar_detector, blazeface_detector, holistic_detector]

    # Name_List = ['MediaPipe Blazeface', 'MediaPipe Holistics']
    # Model_List = [blazeface_detector, holistic_detector]

    Name_List = ['Haar Cascades', 'MP Blazeface', 'MP Holistics','MobileNetSSD','YOLOv8n']
    Model_List = [haar_detector, blazeface_detector, holistic_detector, mobilenet_detector, yolov8n_detector]

    # Name_List = ['MobileNetSSD','YOLOv8n']
    # Model_List = [mobilenet_detector, yolov8n_detector]

    AvgIOU_List = []
    MAP_List = []
    AvgTime_List = []
    F1_Score_List = []

    for face_detector in Model_List:
        print(f'Evaluating {face_detector.name}')
        data_dict = extract_and_filter_data(splits)
        model_start=time.time()
        result = evaluate(face_detector, data_dict, iou_threshold)
        model_end=time.time()
        print(f'Time taken for evaluate {face_detector.name} = {model_end-model_start}')
        print('Average IOU =', result['average_iou'])
        AvgIOU_List.append(result['average_iou'])

        print('mAP =', result['mean_average_precision'])
        MAP_List.append(result['mean_average_precision'])

        print('Average inference time =', result['average_inferencing_time'])
        AvgTime_List.append(result['average_inferencing_time'])


        print('F1 Score =', result['F1_score'])
        F1_Score_List.append(result['F1_score'])
        print('Precision =', result['Precision'])
        print('Recall =', result['Recall'])
        print('TP =', result['TP'])
        print('FP =', result['FP'])
        print('FN =', result['FN'])
        print('TN =', result['TN'])

        plot_heatmap(result['TP'],result['FP'],result['FN'],result['TN'],face_detector.name)

        print(f'Evaluation of {face_detector.name} done\n')

    plotter("Average IOU", "Models", "AvgIOU", Name_List, AvgIOU_List)
    plotter("Mean Average Precision", "Models", "MAP", Name_List, MAP_List)
    plotter("Average inference time", "Models", "AvgTime", Name_List, AvgTime_List)
    plotter("F1 Score", "Models", "TP", Name_List, F1_Score_List)
    main_end=time.time()
    print(f'Total time taken = {main_end-main_start}')

if __name__ == '__main__':
    run_benchmark()
