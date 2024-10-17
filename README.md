# Comparative Analysis of Lightweight Face Detection Models for Edge Devices

This repository contains the codebase and documentation for a comparative analysis of various lightweight face detection models optimized for real-time detection on edge devices. The study evaluates the models based on their accuracy, speed, and performance across different poses and environments, using the WIDER FACE dataset.

## Overview
Face detection is a fundamental task in computer vision, with applications in security, robotics, and human-computer interaction. Deploying these solutions on edge devices requires models that are both efficient and accurate. This project investigates and benchmarks several lightweight face detection models to determine the best options for such constrained environments.

### Models Evaluated
The following models are included in this benchmark:

* __Haar Cascade__ - A classic machine learning-based approach using handcrafted features. Although lightweight, it struggles with pose variations and complex lighting.

* __MediaPipe BlazeFace__ - An SSD-based model optimized for mobile and edge devices, trained specifically on selfie images for fast and accurate detection.

* __MediaPipe Holistic__ - A comprehensive solution integrating facial landmarks, pose estimation, and hand tracking, offering more context but at a higher computational cost.

* __MobileNet SSD__ - A deep learning model leveraging depthwise separable convolutions for efficient detection on mobile platforms.

* __YOLOv8 Nano__ - A compact version of the YOLO framework designed for real-time, high-accuracy detection on resource-constrained devices.

### Dataset
The evaluation uses the WIDER FACE validation subset, consisting of 3,226 images with faces in varied poses, lighting, and environments. This dataset provides a diverse range of conditions for testing model robustness and performance.
http://shuoyang1213.me/WIDERFACE/

### Metrics
The models are evaluated based on the following metrics:

* __Average IoU (Intersection over Union)__: Measures the accuracy of the detected bounding box compared to the ground truth.

* __Mean Average Precision (mAP)__: Assesses detection accuracy across different IoU thresholds (0.5, 0.75, etc.).

* __Average Inference Time__: The time taken for each model to process an image and output results, crucial for determining suitability for real-time applications.

## Installation
To set up the environment and run the code, follow these steps:
Clone the repository
```bash
git clone https://github.com/AkinduID/Face-Detection-Model-Benchmark.git
```
Navigate to the project directory
```bash
cd Face-Detection-Model-Benchmark
```
Install the required dependencies
```bash
pip install -r requirements.txt
```
Download the validation data set from the website mentioned under the Dataset section and place it in the *dataset* folder
```
+-- dataset
|   +-- WIDER_val
|   +-- wider_face_split
```

To evaluate the models, run the following command
```bash
python main.py
```
## Results
The results of the face detection model comparison are stored in the *results* folder. The key findings are summarized in the following comparison graphs generated using Matplotlib
* Average IoU vs Model
* Average Inference Time vs Model
* Average Mean Precision vs Model
<div class="image-container">
  <img src="https://github.com/AkinduID/Face-Detection-Model-Benchmark/blob/main/results/Average%20IOU_comparison_plot.png" width="300" alt="Avg IoU" />
  <img src="https://github.com/AkinduID/Face-Detection-Model-Benchmark/blob/main/results/Average%20inference%20time_comparison_plot.png" width="300" alt="Avg Inference Time" />
  <img src="https://github.com/AkinduID/Face-Detection-Model-Benchmark/blob/main/results/Mean%20Average%20Precision_comparison_plot.png" width="300" alt="Mean Avg Precision" />
</div>

Current result are achieved on a laptop with the follwing specifications:
* Processor - Intel i5-1135G7
* RAM - 8GB
* GPU - Intel Iris Xe 4GB
* Operating System - Windows 11

Total execution for the main script took 45 Minutes.

## Contributing
Contributions are welcome! If you would like to add new models or improve the existing evaluation metrics, feel free to fork this repository and submit a pull request.

## Future Work
* __Model Expansion__: Incorporate additional face detection models to broaden the comparative analysis.
* __Metric Enrichment__: Introduce further evaluation metrics, such as F1 score and confusion matrix, for a more comprehensive assessment.
* __Dataset Optimization__: Explore techniques to reduce the dataset size while preserving its diversity, thereby improving computational efficiency.
* __User Experience Enhancement__: Enhance the user-friendliness of the main program for easier operation and accessibility.

## References
This codebase is built upon the initial work from the following repository:
Modifications were made to adapt the code for the specific needs of this project,including the selection of evaluated models, evaluation metrics, and output formatting.
https://github.com/nodefluxio/face-detector-benchmark
