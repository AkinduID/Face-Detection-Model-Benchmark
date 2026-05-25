import streamlit as st
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path to import the face detection modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detector import *
from main import extract_and_filter_data, evaluate

CHART_LABEL_ROTATION = 35
CHART_LABEL_ALIGNMENT = 'right'

# Initialize face detectors
@st.cache_resource
def initialize_detectors():
    return {
        'Haar Cascades': OpenCVHaarFaceDetector(scaleFactor=1.3, minNeighbors=5, model_path='models/haarcascade_frontalface_default.xml'),
        'MP Blazeface': MediaPipeBlazeFaceDetector(),
        'MP Holistics': MediaPipeHolisticDetector(),
        'MobileNetSSD': TensorFlowMobilNetSSDFaceDetector(),
        'YOLOv8n': YOLOFaceDetector()
    }

@st.cache_data(show_spinner=False)
def load_dataset(splits):
    return extract_and_filter_data(list(splits))

def render_metrics(result):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average IOU", f"{result['average_iou']:.4f}")
    col2.metric("Mean Average Precision", f"{result['mean_average_precision']:.4f}")
    col3.metric("Average Inference Time", f"{result['average_inferencing_time']:.4f}")
    col4.metric("F1 Score", f"{result['F1_score']:.4f}")

def render_heatmap(result, model_name):
    st.subheader("Detection Results Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    heatmap_data = np.array([[result['TP'], result['FN']], [result['FP'], result['TN']]])
    im = ax.imshow(heatmap_data, cmap='Blues', alpha=0.8)
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Detection Results Heatmap - {model_name}')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, heatmap_data[i, j], ha='center', va='center', color='black')

    st.pyplot(fig)
    plt.close(fig)

def render_comparison_charts(results):
    metrics = {
        'Average IOU': 'average_iou',
        'Mean Average Precision': 'mean_average_precision',
        'Average Inference Time': 'average_inferencing_time',
        'F1 Score': 'F1_score'
    }
    for title, metric in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        values = [results[name][metric] for name in results.keys()]
        ax.bar(results.keys(), values, color='#7db6d9')
        ax.set_xlabel('Models')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        plt.xticks(rotation=CHART_LABEL_ROTATION, ha=CHART_LABEL_ALIGNMENT)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def run_benchmark(selected_models, detectors, splits, iou_threshold):
    if not selected_models:
        st.warning("Select at least one model to benchmark.")
        return {}

    overall_status = st.empty()
    model_status = st.empty()
    overall_progress = st.progress(0)
    model_progress = st.progress(0)

    overall_status.text("Loading dataset...")
    data_dict = load_dataset(tuple(splits))
    total_models = len(selected_models)
    results = {}

    for idx, model_name in enumerate(selected_models):
        detector = detectors[model_name]
        overall_status.text(f"Evaluating {model_name} ({idx + 1}/{total_models})")

        def progress_callback(current, total):
            progress_value = current / total if total else 0
            model_progress.progress(progress_value)
            model_status.text(f"{model_name}: {current}/{total} images")

        result = evaluate(detector, data_dict, iou_threshold, progress_callback=progress_callback)
        results[model_name] = result
        model_progress.progress(1.0)
        model_status.text(f"{model_name}: completed")
        overall_progress.progress((idx + 1) / total_models)

        with st.expander(f"{model_name} results", expanded=total_models == 1):
            render_metrics(result)
            render_heatmap(result, model_name)

    overall_status.text("Benchmark complete.")
    return results

def main():
    st.set_page_config(page_title="Face Detection Model Benchmark", layout="wide")
    st.title("Face Detection Model Benchmark")
    st.caption("Select one or more models to benchmark on the WIDER FACE validation split.")

    detectors = initialize_detectors()
    model_options = list(detectors.keys())

    st.sidebar.header("Benchmark Settings")
    selected_models = st.sidebar.multiselect(
        "Select models",
        model_options,
        default=model_options
    )
    iou_threshold = st.sidebar.slider(
        "IoU threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    if st.sidebar.button("Run Benchmark"):
        results = run_benchmark(selected_models, detectors, splits=['val'], iou_threshold=iou_threshold)
        if results:
            st.subheader("Benchmark Comparison")
            render_comparison_charts(results)

if __name__ == "__main__":
    main()