import streamlit as st
import sys
import os

# Add the parent directory to the Python path to import the face detection modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detector import *
from main import extract_and_filter_data, evaluate, plotter, plot_heatmap
import time
import matplotlib.pyplot as plt

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

def run_single_model_evaluation(detector, splits=['val'], iou_threshold=0.5):
    with st.spinner(f'Evaluating {detector.name}...'):
        data_dict = extract_and_filter_data(splits)
        model_start = time.time()
        result = evaluate(detector, data_dict, iou_threshold)
        model_end = time.time()
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average IOU", f"{result['average_iou']:.4f}")
            st.metric("Mean Average Precision", f"{result['mean_average_precision']:.4f}")
        with col2:
            st.metric("Average Inference Time", f"{result['average_inferencing_time']:.4f}")
            st.metric("F1 Score", f"{result['F1_score']:.4f}")
        
        # Display confusion matrix heatmap
        st.subheader("Detection Results Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap_data = np.array([[result['TP'], result['FN']], [result['FP'], result['TN']]])
        im = ax.imshow(heatmap_data, cmap='Blues', alpha=0.8)
        plt.colorbar(im)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Positive', 'Negative'])
        ax.set_yticklabels(['Positive', 'Negative'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Detection Results Heatmap - {detector.name}')
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, heatmap_data[i, j], ha='center', va='center', color='black')
        
        st.pyplot(fig)
        plt.close()

def run_model_comparison(detectors, splits=['val'], iou_threshold=0.5):
    with st.spinner('Running comparison of all models...'):
        results = {}
        progress_bar = st.progress(0)
        
        for idx, (name, detector) in enumerate(detectors.items()):
            data_dict = extract_and_filter_data(splits)
            result = evaluate(detector, data_dict, iou_threshold)
            results[name] = result
            progress_bar.progress((idx + 1) / len(detectors))
        
        # Create comparison plots
        metrics = {
            'Average IOU': 'average_iou',
            'Mean Average Precision': 'mean_average_precision',
            'Average Inference Time': 'average_inferencing_time',
            'F1 Score': 'F1_score'
        }
        
        for title, metric in metrics.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            values = [results[name][metric] for name in detectors.keys()]
            ax.bar(list(detectors.keys()), values, color='#7db6d9')
            ax.set_xlabel('Models')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

def main():
    st.title("Face Detection Model Benchmark")
    
    # Initialize detectors
    detectors = initialize_detectors()
    
    # Create navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Single Model Evaluation", "Model Comparison"]
    )
    
    if page == "Single Model Evaluation":
        st.header("Single Model Evaluation")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Face Detection Model",
            list(detectors.keys())
        )
        
        if st.button("Run Evaluation"):
            run_single_model_evaluation(detectors[selected_model])
            
    else:  # Model Comparison
        st.header("Model Comparison")
        
        if st.button("Run Comparison"):
            run_model_comparison(detectors)

if __name__ == "__main__":
    main() 