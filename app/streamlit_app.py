import os
import sys
import tempfile
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Ensure root and src are on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ultralytics import YOLO
from src.rule_engine import process_detections

# Page Config
st.set_page_config(
    page_title="PPE Workplace Safety Monitor",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

## LOAD MODEL (Cached)
@st.cache_resource
def load_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"))
    return YOLO(model_path)

model = load_model()

## HEADER
st.title("👷 Industrial PPE Workplace Safety Monitor")
st.markdown("Automated Computer Vision pipeline detecting workers, personal protective equipment (PPE), and safety violations in real time.")

## SIDEBAR SETTINGS
st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
mode = st.sidebar.radio("Select Input Mode", ["📸 Image Upload", "🎥 Sample / Custom Video"])

st.sidebar.markdown("---")
st.sidebar.info("**Classes Monitored:**\n- 👷 Worker\n- 🪖 Hard Hat / No-Helmet\n- 🦺 Safety Vest / No-Vest\n- 🧤 Gloves / No-Glove\n- 🥾 Boots / No-Boots\n- 👓 Safety Glasses")

## DRAWING FUNCTION
def annotate_image(image_bgr, conf):
    results = model(image_bgr, conf=conf, verbose=False)[0]
    
    workers, ppe_items = process_detections(
        boxes=results.boxes.xyxy.cpu().numpy(),
        classes=results.boxes.cls.cpu().numpy(),
        confidences=results.boxes.conf.cpu().numpy(),
        names=model.names
    )
    
    annotated = image_bgr.copy()
    
    # 1. Draw individual PPE items
    for ppe in ppe_items:
        px1, py1, px2, py2 = map(int, ppe["box"])
        is_violation = ppe["is_violating"]
        color = (0, 0, 255) if is_violation else (0, 255, 0)
        label = f"{ppe['class']} ({ppe['conf']:.2f})"
        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
        cv2.putText(annotated, label, (px1, max(py1 - 6, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 2. Draw Workers
    for idx, worker in enumerate(workers):
        wx1, wy1, wx2, wy2 = map(int, worker["box"])
        is_violating = worker["is_violating"]
        worker_color = (0, 0, 255) if is_violating else (0, 255, 0)
        status_text = "VIOLATION" if is_violating else "COMPLIANT"
        
        label = f"Worker #{idx+1} [{status_text}]"
        cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), worker_color, 3)
        cv2.putText(annotated, label, (wx1, max(wy1 - 10, 25)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, worker_color, 2)

    return annotated, workers, ppe_items


## MODE 1: IMAGE UPLOAD
if mode == "📸 Image Upload":
    st.subheader("Upload an Image for Safety Compliance Inspection")
    uploaded_file = st.file_uploader("Choose a JPG/PNG image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing PPE compliance with YOLOv8 & Rule Engine..."):
            annotated_bgr, workers, ppe_items = annotate_image(img_bgr, conf_threshold)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Metric Cards
        total_workers = len(workers)
        violations = sum(1 for w in workers if w["is_violating"])
        compliant = total_workers - violations

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👷 Total Workers", total_workers)
        col2.metric("✅ Compliant Workers", compliant)
        col3.metric("🚨 Violations Flagged", violations, delta=-violations if violations > 0 else 0, delta_color="inverse")
        col4.metric("🛡️ Total PPE Items", len(ppe_items))

        # Side by Side Images
        c1, c2 = st.columns(2)
        with c1:
            st.image(pil_img, caption="Original Input Image", use_container_width=True)
        with c2:
            st.image(annotated_rgb, caption="Annotated Compliance Output", use_container_width=True)

        # Detailed Worker Breakdown Table
        if workers:
            st.markdown("### 📋 Detailed Worker Compliance Report")
            table_data = [
                {
                    "Worker ID": f"Worker #{idx+1}",
                    "Status": "🚨 VIOLATION" if w["is_violating"] else "✅ COMPLIANT",
                    "Detected Equipment": ", ".join(w["equipment"]) if w["equipment"] else "None Detected",
                    "Bounding Box": str([int(c) for c in w["box"]])
                }
                for idx, w in enumerate(workers)
            ]
            st.table(table_data)


## MODE 2: VIDEO PROCESSING
elif mode == "🎥 Sample / Custom Video":
    st.subheader("Video Pipeline & Safety Stream Processing")
    
    video_source = st.radio("Choose Video Source:", ["Use Project Sample Video", "Upload Custom Video (.mp4 / .avi)"])
    
    video_path = None
    if video_source == "Use Project Sample Video":
        sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_video.avi"))
        if os.path.exists(sample_path):
            video_path = sample_path
        else:
            st.error("Sample video not found in data/ folder.")
    else:
        uploaded_video = st.file_uploader("Upload video file...", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_path = tfile.name

    if video_path and st.button("🚀 Process Video Stream"):
        cap = cv2.VideoCapture(video_path)
        st_frame = st.empty()
        
        prog_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 30
        curr_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            curr_frame += 1
            annotated_frame, workers, _ = annotate_image(frame, conf_threshold)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            st_frame.image(rgb_frame, caption=f"Live Feed - Frame {curr_frame}/{total_frames}", use_container_width=True)
            prog_bar.progress(min(curr_frame / total_frames, 1.0))
            
        cap.release()
        st.success("Video Stream Processing Complete!")
