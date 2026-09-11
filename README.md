# 👷 Industrial PPE Workplace Safety & Compliance Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=yolo)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-teal?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE.md)

An end-to-end Computer Vision system designed to automate **Personal Protective Equipment (PPE)** compliance monitoring in industrial workplaces and construction sites. The pipeline detects workers, identifies protective equipment, associates safety gear with specific individuals using a **Spatial Rule Engine**, and flags non-compliance violations in real time.

---

## 📌 Key Capabilities

- **Multi-Class PPE Detection:** Detects `Worker`, `Hard_hat`, `Vest`, `Boots`, `Glove`, `Glass`, and violation classes (`No-Helmet`, `No-Vest`, `No-Boots`, `No-Glove`).
- **Spatial Rule Engine (IoU Association):** Evaluates Intersection over Union (IoU) to associate individual safety gear items with worker bounding boxes, determining worker-level compliance.
- **Optimized Video Pipeline:** Implements OpenCV frame extraction with adaptive temporal frame-skipping (reducing compute overhead by **~65%** for real-time ~20+ FPS throughput).
- **Dual Visualizer:** Highlights protective gear with green/red item boxes and renders full-body worker compliance status (`[COMPLIANT]` vs `[VIOLATION]`).
- **Dual-Tier Deployment:**
  - **FastAPI Microservice:** Typed REST endpoint (`/detect`) with Pydantic validation for automated CCTV / machine integration.
  - **Streamlit Dashboard:** Interactive operations interface supporting Image Upload, Video Stream processing, and Live Webcam feeds.

---

## 🏗️ Architecture & Pipeline Flow

```
[ Input Source (Image / CCTV Stream / Webcam) ]
                     │
                     ▼
[ Preprocessing & Stream Optimization ]  (OpenCV, Frame Sampling)
                     │
                     ▼
[ YOLOv8 Deep Learning Backbone ]        (Bounding Box & Class Inference)
                     │
                     ▼
[ Spatial Rule Engine ]                  (IoU Bounding Box Overlap Logic)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
[ FastAPI Microservice ]  [ Streamlit UI ]
(JSON REST Response)      (Live Visuals & Metrics)
```

---

## 🏷️ Monitored Safety Classes

| Category | Compliant Class | Violation / Missing Tag |
| :--- | :--- | :--- |
| **Personnel** | `Worker` | — |
| **Head Protection** | `Hard_hat` | `No-Helmet` |
| **Torso Visibility** | `Vest` | `No-Vest` |
| **Hand Protection** | `Glove` | `No-Glove` |
| **Footwear** | `Boots` | `No-Boots` |
| **Eye Protection** | `Glass` | `No-Glass` |

---

## 📂 Project Structure

```
PPE_Workplace_Safety/
├── api/
│   ├── main.py                  # FastAPI application with REST endpoints
│   └── schemas.py               # Pydantic request/response validation schemas
├── app/
│   └── streamlit_app.py         # Streamlit interactive monitoring dashboard
├── data/
│   ├── sample_video.avi         # Construction site sequence test clip
│   └── data.yaml                # Dataset annotation configuration
├── models/
│   └── best.pt                  # Fine-tuned YOLOv8 model weights
├── notebooks/
│   └── EDA_and_training.ipynb   # Exploratory Data Analysis & experiments
├── src/
│   ├── clean_dataset.py         # Dataset sanitization & class consolidation
│   ├── inference.py             # Image inference with dual-box visualization
│   ├── preprocessing.py         # Video pipeline with frame-skipping logic
│   └── rule_engine.py           # Spatial IoU worker-equipment association logic
├── requirements.txt             # Pinned project dependencies
├── LICENSE.md                   # MIT License
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/<your-username>/PPE_Workplace_Safety.git
cd PPE_Workplace_Safety

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Image Inference Test

```bash
python src/inference.py
```
*Outputs an annotated sample image to `results/inference_test_cv2.jpg`.*

### 3. Run Video Processing Pipeline

```bash
python src/preprocessing.py
```
*Processes `data/sample_video.avi` and exports annotated output to `results/annotated_video.avi`.*

---

## 🌐 Running the Applications

### Option A: Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```
Open **http://localhost:8501** in your browser to inspect images, process videos, or test live webcam feeds.

### Option B: Launch FastAPI REST Microservice

```bash
uvicorn api.main:app --reload --port 8000
```
Open **http://127.0.0.1:8000/docs** to test the interactive Swagger API documentation.

#### API Endpoint Example:

```bash
# Health Check
curl -X GET "http://127.0.0.1:8000/"

# Run Detection on Image
curl -X POST "http://127.0.0.1:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/construction_site.jpg"
```

**JSON Response:**
```json
{
  "filename": "construction_site.jpg",
  "total_workers": 2,
  "total_ppe_items": 4,
  "violations_detected": 1,
  "workers": [
    {
      "worker_id": 1,
      "box": [120.5, 85.0, 340.2, 590.8],
      "status": "COMPLIANT",
      "equipment": ["Hard_hat", "Vest", "Boots"]
    },
    {
      "worker_id": 2,
      "box": [450.1, 110.4, 620.0, 600.5],
      "status": "VIOLATION",
      "equipment": ["No-Helmet", "Vest"]
    }
  ]
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

