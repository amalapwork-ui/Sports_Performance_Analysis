Sports Performance Analysis System

An end-to-end **Sports Analytics Platform** that leverages **Machine Learning**, **Deep Learning (YOLOv8)**, and **modern web frameworks** to analyze football player performance, market value, injury risk, and real-time player tracking from match videos.

---

## 📌 Project Overview

Modern football relies heavily on data-driven decision making for:
- Player performance evaluation
- Market value estimation
- Injury prevention
- Match and tactical analysis

This project builds a **production-ready sports analytics system** that combines:
- **Regression-based predictive analytics** (FIFA dataset)
- **Computer Vision-based player tracking** (YOLOv8)
- **FastAPI backend**
- **Streamlit frontend**

---

## 🚀 Key Features

### 🔹 Machine Learning
- Predicts **Overall Player Rating** using **XGBoost Regressor**
- Predicts **Market Value** using **Random Forest Regressor**
- Trained on **FIFA 19 Complete Player Dataset**

### 🔹 Injury Risk Assessment
- Synthetic injury risk classification
- Factors used:
  - Age
  - Stamina
  - Work Rate
- Risk categories:
  - Low
  - Medium
  - High

### 🔹 Deep Learning (Computer Vision)
- Real-time **player detection & tracking** using **YOLOv8**
- Detects players as `person` class
- Assigns **unique tracking IDs** per player
- Outputs annotated match video

### 🔹 Deployment
- RESTful API using **FastAPI**
- Interactive UI using **Streamlit**
- Cross-platform (Windows / Linux)
- Docker-ready architecture

---

## 🧠 Tech Stack

| Layer | Technology |
|----|----|
| Language | Python |
| ML | scikit-learn, XGBoost |
| DL | YOLOv8 (Ultralytics), OpenCV |
| Backend | FastAPI |
| Frontend | Streamlit |
| Dataset | FIFA 19 Complete Player Dataset |



---

## 📊 Dataset

- **FIFA 19 Complete Player Dataset**
- ~18,000 professional football players
- Attributes include:
  - Physical
  - Technical
  - Mental
  - Financial

> Dataset used only for academic and learning purposes.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/sports-performance-analytics.git
cd sports-performance-analytics
```
### 2️⃣ Create Virtual Environment
python -m venv myvenv
source myvenv/bin/activate   # Linux / Mac
myvenv\Scripts\activate      # Windows


###3️⃣ Install Dependencies
pip install -r requirements.txt


###▶️ Running the Application
Start Backend (FastAPI)
uvicorn app.main:app --reload --port 8001

Start Frontend (Streamlit)
streamlit run frontend.py
