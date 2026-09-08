# 🏗️ ArchMind Pro V2.2

## AI-Powered Construction Material Intelligence

ArchMind Pro is an AI-powered construction intelligence prototype that analyzes 3D structural geometry and project parameters to estimate construction material requirements.

The application combines:

- 3D STL geometry processing
- Feature engineering
- Custom-trained machine learning models
- Construction project parameters
- Streamlit deployment
- Hugging Face model hosting

---

## 🚀 Features

### 3D Geometry Analysis

Users can upload an STL structural model and select the model's unit:

- Feet
- Inches
- Meters
- Millimeters

ArchMind extracts geometry-related information and converts it into model features.

### Project Configuration

Users can configure:

- Building type
- Foundation depth
- Wall thickness
- Number of floors
- Floor height
- Soil condition
- Masonry type
- Structural intensity

### Material Prediction

The V2.2 system predicts:

- Cement bags
- Steel rebar tonnes
- Brick count
- AAC block count

---

## 🧠 Machine Learning

ArchMind Pro V2.2 uses four independently trained ML models.

| Target | Model File |
|---|---|
| Cement | `cement_bags_model.joblib` |
| Steel | `steel_tonnes_model.joblib` |
| Bricks | `brick_count_model.joblib` |
| AAC Blocks | `aac_block_count_model.joblib` |

The models are hosted on Hugging Face and downloaded automatically by the Streamlit application.

---

## ☁️ Model Hosting

Models:

`AloneMrY/archmind-pro-v2-2-models`

The application uses the Hugging Face Hub to retrieve the models at runtime.

This keeps large ML model files out of the GitHub source repository.

---

## 🛠️ Technology Stack

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Joblib
- Trimesh
- Hugging Face Hub

---

## 🔄 Architecture

STL Structural Model
        ↓
3D Geometry Processing
        ↓
Feature Engineering
        ↓
Project Configuration
        ↓
V2.2 ML Models
        ↓
Material Predictions
        ↓
Construction Planning Insights

---

## ⚠️ Important Disclaimer

The V2.2 models were trained using a synthetic engineering-informed dataset.

The outputs are intended for:

- ML prototyping
- Construction planning experiments
- Demonstration
- Hackathon development

They are NOT intended to replace:

- Structural engineering calculations
- Structural drawings
- Bill of Quantities
- Quantity surveying
- Professional engineering certification

---

## 👨‍💻 Project

ArchMind Pro

AI-powered construction material intelligence platform.