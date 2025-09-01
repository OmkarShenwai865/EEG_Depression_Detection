## EEG-Based Depression Detection Using Machine Learning
## 📌 Overview

Depression is one of the leading causes of disability worldwide, affecting over 264 million people globally and 56 million in India alone. Current diagnostic methods rely heavily on subjective assessments (e.g., Beck Depression Inventory, Hamilton Depression Rating Scale), which may lead to misdiagnosis, underdiagnosis, or delayed treatment.

This project explores EEG (Electroencephalography)-based depression detection as an objective, quantifiable approach. EEG is non-invasive, cost-effective, and provides high temporal resolution, making it a promising tool for clinical support.

The project develops an automated machine learning framework to classify depression from EEG signals under Eye Open (EO) and Eye Close (EC) conditions, with both subject-independent and subject-dependent evaluations.

## 🎯 Objectives

Preprocess EEG signals and extract frequency-domain features using FFT.

Train and evaluate multiple machine learning models (SVM, Logistic Regression, KNN, Random Forest, AdaBoost, XGBoost).

Compare classification under Eye Open (EO) and Eye Close (EC) conditions.

Ensure robustness using subject-independent 5-fold cross-validation.

Deploy the best-performing models in an interactive Streamlit application for real-time predictions.

## 📊 Dataset

Dataset Used: Mumtaz2016 EEG Dataset

The EEG dataset used in this project is publicly available on [Figshare](https://figshare.com/articles/dataset/EEG-based_Diagnosis_and_Treatment_Outcome_Prediction).  

Subjects:

87 Healthy controls

95 Patients with Major Depressive Disorder (MDD)

Recording Conditions: Eye Open (EO) and Eye Close (EC)

## ⚙️ Methodology

Preprocessing: EEG signal cleaning and segmentation.

Feature Extraction: Frequency-domain features derived from FFT across delta, theta, alpha, beta, and gamma bands.

Feature Selection: Identification of discriminative features (e.g., T6_beta, T6_theta, O2_theta).

Model Training: ML models trained separately for EO and EC under both validation settings.

Evaluation: Subject-dependent vs. subject-independent performance comparison.

## ⚙️ Installation

### Clone the repository:

git clone https://github.com/yourusername/eeg-depression-detection.git
cd eeg-depression-detection


### Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows


### Install dependencies:

pip install -r requirements.txt

## ▶️ Running the Project
1. Run the Jupyter Notebooks (for training & evaluation)

PROJECT_IITNR(SCRATCH) FOR EO.ipynb → Eye Open condition

PROJECT_IITNR(SCRATCH) FOR EC.ipynb → Eye Closed condition

You can open them in Jupyter Notebook or VS Code and re-run training steps.

2. Run the Streamlit App (for deployment)
streamlit run app.py


The app will launch in your browser (default: http://localhost:8501
), where you can upload EEG features and get predictions (Depressed / Healthy).

## 🚀 Models and Results

### Eye Open (EO)
- **Best Model (Subject-Dependent): XGBoost (97.15%)**
- Independent performance peaks with **KNN (92.9%)**

### Eye Closed (EC)
- **Best Model (Subject-Dependent): AdaBoost (96.39%)**
- Independent performance peaks with **SVM (95.4%)**

---

## 📊 Accuracy Comparison

### Eye Open (EO)
| Model               | Subject-Independent | Subject-Dependent |
|----------------------|---------------------|-------------------|
| SVM                 | 92.0%              | 94.02%           |
| XGBoost             | 91.9%              | **97.15%**       |
| Random Forest       | 91.7%              | 96.33%           |
| KNN                 | **92.9%**          | 96.07%           |
| AdaBoost            | 91.6%              | 96.74%           |
| Logistic Regression | 90.0%              | 93.21%           |

### Eye Closed (EC)
| Model               | Subject-Independent | Subject-Dependent |
|----------------------|---------------------|-------------------|
| SVM                 | **95.4%**          | 92.19%           |
| XGBoost             | 95.1%              | 96.24%           |
| Random Forest       | 94.4%              | 95.37%           |
| KNN                 | 94.9%              | 94.22%           |
| AdaBoost            | 93.4%              | **96.39%**       |
| Logistic Regression | 90.8%              | 90.46%           |

---
## 🚀 Deployment

The selected models (XGBoost for EO and AdaBoost for EC) are integrated into a Streamlit web app.

The app supports real-time prediction by processing EEG input and classifying subjects as Depressed / Healthy.

Currently hosted on localhost as a proof-of-concept, demonstrating feasibility of clinical deployment.

## Repository Structure

.
├── .streamlit/ # Streamlit configuration files
├── models/ # Saved trained models
├── app.py # Streamlit app for deployment
├── PROJECT_IITNR(SCRATCH) FOR EC.ipynb # Jupyter notebook for Eye Closed (EC)
├── PROJECT_IITNR(SCRATCH) FOR EO.ipynb # Jupyter notebook for Eye Open (EO)
├── requirements.txt # Required dependencies

## 🔮 Future Work

Extend study to larger and diverse datasets.

Incorporate deep learning architectures (CNN, RNN, Transformers) for raw EEG signal classification.

Explore real-time EEG acquisition and classification pipelines.

Develop a cloud-based deployment for clinical testing.

## 👨‍💻 Author

Name: Omkar Shenwai

Role: CSE Student (handled ML model design, evaluation,prediction & deployment)

Contributions: Data preprocessing, feature extraction, model development, accuracy comparison, and Streamlit deployment.  

## ⚠️ Disclaimer

This project is a proof-of-concept research framework and not a clinically certified diagnostic tool. It aims to demonstrate technical feasibility for potential clinical support applications.