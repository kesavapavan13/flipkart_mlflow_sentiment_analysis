# 🚀 Flipkart Sentiment Analysis using MLflow & Prefect (MLOps Project)

## 📌 Project Overview
This project demonstrates how to integrate **MLflow** for experiment tracking and model management into a real-world machine learning pipeline.  
The system performs **Sentiment Analysis** on Flipkart product reviews and showcases practical **MLOps workflows**, including experiment tracking, hyperparameter comparison, model registry, and workflow automation using **Prefect**.

The objective was to move beyond traditional model training and build a **reproducible and scalable ML pipeline**.

---

## 🎯 Key Features
- ✅ Experiment tracking using **MLflow**
- ✅ Logging parameters, metrics, and artifacts
- ✅ Custom run naming and organized experiment UI
- ✅ Metric visualization and hyperparameter comparison
- ✅ Model Registry with versioning and tagging
- ✅ Automated workflow orchestration using **Prefect**
- ✅ Streamlit-based sentiment analysis web application

---

## 🧠 Tech Stack
- Python
- Scikit-learn
- MLflow
- Prefect
- Streamlit
- Pandas & NumPy
- TF-IDF Vectorizer
- Logistic Regression

---

## 📂 Project Structure
```
flipkart-mlflow-sentiment-analysis/
│
├── app.py                 # Streamlit application
├── train_mlflow.py        # MLflow experiment training pipeline
├── prefect_flow.py        # Prefect workflow automation
├── cleaned_data.csv       # Processed dataset
├── notebook.ipynb         # Model development notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```
git clone <your-github-repo-link>
cd flipkart-mlflow-sentiment-analysis
```

### 2️⃣ Create Virtual Environment
```
python -m venv myenv
myenv\Scripts\activate
```

### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

---

## 🧪 Run MLflow Experiment Tracking

Start MLflow UI:
```
mlflow ui
```

Open in browser:
```
http://127.0.0.1:5000
```

Run training pipeline:
```
python train_mlflow.py
```

---

## 🔄 Prefect Workflow (Automation)

Start Prefect Server:
```
prefect server start
```

Open Dashboard:
```
http://127.0.0.1:4200
```

Run workflow:
```
python prefect_flow.py
```

---

## 🖥️ Run Streamlit App
```
streamlit run app.py
```

Enter a Flipkart review and the system predicts:

- ✅ Positive Review
- ❌ Negative Review

---

## 📊 MLflow Capabilities Demonstrated
- Experiment Tracking
- Parameter & Metric Logging
- Artifact Storage
- Hyperparameter Visualization
- Model Versioning
- Model Tagging

---

## 💡 Learning Outcomes
This project helped me understand real-world **MLOps practices**, including:

- Managing ML experiments efficiently
- Comparing model performance visually
- Automating ML pipelines
- Organizing model lifecycle using MLflow Registry

---

## 📸 Screenshots
### 🔬 MLflow 
![MLflow Experiment](image\workflow.png)

### ⚙️ Prefect Dashboard
![Prefect Dashboard](image\dashboard.png)

### 🖥️ Streamlit App
![Streamlit App](image\streamlit.png)

---

## 🔗 Connect With Me
If you find this project useful, feel free to connect and collaborate!

---

## ⭐ Acknowledgements
Special thanks to the internship program for providing hands-on exposure to MLflow, Prefect, and modern MLOps workflows.
