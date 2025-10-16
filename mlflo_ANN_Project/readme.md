# 🍷 Wine Quality Prediction using **MLflow**, **TensorFlow**, and **Hyperopt**

### 🚀 End-to-End ML Experiment Tracking and Optimization Pipeline  

This project demonstrates a **complete MLOps workflow** built around **MLflow** — from data preprocessing to model training, hyperparameter tuning, automatic experiment tracking, and model versioning.  

The use case: **Predicting wine quality scores** (0–10) using physicochemical features like acidity, sugar, and alcohol percentage.

---

## 🧠 Project Objective

Develop an automated, traceable, and reproducible pipeline that:  
1. Trains a deep learning regression model using **TensorFlow (Keras)**.  
2. Optimizes hyperparameters using **Hyperopt (TPE algorithm)**.  
3. Tracks all experiments, parameters, and metrics using **MLflow**.  
4. Logs, registers, and retrieves models from the **MLflow Model Registry**.  

---

## 🧾 Data Overview
**Dataset:** Wine Quality (Red or White)  

Each row = one wine sample  
Each column = chemical property of the wine  

| Feature | Description |
|----------|--------------|
| fixed acidity | Tartaric acid concentration |
| volatile acidity | Acetic acid content |
| citric acid | Citric component presence |
| residual sugar | Sugar after fermentation |
| chlorides | Salt content |
| pH | Acidity level |
| sulphates | Sulphate level |
| alcohol | Alcohol percentage |
| quality | Target variable (0–10) |

---

## ⚙️ MLflow-Centric Pipeline Architecture

```text
                ┌──────────────────────────────────────────────┐
                │                MLflow Tracking               │
                │----------------------------------------------│
                │ 1️⃣ Log parameters (lr, momentum)             │
                │ 2️⃣ Log metrics (validation RMSE)             │
                │ 3️⃣ Log TensorFlow model + signature          │
                │ 4️⃣ Auto-version in Model Registry            │
                └──────────────────────────────────────────────┘
                               ▲                 │
                               │                 ▼
                        Hyperopt (TPE)       TensorFlow Model
                           │                      │
                     Optimize params          Generate predictions
                           │                      │
                           ▼                      ▼
                       MLflow Run            MLflow Model Artifact
```

---

## 🔍 Key MLflow Components Demonstrated

| MLflow Feature | Description |
|----------------|--------------|
| **`mlflow.start_run()`** | Automatically starts and groups experiment runs |
| **`mlflow.log_params()`** | Logs hyperparameters for each model run |
| **`mlflow.log_metric()`** | Tracks validation metrics (e.g., RMSE) |
| **`mlflow.tensorflow.log_model()`** | Logs the trained Keras model artifact |
| **`infer_signature()`** | Automatically stores model input/output schema |
| **`mlflow.pyfunc.load_model()`** | Loads logged model for inference |
| **Model Registry (`models:/`)** | Manages versioned models for deployment |

---

## 🧪 Model Inference via MLflow PyFunc

```python
model_uri = "models:/m-b34793258c1340b5a27cbb9073a72b11"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Convert test data to float32 (TensorSpec-safe)
X_test_np = X_test.astype(np.float32)
preds = loaded_model.predict(X_test_np)
```

✅ **Why PyFunc?**  
Because it allows universal deployment of your MLflow model —  
you can load and run the same model with **scikit-learn, TensorFlow, or REST API clients**,  
without needing to reimplement preprocessing or architecture.

---

## 📈 Results and Tracking

- Metrics Logged: `eval_rmse`
- Parameters Logged: `learning_rate`, `momentum`
- Best RMSE: _Automatically tracked in MLflow UI_

Launch MLflow UI:
```bash
mlflow ui
```
Then open [http://localhost:5000](http://localhost:5000)  

🧭 Navigate to:  
- **Experiments tab:** Compare Hyperopt trials  
- **Models tab:** View registered model versions  
- **Artifacts tab:** Download logged models and JSON signatures  

---

## 🧰 Tools Used

| Tool | Purpose |
|------|----------|
| **MLflow** | Experiment tracking, model logging, registry |
| **TensorFlow/Keras** | Model definition and training |
| **Hyperopt** | Bayesian hyperparameter tuning |
| **Scikit-learn** | Data splitting and evaluation |
| **NumPy / Pandas** | Data handling and analysis |

---
## 📊 Final Model Results

| Metric | Value |
|---------|--------|
| **Best RMSE** | **0.7656** |
| **Learning Rate** | 0.001373 |
| **Momentum** | 0.735255 |
| **Optimizer** | SGD with Momentum |
| **Framework** | TensorFlow (Keras) |
| **Tracked in** | MLflow Experiment: /wine-quality |

✅ The tuned model achieves an RMSE of 0.7656,  
showing over 7× improvement compared to the baseline model (RMSE ≈ 5.72).


## 🧠 Key MLflow Learnings

1. ✅ Created **nested experiment structure** to manage multiple tuning runs.  
2. ✅ Logged **TensorFlow models** with automatic signatures using `mlflow.tensorflow.log_model()`.  
3. ✅ Managed **model lifecycle** via **MLflow Model Registry** (`models:/` URI).  
4. ✅ Demonstrated **cross-framework deployment** with MLflow PyFunc.  
5. ✅ Addressed **TensorSpec dtype enforcement** (float64 → float32).  
6. ✅ Automated **parameter tracking and metric visualization** across all trials.  

---

## 🧩 Summary
This project proves practical MLOps knowledge by integrating:
- **MLflow** for full model lifecycle management  
- **Hyperopt** for efficient search  
- **TensorFlow** for model training  
into one reproducible, automated workflow — showcasing your ability to build, optimize, and track deep learning systems **end-to-end**.

---

## 👨‍💻 Author
**Gaurav Sinha (Owner Sir)**  
📍 MLOps Enthusiast | Data Analyst | AI Practitioner  
🚀 Passionate about automation, tracking, and GenAI-based AI workflows.
