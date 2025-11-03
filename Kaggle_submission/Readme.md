# 🚢 Titanic: Machine Learning from Disaster

## 📘 Project Overview
This project is part of the **Kaggle Titanic: Machine Learning from Disaster** competition — a classic binary classification challenge that predicts passenger survival based on socio-economic and travel attributes.

What makes this project unique is that **the entire model was built from the ground up using PyTorch tensors**, focusing on understanding the mechanics of learning rather than relying on prebuilt layers or high-level APIs.

Through **incremental improvements and experimentation**, the model evolved from a simple linear setup to deeper neural architectures — resulting in a final **Kaggle ranking of 3,125 out of 15,414 participants** (Top 20%).

---

## 🎯 Objective
Predict whether a passenger survived the Titanic disaster based on available features such as age, sex, class, and fare.

---

## 🧩 Approach

### 1. Data Preprocessing
- Filled missing values using **mode imputation** for both categorical and continuous variables.  
- Created a new feature `LogFare` to normalize the **long-tailed fare distribution**.  
- Encoded categorical variables (`Sex`, `Pclass`, `Embarked`) using **one-hot encoding**.

### 2. Model Building
- **Phase 1:** Linear Model built manually using tensor operations and gradient descent.  
- **Phase 2:** Introduced **Sigmoid activation** to map outputs into probability range [0, 1].  
- **Phase 3:** Implemented **matrix multiplication** for efficient computation.  
- **Phase 4:** Extended to a **single-layer neural network** with ReLU activation.  
- **Phase 5:** Built a **deep neural network** with multiple hidden layers to observe effects on performance.

### 3. Optimization & Training
- Used **manual gradient computation** (`loss.backward()`) and parameter updates (`coeffs.sub_`) to understand how weights adjust during learning.  
- Experimented with **different learning rates**, normalization, and feature scaling to stabilize convergence.

---

## 📊 Results

| Model Type | Key Technique | Accuracy | Notes |
|-------------|----------------|-----------|-------|
| Linear Model | Gradient Descent + Sigmoid | **0.82** | Best performing, simple yet effective |
| Neural Network | ReLU + 1 Hidden Layer | 0.82 | Moderate improvement with non-linearity |
| Deep Neural Network | multiple Hidden Layers | 0.79 | Required further tuning |

🏅 **Final Kaggle Rank:** 3,125 / 15,414 (Top 20%)  
📈 **Improvement:** From bottom 20% → Top 20% through iterative learning

---

## 🧠 Key Learnings
- Gained deep understanding of **how gradient descent updates weights** and drives convergence.  
- Realized the **critical role of normalization and scaling** in stable model training.  
- Understood **how activations like Sigmoid** transform linear outputs into probabilities.  
- Observed that **depth doesn’t guarantee accuracy** — thoughtful tuning does.  
- Strengthened practical skills in **tensor operations, broadcasting, and matrix algebra** in PyTorch.  
- Developed a mindset of **measurable, stepwise improvement** rather than black-box model building.

---

## ⚙️ Tech Stack
- **Python**
- **PyTorch**
- **Pandas**, **NumPy**
- **FastAI (for data splitting only)**
- **Matplotlib / Sympy** (for visualization)

---

## 🚀 Future Improvements
- Implement **cross-validation** for better generalization.  
- Experiment with **regularization techniques** to reduce overfitting.  
- Explore **ensemble models** combining linear and neural architectures.  
- Integrate **PyTorch Lightning** for cleaner experiment management.

---

## 📁 Repository Structure
