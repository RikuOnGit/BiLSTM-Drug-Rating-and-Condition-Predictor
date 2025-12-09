# Drug Review Rating & Condition Prediction Using Bidirectional LSTM

This project analyzes patient-written drug reviews from the Drugs.com dataset using natural language processing (NLP) and deep learning.

We build two neural models that learn from raw, unstructured text to extract meaningful clinical and emotional signals:

1. **Rating Prediction (Regression)** – Predicting numeric drug ratings (1–10) from review text.
2. **Condition Prediction (Multi-Class Classification)** – Predicting the medical condition associated with each review.

Together, these models demonstrate how sequence-based deep learning can interpret patient narratives and provide insights into treatment experiences.

> ⚠️ **Note:**  
> Trained model files (`best_bilstm_model.h5` and `best_condition_model.h5`) are **not included** in this repository.
> To reproduce the results, **run the notebook end-to-end** using the dataset from kaggle.

---

## 🔍 Project Overview

### **1. Rating Prediction (Regression)**

Predict the numeric drug rating (1–10) directly from text written by the patient.

- **Model:** Bidirectional LSTM  
- **Goal:** Predict satisfaction/effectiveness score  
- **Input:** Cleaned natural-language review  
- **Output:** A numeric rating  
- **Evaluation Metrics:**  
  - Mean Absolute Error (MAE)  
  - Root Mean Squared Error (RMSE)  
  - Pearson correlation  

**Results:**  
- MAE ≈ **1.19**  
- RMSE ≈ **1.84**  
- Correlation ≈ **0.83**

The model successfully learns sentiment, treatment effectiveness cues, and emotional tone from clinical reviews.

---

### **2. Condition Prediction (Classification)**

Predict which medical condition (e.g., Depression, Pain, Birth Control) the review refers to.

- **Model:** Bidirectional LSTM with softmax  
- **Goal:** Multi-class condition classification  
- **Classes:** Over **800+** unique medical conditions  
- **Key Challenges:**  
  - Extremely large number of output classes  
  - Heavy class imbalance (some conditions appear thousands of times, others only once)  
  - Overlapping clinical vocabulary across conditions  
  - Noisy, informal real-world patient text  

**Performance (BiLSTM):**

- **Training accuracy:** ~**67%**  
- **Validation accuracy:** ~**65%**  
- **Best epoch:** Restored via EarlyStopping (epoch 11)

Despite the complexity of the dataset and the large number of classes, the model captures meaningful patterns in patient language, including symptoms, side effects, treatment goals, and clinical context. This allows it to infer the correct condition from raw review text with solid accuracy for a non-transformer sequence model.

---

## 📁 Repository Structure

```
📦 project-root
├── bilstm_rating_condition.ipynb            # Main notebook (full project)
├── README.md                                # Project documentation
├── pyproject.toml                           # Project dependencies (uv)
└── requirements.txt (optional)              # Full environment export
```

> Note: Dataset is not included.

---

## 📊 Models Used

Both tasks use:

- **Text cleaning & preprocessing**
- **Tokenization and sequence padding**
- **Embedding layer** (learned from data)
- **Bidirectional LSTM** for sequence modeling
- **Dense layers** for regression/classification
- **EarlyStopping** and **ModelCheckpoint** for stability

---

## 🚀 Installation & Usage

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### **2. Install dependencies**

Using `uv`:
```bash
uv sync
```

or using pip:
```bash
pip install -r requirements.txt
```

### **3. Run the notebook**
```bash
jupyter notebook
```

---

## 📦 Dataset

This project uses the **Drugs.com Reviews Dataset**, available on Kaggle:

🔗 https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

> ⚠️ The dataset is **not included** in the repository.  
> Please download it manually from Kaggle before running the notebook.

Place the dataset files in the project directory:

```
drugsComTrain_raw.csv
drugsComTest_raw.csv
```

---

## 🧠 Future Improvements

Potential extensions:

- Transformer-based models (ClinicalBERT, BioBERT)
- Attention mechanism on top of the BiLSTM
- Balancing rare condition classes
- Hierarchical condition grouping
- Multi-label condition prediction
- Token-level importance visualization
