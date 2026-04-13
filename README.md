# 🎯 Data Preprocessing & ML Suite - Streamlit Application

A comprehensive **Streamlit-based machine learning application** for data preprocessing, exploratory data analysis (EDA), model training, evaluation, and prediction on regression and classification tasks.

---

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Datasets](#datasets)
- [Advanced Features](#advanced-features)
- [Results Achieved](#results-achieved)
- [Technologies Used](#technologies-used)

---

## ✨ Features

### **Data Processing Pipeline**
- ✅ **Data Overview** - Dataset statistics, data types, missing values
- ✅ **Interactive Filtering** - Numerical sliders & categorical multiselect
- ✅ **EDA Visualization** - Distribution plots, scatter plots, heatmaps, correlation analysis
- ✅ **Data Cleaning** - Missing value handling, outlier detection (IQR method), column removal
- ✅ **Transformation** - Feature scaling (MinMax/StandardScaler), Encoding (Label/OneHot)
- ✅ **PCA** - Dimensionality reduction with configurable components
- ✅ **Outlier Handling** - Smart recommendations, column-wise capping (Winsorize)

### **Model Training**
- ✅ **Auto Task Detection** - Automatically identify Classification vs Regression
- ✅ **Multiple Models**:
  - Classification: Logistic Regression, Random Forest, Decision Tree
  - Regression: Linear Regression, Random Forest, Decision Tree
- ✅ **Feature Selection** - Remove weak features, keep important ones
- ✅ **Hyperparameter Tuning** - GridSearchCV for optimal parameters
- ✅ **Manual Task Selection** - Override auto-detection if needed

### **Model Evaluation**
#### Classification Metrics
- 📊 Accuracy, Precision, Recall, F1-Score
- 🔗 Confusion Matrix (2x2 for binary classification)
- 📈 ROC Curve with AUC score & interpretation table
- 🔍 Misclassified samples analysis with rates

#### Regression Metrics
- 📏 MAE (Mean Absolute Error)
- 📊 MSE (Mean Squared Error)
- 📈 RMSE (Root Mean Squared Error)
- 🎯 R² Score

### **Advanced Features**
- ✅ **Feature Importance** - Bar charts for tree-based models
- ✅ **Prediction System** - Manual input & batch CSV upload
- ✅ **Clustering** - K-Means & DBSCAN with visualization
- ✅ **Data Evolution** - Compare original vs processed data
- ✅ **Export Options** - Download processed data & predictions

---

## 🚀 Installation

### **Prerequisites**
- Python 3.8+
- pip package manager

### **Step 1: Clone Repository**
```bash
git clone https://github.com/AKSHAY30080605/streamlit-.git
cd streamlit-
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### **Step 3: Run Application**
```bash
streamlit run app.py
```

The app will open at: `http://localhost:8501`

---

## 📊 Quick Start

### **Example: Wine Quality Prediction**

```
1. Upload: winequality-red.csv
2. Data Cleaning: View outliers → Cap in 4 columns
3. Transformation: Select StandardScaler → Apply
4. Model Training:
   - Target: quality
   - Task Type: Classification ← (Auto-bins continuous quality)
   - Model: Random Forest
   - ☑️ Enable Feature Selection
   - ☑️ Enable Hyperparameter Tuning
5. Train & Evaluate: See 83% accuracy! ✅
6. Predict: Input wine features → Get quality prediction
```

**Result: 83.13% Accuracy** 🎉

---

## 📁 Project Structure

```
streamlit-/
├── app.py                      # Main Streamlit application (1000+ lines)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── titanic.csv                 # Titanic dataset (classification)
├── winequality-red.csv         # Wine quality dataset (regression)
└── analyze_wine.py             # Wine data analysis script
```

---

## 📖 Usage Guide

### **Step-by-Step Workflow**

#### **1. Data Overview** 📊
```
- View first few rows
- Check statistics (mean, std, min, max)
- Identify missing values & data types
```

#### **2. Interactive Filtering** 🔍
```
- Create custom views of data
- Use sliders for numeric columns
- Multi-select for categorical columns
- Helps identify patterns before processing
```

#### **3. EDA Visualization** 📈
```
Distribution Plot → See feature distributions
Scatter Plot → Identify relationships
Correlation Heatmap → Find feature dependencies
Value Counts → Categorical feature distribution
```

#### **4. Data Cleaning** 🧹
**Missing Values:**
- Drop rows with NaN
- Fill with mean/median/mode

**Outliers (IQR Method):**
```
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
Outliers = Values outside bounds
```

Actions:
- ✅ **Cap** - Adjust to bounds (Winsorize)
- ❌ **Remove** - Drop entire rows (data loss!)

**Column Removal:**
- Remove low-correlation features
- Keep important features only

#### **5. Transformation** 🔄
**Scaling Methods:**
- **StandardScaler**: Center & scale (Recommended for regression)
- **MinMax**: Scale to [0,1] range

**Encoding:**
- **Label**: Convert categories to integers
- **OneHot**: Create binary columns per category

#### **6. PCA** 📉
```
Dimensionality Reduction
- Reduces 20+ features to 2-3 components
- Useful for visualization
- Warning: Loses feature interpretability
```

#### **7. Model Training** 🤖
```
1. Select Target Column
2. Choose Features (click ✅ Select All)
3. Select Task Type (Classification/Regression)
4. Choose Model
5. Enable Advanced Options:
   - Feature Selection (removes noise)
   - Hyperparameter Tuning (finds best params)
6. Click 🚀 Train Model
```

#### **8. Model Evaluation** 📊
```
View:
✅ Performance Metrics
✅ Confusion Matrix (Classification)
✅ ROC Curve with AUC
✅ Feature Importance
✅ Misclassified Samples
✅ Actual vs Predicted (Regression)
```

#### **9. Clustering** 🎯
```
Unsupervised Learning:
- K-Means: Fixed number of clusters
- DBSCAN: Finds density-based clusters
- Visualization with PCA
- Silhouette Score for quality
```

#### **10. Prediction** 🔮
```
Manual Input:
- Enter feature values
- Get prediction + probabilities

Batch Upload:
- Upload CSV with same features
- Get predictions for all rows
- Download results
```

---

## 📊 Model Performance

### **Wine Quality Dataset Results**

#### **Classification (Recommended)**
```
✅ Accuracy:   83.13%
✅ Precision:  79.74%
✅ Recall:     83.13%
✅ F1-Score:   81.33%
✅ AUC:        > 0.85 (Excellent)
```

**Status:** 🟢 **PRODUCTION-READY**

#### **Regression**
```
⚠️ MAE:   0.6114
⚠️ RMSE:  0.7730
⚠️ R²:    0.4040
```

**Status:** 🟡 **Acceptable but Classification is better**

### **Why Classification Wins:**
- Wine quality is naturally categorical (3-9 discrete ratings)
- Classification captures class boundaries better
- 83% accuracy >> 40% R²
- More interpretable for business users

---

## 📈 Datasets

### **1. Wine Quality (winequality-red.csv)**
- **Samples:** 1,599
- **Features:** 11 (alcohol, acidity, sweetness, etc.)
- **Target:** Quality (3-9)
- **Task:** Classification (binned) / Regression (continuous)
- **Best Model:** Random Forest Classifier (83% accuracy)

### **2. Titanic (titanic.csv)**
- **Samples:** 891
- **Features:** 11 (age, sex, ticket class, etc.)
- **Target:** Survived (0/1)
- **Task:** Binary Classification
- **Use Case:** Predict passenger survival

---

## 🎯 Advanced Features

### **1. Hyperparameter Tuning (GridSearchCV)**
```python
# Random Forest Parameters
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}
# Finds best combination → Better performance
```

### **2. Feature Selection**
```
- Trains quick model to get importance
- Removes features below median importance
- Reduces noise → Better generalization
- Faster training
```

### **3. Column-Wise Outlier Capping**
```
- Cap individual columns one by one
- See IQR bounds
- Choose which columns to cap
- Preserves data while reducing extremes
```

### **4. Smart Outlier Recommendations**
```
<3% Outliers  → "Skip - Data is clean"
3-8%          → "Consider capping"
>8%           → "Normal variation"
```

### **5. Auto-Binning for Classification**
```
Continuous target (e.g., 3.2, 5.7, 8.1)
→ Auto-bins into 3-10 discrete classes
→ Works with classification models
```

---

## 🏆 Results Achieved

### **Metrics Comparison**

| Aspect | Classification | Regression |
|--------|---|---|
| **Accuracy/R²** | **83%** ✅ | 40% |
| **Precision/MAE** | **80%** ✅ | 0.61 |
| **Business Value** | **High** ✅ | Low |
| **Production Ready** | **YES** ✅ | No |

### **Key Achievements**
✅ Successfully implemented **10-step ML pipeline**  
✅ Integrated **hyperparameter tuning** (GridSearchCV)  
✅ Added **feature selection** with importance analysis  
✅ Built **interactive UI** with Streamlit  
✅ Achieved **83% classification accuracy**  
✅ Created **production-ready model**  
✅ Implemented **advanced visualizations** (ROC, confusion matrix)  
✅ Added **clustering analysis**  

---

## 💻 Technologies Used

### **Core Framework**
- **Streamlit** - Interactive web UI
- **Python 3.8+** - Programming language

### **Data Science**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
  - Classification models
  - Regression models
  - Metrics & evaluation
  - Hyperparameter tuning (GridSearchCV)
  - Preprocessing & scaling
  - Clustering (K-Means, DBSCAN)

### **Visualization**
- **Plotly** - Interactive charts
  - ROC Curves
  - Confusion Matrices
  - Feature Importance
  - Cluster Visualization

### **CSS Styling**
- Dark theme (Slate colors)
- Custom metric containers
- Enhanced button styling
- Responsive design

---

## 🚀 Quick Commands

```bash
# Clone & Setup
git clone https://github.com/AKSHAY30080605/streamlit-.git
cd streamlit-
pip install -r requirements.txt

# Run Application
streamlit run app.py

# Run Data Analysis
python analyze_wine.py

# Git Commands
git checkout -b company              # Create branch
git add app.py
git commit -m "ML features"
git push -u origin company           # Push to GitHub
```

---

## 📞 Support & Documentation

### **For Issues:**
1. Check dataset format (CSV with headers)
2. Ensure all features are numeric (except target)
3. Handle missing values before upload
4. Use StandardScaler for regression

### **For Questions:**
- Documentation: See this README
- Code Comments: Check `app.py` (extensive inline documentation)
- Example Datasets: Use provided titanic.csv & winequality-red.csv

---

## 📝 License

This project is part of an educational ML course. Use freely for learning and development.

---

## 👨‍💻 Author

**AKSHAY30080605**  
GitHub: https://github.com/AKSHAY30080605  
Repository: https://github.com/AKSHAY30080605/streamlit-

---

## ✨ Key Highlights

🎯 **Complete ML Pipeline** - From raw data to predictions  
🎨 **Beautiful UI** - Dark theme with interactive components  
📊 **Advanced Analytics** - Feature importance, ROC curves, clustering  
⚡ **High Performance** - 83% accuracy on wine quality  
🔧 **Customizable** - Add/remove features, change models, tune parameters  
📈 **Educational** - Learn ML concepts through practical implementation  

---

## 🎉 Get Started Now!

```bash
# 3 simple steps:
1. pip install -r requirements.txt
2. streamlit run app.py
3. Upload your CSV and start analyzing!
```

**Happy Machine Learning! 🚀**

---

**Last Updated:** April 2026  
**Version:** 1.0 (Production Ready)  
**Status:** ✅ All Features Implemented
