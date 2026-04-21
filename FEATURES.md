# 🚀 Data Suite — Feature Architecture

The **Data Suite Prediction Engine v2** is a professional-grade machine learning platform designed to bridge the gap between complex mathematical modeling and intuitive user experience.

---

## 1. Data Intelligence & Ingestion
*   **Smart Ingestion**: Support for CSV datasets with automatic delimiter detection.
*   **Integrity Dashboard**: Real-time stats on dataset shape, feature counts, and target distribution.
*   **Automated Cleaning**: 
    *   One-click missing value imputation (Mean/Median for numbers, Mode for text).
    *   Dynamic feature dropping for irrelevant identifiers (e.g., PassengerId, Names).
*   **Semantic Detection**: Automatically distinguishes between categorical (nominal) and numerical features.

## 2. Professional Preprocessing Pipeline
*   **Encoding Engine**: Robust One-Hot Encoding for categorical data with internal mapping to preserve human-readability.
*   **Scaling Suite**: Choice of professional scalers:
    *   `StandardScaler` (Z-score normalization)
    *   `MinMaxScaler` (0-1 Range)
    *   `RobustScaler` (Outlier-resistant)
*   **Metadata Bridge**: Stores "Raw-to-Model" mappings so you never have to see mathematical scaled values in the UI.

## 3. High-Performance Modeling
*   **Multi-Model Support**:
    *   **Random Forest**: Robust ensemble for complex patterns.
    *   **Decision Tree**: Interpretable logic branches.
    *   **Logistic/Linear Regression**: Reliable statistical baselines.
    *   **KNN & SVM**: Advanced boundary-based learners.
*   **Interactive Hyperparameters**: Real-time tuning of depths, neighbors, kernels, and estimators via sidebar controls.
*   **Auto-Task Detection**: Swaps evaluation logic automatically between **Classification** (labels) and **Regression** (numbers).

## 4. Evaluation & Visualization (Premium)
*   **Interactive Diagnostics**:
    *   📊 **Feature Importance**: See exactly which columns drive your model.
    *   ⚖️ **Model Coefficients**: Horizontal impact charts for linear models.
    *   🗺️ **2D Decision Boundaries**: PCA-reduced contour maps showing exactly where the model draws its lines.
    *   📈 **ROC & AUC**: Visual performance metrics for binary classifiers.
    *   📉 **Residual Plots**: In-depth error analysis for regression tasks.
*   **Statistical Breakdown**: Premium metrics like F1-Score, RMSE, and R² Score presented with high-definition gauges.

## 5. Prediction Engine v2 (Human-Centric)
*   **Semantic Inputs**: Enter data as you see it in the real world (e.g., "Male", "Survived", "Category A"). No manual encoding required.
*   **Confidence Terminal**: 
    *   Displays prediction probabilities as dynamic bar charts.
    *   Premium Verdict Cards with glassmorphic backgrounds.
*   **Batch Prediction**: Upload a secondary CSV to generate predictions for thousands of rows at once.

## 6. UI/UX Design
*   **Premium Aesthetics**: Modern dark mode with a custom HSL color palette (Purple, Slate, Emerald).
*   **Glassmorphism**: Translucent UI components for a state-of-the-art feel.
*   **Instructional Intelligence**: Every step includes tooltips and technical explanations to guide the user.

---
*Created as part of the Data Suite Visualization Upgrade.*
