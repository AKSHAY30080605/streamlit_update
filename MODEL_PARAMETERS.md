# 📘 Antigravity AI — Model Parameters Reference

> Complete reference for all tunable hyperparameters in the Data Suite.

---

## 🌲 Random Forest

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `n_estimators` | Integer | 10 – 1000 | 100 | Number of decision trees in the forest |
| `max_depth` | Integer | 0 – 100 | 0 (None) | Maximum depth of each tree. `0` = unlimited |
| `min_samples_split` | Integer | 2 – 50 | 2 | Minimum samples required to split a node |
| `min_samples_leaf` | Integer | 1 – 50 | 1 | Minimum samples required at a leaf node |
| `max_features` | Select | `sqrt`, `log2`, `None` | `sqrt` | Number of features considered at each split |
| `oob_score` | Boolean | True / False | False | Use out-of-bag samples to estimate accuracy |

**Classification**: `RandomForestClassifier`  
**Regression**: `RandomForestRegressor`

---

## 🌳 Decision Tree

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `criterion` | Select | Classification: `gini`, `entropy`, `log_loss` / Regression: `squared_error`, `friedman_mse`, `absolute_error`, `poisson` | `gini` / `squared_error` | Function to measure split quality |
| `splitter` | Select | `best`, `random` | `best` | Strategy to choose the split at each node |
| `max_depth` | Integer | 0 – 100 | 0 (None) | Maximum depth of the tree. `0` = unlimited |
| `min_samples_split` | Integer | 2 – 50 | 2 | Minimum samples required to split a node |
| `min_samples_leaf` | Integer | 1 – 50 | 1 | Minimum samples required at a leaf node |
| `max_features` | Select | `None`, `sqrt`, `log2` | `None` | Number of features considered at each split |

**Classification**: `DecisionTreeClassifier`  
**Regression**: `DecisionTreeRegressor`

---

## 📈 Logistic Regression (Classification Only)

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `penalty` | Select | `l2`, `l1`, `elasticnet`, `None` | `l2` | Regularization type |
| `C` | Float | 0.01 – 100.0 | 1.0 | Inverse of regularization strength (smaller = stronger) |
| `max_iter` | Integer | 100 – 5000 | 1000 | Maximum number of iterations for solver convergence |
| `fit_intercept` | Boolean | True / False | True | Whether to add a constant (bias) term |
| `multi_class` | Select | `auto`, `ovr`, `multinomial` | `auto` | Strategy for multi-class classification |

---

## 📏 Linear Regression (Regression Only)

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `fit_intercept` | Boolean | True / False | True | Whether to calculate the intercept (bias) |
| `positive` | Boolean | True / False | False | Force all coefficients to be positive |

---

## 👥 K-Nearest Neighbors (KNN)

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `n_neighbors` | Integer | 1 – 100 | 5 | Number of nearest neighbors to use for voting |
| `weights` | Select | `uniform`, `distance` | `uniform` | Weight function: `uniform` = equal vote, `distance` = closer neighbors have more influence |
| `metric` | Select | `minkowski`, `euclidean`, `manhattan` | `minkowski` | Distance metric for neighbor calculation |
| `leaf_size` | Integer | 1 – 100 | 30 | Leaf size for BallTree/KDTree (affects speed) |

**Classification**: `KNeighborsClassifier`  
**Regression**: `KNeighborsRegressor`

### 📊 Visualization
- **Decision Boundaries**: PCA-reduced 2D contour plot with Magma colorscale
- Auto-standardized for balanced principal components

---

## 🛡️ Support Vector Machine (SVM)

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `C` | Float | 0.01 – 100.0 | 1.0 | Regularization parameter. Larger C = tighter fit to training data |
| `kernel` | Select | `rbf`, `linear`, `poly`, `sigmoid` | `rbf` | Kernel function for the hyperplane |
| `gamma` | Select | `scale`, `auto` | `scale` | Kernel coefficient for `rbf`, `poly`, and `sigmoid` |
| `degree` | Slider | 1 – 5 | 3 | Degree of polynomial kernel (only visible when `kernel = poly`) |
| `epsilon` | Float | 0.01 – 1.0 | 0.1 | Margin of tolerance in SVR (only visible for Regression) |
| `probability` | Auto | — | `True` (Classification) | Enables `predict_proba()` for confidence scores |

**Classification**: `SVC`  
**Regression**: `SVR`

### 📊 Visualization
- **Decision Boundaries**: PCA-reduced 2D contour plot with Magma colorscale
- Chart subtitle shows kernel type and C value dynamically

---

## ⚙️ Global Training Options

| Option | Description |
|---|---|
| **Hyperparameter Tuning (GridSearchCV)** | Automatically searches for best parameters using 3-fold cross-validation. Available for Random Forest and Decision Tree. |
| **Feature Selection** | Trains a quick Random Forest to rank features by importance, then drops features below median importance. |
| **Test Size** | Percentage of data held out for testing (default: 20%). |
| **Auto-Imputation** | Missing values are automatically filled: numeric → mean, categorical → mode. |
| **Safe Encoding** | Unseen categories during prediction fall back to `0` instead of crashing. |

---

## 📋 Evaluation Metrics

### Classification
| Metric | Formula | Description |
|---|---|---|
| Accuracy | (TP+TN) / Total | Overall correctness |
| Precision | TP / (TP+FP) | How many positive predictions were correct |
| Recall | TP / (TP+FN) | How many actual positives were found |
| F1-Score | 2 × (P×R)/(P+R) | Harmonic mean of precision and recall |

### Regression
| Metric | Description |
|---|---|
| MAE | Mean Absolute Error – average magnitude of errors |
| MSE | Mean Squared Error – penalizes larger errors |
| RMSE | Root MSE – same scale as target variable |
| R² Score | Proportion of variance explained (1.0 = perfect) |

---

*Generated by Antigravity AI Data Suite*
