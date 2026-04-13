import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans, DBSCAN
import plotly.graph_objects as go

st.set_page_config(page_title="Data Preprocessing & EDA Suite", layout="wide")

# =========================
# HELPER FUNCTIONS
# =========================

def detect_task_type(y_series):
    """
    Auto-detect if task is Classification or Regression
    
    Logic:
    - If target is categorical (object/string) → Classification
    - If target is numeric with ≤ 20 unique values → Classification  
    - If target is numeric with > 20 unique values → Regression
    """
    if y_series.dtype == 'object':
        return 'Classification'
    
    unique_count = y_series.nunique()
    return 'Classification' if unique_count <= 20 else 'Regression'


def get_models_by_task(task_type):
    """Return available models based on task type"""
    if task_type == 'Classification':
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
    else:
        return {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }


# Custom CSS for Modern Design (Dark sleek theme)
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Modify sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0ea5e9 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Main text */
    p, span, div {
        color: #e2e8f0;
    }
    
    /* Metric labels - enhanced visibility */
    [data-testid="metric-container"] {
        background-color: rgba(14, 165, 233, 0.1);
        border-left: 4px solid #0ea5e9;
    }
    
    [data-testid="metric-container"] label {
        color: #38bdf8 !important;
        font-weight: 600 !important;
        font-size: 14px;
    }
    
    /* Markdown text enhancement */
    .markdown-text-container {
        color: #e2e8f0 !important;
    }
    
    /* Subheader styling */
    [data-testid="stVerticalBlock"] h3 {
        color: #0ea5e9 !important;
    }
    
    /* Caption styling */
    .stCaption {
        color: #cbd5e1 !important;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #0ea5e9;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #0284c7;
        color: #ffffff;
        border: 1px solid #38bdf8;
    }
    
    /* Style st.info and success boxes */
    div.stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Data Preprocessing & EDA Suite")

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    # Read the data and establish persistent state
    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != file.name:
        raw_df = pd.read_csv(file)
        
        # Ensure unique column names to prevent Plotly/Narwhals parsing errors
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]
        
        # Ensure strong types to prevent PyArrow serialization issues
        raw_df.columns = raw_df.columns.astype(str)
        for c in raw_df.select_dtypes(include=['object']).columns:
            raw_df[c] = raw_df[c].astype(str)
            
        st.session_state.current_df = raw_df
        st.session_state.original_df = raw_df.copy()
        st.session_state.uploaded_file = file.name

    # Load from persistent state and ALWAYS deduplicate columns
    df = st.session_state.current_df
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = df.columns.astype(str)
    st.session_state.current_df = df
    original_df = st.session_state.original_df
    original_df = original_df.loc[:, ~original_df.columns.duplicated()]

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("🔄 Reset Data to Original"):
        st.session_state.current_df = st.session_state.original_df.copy()
        st.rerun()
        
    st.sidebar.markdown("---")
    
    step = st.sidebar.radio("Select Suite Step", [
        "Data Overview",
        "Interactive Filtering",
        "EDA Visualization",
        "Data Cleaning",
        "Transformation",
        "PCA",
        "Model Training",
        "Model Evaluation",
        "Clustering"
    ])

    # =========================
    # DATA OVERVIEW
    # =========================
    if step == "Data Overview":
        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Statistics")
        st.write(df.describe())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Data Types")
        st.write(df.dtypes)

    # =========================
    # INTERACTIVE FILTERING
    # =========================
    elif step == "Interactive Filtering":
        st.subheader("Interactive Dataset Filtering")
        filtered_df = df.copy()
        
        col1, col2 = st.columns(2)
        
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        with col1:
            st.markdown("#### Numeric Filters")
            for col in num_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                with st.expander(f"Filter: {col}"):
                    if min_val == max_val:
                        st.info(f"All values in '{col}' are {min_val} — no range to filter.")
                    else:
                        range_val = st.slider(f"Select Range", min_val, max_val, (min_val, max_val), key=f'slider_{col}')
                        filtered_df = filtered_df[(filtered_df[col] >= range_val[0]) & (filtered_df[col] <= range_val[1])]
                    
        with col2:
            st.markdown("#### Categorical Filters")
            for col in cat_cols:
                unique_vals = df[col].unique().tolist()
                with st.expander(f"Filter: {col}"):
                    selected_vals = st.multiselect(f"Select Values", unique_vals, default=unique_vals, key=f'multi_{col}')
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
                    
        st.metric(label="Rows Remaining", value=len(filtered_df), delta=f"Initial: {len(df)}", delta_color="off")
        st.dataframe(filtered_df)
        
        if st.button("Save Filtered Data"):
            st.session_state.current_df = filtered_df
            st.rerun()

    # =========================
    # EDA VISUALIZATION
    # =========================
    elif step == "EDA Visualization":
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Plotly toolbar config: only show Download and Autoscale
        chart_cfg = {"modeBarButtonsToKeep": ["toImage", "autoScale2d"], "displaylogo": False}
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribution Plot")
            dist_col = st.selectbox("Select Numeric Column", num_cols, key="dist")
            if st.button("Show Distribution"):
                fig = px.histogram(df, x=dist_col, marginal="box", template="plotly_dark", color_discrete_sequence=['#38bdf8'])
                st.plotly_chart(fig, use_container_width=True, config=chart_cfg)

        with c2:
            st.subheader("Scatter Plot")
            scat_x = st.selectbox("X-axis", num_cols, key="scat_x")
            scat_y = st.selectbox("Y-axis", num_cols, key="scat_y")
            if st.button("Show Scatter"):
                if scat_x == scat_y:
                    st.warning("Please select two different columns for the scatter plot.")
                else:
                    pearson_corr = df[[scat_x, scat_y]].corr().iloc[0, 1]
                    st.metric(f"Pearson Correlation ({scat_x} vs {scat_y})", f"{pearson_corr:.3f}")
                    scatter_df = df[[scat_x, scat_y]].copy()
                    scatter_df.columns = [scat_x, scat_y]
                    fig = px.scatter(scatter_df, x=scat_x, y=scat_y, 
                                     template="plotly_dark", 
                                     color_discrete_sequence=['#a855f7'])
                    st.plotly_chart(fig, use_container_width=True, config=chart_cfg)
                
        st.markdown("---")
        
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Correlation Heatmap")
            if st.button("Show Heatmap"):
                numeric_df = df.select_dtypes(include=np.number).dropna()
                fig = px.imshow(numeric_df.corr(), text_auto=".2f", aspect="auto", template="plotly_dark", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True, config=chart_cfg)
                
        with c4:
            st.subheader("Categorical Value Counts")
            if len(cat_cols) > 0:
                cat_col = st.selectbox("Categorical Column", cat_cols)
                if st.button("Show Counts"):
                    counts = df[cat_col].value_counts().reset_index()
                    counts.columns = ['Value', 'Count']
                    fig = px.bar(counts, x='Value', y='Count', template="plotly_dark", color='Count', color_continuous_scale="Teal")
                    st.plotly_chart(fig, use_container_width=True, config=chart_cfg)


    # =========================
    # DATA CLEANING
    # =========================
    elif step == "Data Cleaning":
        st.subheader("Missing Values")
        method = st.selectbox("Method", ["Drop", "Mean", "Median", "Mode"])

        if st.button("Apply Missing Handling"):
            if method == "Drop":
                df = df.dropna()
            elif method == "Mean":
                df = df.fillna(df.mean(numeric_only=True))
            elif method == "Median":
                df = df.fillna(df.median(numeric_only=True))
            elif method == "Mode":
                df = df.fillna(df.mode().iloc[0])
            st.session_state.current_df = df
            st.rerun()

        st.markdown("---")
        st.subheader("Remove Columns")
        cols_to_remove = st.multiselect("Select Columns to Remove", df.columns.tolist(), key="remove_cols")
        if cols_to_remove:
            if st.button("Drop Selected Columns"):
                df = df.drop(columns=cols_to_remove)
                st.session_state.current_df = df
                st.rerun()

        st.markdown("---")
        st.subheader("Outlier Detection & Removal (IQR Method)")
        
        num_df = df.select_dtypes(include=np.number)
        
        if not num_df.empty:
            st.markdown("#### Outlier Analysis")
            outlier_results = []
            for col in num_df.columns:
                Q1 = num_df[col].quantile(0.25)
                Q3 = num_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = num_df[(num_df[col] < lower_bound) | (num_df[col] > upper_bound)]
                num_outliers = len(outliers)
                outlier_results.append({
                    'Column': col, 
                    'Number of Outliers': num_outliers, 
                    'Percent Outliers (%)': round((num_outliers / len(df)) * 100, 2)
                })
                
            # Highlight the column with the highest number of outliers
            outliers_df = pd.DataFrame(outlier_results)
            st.dataframe(outliers_df.set_index('Column').style.highlight_max(axis=0, subset=['Number of Outliers'], color='#a855f7'))
            
            # Smart recommendation for outlier handling
            max_outlier_pct = outliers_df['Percent Outliers (%)'].max()
            
            st.markdown("#### 💡 Outlier Handling Recommendation")
            if max_outlier_pct < 3:
                st.info("✅ **Very Few Outliers** (<3%): Skip outlier handling. Your data is clean!")
            elif max_outlier_pct < 8:
                st.warning(f"⚠️ **Moderate Outliers** ({max_outlier_pct:.1f}%): Consider **capping** (not removing) to preserve data.")
            else:
                st.warning(f"⚠️ **High Outliers** ({max_outlier_pct:.1f}%): This is normal for wine quality. Outliers represent natural variation. **Cap only if needed.**")
            
            st.markdown("**For Wine Quality Datasets:**")
            st.markdown("- ✅ **CAP Outliers** (Winsorize): Adjust extreme values, keep rows")
            st.markdown("- ❌ **Don't Remove**: Removing rows loses valuable data")
            st.markdown("- ⏭️ **Skip if <5%**: Very few outliers = no action needed")

            st.markdown("#### Visualize Outliers")
            col_outlier_viz = st.selectbox("Select Column to Plot:", num_df.columns)
            if col_outlier_viz:
                fig_outlier = px.box(df, y=col_outlier_viz, template="plotly_dark", color_discrete_sequence=['#f43f5e'], title=f"Outlier Plot: {col_outlier_viz}")
                st.plotly_chart(fig_outlier, use_container_width=True, config={"modeBarButtonsToKeep": ["toImage", "autoScale2d"], "displaylogo": False})

            st.markdown("#### Apply Outlier Handling")
            
            selected_outlier_cols = st.multiselect(
                "Select Columns to Apply Outlier Handling",
                num_df.columns.tolist(),
                default=num_df.columns.tolist(),
                key="outlier_cols"
            )
            
            if selected_outlier_cols:
                selected_num_df = num_df[selected_outlier_cols]
                Q1 = selected_num_df.quantile(0.25)
                Q3 = selected_num_df.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    st.markdown("**Remove Outliers**")
                    st.caption("Drops rows with outliers in selected columns.")
                    if st.button("Remove Outliers", key="remove_outliers"):
                        df = df[~((selected_num_df < lower_bound) | (selected_num_df > upper_bound)).any(axis=1)]
                        st.session_state.current_df = df
                        st.rerun()
                        
                with btn_col2:
                    st.markdown("**Cap All Outliers (Winsorize)**")
                    st.caption("Clips extreme values in all selected columns to IQR bounds.")
                    if st.button("Cap All Columns", key="cap_outliers"):
                        df_capped = df.copy()
                        for col in selected_outlier_cols:
                            df_capped[col] = df_capped[col].clip(lower=lower_bound[col], upper=upper_bound[col])
                        df = df_capped
                        st.session_state.current_df = df
                        st.rerun()
                
                st.markdown("---")
                st.markdown("#### Column-Wise Capping")
                st.caption("Cap outliers for individual columns")
                
                col_cap_col1, col_cap_col2 = st.columns([3, 1])
                
                with col_cap_col1:
                    cap_column = st.selectbox("Select Column to Cap:", selected_outlier_cols, key="cap_col_select")
                
                with col_cap_col2:
                    st.markdown("###")
                    if st.button("🔧 Cap This Column", key=f"cap_col_{cap_column}"):
                        df_capped = df.copy()
                        df_capped[cap_column] = df_capped[cap_column].clip(
                            lower=lower_bound[cap_column], 
                            upper=upper_bound[cap_column]
                        )
                        df = df_capped
                        st.session_state.current_df = df
                        st.success(f"✅ Capped outliers in '{cap_column}'")
                        st.rerun()
                
                # Show bounds for selected column
                if cap_column:
                    st.info(f"📊 **{cap_column}** - Bounds: [{lower_bound[cap_column]:.2f}, {upper_bound[cap_column]:.2f}]")
            else:
                st.warning("Please select at least one column.")
        else:
            st.info("No numerical columns found to analyze for outliers.")

        st.markdown("---")
        st.write("Current Dataset Preview:")
        st.write(df)


    # =========================
    # TRANSFORMATION
    # =========================
    elif step == "Transformation":
        st.subheader("Scaling")
        method = st.selectbox("Scaling Method", ["MinMax", "Standard"])

        if st.button("Apply Scaling"):
            num_cols = df.select_dtypes(include=np.number).columns
            scaler = MinMaxScaler() if method == "MinMax" else StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            st.session_state.current_df = df
            st.rerun()

        st.subheader("Encoding")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(cat_cols) > 0:
            enc = st.selectbox("Encoding Type", ["Label", "OneHot"])
            if st.button("Apply Encoding"):
                if enc == "Label":
                    for col in cat_cols:
                        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                else:
                    df = pd.get_dummies(df, columns=cat_cols)
                st.session_state.current_df = df
                st.rerun()

        st.write(df)


    # =========================
    # PCA
    # =========================
    elif step == "PCA":
        st.subheader("PCA")
        n = st.slider("Number of Components", 1, len(df.columns), 2)

        if st.button("Apply PCA"):
            df_clean = df.dropna()
            df_numeric = pd.get_dummies(df_clean)
            n_actual = min(n, len(df_numeric.columns))
            pca = PCA(n_components=n_actual)
            df = pd.DataFrame(pca.fit_transform(df_numeric), columns=[f"PC{i+1}" for i in range(n_actual)])
            st.session_state.current_df = df
            st.rerun()

        st.write(df)


    # =========================
    # MODEL TRAINING
    # =========================
    elif step == "Model Training":
        st.subheader("🤖 Model Training")
        
        # Initialize session state for model training
        if "trained_model" not in st.session_state:
            st.session_state.trained_model = None
            st.session_state.task_type = None
            st.session_state.feature_names = None
            st.session_state.X_train = None
            st.session_state.X_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Step 1: Select Target Column (y)")
            target_col = st.selectbox("Choose Target Column", df.columns.tolist(), key="target_col_select")
        
        with col2:
            st.markdown("#### Step 2: Train-Test Split")
            test_size = st.slider("Test Size (%)", 10, 40, 20, key="test_size")
            test_size = test_size / 100
        
        if target_col:
            # Feature selection - Define all_features FIRST
            st.markdown("#### Step 3: Select Features (X)")
            all_features = [col for col in df.columns.tolist() if col != target_col]
            
            # Initialize session state for features BEFORE creating widget
            if "features_select" not in st.session_state:
                st.session_state["features_select"] = all_features
            
            col_feat1, col_feat2 = st.columns([4, 1])
            
            with col_feat1:
                selected_features = st.multiselect(
                    "Choose Features", 
                    all_features,
                    default=st.session_state.get("features_select", all_features),
                    key="features_select"
                )
            
            with col_feat2:
                st.markdown("###")
                if st.button("✅ Select All", key="select_all_btn"):
                    st.session_state["features_select"] = all_features
                    st.rerun()
            
            if not selected_features:
                st.warning("⚠️ Please select at least one feature!")
            else:
                y = df[target_col]
                X = df[selected_features]
                
                st.markdown("---")
                st.markdown("#### Step 4: Select Task Type")
                task_type = st.selectbox("Choose Task Type", ["Classification", "Regression"], key="task_type_select")
                st.session_state.task_type = task_type
                st.caption(f"Target column '{target_col}' has {y.nunique()} unique values")
                
                st.markdown("---")
                st.markdown("#### Step 5: Select Model")
                
                available_models = get_models_by_task(task_type)
                model_name = st.selectbox("Choose Model", list(available_models.keys()), key="model_select")
                
                st.markdown("---")
                
                # Auto-convert target for classification/regression
                y_processed = y.copy()
                info_msg = ""
                
                if task_type == "Classification":
                    # Check if target is continuous and auto-bin it
                    if y.dtype in ['float64', 'float32']:
                        # Auto-bin continuous values into discrete classes
                        n_bins = min(10, max(3, y.nunique() // 5))
                        y_processed = pd.cut(y_processed, bins=n_bins, labels=False)
                        info_msg = f"ℹ️ **Auto-binned continuous target** into {y_processed.nunique()} classes for classification"
                    elif y.nunique() > 50:
                        st.warning(f"⚠️ **Too many classes** ({y.nunique()}) for Classification. Consider selecting Regression instead.")
                    else:
                        info_msg = f"✅ Classification task: {y_processed.nunique()} discrete classes"
                else:  # Regression
                    # Ensure numeric target for regression
                    if y.dtype not in ['float64', 'float32', 'int64', 'int32', 'int16', 'int8']:
                        st.error(f"❌ **Regression requires numeric target**, but target is categorical. Select Classification instead.")
                    else:
                        info_msg = f"✅ Regression task: continuous target"
                
                if info_msg:
                    st.info(info_msg)
                
                st.markdown("---")
                st.markdown("#### Step 6: Advanced Options")
                
                col_hp, col_fs = st.columns(2)
                
                with col_hp:
                    use_hyperparameter_tuning = st.checkbox("🔧 Enable Hyperparameter Tuning (GridSearchCV)", value=False)
                    st.caption("Automatically finds best parameters - may take longer")
                
                with col_fs:
                    use_feature_selection = st.checkbox("🎯 Feature Selection (Remove Weak Features)", value=False)
                    st.caption("Keep only important features - improves performance")
                
                st.markdown("---")
                
                if st.button("🚀 Train Model", key="train_btn"):
                    try:
                        X_for_training = X.copy()
                        progress_placeholder = st.empty()
                        
                        # Feature Selection: Remove weak features
                        if use_feature_selection:
                            progress_placeholder.info("📊 Step 1/3: Analyzing feature importance...")
                            
                            # Train a quick model to get feature importance
                            if model_name in ["Random Forest", "Decision Tree"]:
                                temp_model = available_models[model_name]
                                X_temp_train, _, y_temp_train, _ = train_test_split(
                                    X_for_training, y_processed, test_size=test_size, random_state=42
                                )
                                temp_model.fit(X_temp_train, y_temp_train)
                                
                                # Get feature importance
                                feature_importance = temp_model.feature_importances_
                                importance_df = pd.DataFrame({
                                    'Feature': X_for_training.columns,
                                    'Importance': feature_importance
                                }).sort_values('Importance', ascending=False)
                                
                                # Keep features with importance > median
                                median_importance = importance_df['Importance'].median()
                                important_features = importance_df[importance_df['Importance'] > median_importance]['Feature'].tolist()
                                
                                if len(important_features) > 0:
                                    X_for_training = X_for_training[important_features]
                                    st.success(f"✅ Selected {len(important_features)} important features (removed {len(X.columns) - len(important_features)})")
                                    st.dataframe(importance_df, use_container_width=True)
                            else:
                                st.warning("⚠️ Feature selection only works with Tree-based models. Using all features.")
                        
                        # Split data
                        progress_placeholder.info("📊 Step 2/3: Splitting data...")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_for_training, y_processed, test_size=test_size, random_state=42
                        )
                        
                        # Hyperparameter Tuning
                        if use_hyperparameter_tuning:
                            progress_placeholder.info("📊 Step 3/3: Tuning hyperparameters (this may take a minute)...")
                            
                            # Define parameter grids for different models
                            param_grids = {
                                "Random Forest": {
                                    "n_estimators": [100, 200],
                                    "max_depth": [5, 10, None],
                                    "min_samples_split": [2, 5]
                                },
                                "Decision Tree": {
                                    "max_depth": [5, 10, 15],
                                    "min_samples_split": [2, 5, 10]
                                }
                            }
                            
                            if model_name in param_grids:
                                grid_search = GridSearchCV(
                                    available_models[model_name],
                                    param_grids[model_name],
                                    cv=3,
                                    n_jobs=-1,
                                    verbose=0
                                )
                                grid_search.fit(X_train, y_train)
                                model = grid_search.best_estimator_
                                
                                st.success(f"✅ Best parameters found: {grid_search.best_params_}")
                                st.info(f"Best CV Score: {grid_search.best_score_:.4f}")
                            else:
                                st.warning(f"⚠️ Hyperparameter tuning not available for {model_name}. Using default parameters.")
                                model = available_models[model_name]
                                model.fit(X_train, y_train)
                        else:
                            progress_placeholder.info("📊 Step 3/3: Training model with default parameters...")
                            model = available_models[model_name]
                            model.fit(X_train, y_train)
                        
                        # Store in session state
                        st.session_state.trained_model = model
                        st.session_state.feature_names = X_train.columns.tolist()
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        
                        progress_placeholder.empty()
                        st.success(f"✅ Model trained successfully! ({model_name})")
                        st.info(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
                        
                    except Exception as e:
                        st.error(f"❌ Error training model: {str(e)}")
                
                st.markdown("---")
                st.markdown("#### Preview: Training Data")
                st.write(f"**Features (X):** {X.shape}")
                st.write(f"**Target (y):** {y.shape}")
                st.dataframe(X.head())

    # =========================
    # MODEL EVALUATION
    # =========================
    elif step == "Model Evaluation":
        st.subheader("📊 Model Evaluation & Metrics")
        
        if st.session_state.trained_model is None:
            st.warning("⚠️ No trained model found. Please train a model in 'Model Training' section first.")
        else:
            model = st.session_state.trained_model
            task_type = st.session_state.task_type
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            st.success(f"✅ Task Type: **{task_type}**")
            st.markdown("---")
            
            if task_type == 'Classification':
                # Ensure y_test is proper integer type for classification metrics
                y_test_class = y_test.astype(int) if hasattr(y_test, 'astype') else y_test
                y_pred_class = y_pred.astype(int) if hasattr(y_pred, 'astype') else y_pred
                
                # Classification metrics
                st.markdown("#### 📊 Classification Metrics")
                accuracy = accuracy_score(y_test_class, y_pred_class)
                precision = precision_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
                recall = recall_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
                f1 = f1_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📈 Accuracy", f"{accuracy:.4f}", delta=None)
                with col2:
                    st.metric("🎯 Precision", f"{precision:.4f}", delta=None)
                with col3:
                    st.metric("🔍 Recall", f"{recall:.4f}", delta=None)
                with col4:
                    st.metric("⚖️ F1-Score", f"{f1:.4f}", delta=None)
                
                st.markdown("---")
                
                # Confusion Matrix
                st.markdown("#### 🔗 Confusion Matrix")
                try:
                    # Check if binary classification
                    unique_classes = np.unique(y_test_class)
                    if len(unique_classes) == 2:
                        cm = confusion_matrix(y_test_class, y_pred_class)
                        
                        # Create custom annotations with values and percentages
                        labels = unique_classes
                        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                        
                        # Create annotation text with both count and percentage
                        annotation_text = []
                        for i in range(cm.shape[0]):
                            row = []
                            for j in range(cm.shape[1]):
                                row.append(f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)")
                            annotation_text.append(row)
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=labels,
                            y=labels,
                            text=annotation_text,
                            texttemplate="%{text}",
                            textfont={"size": 14, "color": "white"},
                            colorscale="Blues",
                            colorbar=dict(title="Count")
                        ))
                        
                        fig.update_layout(
                            title="Confusion Matrix (2x2)",
                            xaxis_title="Predicted Label",
                            yaxis_title="True Label",
                            template="plotly_dark",
                            height=500,
                            width=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"ℹ️ Confusion Matrix (2x2) is available only for binary classification. Your model has {len(unique_classes)} classes.")
                except Exception as e:
                    st.error(f"❌ Could not generate confusion matrix: {str(e)}")
                
                st.markdown("---")
                st.markdown("#### 📈 ROC Curve")
                try:
                    # Check if binary classification
                    unique_classes = np.unique(y_test_class)
                    if len(unique_classes) == 2:
                        # Get prediction probabilities
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test)[:, 1]
                        else:
                            y_proba = model.decision_function(X_test)
                        
                        # Calculate ROC curve
                        fpr, tpr, _ = roc_curve(y_test_class, y_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        # Plot ROC curve
                        fig_roc = go.Figure()
                        
                        # ROC curve
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                    name=f'ROC Curve (AUC = {roc_auc:.4f})',
                                                    line=dict(color='#0ea5e9', width=3)))
                        
                        # Diagonal line (random classifier)
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                    name='Random Classifier',
                                                    line=dict(color='#f43f5e', width=2, dash='dash')))
                        
                        fig_roc.update_layout(
                            title=f'ROC Curve (AUC = {roc_auc:.4f})',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            template='plotly_dark',
                            height=500,
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig_roc, use_container_width=True)
                        st.metric("AUC Score", f"{roc_auc:.4f}")
                        
                        # AUC Interpretation Table
                        st.markdown("#### 📊 AUC Interpretation")
                        auc_interpretation = pd.DataFrame({
                            'AUC Value': ['1.0', '0.9+', '0.7–0.8', '0.5'],
                            'Meaning': ['Perfect model 🔥', 'Excellent', 'Good', 'Random (bad ❌)']
                        })
                        st.table(auc_interpretation)
                        
                        # Show current model's AUC category
                        if roc_auc >= 0.9:
                            category = "Excellent 🚀"
                        elif roc_auc >= 0.7:
                            category = "Good ✅"
                        elif roc_auc >= 0.5:
                            category = "Random ⚠️"
                        else:
                            category = "Very Bad ❌"
                        
                        st.success(f"Your model's AUC ({roc_auc:.4f}) is: **{category}**")
                    else:
                        st.info("ℹ️ ROC Curve available only for binary classification (2 classes)")
                except Exception as e:
                    st.error(f"❌ Could not generate ROC Curve: {str(e)}")
                
                # Misclassified Samples
                st.markdown("#### 🔍 Misclassified Samples")
                misclassified_mask = y_test_class != y_pred_class
                num_misclassified = misclassified_mask.sum()
                
                col1, col2 = st.columns(2)
                col1.metric("Total Misclassified", num_misclassified)
                col2.metric("Misclassification Rate (%)", f"{(num_misclassified / len(y_test_class)) * 100:.2f}%")
                
                if num_misclassified > 0:
                    st.markdown("**Sample Misclassified Predictions:**")
                    misclassified_df = pd.DataFrame({
                        'Actual': y_test_class.values[misclassified_mask] if hasattr(y_test_class, 'values') else y_test_class[misclassified_mask],
                        'Predicted': y_pred_class[misclassified_mask]
                    }).head(10)
                    st.dataframe(misclassified_df, use_container_width=True)
                else:
                    st.success("✅ Perfect classification! No misclassified samples.")
                
            else:
                # Regression metrics
                st.markdown("#### 📊 Regression Metrics")
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📏 MAE", f"{mae:.4f}", delta=None)
                with col2:
                    st.metric("📊 MSE", f"{mse:.4f}", delta=None)
                with col3:
                    st.metric("📈 RMSE", f"{rmse:.4f}", delta=None)
                with col4:
                    st.metric("🎯 R² Score", f"{r2:.4f}", delta=None)
                
                st.markdown("---")
                
                # Actual vs Predicted plot
                st.markdown("#### 📈 Actual vs Predicted")
                try:
                    pred_df = pd.DataFrame({
                        'Actual': y_test.values if hasattr(y_test, 'values') else y_test,
                        'Predicted': y_pred,
                        'Index': range(len(y_test))
                    })
                    fig = px.scatter(pred_df, x='Actual', y='Predicted', 
                                   template="plotly_dark", color_discrete_sequence=['#38bdf8'],
                                   trendline="ols", title="Predictions vs Actuals")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Could not generate Actual vs Predicted plot: {str(e)}")
                
                # Residual plot
                st.markdown("#### 📉 Residual Plot")
                try:
                    pred_df = pd.DataFrame({
                        'Actual': y_test.values if hasattr(y_test, 'values') else y_test,
                        'Predicted': y_pred,
                        'Index': range(len(y_test))
                    })
                    pred_df['Residuals'] = pred_df['Actual'] - pred_df['Predicted']
                    fig = px.scatter(pred_df, x='Predicted', y='Residuals',
                                   template="plotly_dark", color_discrete_sequence=['#f43f5e'],
                                   title="Residuals Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Could not generate Residual plot: {str(e)}")
            
            st.markdown("---")
            if hasattr(model, 'feature_importances_'):
                st.markdown("#### 🎯 Feature Importance")
                try:
                    feature_imp = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                               template="plotly_dark", color_continuous_scale="Viridis",
                               title="Feature Importance Score")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Could not generate Feature Importance chart: {str(e)}")
            
            st.markdown("---")
            st.markdown("## 🤖 PHASE 4: Prediction System")
            
            # Prediction UI
            st.subheader("Make Predictions on New Data")
            
            prediction_mode = st.radio("Select Prediction Mode", ["Manual Input", "Upload Dataset"], horizontal=True)
            
            if prediction_mode == "Manual Input":
                st.markdown("#### Manual Input Fields")
                st.caption("Enter values for each feature")
                
                input_data = {}
                cols = st.columns(2)
                
                for idx, feature in enumerate(st.session_state.feature_names):
                    with cols[idx % 2]:
                        input_data[feature] = st.number_input(f"{feature}", value=0.0, key=f"input_{feature}")
                
                if st.button("🔮 Predict", key="predict_manual"):
                    try:
                        # Create dataframe from input
                        pred_input = pd.DataFrame([input_data])
                        
                        # Make prediction
                        prediction = model.predict(pred_input)[0]
                        
                        if task_type == 'Classification':
                            if hasattr(model, 'predict_proba'):
                                probabilities = model.predict_proba(pred_input)[0]
                                st.success(f"✅ **Prediction: {prediction}**")
                                
                                # Show probabilities
                                prob_df = pd.DataFrame({
                                    'Class': np.unique(y_test),
                                    'Probability': probabilities
                                })
                                st.dataframe(prob_df, use_container_width=True)
                            else:
                                st.success(f"✅ **Prediction: {prediction}**")
                        else:
                            st.success(f"✅ **Predicted Value: {prediction:.4f}**")
                    
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
            
            else:
                st.markdown("#### Upload Dataset for Batch Predictions")
                uploaded_pred_file = st.file_uploader("Upload CSV File", type=["csv"], key="pred_upload")
                
                if uploaded_pred_file is not None:
                    try:
                        pred_df = pd.read_csv(uploaded_pred_file)
                        
                        # Validate columns
                        missing_cols = set(st.session_state.feature_names) - set(pred_df.columns)
                        if missing_cols:
                            st.error(f"❌ Missing columns: {missing_cols}")
                        else:
                            # Select only required features
                            pred_df_subset = pred_df[st.session_state.feature_names]
                            
                            if st.button("🔮 Predict All", key="predict_batch"):
                                # Make predictions
                                predictions = model.predict(pred_df_subset)
                                
                                # Create results dataframe
                                results_df = pred_df.copy()
                                results_df['Prediction'] = predictions
                                
                                if task_type == 'Classification' and hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(pred_df_subset)
                                    for i, class_label in enumerate(np.unique(y_test)):
                                        results_df[f'Prob_{class_label}'] = probabilities[:, i]
                                
                                st.success(f"✅ Predictions Complete! ({len(predictions)} samples)")
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Download button
                                csv = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")


    # =========================
    # CLUSTERING
    # =========================
    elif step == "Clustering":
        st.subheader("🎯 Clustering Analysis")
        st.markdown("Unsupervised learning to group similar data points")
        
        # Enable proper numeric conversion
        df_numeric = df.copy()
        for col in df_numeric.select_dtypes(include=['object']).columns:
            try:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
            except:
                pass
        
        # Drop rows with NaN after conversion
        df_numeric = df_numeric.dropna()
        
        if df_numeric.shape[0] < 2:
            st.error("❌ Not enough numeric data for clustering")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                clustering_method = st.selectbox("Select Clustering Method", ["K-Means", "DBSCAN"])
            
            with col2:
                if clustering_method == "K-Means":
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                else:
                    eps = st.slider("EPS (Epsilon)", 0.1, 2.0, 0.5, step=0.1)
                    min_samples = st.slider("Min Samples", 2, 20, 5)
            
            if st.button("🚀 Run Clustering"):
                try:
                    if clustering_method == "K-Means":
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(df_numeric)
                        silhouette_avg = silhouette_score(df_numeric, clusters)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Number of Clusters", n_clusters)
                        col2.metric("Silhouette Score", f"{silhouette_avg:.4f}")
                        
                        st.info(f"✅ Clustering complete! Silhouette Score: {silhouette_avg:.4f}")
                        st.caption("Silhouette Score: -1 (bad) to 1 (perfect) - measures cluster quality")
                        
                    else:  # DBSCAN
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        clusters = dbscan.fit_predict(df_numeric)
                        n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
                        n_noise = list(clusters).count(-1)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Clusters Found", n_clusters_found)
                        col2.metric("Noise Points", n_noise)
                        col3.metric("Total Points", len(clusters))
                        
                        st.info(f"✅ DBSCAN clustering complete! Found {n_clusters_found} clusters")
                    
                    # Visualize clusters with first 2 PCA components
                    if df_numeric.shape[1] > 1:
                        pca = PCA(n_components=2)
                        df_pca = pca.fit_transform(df_numeric)
                        
                        fig = go.Figure()
                        for cluster_id in sorted(set(clusters)):
                            if cluster_id == -1:
                                label = "Noise"
                                color = "gray"
                            else:
                                label = f"Cluster {cluster_id}"
                                color = None
                            
                            mask = clusters == cluster_id
                            fig.add_trace(go.Scatter(
                                x=df_pca[mask, 0],
                                y=df_pca[mask, 1],
                                mode='markers',
                                name=label,
                                marker=dict(size=8, color=color)
                            ))
                        
                        fig.update_layout(
                            title="Cluster Visualization (PCA)",
                            xaxis_title="PC1",
                            yaxis_title="PC2",
                            template="plotly_dark",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add clusters to dataframe
                    df['Cluster'] = clusters
                    st.session_state.current_df = df
                    
                    st.markdown("---")
                    st.markdown("#### Cluster Distribution")
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    st.bar_chart(cluster_counts)
                    
                    st.markdown("---")
                    st.markdown("#### Sample from Each Cluster")
                    for cluster_id in sorted(set(clusters)):
                        mask = clusters == cluster_id
                        st.write(f"**Cluster {cluster_id}** ({mask.sum()} samples)")
                        st.dataframe(df[mask].head(3), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Clustering error: {str(e)}")


    # =========================
    # BEFORE VS AFTER
    # =========================
    st.markdown("---")
    st.subheader("Data Evolution")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Dataset")
        st.write(original_df)

    with col2:
        st.write("Current Processed Dataset")
        st.write(df)

    # =========================
    # DOWNLOAD
    # =========================
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Current Pipeline State", csv, "processed.csv", "text/csv")
