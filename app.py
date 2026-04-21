import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import inspect
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Antigravity AI | Data Suite", layout="wide", page_icon="🚀")

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
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True)
        }
    elif task_type == 'Regression':
        return {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'KNN': KNeighborsRegressor(),
            'SVM': SVR()
        }
    else: # Clustering
        return {
            'K-Means': KMeans(n_clusters=3, random_state=42, n_init=10)
        }


# Custom CSS for Modern Design (Dark sleek theme with improved contrast)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap');

    /* Global Styles */
    :root {
        --primary: #8b5cf6;
        --secondary: #6366f1;
        --bg-main: #0a1128;
        --bg-sidebar: #050505;
        --bg-glass: rgba(15, 23, 42, 0.7);
        --border-glass: rgba(139, 92, 246, 0.3);
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
        
        /* Force Streamlit Theme Variables */
        --primary-color: #8b5cf6;
        --background-color: #0a1128;
        --secondary-background-color: #050505;
        --text-color: #f8fafc;
        --font: 'Inter', sans-serif;
    }

    /* THEME LOCK: Force Dark Mode even if user is in Light Mode */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-glass) !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: var(--bg-main) !important;
    }

    .stApp {
        background-color: var(--bg-main) !important;
    }

    /* Force all text elements to be light */
    .stMarkdown, p, span, label, li, h1, h2, h3, h4, h5, h6, .stMetric, [data-testid="stExpander"] {
        color: var(--text-main) !important;
    }

    /* Table/DataFrame Styling Overrides */
    [data-testid="stTable"], [data-testid="stDataFrame"], .stTable, .stDataFrame {
        background-color: rgba(255, 255, 255, 0.02) !important;
        color: var(--text-main) !important;
    }
    
    /* Navigation Step Buttons */
    div[role="radiogroup"] label[data-baseweb="radio"] {
        background-color: rgba(139, 92, 246, 0.05) !important;
        border: 1px solid rgba(139, 92, 246, 0.1) !important;
        border-radius: 8px !important;
        padding: 5px 12px !important;
        margin-bottom: 6px !important;
    }

    div[role="radiogroup"] label[data-baseweb="radio"]:hover {
        border-color: var(--primary) !important;
        background-color: rgba(139, 92, 246, 0.15) !important;
    }

    /* Metric Boxes Glassmorphism */
    div[data-testid="stMetricContainer"] {
        background: var(--bg-glass) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.4) !important;
    }

    /* Title & Header Gradients */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 700 !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
        filter: brightness(1.1);
    }

    /* Input Fields */
    .stTextInput input, .stSelectbox [data-baseweb="select"], .stNumberInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Custom Cards for Predictions */
    .prediction-card {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid var(--primary);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
    }

    .prediction-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary);
        margin: 10px 0;
        text-shadow: 0 0 15px rgba(139, 92, 246, 0.5);
    }
    
    /* Toast Styling */
    [data-testid="stToast"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--primary) !important;
        color: white !important;
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
                fig = px.histogram(df, x=dist_col, marginal="box", template="plotly_dark", color_discrete_sequence=['#8b5cf6'])
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
                                     color_discrete_sequence=['#c084fc'])
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
            
            # Store scaler and numeric columns in session state for later use in predictions
            st.session_state.scaler = scaler
            st.session_state.numeric_columns_for_scaling = list(num_cols)
            st.session_state.scaling_method = method
            
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
            st.session_state.categorical_mappings = {}  # Store mappings for categorical→numeric
            st.session_state.scaler = None  # Store scaler object for predictions
            st.session_state.numeric_columns_for_scaling = []  # Track which columns were scaled
            st.session_state.scaling_method = None  # Track scaling method (MinMax or Standard)
        
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
            
            # Filter session state to only include valid features (in case target column changed)
            valid_default_features = [f for f in st.session_state.get("features_select", all_features) if f in all_features]
            if not valid_default_features:
                valid_default_features = all_features
            
            selected_features = st.multiselect(
                "Choose Features", 
                all_features,
                default=valid_default_features,
                key="features_select",
                help="Select the features you want to use for training. All features are selected by default."
            )
            
            if not selected_features:
                st.warning("⚠️ Please select at least one feature!")
            else:
                y = df[target_col]
                X = df[selected_features]
                
                # Store original data for later preprocessing reference
                st.session_state.original_X = X.copy()
                st.session_state.original_df_for_ref = df[selected_features].copy()
                
                # Store original feature types before any encoding (update every time for accuracy)
                original_cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                st.session_state.original_feature_types = {
                    'categorical': original_cat_features,
                    'numeric': [f for f in selected_features if f not in original_cat_features]
                }
                
                # Show feature types diagnostic
                st.write("**📊 Feature Summary:**")
                feature_types_summary = []
                for feat in selected_features:
                    dtype = str(X[feat].dtype)
                    is_cat = dtype in ['object', 'category']
                    feature_types_summary.append({
                        'Feature': feat,
                        'Type': 'Categorical' if is_cat else 'Numeric',
                        'Data Type': dtype,
                        'Sample Values': str(X[feat].unique()[:3].tolist())
                    })
                st.dataframe(pd.DataFrame(feature_types_summary), use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("#### Step 4: Select Task Type")
                task_type_opts = ["Classification", "Regression", "Clustering"]
                # Default to detecting supervised task, but allow manual Clustering
                detected = detect_task_type(y)
                default_idx = task_type_opts.index(detected) if detected in task_type_opts else 0
                
                task_type = st.selectbox("Choose Task Type", task_type_opts, index=default_idx, key="task_type_select")
                st.session_state.task_type = task_type
                st.caption(f"Target column '{target_col}' has {y.nunique()} unique values")
                
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
                st.markdown("#### Step 5: Select Model")
                
                available_models = get_models_by_task(task_type)
                model_name = st.selectbox("Choose Model", list(available_models.keys()), key="model_select")
                
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
                st.markdown("#### Step 7: Model Parameters Customization (Optional)")
                
                # Collapsible section for custom parameters
                with st.expander("⚙️ **Customize Model Parameters**", expanded=False):
                    custom_params = {}
                    
                    if model_name == "Random Forest":
                        st.markdown("**🌲 Random Forest Parameters:**")
                        rf_cols = st.columns(3)
                        with rf_cols[0]:
                            rf_n_estimators = st.number_input("n_estimators", min_value=10, max_value=1000, value=100, help="Number of trees (default: 100)")
                            rf_max_features = st.selectbox("max_features", ["sqrt", "log2", None], index=0, help="Features to consider at split (default: 'sqrt')")
                        with rf_cols[1]:
                            rf_max_depth = st.number_input("max_depth", min_value=0, max_value=100, value=0, help="Max depth (0 for None)")
                            rf_min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=50, value=1, help="Min samples at leaf (default: 1)")
                        with rf_cols[2]:
                            rf_min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=50, value=2, help="Min samples to split (default: 2)")
                            rf_oob_score = st.checkbox("oob_score", value=False, help="Use out-of-bag samples (default: False)")
                        
                        custom_params = {
                            "n_estimators": rf_n_estimators,
                            "max_depth": rf_max_depth if rf_max_depth > 0 else None,
                            "min_samples_split": rf_min_samples_split,
                            "min_samples_leaf": rf_min_samples_leaf,
                            "max_features": rf_max_features,
                            "oob_score": rf_oob_score
                        }

                    elif model_name == "Decision Tree":
                        st.markdown("**🌳 Decision Tree Parameters:**")
                        dt_cols = st.columns(2)
                        with dt_cols[0]:
                            dt_crit_options = ["gini", "entropy", "log_loss"] if task_type == "Classification" else ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                            dt_criterion = st.selectbox("criterion", dt_crit_options, index=0)
                            dt_max_depth = st.number_input("max_depth", min_value=0, max_value=100, value=0, help="Max depth (0 for None)")
                            dt_max_features = st.selectbox("max_features", [None, "sqrt", "log2"], index=0)
                        with dt_cols[1]:
                            dt_splitter = st.selectbox("splitter", ["best", "random"], index=0)
                            dt_min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=50, value=2)
                            dt_min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=50, value=1)
                        
                        custom_params = {
                            "criterion": dt_criterion,
                            "splitter": dt_splitter,
                            "max_depth": dt_max_depth if dt_max_depth > 0 else None,
                            "min_samples_split": dt_min_samples_split,
                            "min_samples_leaf": dt_min_samples_leaf,
                            "max_features": dt_max_features
                        }

                    elif model_name == "Logistic Regression":
                        st.markdown("**📈 Logistic Regression Parameters:**")
                        lr_cols = st.columns(2)
                        with lr_cols[0]:
                            lr_penalty = st.selectbox("penalty", ["l2", "l1", "elasticnet", None], index=0)
                            lr_C = st.number_input("C (Regularization strength)", min_value=0.01, max_value=100.0, value=1.0)
                        with lr_cols[1]:
                            lr_max_iter = st.number_input("max_iter", min_value=100, max_value=5000, value=1000)
                            lr_fit_intercept = st.checkbox("fit_intercept", value=True)
                        lr_multi_class = st.selectbox("multi_class", ["auto", "ovr", "multinomial"], index=0)
                        
                        custom_params = {
                            "penalty": lr_penalty,
                            "C": lr_C,
                            "max_iter": lr_max_iter,
                            "fit_intercept": lr_fit_intercept,
                            "multi_class": lr_multi_class
                        }

                    elif model_name == "Linear Regression":
                         st.markdown("**📏 Linear Regression Parameters:**")
                         lin_cols = st.columns(2)
                         with lin_cols[0]:
                             lin_fit_intercept = st.checkbox("fit_intercept", value=True)
                         with lin_cols[1]:
                             lin_positive = st.checkbox("positive", value=False)
                         
                         custom_params = {
                             "fit_intercept": lin_fit_intercept,
                             "positive": lin_positive
                         }
                    
                    elif model_name == "KNN":
                        st.markdown("**👥 K-Nearest Neighbors Parameters:**")
                        knn_cols = st.columns(2)
                        with knn_cols[0]:
                            knn_neighbors = st.number_input("n_neighbors", min_value=1, max_value=100, value=5)
                            knn_weights = st.selectbox("weights", ["uniform", "distance"], index=0)
                        with knn_cols[1]:
                            knn_metric = st.selectbox("metric", ["minkowski", "euclidean", "manhattan"], index=0)
                            knn_leaf_size = st.number_input("leaf_size", min_value=1, max_value=100, value=30)
                        
                        custom_params = {
                            "n_neighbors": knn_neighbors,
                            "weights": knn_weights,
                            "metric": knn_metric,
                            "leaf_size": knn_leaf_size
                        }
                    
                    elif model_name == "SVM":
                        st.markdown("**🛡️ SVM Parameters:**")
                        svm_cols = st.columns(2)
                        with svm_cols[0]:
                            svm_C = st.number_input("C (Regularization)", min_value=0.01, max_value=100.0, value=1.0)
                            svm_kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
                        with svm_cols[1]:
                            svm_gamma = st.selectbox("gamma", ["scale", "auto"], index=0)
                            if svm_kernel == "poly":
                                svm_degree = st.slider("degree (for poly kernel)", 1, 5, 3)
                            if task_type == "Regression":
                                svm_epsilon = st.number_input("epsilon", min_value=0.01, max_value=1.0, value=0.1)
                        
                        custom_params = {
                            "C": svm_C,
                            "kernel": svm_kernel,
                            "gamma": svm_gamma,
                            "probability": True if task_type == "Classification" else False
                        }
                        if svm_kernel == "poly":
                            custom_params["degree"] = svm_degree
                        if task_type == "Regression":
                            custom_params["epsilon"] = svm_epsilon
                    
                    elif model_name == "K-Means":
                        st.markdown("**🧠 K-Means Clustering Parameters:**")
                        km_cols = st.columns(2)
                        with km_cols[0]:
                            km_n_clusters = st.number_input("n_clusters (k)", min_value=2, max_value=20, value=3)
                            km_init = st.selectbox("init", ["k-means++", "random"], index=0)
                        with km_cols[1]:
                            km_max_iter = st.number_input("max_iter", min_value=100, max_value=1000, value=300)
                            km_n_init = st.number_input("n_init", min_value=1, max_value=50, value=10)
                        
                        custom_params = {
                            "n_clusters": km_n_clusters,
                            "init": km_init,
                            "max_iter": km_max_iter,
                            "n_init": km_n_init
                        }
                        
                        st.markdown("---")
                        st.markdown("#### 🎯 Elbow Method (Find Optimal K)")
                        st.caption("We'll run K-Means for k=1 to 10 and plot the 'Inertia'. Look for the point where the curve starts to flatten (the 'elbow').")
                        
                        if st.button("📈 Find Optimal K", key="elbow_btn"):
                            with st.spinner("Calculating inertia for k=1..10..."):
                                inertias = []
                                K_range = range(1, 11)
                                
                                # Use a copy of X for calculation
                                X_elbow = X.copy()
                                # Quick cleanup for elbow
                                X_elbow = X_elbow.select_dtypes(include=np.number).fillna(X_elbow.mean(numeric_only=True))
                                
                                if X_elbow.empty:
                                    st.error("❌ No numeric features found for Elbow Method.")
                                else:
                                    for k in K_range:
                                        km_temp = KMeans(n_clusters=k, random_state=42, n_init=5)
                                        km_temp.fit(X_elbow)
                                        inertias.append(km_temp.inertia_)
                                    
                                    fig_elbow = px.line(x=list(K_range), y=inertias, markers=True, 
                                                       template="plotly_dark", color_discrete_sequence=['#a78bfa'])
                                    fig_elbow.update_layout(title="Elbow Method: Inertia vs Number of Clusters", 
                                                          xaxis_title="Number of Clusters (k)", yaxis_title="Inertia", height=400)
                                    st.plotly_chart(fig_elbow, use_container_width=True)
                                    st.info("💡 **Interpretation:** Choose the 'k' where the drop in inertia slows down significantly.")
                    
                    
                    st.divider()
                    st.info("💡 **Tip:** Custom values will be applied to the selected model. Non-customized models will use default parameters.")
                
                if st.button("🚀 Train Model", key="train_btn"):
                    # Display selected parameters
                    with st.expander("📋 **Selected Parameters**", expanded=True):
                        st.markdown(f"**Parameters for {model_name}:**")
                        if custom_params:
                            for p_name, p_val in custom_params.items():
                                st.info(f"🔹 {p_name}: **{p_val}**")
                        else:
                            st.success("✅ Using **default parameters**")
                    try:
                        X_for_training = X.copy()
                        progress_placeholder = st.empty()
                        
                        # Handle Missing Values (Imputation)
                        missing_values = X_for_training.isnull().sum().sum()
                        if missing_values > 0:
                            progress_placeholder.info(f"🧹 Auto-cleaning: Handling {missing_values} missing values...")
                            # Fill numeric with mean
                            num_impute_cols = X_for_training.select_dtypes(include=np.number).columns
                            if not num_impute_cols.empty:
                                X_for_training[num_impute_cols] = X_for_training[num_impute_cols].fillna(X_for_training[num_impute_cols].mean())
                            # Fill categorical with mode
                            cat_impute_cols = X_for_training.select_dtypes(include=['object', 'category']).columns
                            for col in cat_impute_cols:
                                if not X_for_training[col].mode().empty:
                                    X_for_training[col] = X_for_training[col].fillna(X_for_training[col].mode().iloc[0])
                                else:
                                    X_for_training[col] = X_for_training[col].fillna("Unknown")
                            st.success(f"✅ Data auto-cleaned! Handled {missing_values} missing values.")
                        
                        # Feature Selection: Remove weak features
                        if use_feature_selection:
                            progress_placeholder.info("📊 Step 1/5: Analyzing feature importance...")
                            
                            # Train a quick model to get feature importance
                            if task_type == "Classification":
                                temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
                            else:
                                temp_model = RandomForestRegressor(n_estimators=50, random_state=42)
                            
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
                        
                        # Encode categorical features
                        progress_placeholder.info("📊 Step 2/5: Encoding categorical features...")
                        X_for_training = X_for_training.copy()
                        categorical_features = X_for_training.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        # Explicit initialization
                        if not hasattr(st.session_state, 'categorical_mappings'):
                            st.session_state.categorical_mappings = {}
                        
                        if categorical_features:
                            st.session_state.categorical_mappings = {}  # Reset to ensure clean state
                            encoding_info = []
                            for cat_col in categorical_features:
                                le = LabelEncoder()
                                encoded_vals = le.fit_transform(X_for_training[cat_col].astype(str))
                                X_for_training[cat_col] = encoded_vals
                                
                                # Store the mapping for later use in predictions - maps original values to numeric
                                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                                st.session_state.categorical_mappings[cat_col] = mapping
                                # Only show feature name and count of unique values (concise)
                                encoding_info.append(f"✅ {cat_col} ({len(mapping)} unique values)")
                            
                            st.success(f"🔄 **Encoded {len(categorical_features)} categorical features:**\n" + " | ".join(encoding_info))
                        else:
                            st.info("ℹ️ No categorical features found in the selected features.")
                        
                        # =============================================
                        # BUILD FEATURE PROFILE (HUMAN-READABLE)
                        # Uses original_df (raw untouched CSV) for real values
                        # =============================================
                        raw_df = st.session_state.original_df  # UNTOUCHED raw CSV
                        feature_profile = {}
                        
                        for col in X_for_training.columns:
                            # Check if this column was originally text/categorical in the RAW CSV
                            if col in raw_df.columns and raw_df[col].dtype == 'object':
                                # This is a categorical column - show original text values
                                raw_vals = sorted([str(v) for v in raw_df[col].dropna().unique().tolist()])
                                # Build/update mapping from string -> number
                                if col not in st.session_state.categorical_mappings:
                                    st.session_state.categorical_mappings[col] = {v: i for i, v in enumerate(raw_vals)}
                                feature_profile[col] = {
                                    'type': 'categorical',
                                    'options': raw_vals,
                                    'default': raw_vals[0] if raw_vals else ''
                                }
                            elif col in st.session_state.categorical_mappings:
                                # Already have a mapping from the encoding step
                                options = list(st.session_state.categorical_mappings[col].keys())
                                feature_profile[col] = {
                                    'type': 'categorical',
                                    'options': options,
                                    'default': options[0] if options else ''
                                }
                            else:
                                # Numeric: pull ORIGINAL ranges from untouched CSV
                                if col in raw_df.columns:
                                    col_data = pd.to_numeric(raw_df[col], errors='coerce').dropna()
                                    feature_profile[col] = {
                                        'type': 'numeric',
                                        'min': float(col_data.min()) if len(col_data) > 0 else 0.0,
                                        'max': float(col_data.max()) if len(col_data) > 0 else 100.0,
                                        'default': float(col_data.median()) if len(col_data) > 0 else 0.0
                                    }
                                else:
                                    feature_profile[col] = {
                                        'type': 'numeric', 'min': 0.0, 'max': 100.0, 'default': 0.0
                                    }
                        
                        st.session_state.feature_profile = feature_profile
                        st.session_state.feature_names_for_prediction = X_for_training.columns.tolist()
                        
                        # Apply scaling to numeric features if available in session state
                        if "scaler" in st.session_state and st.session_state.scaler is not None:
                            numeric_cols_for_scaling = X_for_training.select_dtypes(include=np.number).columns.tolist()
                            if numeric_cols_for_scaling:
                                # Re-fit the scaler on the CURRENT columns to avoid "Feature name mismatch" errors
                                # if the user changed feature selection since the last transformation step.
                                try:
                                    X_for_training[numeric_cols_for_scaling] = st.session_state.scaler.fit_transform(X_for_training[numeric_cols_for_scaling])
                                    # Update metadata for the prediction engine
                                    st.session_state.numeric_columns_for_scaling = numeric_cols_for_scaling
                                    progress_placeholder.info(f"📊 Applied {st.session_state.scaling_method} scaling to {len(numeric_cols_for_scaling)} features")
                                except Exception as scaling_err:
                                    st.warning(f"⚠️ Scaler alignment issue: {str(scaling_err)}. Re-initializing scaler...")
                                    # Fallback: re-initialize and fit
                                    if st.session_state.scaling_method == "Standardization":
                                        st.session_state.scaler = StandardScaler()
                                    else:
                                        st.session_state.scaler = MinMaxScaler()
                                    X_for_training[numeric_cols_for_scaling] = st.session_state.scaler.fit_transform(X_for_training[numeric_cols_for_scaling])
                                    st.session_state.numeric_columns_for_scaling = numeric_cols_for_scaling
                        
                        # Store numeric columns for prediction preprocessing
                        numeric_training_cols = X_for_training.select_dtypes(include=np.number).columns.tolist()
                        st.session_state.numeric_columns_in_training = numeric_training_cols
                        
                        # Split data
                        progress_placeholder.info("📊 Step 3/5: Splitting data...")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_for_training, y_processed, test_size=test_size, random_state=42
                        )
                        
                        # Model Training
                        available_models = get_models_by_task(task_type)
                        
                        # Function to apply custom parameters to models
                        def apply_custom_params(model_name, model_class, custom_params):
                            """Apply custom parameters to model if provided, otherwise use defaults"""
                            params = custom_params.copy()
                            
                            # Safely apply random_state only if supported by the model
                            sig = inspect.signature(model_class)
                            if 'random_state' in sig.parameters:
                                params["random_state"] = 42
                            
                            try:
                                return model_class(**params)
                            except Exception as e:
                                st.warning(f"⚠️ Error applying some parameters: {str(e)}. Using defaults.")
                                # Fallback to default instantiation (still trying random_state if supported)
                                fallback_params = {"random_state": 42} if 'random_state' in sig.parameters else {}
                                return model_class(**fallback_params)
                        
                        # Train single model
                        if True:
                            # Train single model
                            progress_placeholder.info("📊 Step 4/5: Training model...")
                            
                            # Apply custom parameters
                            model = apply_custom_params(model_name, available_models[model_name].__class__, custom_params)
                            
                            # Hyperparameter Tuning
                            if use_hyperparameter_tuning:
                                progress_placeholder.info("📊 Step 5/5: Tuning hyperparameters (this may take a minute)...")
                                
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
                                        model,
                                        param_grids[model_name],
                                        cv=3,
                                        n_jobs=-1,
                                        verbose=0
                                    )
                                    if task_type == "Clustering":
                                        grid_search.fit(X_train)
                                    else:
                                        grid_search.fit(X_train, y_train)
                                    model = grid_search.best_estimator_
                                    
                                    st.success(f"✅ Best parameters found: {grid_search.best_params_}")
                                    st.info(f"Best CV Score: {grid_search.best_score_:.4f}")
                                else:
                                    st.warning(f"⚠️ Hyperparameter tuning not available for {model_name}. Using default parameters.")
                                    model.fit(X_train, y_train)
                            else:
                                progress_placeholder.info(f"📊 Step 4/4: {'Grouping' if task_type == 'Clustering' else 'Training'} model...")
                                if task_type == "Clustering":
                                    model.fit(X_train)
                                else:
                                    model.fit(X_train, y_train)
                            
                            # Store in session state
                            st.session_state.trained_model = model
                            st.session_state.feature_names = X_train.columns.tolist()
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            
                            # Store target metadata for human-readable prediction labels
                            st.session_state.target_col_name = target_col
                            original_target = st.session_state.original_df[target_col]
                            st.session_state.target_classes = sorted(original_target.dropna().unique().tolist())
                            
                            # Feature profile already built BEFORE scaling (above) - no need to rebuild.
                            # Keep the categorical mappings from encoding phase (already stored in session state)
                            
                            progress_placeholder.empty()
                            st.success(f"✅ Model trained successfully! ({model_name})")
                            st.info(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
                        
                    except Exception as e:
                        st.error(f"❌ Error training model: {str(e)}")
                
                # --- AUTO-ML LEADERBOARD SECTION ---
                st.markdown("---")
                st.markdown("#### 🧪 AutoML Leaderboard (Compare All)")
                st.caption("Train and rank every model available for this task type automatically.")
                
                if st.button("🏁 Run Multi-Model Comparison", key="run_all_btn"):
                    all_results = []
                    prog_bar = st.progress(0)
                    available_models = get_models_by_task(task_type)
                    total_m = len(available_models)
                    
                    for idx, (m_name, m_obj) in enumerate(available_models.items()):
                        try:
                            prog_bar.progress((idx + 1) / total_m)
                            st.caption(f"Testing {m_name}...")
                            
                            # Standard fit (simplest version for speed)
                            m_obj.fit(X_train, y_train.astype(int) if task_type == 'Classification' else y_train)
                            y_p = m_obj.predict(X_test)
                            
                            res = {'Model': m_name}
                            if task_type == 'Classification':
                                res['Accuracy'] = accuracy_score(y_test, y_p)
                                res['F1-Score'] = f1_score(y_test, y_p, average='weighted', zero_division=0)
                                sort_key = 'Accuracy'
                            elif task_type == 'Regression':
                                res['R² Score'] = r2_score(y_test, y_p)
                                res['RMSE'] = np.sqrt(mean_squared_error(y_test, y_p))
                                sort_key = 'R² Score'
                            else: # Clustering
                                res['Silhouette'] = silhouette_score(X_test, y_p)
                                sort_key = 'Silhouette'
                            
                            all_results.append(res)
                        except:
                            continue
                    
                    prog_bar.empty()
                    if all_results:
                        leaderboard_df = pd.DataFrame(all_results).sort_values(sort_key, ascending=False if sort_key != 'RMSE' else True)
                        st.success(f"🏆 **Leaderboard Complete!** Best model: **{leaderboard_df.iloc[0]['Model']}**")
                        
                        fig_ldr = px.bar(leaderboard_df, x=sort_key, y='Model', orientation='h', 
                                       template='plotly_dark', color=sort_key, color_continuous_scale='Purp')
                        fig_ldr.update_layout(height=max(300, len(leaderboard_df)*40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_ldr, use_container_width=True)
                        st.dataframe(leaderboard_df.style.highlight_max(axis=0, color='#8b5cf6'), use_container_width=True, hide_index=True)
                    else:
                        st.error("❌ Leaderboard failed. Ensure data is processed correctly.")
                
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
                
                # Model Download (Persistence)
                st.markdown("---")
                try:
                    import io
                    model_buffer = io.BytesIO()
                    pickle.dump(model, model_buffer)
                    st.download_button(
                        label="💾 Download Trained Model (.pkl)",
                        data=model_buffer.getvalue(),
                        file_name=f"trained_{type(model).__name__}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                except:
                    pass
                
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
                            colorscale="Purples",
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
                
                # Uni-Classification Boundary Logic (Used later in Model Visualization)
                is_classification = True

            elif task_type == 'Regression':
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
                
                # Uni-Regression Diagnostics
                is_classification = False

            else: # Clustering
                st.markdown("#### 🧶 Cluster Analysis")
                c_col1, c_col2 = st.columns(2)
                
                with c_col1:
                    # Silhouette Score
                    try:
                        s_score = silhouette_score(X_test, y_pred)
                        st.metric("✨ Silhouette Score", f"{s_score:.4f}", help="Range: -1 to 1. Higher is better. > 0.5 is good.")
                    except:
                        st.warning("⚠️ Could not compute Silhouette score (requires >1 cluster)")
                    
                    # Inertia
                    if hasattr(model, 'inertia_'):
                        st.metric("📉 Final Inertia", f"{model.inertia_:.2f}")
                
                with c_col2:
                    # Cluster Distribution
                    counts = pd.Series(y_pred).value_counts().sort_index()
                    dist_df = pd.DataFrame({'Cluster': [f"Cluster {i}" for i in counts.index], 'Samples': counts.values})
                    fig_dist = px.pie(dist_df, values='Samples', names='Cluster', 
                                    template="plotly_dark", color_discrete_sequence=['#8b5cf6', '#c084fc', '#a78bfa', '#6366f1', '#d946ef'])
                    fig_dist.update_layout(title="Cluster Distribution", height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 🗺️ PCA Cluster Map (2D Projection)")
                
                try:
                    # PCA to 2D
                    pca_c = PCA(n_components=2)
                    X_c_2d = pca_c.fit_transform(X_test)
                    
                    viz_df = pd.DataFrame({
                        'PC1': X_c_2d[:, 0],
                        'PC2': X_c_2d[:, 1],
                        'Cluster': [f"Cluster {c}" for c in y_pred]
                    })
                    
                    # Add ground truth if available and relevant (if target matches unique cluster count roughly)
                    if hasattr(st.session_state, 'target_col_name'):
                        viz_df['Original Label'] = y_test.values if hasattr(y_test, 'values') else y_test
                        fig_c = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster', symbol='Original Label',
                                         template="plotly_dark", height=600)
                    else:
                        fig_c = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster',
                                         template="plotly_dark", height=600)
                        
                    fig_c.update_layout(title="K-Means Clusters in PCA Space")
                    st.plotly_chart(fig_c, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Could not generate cluster map: {str(e)}")
                
                st.markdown("---")
            
            # Model-specific visualizations
            st.markdown("#### 🎨 Model Visualization")
            
            model_type = type(model).__name__
            
            try:
                if 'DecisionTree' in model_type:
                    # Decision Tree Visualization with depth customization
                    st.markdown("**Decision Tree Structure:**")
                    from sklearn.tree import plot_tree
                    import matplotlib.pyplot as plt
                    
                    # Add depth limit control
                    col1, col2 = st.columns(2)
                    with col1:
                        max_tree_depth = st.slider(
                            "Maximum Tree Depth to Display",
                            min_value=1,
                            max_value=min(model.get_depth(), 20),
                            value=min(model.get_depth(), 10),
                            help="Limit visualization depth to see higher-level patterns"
                        )
                    with col2:
                        st.metric("Tree Statistics", f"Depth: {model.get_depth()} | Leaves: {model.get_n_leaves()}")
                    
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(model, feature_names=st.session_state.feature_names, 
                             filled=True, ax=ax, rounded=True, fontsize=10,
                             max_depth=max_tree_depth)
                    st.pyplot(fig)
                    
                elif 'RandomForest' in model_type:
                    # Random Forest - Show tree with customization
                    st.markdown("**Random Forest - Tree Visualization:**")
                    from sklearn.tree import plot_tree
                    import matplotlib.pyplot as plt
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        tree_index = st.slider(
                            "Select Tree to Visualize",
                            min_value=0,
                            max_value=min(model.n_estimators - 1, 49),
                            value=0,
                            help="Choose which tree from the forest to display"
                        )
                    with col2:
                        max_tree_depth = st.slider(
                            "Maximum Tree Depth",
                            min_value=1,
                            max_value=20,
                            value=10,
                            help="Limit visualization depth"
                        )
                    with col3:
                        st.metric("Forest Info", f"Trees: {model.n_estimators}")
                    
                    # Plot selected tree
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(model.estimators_[tree_index], feature_names=st.session_state.feature_names,
                             filled=True, ax=ax, rounded=True, fontsize=8,
                             max_depth=max_tree_depth)
                    st.pyplot(fig)
                    
                    st.info(f"🌲 **Showing Tree #{tree_index + 1}** of {model.n_estimators} trees in the forest")
                
                elif 'LogisticRegression' in model_type:
                    if task_type == 'Classification':
                        try:
                            # ========== 1. Feature Coefficients ==========
                            st.markdown("**📊 Feature Coefficients (Model Weights):**")
                            st.caption("Shows how strongly each feature influences the model's prediction. Positive = pushes towards class 1, Negative = pushes towards class 0.")
                            
                            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                            feat_names = st.session_state.feature_names
                            coef_df = pd.DataFrame({
                                'Feature': feat_names,
                                'Coefficient': coef
                            }).sort_values('Coefficient', ascending=True)
                            
                            colors = ['#f43f5e' if c < 0 else '#22c55e' for c in coef_df['Coefficient']]
                            
                            fig_coef = go.Figure(go.Bar(
                                x=coef_df['Coefficient'],
                                y=coef_df['Feature'],
                                orientation='h',
                                marker=dict(color=colors, line=dict(width=0)),
                                text=[f"{c:+.3f}" for c in coef_df['Coefficient']],
                                textposition='outside',
                                textfont=dict(color='white', size=11)
                            ))
                            fig_coef.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=max(300, len(feat_names) * 35),
                                margin=dict(l=10, r=60, t=30, b=30),
                                xaxis=dict(title='Coefficient Value', gridcolor='rgba(255,255,255,0.05)', zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'),
                                yaxis=dict(title=''),
                                title=dict(text='Feature Impact on Prediction', font=dict(size=14, color='#c4b5fd'))
                            )
                            st.plotly_chart(fig_coef, use_container_width=True)
                            
                            if hasattr(model, 'intercept_'):
                                st.info(f"📐 **Intercept (bias):** {model.intercept_[0]:.4f}")
                            
                            st.markdown("---")
                            
                            # ========== 2. Decision Boundary (PCA-reduced) ==========
                            st.markdown("**🗺️ Decision Boundary (PCA-reduced to 2D):**")
                            st.caption("Logistic Regression separating hyperplane projected onto 2 principal components.")
                            
                            X_pca_lr = st.session_state.X_train.copy()
                            y_pca_lr = st.session_state.y_train
                            
                            if X_pca_lr.shape[1] >= 2:
                                if X_pca_lr.isnull().any().any():
                                    X_pca_lr = X_pca_lr.fillna(X_pca_lr.mean(numeric_only=True))
                                
                                scaler_lr = StandardScaler()
                                X_scaled_lr = scaler_lr.fit_transform(X_pca_lr)
                                pca_lr = PCA(n_components=2)
                                X_pca_2d = pca_lr.fit_transform(X_scaled_lr)
                                
                                # Retrain a lightweight LR on the 2D space for boundary
                                from sklearn.linear_model import LogisticRegression as LR2D
                                lr_2d = LR2D(max_iter=1000, random_state=42)
                                lr_2d.fit(X_pca_2d, y_pca_lr)
                                
                                # Mesh grid
                                padding = 1.0
                                x_min, x_max = X_pca_2d[:, 0].min() - padding, X_pca_2d[:, 0].max() + padding
                                y_min, y_max = X_pca_2d[:, 1].min() - padding, X_pca_2d[:, 1].max() + padding
                                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
                                grid_points = np.c_[xx.ravel(), yy.ravel()]
                                
                                if hasattr(lr_2d, 'predict_proba'):
                                    Z = lr_2d.predict_proba(grid_points)[:, 1].reshape(xx.shape)
                                else:
                                    Z = lr_2d.predict(grid_points).reshape(xx.shape)
                                
                                fig_boundary = go.Figure()
                                
                                # Decision surface
                                is_binary = (len(unique_y) == 2)
                                fig_boundary.add_trace(go.Contour(
                                    x=np.linspace(x_min, x_max, 200),
                                    y=np.linspace(y_min, y_max, 200),
                                    z=Z,
                                    colorscale='RdBu',
                                    opacity=0.6,
                                    showscale=is_binary, # Hide colorbar for multiclass to avoid overlap
                                    colorbar=dict(
                                        title=dict(text='P(Class 1)', font=dict(color='white')), 
                                        tickfont=dict(color='white'),
                                        x=1.1 # Move slightly further right if shown
                                    ),
                                    contours=dict(showlines=False)
                                ))
                                
                                # Data points
                                palette = ['#f43f5e', '#22c55e', '#3b82f6', '#f59e0b', '#a78bfa', '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
                                for idx_c, cls in enumerate(unique_y):
                                    mask = y_pca_lr == cls
                                    mask_arr = mask.values if hasattr(mask, 'values') else mask
                                    fig_boundary.add_trace(go.Scatter(
                                        x=X_pca_2d[mask_arr, 0],
                                        y=X_pca_2d[mask_arr, 1],
                                        mode='markers',
                                        name=f'Class {int(cls)}',
                                        marker=dict(size=6, color=palette[idx_c % len(palette)], line=dict(width=0.5, color='white'))
                                    ))
                                
                                fig_boundary.update_layout(
                                    template='plotly_dark',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    height=550,
                                    xaxis_title='Principal Component 1',
                                    yaxis_title='Principal Component 2',
                                    title=dict(text='Logistic Regression — Decision Boundary', font=dict(size=14, color='#c4b5fd')),
                                    legend=dict(
                                        font=dict(color='white'),
                                        orientation='h', # Horizontal legend
                                        yanchor='bottom',
                                        y=-0.3, # Position below the plot
                                        xanchor='center',
                                        x=0.5
                                    ),
                                    margin=dict(l=50, r=50, t=50, b=100)
                                )
                                st.plotly_chart(fig_boundary, use_container_width=True)
                            else:
                                st.warning("⚠️ Need at least 2 features for decision boundary visualization.")
                            
                            st.markdown("---")
                            
                            # ========== 3. ROC Curve ==========
                            st.markdown("**📈 ROC Curve:**")
                            from sklearn.metrics import roc_curve, auc
                            
                            unique_classes = np.unique(y_test)
                            if len(unique_classes) == 2:
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                                roc_auc = auc(fpr, tpr)
                                
                                fig_roc = go.Figure()
                                fig_roc.add_trace(go.Scatter(
                                    x=fpr, y=tpr, mode='lines',
                                    name=f'ROC Curve (AUC = {roc_auc:.4f})',
                                    line=dict(color='#a78bfa', width=3),
                                    fill='tozeroy',
                                    fillcolor='rgba(167, 139, 250, 0.15)'
                                ))
                                fig_roc.add_trace(go.Scatter(
                                    x=[0, 1], y=[0, 1], mode='lines',
                                    name='Random Classifier',
                                    line=dict(color='#475569', dash='dash', width=2)
                                ))
                                fig_roc.update_layout(
                                    template='plotly_dark',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    height=450,
                                    xaxis_title='False Positive Rate',
                                    yaxis_title='True Positive Rate',
                                    title=dict(text=f'Receiver Operating Characteristic — AUC: {roc_auc:.4f}', font=dict(size=14, color='#c4b5fd')),
                                    legend=dict(font=dict(color='white'))
                                )
                                st.plotly_chart(fig_roc, use_container_width=True)
                                
                                # AUC Score Interpretation
                                if roc_auc >= 0.9:
                                    st.success(f"🏆 **Excellent** classifier (AUC = {roc_auc:.4f})")
                                elif roc_auc >= 0.8:
                                    st.success(f"✅ **Good** classifier (AUC = {roc_auc:.4f})")
                                elif roc_auc >= 0.7:
                                    st.info(f"📊 **Fair** classifier (AUC = {roc_auc:.4f})")
                                else:
                                    st.warning(f"⚠️ **Poor** classifier (AUC = {roc_auc:.4f}) — consider feature engineering")
                            else:
                                st.info(f"ℹ️ ROC Curve is available for binary classification only. Your model has {len(unique_classes)} classes.")
                        
                        except Exception as e:
                            st.error(f"❌ Logistic Regression visualization error: {str(e)}")
                
                elif any(m in model_type for m in ['KNeighbors', 'SVC', 'SVR']):
                    # Decision Boundary Visualization (PCA-reduced)
                    title_prefix = "KNN" if "KNeighbors" in model_type else "SVM"
                    st.markdown(f"**{title_prefix} Decision Boundaries (PCA-reduced to 2D):**")
                    
                    try:
                        # 1. Prepare data for PCA
                        X_pca_input = st.session_state.X_train.copy()
                        y_pca_input = st.session_state.y_train
                        
                        # Guard: Check for minimum features
                        if X_pca_input.shape[1] < 2:
                            st.warning("⚠️ Visualization requires at least 2 features to perform PCA reduction. Skipping plot.")
                        else:
                            # Guard: Handle any lingering NaNs for visualization safety
                            if X_pca_input.isnull().any().any():
                                X_pca_input = X_pca_input.fillna(X_pca_input.mean(numeric_only=True))
                            
                            # 2. Scale data and PCA to 2 dimensions
                            scaler_viz = StandardScaler()
                            X_scaled_viz = scaler_viz.fit_transform(X_pca_input)
                            
                            pca_viz = PCA(n_components=2)
                            X_train_pca = pca_viz.fit_transform(X_scaled_viz)
                            
                            # Prepare test data for plotting
                            X_test_clean = st.session_state.X_test.fillna(st.session_state.X_test.mean(numeric_only=True))
                            X_test_scaled = scaler_viz.transform(X_test_clean)
                            X_test_pca = pca_viz.transform(X_test_scaled)
                            
                            # 3. Retrain a temporary model on 2D data for visualization
                            model_params = model.get_params()
                            if task_type == 'Classification':
                                if 'SVC' in model_type:
                                    temp_model = SVC(**model_params)
                                else:
                                    temp_model = KNeighborsClassifier(**model_params)
                            else:
                                if 'SVR' in model_type:
                                    temp_model = SVR(**model_params)
                                else:
                                    temp_model = KNeighborsRegressor(**model_params)
                            
                            temp_model.fit(X_train_pca, y_pca_input)
                            
                            # 4. Create Meshgrid for background
                            mesh_res = 200 # High Resolution
                            x_min, x_max = X_test_pca[:, 0].min() - 0.5, X_test_pca[:, 0].max() + 0.5
                            y_min, y_max = X_test_pca[:, 1].min() - 0.5, X_test_pca[:, 1].max() + 0.5
                            
                            h_x = (x_max - x_min) / mesh_res
                            h_y = (y_max - y_min) / mesh_res
                            xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))
                            
                            # 5. Predict on meshgrid
                            Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
                            Z = Z.reshape(xx.shape)
                            
                            # 6. Plot using Plotly
                            fig = go.Figure()
                            
                            # Background Contours
                            if task_type == 'Classification':
                                fig.add_trace(go.Contour(
                                    x=np.arange(x_min, x_max, h_x),
                                    y=np.arange(y_min, y_max, h_y),
                                    z=Z,
                                    opacity=0.6,
                                    showscale=False,
                                    colorscale='Magma',
                                    line_width=0,
                                    hoverinfo='skip'
                                ))
                            else:
                                fig.add_trace(go.Heatmap(
                                    x=np.arange(x_min, x_max, h_x),
                                    y=np.arange(y_min, y_max, h_y),
                                    z=Z,
                                    opacity=0.6,
                                    showscale=True,
                                    colorscale='Magma',
                                    colorbar=dict(title="Predicted", tickfont=dict(color='white'))
                                ))
                                
                            # Actual Test Data Points
                            if task_type == 'Classification':
                                class_colors = ['#f472b6', '#60a5fa', '#34d399', '#fbbf24', '#a855f7']
                                unique_targets = np.unique(st.session_state.y_test)
                                for i, target in enumerate(unique_targets):
                                    mask = st.session_state.y_test == target
                                    color = class_colors[i % len(class_colors)]
                                    fig.add_trace(go.Scatter(
                                        x=X_test_pca[mask, 0],
                                        y=X_test_pca[mask, 1],
                                        mode='markers',
                                        name=f'Class {int(target)}',
                                        marker=dict(size=10, color=color, line=dict(width=1.5, color='white'))
                                    ))
                            else:
                                fig.add_trace(go.Scatter(
                                    x=X_test_pca[:, 0],
                                    y=X_test_pca[:, 1],
                                    mode='markers',
                                    marker=dict(size=10, color=st.session_state.y_test, colorscale='Magma', line=dict(width=1.5, color='white')),
                                    text=st.session_state.y_test,
                                    name="Test Data"
                                ))
                                
                            subtitle = ""
                            if "KNeighbors" in model_type:
                                subtitle = f" (K={model.n_neighbors})"
                            elif "SVC" in model_type or "SVR" in model_type:
                                subtitle = f" (Kernel={model.kernel}, C={model.C})"
                                
                            fig.update_layout(
                                title=f"{title_prefix}{subtitle} Decision Boundaries",
                                xaxis_title="Standardized PC 1",
                                yaxis_title="Standardized PC 2",
                                template="plotly_dark",
                                height=600,
                                font=dict(family="Outfit", color="white"),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Could not generate Visualization: {str(e)}")
                
                
                elif 'LinearRegression' in model_type:
                    # Linear Regression - Coefficients
                    st.markdown("**Linear Regression Coefficients:**")
                    coef_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Coefficient': model.coef_
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    
                    fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                               template="plotly_dark", color='Coefficient',
                               color_continuous_scale="RdBu",
                               title="Linear Regression Coefficients")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Intercept", f"{model.intercept_:.4f}")
                
            except Exception as e:
                st.info(f"ℹ️ Model visualization not available for this model type: {str(e)}")
            
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
                # Ensure feature profile exists
                if "feature_profile" not in st.session_state:
                    st.error("❌ Prediction Metadata missing. Please re-train your model to enable v2 Prediction Engine.")
                    st.stop()
                
                profile = st.session_state.feature_profile
                
                # Header with modern glassmorphism
                st.markdown("""
                    <div style='background: rgba(255, 255, 255, 0.03); border-radius: 15px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 25px;'>
                        <h4 style='margin:0; color:#a78bfa;'>📋 Manual Entry Terminal</h4>
                        <p style='color:#94a3b8; font-size: 0.9em; margin-bottom: 0;'>Enter raw feature values as they appear in the original real-world data. The engine will handle all mathematical transformations internally.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                input_data = {}
                feature_names = st.session_state.feature_names
                
                # Two-column layout for inputs
                input_cols = st.columns(2)
                
                for idx, feat in enumerate(feature_names):
                    feat_meta = profile.get(feat, {'type': 'numeric', 'min': 0.0, 'max': 100.0, 'default': 0.0})
                    
                    with input_cols[idx % 2]:
                        if feat_meta['type'] == 'categorical':
                            # Categorical Dropdown
                            input_data[feat] = st.selectbox(
                                label=f"🏷️ {feat}",
                                options=feat_meta['options'],
                                index=0 if feat_meta['options'] else None,
                                key=f"input_v2_{feat}",
                                help=f"Original categories for {feat}"
                            )
                        else:
                            # Numeric Input with original ranges
                            input_data[feat] = st.number_input(
                                label=f"🔢 {feat}",
                                min_value=float(feat_meta['min']),
                                max_value=float(feat_meta['max']),
                                value=float(feat_meta['default']),
                                key=f"input_v2_{feat}",
                                help=f"Natural range: {feat_meta['min']} to {feat_meta['max']}"
                            )
                
                st.markdown("---")
                
                # Bottom Action Area
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    if st.button("✨ Generate Prediction", key="predict_v2_btn", use_container_width=True):
                        try:
                            # PREPROCESSING BRIDGE: Raw Input -> Model Vector
                            # 1. Create Dataframe
                            raw_input_df = pd.DataFrame([input_data])
                            processed_df = raw_input_df.copy()
                            
                            # 2. Apply Categorical Encoding
                            if st.session_state.categorical_mappings:
                                for cat_feat, mapping in st.session_state.categorical_mappings.items():
                                    if cat_feat in processed_df.columns:
                                        val = str(processed_df[cat_feat].iloc[0])
                                        processed_df[cat_feat] = mapping.get(val, 0) # Fallback to 0 if unknown
                            
                            # 3. Apply Scaling
                            if "scaler" in st.session_state and st.session_state.scaler is not None:
                                num_cols = st.session_state.numeric_columns_for_scaling
                                if num_cols:
                                    scale_targets = [c for c in num_cols if c in processed_df.columns]
                                    if scale_targets:
                                        processed_df[scale_targets] = st.session_state.scaler.transform(processed_df[scale_targets])
                            
                            # 4. Predict
                            with st.spinner("Analyzing data patterns..."):
                                raw_prediction = model.predict(processed_df)[0]
                                
                                # Build human-readable label
                                target_name = st.session_state.get('target_col_name', 'Target')
                                target_classes = st.session_state.get('target_classes', [])
                                
                                if task_type == "Classification":
                                    pred_int = int(raw_prediction)
                                    # Try to map back to original value
                                    if target_classes and pred_int < len(target_classes):
                                        original_value = target_classes[pred_int]
                                        display_label = f"{original_value}"
                                        display_subtitle = f"{target_name} = {original_value} (Class {pred_int})"
                                    else:
                                        display_label = f"Class {pred_int}"
                                        display_subtitle = f"{target_name} = {pred_int}"
                                else:
                                    display_label = f"{raw_prediction:.4f}"
                                    display_subtitle = f"{target_name} = {raw_prediction:.4f}"
                                
                                st.session_state.last_prediction = {
                                    'value': display_label,
                                    'subtitle': display_subtitle,
                                    'type': task_type,
                                    'raw': raw_prediction,
                                    'target_name': target_name,
                                    'target_classes': target_classes
                                }
                                
                                if task_type == 'Classification' and hasattr(model, 'predict_proba'):
                                    st.session_state.last_prediction['probs'] = model.predict_proba(processed_df)[0]
                        
                        except Exception as e:
                            st.error(f"❌ Processing Error: {str(e)}")
                
                # Results Display Area (Glassmorphic)
                with res_col2:
                    if "last_prediction" in st.session_state:
                        pred = st.session_state.last_prediction
                        
                        # Premium Result Design
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(30, 41, 59, 0.8) 100%); 
                                        padding: 25px; border-radius: 20px; border: 1px solid rgba(139, 92, 246, 0.3); text-align: center;'>
                                <div style='color: #c4b5fd; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;'>{pred.get('target_name', 'Prediction')}</div>
                                <div style='font-size: 3.5rem; font-weight: 800; color: white; line-height: 1; margin-bottom: 10px;'>{pred['value']}</div>
                                <div style='height: 4px; width: 60px; background: #8b5cf6; margin: 15px auto; border-radius: 2px;'></div>
                                <div style='color: #94a3b8;'>{pred.get('subtitle', '')} • {pred['type']} ✅</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence Visualization
                        if 'probs' in pred:
                            # Use human-readable class names
                            tc = pred.get('target_classes', [])
                            # Get classes directly from the model to guarantee matching length with 'probs'
                            model_classes = st.session_state.trained_model.classes_
                            class_labels = []
                            for c in model_classes:
                                c_int = int(c)
                                # Map back to human readable if possible
                                if tc and c_int < len(tc):
                                    class_labels.append(f"{tc[c_int]}")
                                else:
                                    class_labels.append(f"Class {c_int}")
                            
                            prob_df = pd.DataFrame({'Class': class_labels, 'Confidence': pred['probs']})
                            fig = px.bar(prob_df, x='Confidence', y='Class', orientation='h',
                                       color='Confidence', color_continuous_scale='Purp',
                                       height=250, template='plotly_dark')
                            fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.markdown("#### Upload Dataset for Batch Predictions")
                
                with st.expander("📋 **What format should my CSV have?**", expanded=True):
                    st.markdown("**Required CSV Format:**")
                    st.info(f"Your CSV file must contain the exact same column names and original values as in your dataset.")
                    
                    # Detect categorical features from original dataset (same priority logic as manual input)
                    all_categorical_features = []
                    if "original_df_for_ref" in st.session_state:
                        all_categorical_features = st.session_state.original_df_for_ref.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if not all_categorical_features and st.session_state.categorical_mappings:
                        all_categorical_features = list(st.session_state.categorical_mappings.keys())
                    
                    if not all_categorical_features and "original_feature_types" in st.session_state:
                        all_categorical_features = st.session_state.original_feature_types.get('categorical', [])
                    
                    cols_display_list = []
                    for col in st.session_state.feature_names:
                        if col in all_categorical_features and "original_df_for_ref" in st.session_state:
                            try:
                                unique_vals = st.session_state.original_df_for_ref[col].unique().tolist()
                                columns_dict = {
                                    'Column Name': col,
                                    'Data Type': 'Categorical',
                                    'Unique Values': len(unique_vals),
                                    'Example': str(unique_vals[0]) if unique_vals else 'N/A',
                                    'Required': '✅ Yes'
                                }
                            except:
                                columns_dict = {
                                    'Column Name': col,
                                    'Data Type': 'Categorical',
                                    'Unique Values': '?',
                                    'Example': 'See dataset',
                                    'Required': '✅ Yes'
                                }
                        else:
                            columns_dict = {
                                'Column Name': col,
                                'Data Type': 'Numeric',
                                'Unique Values': 'Any',
                                'Example': 'e.g., 25',
                                'Required': '✅ Yes'
                            }
                        cols_display_list.append(columns_dict)
                    
                    cols_display = pd.DataFrame(cols_display_list)
                    st.dataframe(cols_display, use_container_width=True, hide_index=True)
                    
                    st.markdown("**Example CSV Format:**")
                    st.info("Use original values from your dataset, not preprocessed/numeric values:")
                    example_rows = {}
                    for col in st.session_state.feature_names[:min(3, len(st.session_state.feature_names))]:
                        if col in all_categorical_features and "original_df_for_ref" in st.session_state:
                            try:
                                unique_vals = st.session_state.original_df_for_ref[col].unique().tolist()
                                example_rows[col] = str(unique_vals[0])
                            except:
                                example_rows[col] = 'N/A'
                        else:
                            example_rows[col] = st.session_state.X_test[col].mean()
                    if len(st.session_state.feature_names) > 3:
                        st.caption("(showing first 3 features)")
                    st.code(pd.DataFrame([example_rows]).to_csv(index=False), language="csv")
                
                st.markdown("**Upload your CSV file:**")
                uploaded_pred_file = st.file_uploader("Upload CSV File", type=["csv"], key="pred_upload")
                
                if uploaded_pred_file is not None:
                    try:
                        pred_df = pd.read_csv(uploaded_pred_file)
                        
                        st.info(f"📊 Loaded {len(pred_df)} rows with {len(pred_df.columns)} columns")
                        
                        # Validate columns
                        missing_cols = set(st.session_state.feature_names) - set(pred_df.columns)
                        extra_cols = set(pred_df.columns) - set(st.session_state.feature_names)
                        
                        if missing_cols:
                            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                        else:
                            if extra_cols:
                                st.warning(f"⚠️ Extra columns will be ignored: {', '.join(extra_cols)}")
                            
                            st.success("✅ CSV format is valid!")
                            # Select only required features
                            pred_df_subset = pred_df[st.session_state.feature_names].copy()
                            
                            # Robust Preprocessing for Batch Data
                            # 1. Fill missing values using training set logic
                            missing_count = pred_df_subset.isnull().sum().sum()
                            if missing_count > 0:
                                st.info(f"🧹 Auto-cleaning: Filling {missing_count} missing values in prediction data...")
                                # Use training set means for numeric
                                if "numeric_columns_in_training" in st.session_state:
                                    for col in st.session_state.numeric_columns_in_training:
                                        if col in pred_df_subset.columns:
                                            # If training mean exists in current_df...
                                            mean_val = st.session_state.current_df[col].mean() if col in st.session_state.current_df.columns else 0
                                            pred_df_subset[col] = pred_df_subset[col].fillna(mean_val)
                            
                            # 2. Safe Encoding (Handle Unseen Categories)
                            if st.session_state.categorical_mappings:
                                for feature in st.session_state.categorical_mappings:
                                    if feature in pred_df_subset.columns:
                                        mapping = st.session_state.categorical_mappings[feature]
                                        
                                        # Function to safely map values
                                        def safe_map(val):
                                            val_str = str(val)
                                            if val_str in mapping:
                                                return mapping[val_str]
                                            else:
                                                # Use the mode (most common code) as fallback for unseen values
                                                return list(mapping.values())[0] if mapping else 0
                                        
                                        pred_df_subset[feature] = pred_df_subset[feature].apply(safe_map)
                            
                            if st.button("🔮 Predict All", key="predict_batch"):
                                try:
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
                                    st.error(f"❌ Prediction error: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")


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
                st.info("K-Means Clustering: Grouping data based on proximity to centroids.")
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            
            if st.button("🚀 Run Clustering"):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(df_numeric)
                    silhouette_avg = silhouette_score(df_numeric, clusters)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Number of Clusters", n_clusters)
                    col2.metric("Silhouette Score", f"{silhouette_avg:.4f}")
                    
                    st.info(f"✅ Clustering complete! Silhouette Score: {silhouette_avg:.4f}")
                    st.caption("Silhouette Score: -1 (bad) to 1 (perfect) - measures cluster quality")
                    
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
