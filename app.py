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


# Custom CSS for Modern Design (Dark sleek theme with improved contrast)
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Modify sidebar with better contrast */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers - Improved contrast */
    h1 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
        border-bottom: 2px solid #30363d;
        padding-bottom: 10px;
    }
    
    h3, h4, h5, h6 {
        color: #79c0ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
    }
    
    /* Main text - Better contrast */
    p, span, div {
        color: #e6edf3 !important;
    }
    
    /* Metric labels - Enhanced visibility with better contrast */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.08) 0%, rgba(79, 195, 247, 0.05) 100%);
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 12px !important;
    }
    
    [data-testid="metric-container"] label {
        color: #79c0ff !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="metric-container"] div {
        color: #f0f6fc !important;
        font-weight: 600 !important;
        font-size: 24px !important;
    }
    
    /* Markdown text enhancement */
    .markdown-text-container {
        color: #e6edf3 !important;
    }
    
    /* Caption styling - Improved contrast */
    .stCaption {
        color: #8b949e !important;
        font-weight: 500 !important;
    }
    
    /* Buttons - Better visual feedback */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0969da 0%, #0860ca 100%);
        color: #000000 !important;
        border: 1px solid #1f6feb !important;
        border-radius: 8px;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0860ca 0%, #073a99 100%);
        color: #000000 !important;
        border: 1px solid #58a6ff !important;
        box-shadow: 0 3px 12px rgba(9, 105, 218, 0.4);
        transform: translateY(-1px);
    }
    
    /* Style st.info boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #30363d;
        background-color: rgba(88, 166, 255, 0.05) !important;
        backdrop-filter: blur(10px);
    }
    
    .stAlert > div {
        color: #e6edf3 !important;
    }
    
    /* Sidebar text contrast */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
        color: #e6edf3 !important;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #79c0ff !important;
    }
    
    /* Radio and checkbox labels */
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
        color: #e6edf3 !important;
        font-weight: 500 !important;
    }
    
    /* Input field styling */
    input, select, textarea {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    
    input:focus, select:focus, textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1) !important;
    }
    
    /* Selectbox text color */
    [data-testid="stSelectbox"] label {
        color: #e6edf3 !important;
    }
    
    [data-baseweb="select"] div {
        color: #e6edf3 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: transparent;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(88, 166, 255, 0.05);
    }
    
    /* Streamlit tabs styling */
    [data-baseweb="tab"] {
        color: #8b949e !important;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    
    /* Success and warning alerts */
    .stSuccess {
        background-color: rgba(3, 102, 214, 0.1) !important;
        color: #58a6ff !important;
    }
    
    .stWarning {
        background-color: rgba(187, 128, 9, 0.1) !important;
        color: #d29922 !important;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: rgba(88, 166, 255, 0.08) !important;
        border: 2px dashed #58a6ff !important;
        border-radius: 8px !important;
        padding: 20px !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #79c0ff !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background-color: transparent !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #0969da !important;
        color: #000000 !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #0860ca !important;
        box-shadow: 0 2px 8px rgba(9, 105, 218, 0.4) !important;
    }
    
    /* Dropdown menu styling */
    [data-baseweb="menu"] li div {
        color: #e6edf3 !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #1c2128 !important;
    }
    
    [data-baseweb="listbox"] li {
        color: #e6edf3 !important;
        background-color: #161b22 !important;
    }
    
    /* Multiselect and option styling - Fix all dropdown items */
    [role="option"] {
        color: #e6edf3 !important;
        background-color: #161b22 !important;
    }
    
    [role="option"]:hover {
        background-color: #0d419f !important;
    }
    
    /* Streamlit multiselect tokens/tags */
    [data-testid="stMultiSelect"] div {
        color: #e6edf3 !important;
    }
    
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background-color: #0969da !important;
        color: #ffffff !important;
    }
    
    /* All option/select elements */
    option {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }
    
    optgroup {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }
    
    /* Ensure all text in popovers and menus is visible */
    [data-baseweb="popover"] {
        background-color: #1c2128 !important;
        color: #e6edf3 !important;
    }
    
    [data-baseweb="popover"] * {
        color: #e6edf3 !important;
    }
    
    /* List items in menus */
    [data-baseweb="menu"] li {
        background-color: #1c2128 !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #0d419f !important;
    }
    
    /* Download button styling */
    div.stDownloadButton > button:first-child {
        background: linear-gradient(135deg, #0969da 0%, #0860ca 100%);
        color: #000000 !important;
        border: 1px solid #1f6feb !important;
        border-radius: 8px;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    div.stDownloadButton > button:first-child:hover {
        background: linear-gradient(135deg, #0860ca 0%, #073a99 100%);
        color: #000000 !important;
        border: 1px solid #58a6ff !important;
        box-shadow: 0 3px 12px rgba(9, 105, 218, 0.4);
        transform: translateY(-1px);
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
                task_type = st.selectbox("Choose Task Type", ["Classification", "Regression"], key="task_type_select")
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
                with st.expander("⚙️ **Customize Model Parameters** (Leave empty to use defaults)", expanded=False):
                    st.markdown("**📌 Random Forest Parameters:**")
                    rf_cols = st.columns(3)
                    with rf_cols[0]:
                        rf_n_estimators = st.number_input("Random Forest - n_estimators", min_value=10, max_value=500, value=None, help="Number of trees in forest (default: 100)")
                    with rf_cols[1]:
                        rf_max_depth = st.number_input("Random Forest - max_depth", min_value=1, max_value=50, value=None, help="Max depth of tree (default: None)")
                    with rf_cols[2]:
                        rf_min_samples = st.number_input("Random Forest - min_samples_split", min_value=2, max_value=20, value=None, help="Min samples to split (default: 2)")
                    
                    st.divider()
                    st.markdown("**📌 Decision Tree Parameters:**")
                    dt_cols = st.columns(2)
                    with dt_cols[0]:
                        dt_max_depth = st.number_input("Decision Tree - max_depth", min_value=1, max_value=50, value=None, help="Max depth of tree (default: None)")
                    with dt_cols[1]:
                        dt_min_samples = st.number_input("Decision Tree - min_samples_split", min_value=2, max_value=20, value=None, help="Min samples to split (default: 2)")
                    
                    if task_type == "Regression":
                        st.divider()
                        st.markdown("**📌 Linear Regression Parameters:**")
                        st.info("ℹ️ Linear Regression uses default parameters (no customization needed)")
                    
                    st.divider()
                    st.info("💡 **Tip:** Leave fields empty to use default parameters. Custom values override defaults.")
                
                # Store custom parameters in session state
                custom_params = {
                    "rf_n_estimators": rf_n_estimators,
                    "rf_max_depth": rf_max_depth if rf_max_depth is not None else None,
                    "rf_min_samples": rf_min_samples,
                    "dt_max_depth": dt_max_depth if dt_max_depth is not None else None,
                    "dt_min_samples": dt_min_samples
                }
                
                st.markdown("---")
                
                if st.button("🚀 Train Model", key="train_btn"):
                    # Display selected parameters
                    with st.expander("📋 **Selected Parameters**", expanded=True):
                        st.markdown("**Custom Parameters to be used:**")
                        
                        any_custom = False
                        if custom_params.get("rf_n_estimators"):
                            st.info(f"🌲 Random Forest - n_estimators: **{custom_params['rf_n_estimators']}**")
                            any_custom = True
                        if custom_params.get("rf_max_depth") is not None:
                            st.info(f"🌲 Random Forest - max_depth: **{custom_params['rf_max_depth']}**")
                            any_custom = True
                        if custom_params.get("rf_min_samples"):
                            st.info(f"🌲 Random Forest - min_samples_split: **{custom_params['rf_min_samples']}**")
                            any_custom = True
                        if custom_params.get("dt_max_depth") is not None:
                            st.info(f"🌳 Decision Tree - max_depth: **{custom_params['dt_max_depth']}**")
                            any_custom = True
                        if custom_params.get("dt_min_samples"):
                            st.info(f"🌳 Decision Tree - min_samples_split: **{custom_params['dt_min_samples']}**")
                            any_custom = True
                        
                        if not any_custom:
                            st.success("✅ Using **default parameters** for all models (no custom values set)")
                    try:
                        X_for_training = X.copy()
                        progress_placeholder = st.empty()
                        
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
                        
                        # Apply scaling to numeric features if available in session state
                        if "scaler" in st.session_state and st.session_state.scaler is not None:
                            numeric_cols = st.session_state.numeric_columns_for_scaling
                            if numeric_cols and any(col in X_for_training.columns for col in numeric_cols):
                                cols_to_scale = [col for col in numeric_cols if col in X_for_training.columns]
                                X_for_training[cols_to_scale] = st.session_state.scaler.transform(X_for_training[cols_to_scale])
                                progress_placeholder.info(f"📊 Applied {st.session_state.scaling_method} scaling to numeric features")
                        
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
                            params = {}
                            
                            if model_name == "Random Forest":
                                if custom_params.get("rf_n_estimators"):
                                    params["n_estimators"] = custom_params["rf_n_estimators"]
                                if custom_params.get("rf_max_depth") is not None:
                                    params["max_depth"] = custom_params["rf_max_depth"]
                                if custom_params.get("rf_min_samples"):
                                    params["min_samples_split"] = custom_params["rf_min_samples"]
                                params["random_state"] = 42
                                
                            elif model_name == "Decision Tree":
                                if custom_params.get("dt_max_depth") is not None:
                                    params["max_depth"] = custom_params["dt_max_depth"]
                                if custom_params.get("dt_min_samples"):
                                    params["min_samples_split"] = custom_params["dt_min_samples"]
                                params["random_state"] = 42
                            
                            # Create new instance with custom or default params
                            if params:
                                return model_class(**params)
                            else:
                                return model_class()
                        
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
                                    grid_search.fit(X_train, y_train)
                                    model = grid_search.best_estimator_
                                    
                                    st.success(f"✅ Best parameters found: {grid_search.best_params_}")
                                    st.info(f"Best CV Score: {grid_search.best_score_:.4f}")
                                else:
                                    st.warning(f"⚠️ Hyperparameter tuning not available for {model_name}. Using default parameters.")
                                    model.fit(X_train, y_train)
                            else:
                                progress_placeholder.info("📊 Step 4/4: Training model with custom/default parameters...")
                                model.fit(X_train, y_train)
                            
                            # Store in session state
                            st.session_state.trained_model = model
                            st.session_state.feature_names = X_train.columns.tolist()
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            
                            # Keep the categorical mappings from encoding phase (already stored in session state)
                            
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
                    # Logistic Regression - ROC Curve for binary classification
                    if task_type == 'Classification':
                        st.markdown("**ROC Curve (Logistic Regression):**")
                        from sklearn.metrics import roc_curve, auc
                        
                        unique_classes = np.unique(y_test)
                        if len(unique_classes) == 2:
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                                    name=f'ROC Curve (AUC = {roc_auc:.4f})',
                                                    line=dict(color='#58a6ff', width=3)))
                            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                    name='Random Classifier',
                                                    line=dict(color='#8b949e', dash='dash')))
                            
                            fig.update_layout(title='ROC Curve - Logistic Regression',
                                            xaxis_title='False Positive Rate',
                                            yaxis_title='True Positive Rate',
                                            template='plotly_dark',
                                            hovermode='closest',
                                            height=500)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"ℹ️ ROC Curve is available for binary classification only.")
                
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
                st.markdown("#### Manual Input Fields")
                st.info(f"ℹ️ Enter values exactly as they appear in the original dataset. They will be automatically preprocessed before prediction.")
                
                # Detect categorical features from original dataset
                # Priority 1: Check original_df_for_ref columns (most reliable)
                all_categorical = []
                if "original_df_for_ref" in st.session_state:
                    all_categorical = st.session_state.original_df_for_ref.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Priority 2: Fall back to stored categorical_mappings
                if not all_categorical and st.session_state.categorical_mappings:
                    all_categorical = list(st.session_state.categorical_mappings.keys())
                
                # Priority 3: Fall back to original_feature_types
                if not all_categorical and "original_feature_types" in st.session_state:
                    all_categorical = st.session_state.original_feature_types.get('categorical', [])
                
                # DEBUG: Show what was detected
                with st.expander("🔍 DEBUG: Feature Detection", expanded=True):
                    st.write("**Categorical Mappings:**")
                    st.write(st.session_state.categorical_mappings)
                    st.write("\n**Original Feature Types:**")
                    st.write(st.session_state.original_feature_types if "original_feature_types" in st.session_state else "NOT SET")
                    st.write(f"\n**All Features:** {st.session_state.feature_names}")
                    st.write(f"\n**Detected Categorical (from original_df):** {all_categorical}")
                
                # Get unique values from original data for each categorical feature
                categorical_options = {}
                for feat in st.session_state.feature_names:
                    if feat in all_categorical and "original_df_for_ref" in st.session_state:
                        try:
                            unique_vals = sorted(st.session_state.original_df_for_ref[feat].unique().tolist())
                            categorical_options[feat] = unique_vals
                        except:
                            pass
                
                # Show preprocessing pipeline info
                st.markdown("**📋 Applied Preprocessing:**")
                preprocess_cols = st.columns(3)
                with preprocess_cols[0]:
                    if st.session_state.categorical_mappings:
                        st.caption(f"✅ Categorical Encoding: {len(st.session_state.categorical_mappings)} features")
                    else:
                        st.caption("⭕ Categorical Encoding: None")
                with preprocess_cols[1]:
                    if "scaler" in st.session_state and st.session_state.scaler is not None:
                        st.caption(f"✅ Scaling: {st.session_state.scaling_method}")
                    else:
                        st.caption("⭕ Scaling: None")
                with preprocess_cols[2]:
                    st.caption(f"📊 Features: {len(st.session_state.feature_names)}")
                
                # Display feature info table with proper type detection
                feature_info_list = []
                for feature in st.session_state.feature_names:
                    if feature in categorical_options:
                        num_options = len(categorical_options[feature])
                        feature_info_list.append({
                            'Feature': feature, 
                            'Type': 'Categorical', 
                            'Input': 'Dropdown',
                            'Unique Values': num_options,
                            'Example': str(categorical_options[feature][0]) if categorical_options[feature] else 'N/A'
                        })
                    else:
                        feature_info_list.append({
                            'Feature': feature, 
                            'Type': 'Numeric', 
                            'Input': 'Number Input',
                            'Unique Values': 'Any',
                            'Example': 'e.g., 25'
                        })
                
                feature_info = pd.DataFrame(feature_info_list)
                st.dataframe(feature_info, use_container_width=True, hide_index=True)
                
                st.markdown("**📝 Input all feature values below:**")
                
                input_data = {}
                cols = st.columns(2)
                
                for idx, feature in enumerate(st.session_state.feature_names):
                    with cols[idx % 2]:
                        if feature in categorical_options:
                            # Categorical input - dropdown with original values
                            selected_val = st.selectbox(
                                label=f"{feature} (Categorical)",
                                options=categorical_options[feature],
                                help=f"Select from available values",
                                key=f"input_{feature}"
                            )
                            input_data[feature] = selected_val
                        else:
                            # Numeric input
                            min_val = st.session_state.X_test[feature].min()
                            max_val = st.session_state.X_test[feature].max()
                            avg_val = st.session_state.X_test[feature].mean()
                            
                            input_data[feature] = st.number_input(
                                label=f"{feature} (Numeric)",
                                value=float(avg_val),
                                help=f"Range in training data: {min_val:.2f} to {max_val:.2f}",
                                key=f"input_{feature}"
                            )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🔮 Predict", key="predict_manual"):
                        try:
                            # Create dataframe from input (with original values)
                            pred_input = pd.DataFrame([input_data])
                            
                            # Apply same preprocessing as training data
                            # Step 1: Encode categorical features using stored mappings
                            if st.session_state.categorical_mappings:
                                for feature, mapping in st.session_state.categorical_mappings.items():
                                    if feature in pred_input.columns:
                                        original_val = pred_input[feature].iloc[0]
                                        # Convert using the mapping: original value → numeric
                                        numeric_val = mapping.get(str(original_val), original_val)
                                        pred_input[feature] = numeric_val
                            
                            # Step 2: Apply scaling to numeric features using stored scaler
                            if "scaler" in st.session_state and st.session_state.scaler is not None:
                                numeric_cols = st.session_state.numeric_columns_for_scaling
                                if numeric_cols:
                                    cols_to_scale = [col for col in numeric_cols if col in pred_input.columns]
                                    if cols_to_scale:
                                        pred_input[cols_to_scale] = st.session_state.scaler.transform(pred_input[cols_to_scale])
                            
                            # Make prediction
                            prediction = model.predict(pred_input)[0]
                            
                            if task_type == 'Classification':
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(pred_input)[0]
                                    st.success(f"✅ **Prediction: {int(prediction)}**")
                                    
                                    # Show probabilities as bar chart
                                    prob_df = pd.DataFrame({
                                        'Class': [str(int(c)) for c in np.unique(y_test)],
                                        'Probability': probabilities
                                    })
                                    st.dataframe(prob_df, use_container_width=True)
                                    
                                    fig = px.bar(prob_df, x='Class', y='Probability', 
                                               template="plotly_dark", color_discrete_sequence=['#58a6ff'],
                                               title="Prediction Confidence by Class")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.success(f"✅ **Prediction: {int(prediction)}**")
                            else:
                                st.success(f"✅ **Predicted Value: {prediction:.4f}**")
                        
                        except Exception as e:
                            st.error(f"❌ Prediction error: {str(e)}")
                
                with col2:
                    st.markdown("**Input Summary:**")
                    st.caption(f"✅ Total features: {len(st.session_state.feature_names)}")
                    st.caption(f"✅ Categorical: {len(categorical_options)} features")
                    st.caption(f"📊 Numeric: {len(st.session_state.feature_names) - len(categorical_options)} features")
                    st.caption("💡 Tip: Values outside training range may give unusual predictions")
            
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
                            
                            # Apply encoding to the data (without showing conversion info)
                            if st.session_state.categorical_mappings:
                                for feature in st.session_state.categorical_mappings:
                                    if feature in pred_df_subset.columns:
                                        pred_df_subset[feature] = pred_df_subset[feature].astype(str).map(st.session_state.categorical_mappings[feature])
                            
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
