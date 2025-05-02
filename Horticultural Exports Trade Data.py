import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Horticultural Trends App", page_icon=":seedling:", layout="wide")

# --- WHITE BACKGROUND + CUSTOM STYLING ---
st.markdown(
    """
    <style>
    /* General Background */
    .stApp {
        background-color: white;
        color: black;  /* Set text color to black */
    }

    /* Styling for the table itself */
    .stDataFrame table {
        background-color: white !important;  /* White background for table */
        border-collapse: collapse;  /* Ensures borders are shown */
        width: 100%;  /* Table width set to 100% */
    }

    /* Styling for table headers (th) */
    .stDataFrame table th {
        background-color: #f1f1f1;  /* Light grey background for headers */
        color: black;  /* Black text color for headers */
        padding: 10px;
        border: 1px solid black !important;  /* Black borders for header cells */
        text-align: center;
    }

    /* Styling for table data (td) */
    .stDataFrame table td {
        background-color: white !important;  /* White background for cells */
        color: black;  /* Black text */
        padding: 10px;
        border: 1px solid black !important;  /* Black borders for cells */
        text-align: center;
    }

    /* Table border styling */
    .stDataFrame table {
        border: 1px solid black !important;  /* Black border around the table */
    }

    /* Custom Text (For headings) */
    .stText {
        font-weight: bold;
        color: #006400; /* Dark Green for Text */
    }

    /* Center Images */
    .stImage {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- DISPLAY TWO IMAGES CENTERED AND LARGER ---
col1, col2 = st.columns(2)  # Create two columns to display images side by side

with col1:
    st.image(r"C:\Users\YVONNE ODERA\Documents\API\Image 3.jpg", use_container_width=True)  # Adjust width as needed
    #st.image(r"CC:\Users\YVONNE ODERA\Documents\API\Image 6.jpg", use_container_width=True)  # Adjust width as needed

with col2:
    st.image(r"C:\Users\YVONNE ODERA\Documents\API\Image 6.jpg", use_container_width=True)  # Adjust width as needed



# --- HEADER & INTRODUCTORY REMARK ---
st.markdown("""
# 🌱 Welcome to the Horticultural Market Trends Analysis App!

This platform helps users **uncover market trends** and **pricing dynamics** within the horticultural sector.

By using key indicators like:

- **GDP**
- **Kenya's GDP (KEN_GDP)**
- **Average Dollar Exchange Rate (AvgDollarRate)**
- **Rainfall Patterns**
- **Greenhouse Gas Emissions**
- **Export Quantities**
- **Distance-related Costs**
- **Labour Force Metrics**
- **Preferential Trade Agreements (PTA)**

We analyze how these factors influence **horticultural trade values**.

Using **Ridge Regression modeling**, this app enables users to understand the impact of **macroeconomic** and **environmental changes** on horticultural market performance.
""", unsafe_allow_html=True)





# --- Load datasets in the background ---

# @st.cache_data
# def load_key_data(*args, **kwargs):
#     return pd.read_csv(r"C:/Users/YVONNE ODERA/Documents/API\Horticultural Exports Trade/world.csv")

# @st.cache_data
# def load_world_data():
#     return pd.read_csv(r"C:/Users/YVONNE ODERA/Documents/API/Horticultural Exports Trade/world.csv")

# @st.cache_data
# def load_kenya_data(*args, **kwargs):
#     return pd.read_csv(r"C:\Users\YVONNE ODERA\Documents\API\Horticultural Exports Trade\Horticulturedata.csv")

# @st.cache_data
# def load_main_data():
#     return pd.read_csv(r"C:\Users\YVONNE ODERA\Documents\API\Horticultural Exports Trade\data_model1_ammended2.csv")

# @st.cache_data
# def load_prediction_data():
#     return pd.read_csv(r"C:\Users\YVONNE ODERA\Documents\API\Horticultural Exports Trade\data_model1_ammended2.csv")

# # --- Load datasets into session state ---
# if 'world_df' not in st.session_state:
#     st.session_state.world_df = load_world_data()

# if 'main_df' not in st.session_state:
#     st.session_state.main_df = load_main_data()

# if 'key_df' not in st.session_state:
#     st.session_state.key_df = load_key_data()

# if 'kenya_df' not in st.session_state:
#     st.session_state.analysis_df = load_kenya_data()

# if 'prediction_df' not in st.session_state:
#     st.session_state.prediction_df = load_prediction_data()
# Sidebar for data upload
with st.sidebar:
    st.header("Upload your datasets")
    key_data = st.file_uploader("Uoload your file.csv", type="csv", key="key")
    world_data = st.file_uploader("Upload your file.csv", type="csv", key="world")
    kenya_data = st.file_uploader("Upload your file.csv", type="csv", key="kenya")
    main_data = st.file_uploader("Upload your file.csv", type="csv", key="main")
    prediction_data = st.file_uploader("Upload your file.csv", type="csv", key="prediction")

# Load data into session_state
if key_data: st.session_state.key_df = pd.read_csv(key_data)
if world_data: st.session_state.world_df = pd.read_csv(world_data)
if kenya_data: st.session_state.kenya_df = pd.read_csv(kenya_data)
if main_data: st.session_state.main_df = pd.read_csv(main_data)
if prediction_data: st.session_state.prediction_df = pd.read_csv(prediction_data)

# Tabs
selected_tab = st.selectbox("Select a tab", [
    "Export Goods Key",
    "World Stats",
    "Kenya Stats",
    "Data Analysis",
    "Prediction"
])

# Export Goods Key Tab
if selected_tab == "Export Goods Key":
    st.subheader("Export Goods Key")
    df = st.session_state.get('key_df')
    if df is not None:
        if {"ProductCode", "ProductDescription"}.issubset(df.columns):
            export_goods_key = df[["ProductCode", "ProductDescription"]].drop_duplicates()
            styled_table = export_goods_key.style.set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#4CAF50'), ('color', 'black')]}
            ])
            st.dataframe(styled_table, use_container_width=True)
        else:
            st.warning("Missing columns in key data.")
    else:
        st.warning("Please upload the Export Goods Key file.")

# World Stats Tab
elif selected_tab == "World Stats":
    st.subheader("World Statistics")
    df2 = st.session_state.get('world_df')
    if df2 is not None:
        section = st.selectbox("Select a section", ["Data Analysis", "Graphs", "Trends"])

        if section == "Data Analysis":
            st.dataframe(df2.head(), use_container_width=True)
            st.dataframe(df2.describe(), use_container_width=True)

        elif section == "Graphs":
            try:
                top_importers = df2.groupby('Importer')['TradeValue in 1000 USD'].sum().nlargest(15)
                top_products = df2.groupby('ProductDescription')['TradeValue in 1000 USD'].sum().nlargest(10)
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                axes[0].barh(top_importers.index, top_importers.values)
                axes[1].bar(top_products.index, top_products.values)
                axes[1].tick_params(axis='x', rotation=90)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating graphs: {e}")

        elif section == "Trends":
            try:
                # Ensure the TradeValue is numeric
                df2['TradeValue in 1000 USD'] = pd.to_numeric(df2['TradeValue in 1000 USD'], errors='coerce')

                # Drop rows with missing values in required columns
                df_filtered = df2.dropna(subset=['ProductCode', 'Year', 'TradeValue in 1000 USD'])

                # Group the data
                df_grouped = df_filtered.groupby(['ProductCode', 'Year'])['TradeValue in 1000 USD'].mean().reset_index()

                # Convert ProductCode to string for consistent display
                df_grouped['ProductCode'] = df_grouped['ProductCode'].astype(str)
                available_codes = sorted(df_grouped['ProductCode'].unique())

                # Multiselect for user to choose codes
                selected_codes = st.multiselect(
                    "Select Product Codes to Plot",
                    options=available_codes,
                    default=[code for code in ['810', '709', '804', '902', '603'] if code in available_codes]
                )

                if selected_codes:
                    plt.figure(figsize=(10, 6))
                    for code in selected_codes:
                        product_data = df_grouped[df_grouped['ProductCode'] == code]
                        plt.plot(product_data['Year'], product_data['TradeValue in 1000 USD'], label=code)

                    plt.title('Trend of Trade Value by Product Code')
                    plt.xlabel('Year')
                    plt.ylabel('Trade Value (1000 USD)')
                    plt.legend(title='Product Code', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                else:
                    st.info("Please select at least one Product Code to view trends.")
            except Exception as e:
                st.error(f"Error generating trends: {e}")
    else:
        st.warning("Please upload the World Stats file.")

# Kenya Stats Tab
elif selected_tab == "Kenya Stats":
    st.subheader("Kenya Statistics")
    df3 = st.session_state.get('kenya_df')
    if df3 is not None:
        section = st.selectbox("Select a section", ["Data Analysis", "Graphs", "Trends"])

        if section == "Data Analysis":
            st.dataframe(df3.head(), use_container_width=True)

        elif section == "Graphs":
            try:
                df3['HorticulturalValue'] = pd.to_numeric(df3['HorticulturalValue'], errors='coerce')
                top_importers = df3.groupby('Country')['HorticulturalValue'].sum().nlargest(15)
                top_products = df3.groupby('ProductDescription')['HorticulturalValue'].sum().nlargest(10)
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                axes[0].barh(top_importers.index, top_importers.values)
                axes[1].bar(top_products.index, top_products.values)
                axes[1].tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating graphs: {e}")

        elif section == "Trends":
            try:
                # Ensure the TradeValue is numeric
                df3['HorticulturalValue'] = pd.to_numeric(df3['HorticulturalValue'], errors='coerce')

                # Drop rows with missing values in required columns
                df_filtered1 = df3.dropna(subset=['HortiProductCode', 'Year', 'HorticulturalValue'])

                # Group the data
                df_grouped1 = df_filtered1.groupby(['HortiProductCode', 'Year'])['HorticulturalValue'].mean().reset_index()

                # Convert ProductCode to string for consistent display
                df_grouped1['HortiProductCode'] = df_grouped1['HortiProductCode'].astype(str)
                available_codes = sorted(df_grouped1['HortiProductCode'].unique())

                # Multiselect for user to choose codes
                selected_codes = st.multiselect(
                    "Select Product Codes to Plot",
                    options=available_codes,
                    default=[code for code in ['810', '709', '804', '902', '603'] if code in available_codes]
                )

                if selected_codes:
                    plt.figure(figsize=(10, 6))
                    for code in selected_codes:
                        product_data = df_grouped1[df_grouped1['HortiProductCode'] == code]
                        plt.plot(product_data['Year'], product_data['HorticulturalValue'], label=code)

                    plt.title('Trend of Trade Value by Product Code')
                    plt.xlabel('Year')
                    plt.ylabel('HorticuturalValue')
                    plt.legend(title='Product Code', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                else:
                    st.info("Please select at least one Product Code to view trends.")
            except Exception as e:
                st.error(f"Error generating trends: {e}")
    else:
        st.warning("Please upload the World Stats file.")

# Data Analysis Tab
elif selected_tab == "Data Analysis":
    st.subheader("Data Analysis")
    df4 = st.session_state.get('main_df')
    if df4 is not None:
        section = st.selectbox("Select a section", ["Data Analysis", "Graphs", "Trends"])

        if section == "Data Analysis":
            st.dataframe(df4.head(), use_container_width=True)

        elif section == "Graphs":
            try:
                df4['HorticulturalValue'] = pd.to_numeric(df4['HorticulturalValue'], errors='coerce')
                top_importers = df4.groupby('Country')['HorticulturalValue'].sum().nlargest(15)
                top_products = df4.groupby('ProductDescription')['HorticulturalValue'].sum().nlargest(10)
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                axes[0].barh(top_importers.index, top_importers.values)
                axes[1].bar(top_products.index, top_products.values)
                axes[1].tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating graphs: {e}")

        elif section == "Trends":
            st.info("Trends analysis to be developed.")
            try:
                df_grouped = df4.groupby(['HortiProductCode', 'Year'])['HorticulturalValue'].mean().reset_index()
                plt.figure(figsize=(10, 6))
                for product_code in df_grouped['HortiProductCode'].unique():
                    product_data = df_grouped[df_grouped['HortiProductCode'] == product_code]
                    plt.plot(product_data['Year'], product_data['HorticulturalValue'], label=str(product_code))
                plt.title('Trend of Trade Value by Product Code')
                plt.xlabel('Year')
                plt.ylabel('Trade Value (1000 USD)')
                plt.legend(title='Product Code', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True)
                plt.tight_layout()
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error generating trends: {e}")
    else:
        st.warning("Please upload the World Stats file.")

# Prediction Tab
elif selected_tab == "Prediction":
    st.subheader("Prediction")
    df5 = st.session_state.get('prediction_df')

    if df5 is not None:
        section = st.selectbox("Choose Analysis Section", ["Feature Importance", "Metrics", "Predictions"])

        # Allow user to select features
        features = st.multiselect("Select Features for Analysis", options=df5.columns.tolist(), default=df5.columns.tolist())
        target_variable = st.selectbox("Select Target Variable", options=df5.columns.tolist())

        if section == "Feature Importance":
            try:
                # Load dataset
                X = df5[features].copy()
                y = df5[target_variable]

                # Convert X to numeric and drop NaNs
                X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

                # Log transform features
                X_log = X.copy()
                for col in X.columns:
                    if col == 'distance_costs':
                        X_log[col] = -np.log1p(X[col].abs())
                    else:
                        X_log[col] = np.sign(X[col]) * np.log1p(abs(X[col]))

                # Replace infs and fill NaNs
                X_log = X_log.replace([np.inf, -np.inf], np.nan).fillna(X_log.max())

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_log)
                X_scaled = pd.DataFrame(X_scaled, columns=X_log.columns)

                # Compute VIF
                vif_data = pd.DataFrame()
                vif_data['Feature'] = X_scaled.columns
                vif_data['VIF'] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]

                # Drop features with high VIF (>10)
                X_filtered = X_scaled[vif_data[vif_data['VIF'] < 10]['Feature']]

                # ----- RandomForest Regressor -----
                # Fit RandomForestRegressor
                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_regressor.fit(X_filtered, y)

                # Feature importance from RandomForest Regressor
                feature_importance_regressor = pd.DataFrame({
                    'Feature': X_filtered.columns,
                    'Importance': rf_regressor.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                # ----- RandomForest Classifier -----
                # Encode categorical target variable if it's categorical
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)

                # Fit RandomForestClassifier
                rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_classifier.fit(X_filtered, y_encoded)

                # Feature importance from RandomForest Classifier
                feature_importance_classifier = pd.DataFrame({
                    'Feature': X_filtered.columns,
                    'Importance': rf_classifier.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                # ----- Plotting Comparison of Feature Importances -----
                top_n = 10
                fig, ax = plt.subplots(1, 2, figsize=(16, 6))

                # Plot for Regressor
                ax[0].barh(feature_importance_regressor['Feature'][:top_n][::-1], feature_importance_regressor['Importance'][:top_n][::-1])
                ax[0].set_xlabel("Importance")
                ax[0].set_title("Top 10 Feature Importances (Random Forest Regressor)")

                # Plot for Classifier
                ax[1].barh(feature_importance_classifier['Feature'][:top_n][::-1], feature_importance_classifier['Importance'][:top_n][::-1])
                ax[1].set_xlabel("Importance")
                ax[1].set_title("Top 10 Feature Importances (Random Forest Classifier)")

                plt.tight_layout()
                st.pyplot(plt.gcf())

                # Display the top 10 features for both models
                st.write("Top 10 Features for RandomForest Regressor:")
                st.dataframe(feature_importance_regressor.head(top_n))

                st.write("Top 10 Features for RandomForest Classifier:")
                st.dataframe(feature_importance_classifier.head(top_n))

                # Display VIF table
                st.write("VIF Data after scaling and filtering:")
                st.dataframe(vif_data)

            except Exception as e:
                st.error(f"Error: {e}")

        elif section == "Metrics":
            try:
                numeric_cols = ['GDP', 'KEN_GDP', 'AvgDollarRate', 'Rainfall',
                                'greenhouse_gas_emissions_CO2_equivalent_kt', 'Quantity',
                                'distance_costs', 'labourforce', 'PTA']
                dummy_cols = [col for col in df5.columns if col.startswith("Product_")]
                X = df5[numeric_cols + dummy_cols].fillna(0)
                y = df5['HorticulturalValue'].fillna(0)

                X = sm.add_constant(X)
                ridge = Ridge(alpha=1.0)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                mse_scores, r2_scores = [], []
                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    ridge.fit(X_train, y_train)
                    y_pred = ridge.predict(X_test)
                    mse_scores.append(mean_squared_error(y_test, y_pred))
                    r2_scores.append(r2_score(y_test, y_pred))

                st.write(f"**Average MSE:** {np.mean(mse_scores):.4f}")
                st.write(f"**Average R² Score:** {np.mean(r2_scores):.4f}")

            except Exception as e:
                st.error(f"Error: {e}")

        elif section == "Predictions":
            try:
                st.info("Running prediction using Random Forest...")

                # Select features excluding target
                features = [col for col in df5.columns if col not in ['Year', 'Country', 'HorticulturalValue']]
                X = df5[features].apply(pd.to_numeric, errors='coerce')
                y = pd.to_numeric(df5['HorticulturalValue'], errors='coerce')

                # Drop columns with all NaNs
                X = X.dropna(axis=1, how='all')

                # Fill NaNs with column means (fallback)
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

                # Diagnostics
                st.write("Initial X shape:", X.shape)
                st.write("Initial y shape:", y.shape)
                st.write("Number of NaNs in X:", X.isna().sum().sum())
                st.write("Number of NaNs in y:", y.isna().sum())
                st.write("X columns:", X.columns.tolist())

                # Filter valid rows
                valid_idx = X.notna().all(axis=1) & y.notna()
                X_valid = X[valid_idx]
                y_valid = y[valid_idx]

                if X_valid.empty or y_valid.empty:
                    st.warning("No valid data to train the model. Please check for missing or invalid values in the dataset.")
                else:
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X_valid, y_valid)
                    y_pred = rf.predict(X_valid)

                    df_pred = df5.loc[valid_idx].copy()
                    df_pred['PredictedValue'] = y_pred
                    st.dataframe(df_pred[['Year', 'Country', 'HorticulturalValue', 'PredictedValue']].head(20))

                    plt.figure(figsize=(10, 5))
                    plt.scatter(df_pred['HorticulturalValue'], df_pred['PredictedValue'], alpha=0.6)
                    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
                    plt.xlabel("Actual Horticultural Value")
                    plt.ylabel("Predicted Value")
                    plt.title("Prediction Accuracy")
                    plt.tight_layout()
                    st.pyplot(plt.gcf())

            except Exception as e:
                st.error(f"Prediction error: {e}")

