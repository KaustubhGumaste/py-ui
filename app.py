# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title and Header
st.set_page_config(page_title="Earthquake Prediction", page_icon="🌍", layout="wide")
st.title("🌍 Earthquake Magnitude Prediction")
st.markdown(
    """
    This application predicts the magnitude of earthquakes based on provided seismic data.
    You can upload your dataset, test with sample data, or provide values manually for predictions!
    """
)

# Sidebar for user options
st.sidebar.header("Options")
user_choice = st.sidebar.radio(
    "Choose your option:", ["Upload Dataset", "Use Sample Data", "Enter Values Manually"]
)

# Function to train the model
@st.cache_resource
def train_model(data, target_column, excluded_columns, test_size, n_estimators, random_state):
    X = data.drop(excluded_columns + [target_column], axis=1)
    y = data[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse

# Columns to exclude by default
default_excluded_columns = ['title', 'date_time', 'net', 'magType', 'location', 'continent', 'country', 'alert']
target_column = 'magnitude'

# Dataset handling
if user_choice in ["Upload Dataset", "Use Sample Data"]:
    if user_choice == "Upload Dataset":
        uploaded_file = st.file_uploader("Upload your earthquake dataset (CSV file)", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(data.head())
        else:
            st.warning("Please upload a valid CSV file.")
            data = None
    else:
        st.info("Using built-in sample dataset.")
        data = pd.read_csv("C:/Users/Krishna Gumaste/Desktop/ai/imp/DisasterPrediction_AI/earthquake_data.csv")
        st.dataframe(data.head())

    # Train the model
    if data is not None:
        excluded_columns = st.sidebar.multiselect(
            "Columns to Exclude", data.columns, default=default_excluded_columns
        )
        test_size = st.sidebar.slider("Test Set Size (as a fraction)", 0.1, 0.5, 0.2, step=0.05)
        n_estimators = st.sidebar.slider("Number of Trees in Forest", 50, 300, 100, step=10)
        random_state = st.sidebar.number_input("Random State", value=42, step=1)

        with st.spinner("Training the model..."):
            model, mse = train_model(data, target_column, excluded_columns, test_size, n_estimators, random_state)

        st.success(f"Model trained successfully! Mean Squared Error: {mse:.4f}")

        # Predict on new data
        st.markdown("## Test the Model")
        uploaded_test_file = st.file_uploader("Upload test data (CSV file for prediction)", type=["csv"], key="test")
        if uploaded_test_file is not None:
            test_data = pd.read_csv(uploaded_test_file)
            st.write("Test Data Preview:")
            st.dataframe(test_data.head())

            X_test = test_data.drop(excluded_columns, axis=1, errors="ignore")
            predictions = model.predict(X_test)
            test_data["Predicted Magnitude"] = predictions
            st.write("Predictions:")
            st.dataframe(test_data)

            # Download option
            st.markdown("### Download Results")
            csv = test_data.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        else:
            st.info("Upload test data to see predictions.")

elif user_choice == "Enter Values Manually":
    st.markdown("## Enter Values Manually for Prediction")
    # Placeholder for manual input based on sample columns
    feature_values = {}
    columns = ['latitude', 'longitude', 'depth', 'other_feature']  # Replace with actual columns in dataset
    for col in columns:
        feature_values[col] = st.number_input(f"Enter value for {col}:", value=0.0)

    # Predict based on manual input
    if st.button("Predict"):
        sample_input = pd.DataFrame([feature_values])
        sample_prediction = model.predict(sample_input)
        st.success(f"Predicted Earthquake Magnitude: {sample_prediction[0]:.2f}")

# Footer
st.markdown("---")
st.markdown(
    "Created by **Krishna Gumaste** | Powered by [Streamlit](https://streamlit.io) 🌟"
)
