import streamlit as st
import pandas as pd
import io

# Function to upload dataset
def load_data():
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    return None

# Function to display dataset info
def display_dataset_info(df, title="Raw Dataset"):
    st.subheader(f"📋 {title} Information")
    st.write(f"Shape: {df.shape}")
    st.write("First 5 Rows:")
    st.dataframe(df.head())
    st.write("Statistical Summary:")
    st.dataframe(df.describe(include="all"))
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)