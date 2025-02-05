import streamlit as st
import pandas as pd
import io

# Function to download data
def download_data(df):
    csv = df.to_csv(index=False)
    excel_io = io.BytesIO()
    df.to_excel(excel_io, index=False, engine="openpyxl")
    excel_io.seek(0)
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")
    st.download_button("Download Excel", excel_io, "cleaned_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")