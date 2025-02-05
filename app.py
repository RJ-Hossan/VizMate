import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

# Function to plot Bar Chart
def plot_bar_chart(df):
    st.subheader("📊 Bar Charts for Categorical Features")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        st.subheader(f"Bar Chart for {col}")
        value_counts_df = df[col].value_counts().reset_index()
        value_counts_df.columns = [col, "count"]
        fig = px.bar(
            value_counts_df,
            x=col,
            y="count",
            color=col,
            template="plotly_white",
            title=f"Distribution of {col}",
        )
        fig.update_layout(xaxis_title="Categories", yaxis_title="Count", font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

# Function to plot Pie Chart
def plot_pie_chart(df):
    st.subheader("🥧 Pie Charts for Categorical Features")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        st.subheader(f"Pie Chart for {col}")
        fig = px.pie(
            df,
            names=col,
            title=f"Distribution of {col}",
            hole=0.3,
            template="plotly_white",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

# Function to plot Violin Plot
def plot_violin_plot(df):
    st.subheader("🎻 Violin Plots for Numerical Features")
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        st.warning("No numerical features available for violin plot.")
        return
    for col in num_cols:
        st.subheader(f"Violin Plot for {col}")
        fig = px.violin(
            df,
            y=col,
            box=True,
            points="all",
            color_discrete_sequence=["#636EFA"],
            template="plotly_white",
            title=f"Distribution of {col}",
        )
        fig.update_layout(xaxis_title="Feature", yaxis_title=col, font=dict(size=12), hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

# Function to plot Radar Chart
def plot_radar_chart(df):
    st.subheader("🌐 Radar Chart for Numerical Features")
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        st.warning("No numerical features available for radar plot.")
        return
    avg_values = df[num_cols].mean()
    fig = px.line_polar(
        r=avg_values,
        theta=num_cols,
        line_close=True,
        title="Radar Chart of Average Values",
        template="plotly_dark",
    )
    fig.update_traces(fill="toself", line=dict(color="#EF553B"))
    st.plotly_chart(fig, use_container_width=True)

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    st.subheader("🔥 Correlation Heatmap")
    
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) < 2:
        st.warning("Not enough numerical features to calculate correlation.")
        return
    
    corr_matrix = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues",  
        cbar=True, 
        square=True, 
        linewidths=0.5, 
        linecolor="gray", 
        annot_kws={"color": "black"},  
        ax=ax
    )
    
    st.pyplot(fig)

# Function to perform preprocessing
def preprocess_data(df):
    report = []
    
    # Handling Missing Values
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0].index.tolist()
    for col in missing_cols:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
            report.append(f"Filled missing values in {col} with mode")
        else:
            df[col].fillna(df[col].median(), inplace=True)
            report.append(f"Filled missing values in {col} with median")
    
    # Encoding Categorical Variables
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    report.append(f"Applied One-Hot Encoding to categorical variables: {list(cat_cols)}")
    
    # Scaling Numerical Features
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())
    report.append(f"Applied Min-Max Scaling to numerical features: {list(num_cols)}")
    
    return df, report

# Function to download data
def download_data(df):
    csv = df.to_csv(index=False)
    excel_io = io.BytesIO()
    df.to_excel(excel_io, index=False, engine="openpyxl")
    excel_io.seek(0)
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")
    st.download_button("Download Excel", excel_io, "cleaned_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Main Function for Streamlit App
def main():
    st.title("📊 Data Analysis & Preprocessing App")
    st.write("Upload your dataset, visualize features, preprocess data, and download the cleaned dataset.")
    
    # Load Data
    df = load_data()
    if df is not None:
        # Display Raw Dataset Information
        display_dataset_info(df, title="Raw Dataset")
        
        # Plot Data Visualizations
        st.subheader("📊 Data Visualizations")
        plot_bar_chart(df)
        plot_pie_chart(df)
        plot_violin_plot(df)
        plot_radar_chart(df)
        plot_correlation_heatmap(df)
        
        # Data Preprocessing
        st.subheader("⚙️ Data Preprocessing")
        cleaned_df, preprocess_report = preprocess_data(df)
        
        # Display Cleaned Dataset Information
        display_dataset_info(cleaned_df, title="Cleaned Dataset")
        
        # Show Preprocessing Report
        st.subheader("📄 Preprocessing Steps & Report")
        for step in preprocess_report:
            st.write("- " + step)
        
        # Download Cleaned Dataset
        st.subheader("📥 Download Cleaned Dataset")
        download_data(cleaned_df)

if __name__ == "__main__":
    main()