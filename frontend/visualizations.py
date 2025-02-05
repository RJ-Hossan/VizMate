import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to plot Correlation Heatmap
def plot_correlation_heatmap(df):
    st.subheader("🔥 Correlation Heatmap")
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) < 2:
        st.warning("Not enough numerical features to calculate correlation.")
        return
    corr_matrix = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=ax)
    st.pyplot(fig)