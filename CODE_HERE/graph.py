import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import re

# Setup

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Graph Analysis", layout="wide")
st.title("📊 Graph Analysis ")


# Helper functions

def parse_command(command, df):
    command = command.lower()
    chart_type = None

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    y_col = numeric_cols[0] if numeric_cols else None
    x_col = categorical_cols[0] if categorical_cols else numeric_cols[0] if numeric_cols else None

    # Detect chart type
    if "pie" in command:
        chart_type = "Pie Chart"
    elif "bar" in command:
        chart_type = "Bar Chart"
    elif "histogram" in command:
        chart_type = "Histogram"
    elif "line" in command:
        chart_type = "Line Chart"

    # Detect column mentioned in command
    for col in df.columns:
        if col.lower() in command:
            if col in categorical_cols:
                x_col = col
            elif col in numeric_cols:
                y_col = col

    return chart_type, x_col, y_col


def filter_data(command, df):
    """Filter numeric columns if user types a number"""
    filtered_df = df.copy()
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", command)
    if numbers:
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            for num in numbers:
                filtered_df = filtered_df[filtered_df[col] == float(num)]
    return filtered_df


# Load CSV

st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV file loaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

if df.empty:
    st.error("CSV is empty!")
    st.stop()

# NLP Command Input

user_command = st.text_input("💬 Type your chart, data, or summary command:", "")

show_table = st.checkbox("Show Table (Top 5 rows only)")

if user_command:
    # Filter data based on numeric values in command
    filtered_df = filter_data(user_command, df)

    # Show filtered table only if checkbox is checked
    if show_table and not filtered_df.empty:
        st.subheader("🗃 Filtered Data (Top 5 rows)")
        st.dataframe(filtered_df.head(5))  # show only top 5 rows
    elif show_table:
        st.warning("No data matches your query.")


    # Summarize Data

    if "summarize" in user_command.lower() or "summary" in user_command.lower():
        with st.spinner("Generating summary..."):
            chat = ChatOpenAI(model="gpt-4", max_tokens=300)
            result = chat.invoke([HumanMessage(
                content=f"Please provide a summary of this dataset:\n{filtered_df.head(5).to_dict()}"
            )])
            st.subheader("📝 Data Summary:")
            st.write(result.content)

    # Parse chart info

    chart_type, x_col, y_col = parse_command(user_command, filtered_df)

    # Generate Chart

    fig = None
    if chart_type:
        try:
            if chart_type == "Pie Chart":
                if y_col is None or x_col is None:
                    st.error("No columns available for Pie chart.")
                else:
                    data_agg = filtered_df.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.pie(data_agg, names=x_col, values=y_col, template="plotly_dark")
            elif chart_type == "Histogram":
                if y_col is None:
                    st.error("No numeric column available for Histogram.")
                else:
                    fig = px.histogram(filtered_df, x=y_col, color=x_col, template="plotly_dark", nbins=10)
            elif chart_type == "Bar Chart":
                fig = px.bar(filtered_df, x=x_col, y=y_col, color=x_col, barmode="group", template="plotly_dark")
            else:  # Line Chart
                fig = px.line(filtered_df, x=x_col, y=y_col, color=x_col, template="plotly_dark")

            st.subheader(f"📈 {chart_type}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating chart: {e}")


    # AI Insights (Text-based GPT-4)

    if fig is not None and st.button("✨ Generate AI Insights"):
        with st.spinner("Analyzing chart..."):
            description = f"""
            Chart type: {chart_type}
            X-axis: {x_col}
            Y-axis: {y_col}
            Top 5 rows of filtered data:
            {filtered_df.head().to_dict()}
            """
            chat = ChatOpenAI(model="gpt-4", max_tokens=300)
            result = chat.invoke(
                [HumanMessage(content=f"Analyze this chart data and provide insights:\n{description}")])
            st.subheader("🤖 AI Insights:")
            st.write(result.content)
