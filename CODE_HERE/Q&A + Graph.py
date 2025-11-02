import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import re


# Load API Key

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in .env file!")
    st.stop()


# Streamlit App Setup

st.set_page_config(page_title="Q&A + Graphs + AI Insights", layout="wide")
st.title("📊 Q&A + Graphs + AI Insights")

# --- Upload CSV ---
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if not uploaded_file:
    st.warning("Please upload a CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Dataset Preview (Top 5 rows)")
st.dataframe(df.head())


# Helper Functions
def parse_command(command, df):
    """Detect chart type and columns from user command"""
    command = command.lower()
    chart_type = None
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    y_col = numeric_cols[0] if numeric_cols else None
    x_col = categorical_cols[0] if categorical_cols else numeric_cols[0] if numeric_cols else None

    if "pie" in command:
        chart_type = "Pie Chart"
    elif "bar" in command:
        chart_type = "Bar Chart"
    elif "histogram" in command:
        chart_type = "Histogram"
    elif "line" in command:
        chart_type = "Line Chart"

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


# User Interaction

tab1, tab2 = st.tabs(["💬 Ask Questions", "📈 Graphs & Insights"])

with tab1:
    st.subheader("Ask Questions about Your CSV")
    user_question = st.text_input("Type your question here:")

    if user_question:
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            openai_api_key=OPENAI_API_KEY,
            streaming=True
        )

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
        )

        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = agent.run(user_question, callbacks=[st_cb])
            st.subheader("🤖 Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"Error generating answer: {e}")

with tab2:
    st.subheader("Generate Charts & AI Insights")
    user_command = st.text_input("Type your chart or summary command:", key="graph_command")
    show_table = st.checkbox("Show filtered table (Top 5 rows only)")

    if user_command:
        filtered_df = filter_data(user_command, df)

        if show_table:
            st.subheader("🗃 Filtered Data")
            if filtered_df.empty:
                st.warning("No data matches your query.")
            else:
                st.dataframe(filtered_df.head(5))

        chart_type, x_col, y_col = parse_command(user_command, filtered_df)
        fig = None

        try:
            if chart_type:
                if chart_type == "Pie Chart":
                    data_agg = filtered_df.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.pie(data_agg, names=x_col, values=y_col, template="plotly_dark")
                elif chart_type == "Histogram":
                    fig = px.histogram(filtered_df, x=y_col, color=x_col, template="plotly_dark", nbins=10)
                elif chart_type == "Bar Chart":
                    fig = px.bar(filtered_df, x=x_col, y=y_col, color=x_col, barmode="group", template="plotly_dark")
                else:
                    fig = px.line(filtered_df, x=x_col, y=y_col, color=x_col, template="plotly_dark")

                st.subheader(f"📈 {chart_type}")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating chart: {e}")

        # AI Insights
        if st.button("✨ Generate AI Insights"):
            with st.spinner("Analyzing chart/data..."):
                chat = ChatOpenAI(model="gpt-4", max_tokens=300)
                description = f"""
                Chart type: {chart_type}
                X-axis: {x_col}
                Y-axis: {y_col}
                Top 5 rows of filtered data:
                {filtered_df.head().to_dict()}
                """
                result = chat.invoke([HumanMessage(content=f"Analyze this chart/data and provide insights:\n{description}")])
                st.subheader("🤖 AI Insights:")
                st.write(result.content)

