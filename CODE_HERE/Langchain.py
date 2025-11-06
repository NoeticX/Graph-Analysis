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
import datetime
import io


# Configuration & Setup
# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit Setup
st.set_page_config(
    page_title="DataSense AI Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global constants for better maintainability
DEFAULT_AI_MODEL = "gpt-4"
INSIGHTS_MODEL = "gpt-4"

# Define Custom Avatars
USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"


# State Management

if "df" not in st.session_state:
    st.session_state.df = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "messages" not in st.session_state:
    # Initialize chat history
    st.session_state.messages = []


# ================================
# 3. 🎨 UI Components & Layout
# ================================

def setup_sidebar():
    """Handles file upload and query history in the sidebar."""
    st.sidebar.title("⚙️ DataSense AI Analyst")
    st.sidebar.markdown("---")

    # --- File Uploader ---
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file to begin analysis.",
        type=["csv"],
        key="csv_uploader"
    )

    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

    # --- Query History ---
    st.sidebar.markdown("---")
    st.sidebar.header("📜 Recent Queries")
    if st.session_state.query_history:
        for q in reversed(st.session_state.query_history[-5:]):
            st.sidebar.markdown(f"**[{q['time']}]** `{q['command']}`")
            with st.sidebar.expander("Summary"):
                st.caption(q["summary"])
    else:
        st.sidebar.info("No queries yet. Ask an AI or generate a chart!")


def display_main_content():
    """Handles the main content area for analysis and display."""
    st.title("📊Data Analytics")
    st.markdown(
        "Ask a question (e.g., 'What is the average age?') or request a chart (e.g., 'Bar chart of category by sales').")

    if st.session_state.df is None:
        st.info("Please upload a CSV file in the sidebar to start.")
        return

    # --- Dataset Preview in an expander ---
    # FIX: Removed the 'key' argument to fix the TypeError in older Streamlit versions.
    with st.expander("🔍 Dataset Preview & Information"):
        st.subheader("First 5 Rows")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        st.markdown("---")

    # --- Chat History Display ---
    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    # --- The main input box ---
    if prompt := st.chat_input("🔍 Enter your analysis command or question..."):
        # 1. Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar=USER_AVATAR).write(prompt)

        # 2. Process the prompt
        process_user_input(prompt)



# Core Logic & Helper Functions


def log_query(command, summary):
    """Logs the command and a brief summary to the session history."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.query_history.append({
        "time": timestamp,
        "command": command,
        "summary": summary
    })


@st.cache_data
def get_column_types(df):
    """Caches column type lists to avoid recalculation."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numeric_cols, categorical_cols


def parse_chart_request(command, df):
    """Intelligently detects chart type and columns from user command."""
    command = command.lower()
    chart_type = None
    x_col = None
    y_col = None

    numeric_cols, categorical_cols = get_column_types(df)

    if "pie" in command:
        chart_type = "Pie Chart"
    elif "bar" in command:
        chart_type = "Bar Chart"
    elif "histogram" in command:
        chart_type = "Histogram"
    elif "line" in command or "trend" in command:
        chart_type = "Line Chart"
    elif "scatter" in command:
        chart_type = "Scatter Plot"

    for col in df.columns:
        col_lower = col.lower().replace('_', ' ')
        if col_lower in command or col.lower() in command:
            if col in numeric_cols and y_col is None:
                y_col = col
            elif col in categorical_cols and x_col is None:
                x_col = col
            elif x_col is None and y_col is not None:
                x_col = col

    if chart_type and x_col is None and categorical_cols:
        x_col = categorical_cols[0]
    if chart_type and y_col is None and numeric_cols:
        y_col = numeric_cols[0]

    return chart_type, x_col, y_col


def filter_data(command, df):
    """Filters data based on simple numeric conditions (>, <, =)."""
    filtered_df = df.copy()
    command = command.lower()
    numeric_cols, _ = get_column_types(df)

    for col in numeric_cols:
        match = re.search(rf'({col})\s*(>|greater than|above|<|less than|below|=)\s*(\d+\.?\d*)', command)
        if match:
            col_name = match.group(1)
            operator_str = match.group(2)
            value = float(match.group(3))

            if '>' in operator_str or 'greater' in operator_str or 'above' in operator_str:
                filtered_df = filtered_df[filtered_df[col_name] > value]
            elif '<' in operator_str or 'less' in operator_str or 'below' in operator_str:
                filtered_df = filtered_df[filtered_df[col_name] < value]
            elif '=' in operator_str:
                filtered_df = filtered_df[filtered_df[col_name] == value]

    return filtered_df


def generate_ai_insights(fig_description, filtered_df, user_input):
    """Uses LLM to generate descriptive insights from the chart or data."""
    try:
        chat = ChatOpenAI(model=INSIGHTS_MODEL, temperature=0.5, max_tokens=500, openai_api_key=OPENAI_API_KEY)
        prompt = f"""
        Analyze the data visualization described below based on the user's request.
        The original command was: '{user_input}'

        Chart/Data Description:
        {fig_description}

        Provide a concise, insightful summary (max 3-4 bullet points) that highlights the main trends, outliers, or key observations.
        """
        result = chat.invoke([HumanMessage(content=prompt)])
        return result.content
    except Exception as e:
        return f"Could not generate insights: {e}"



# Chart Generation

def create_plotly_chart(chart_type, x_col, y_col, filtered_df):
    """Generates the appropriate Plotly figure based on parsed parameters."""
    fig = None
    title = f"{chart_type} of {y_col} by {x_col}" if x_col and y_col else f"{chart_type} of {y_col}"

    try:
        if chart_type == "Pie Chart":
            if x_col and y_col:
                data_agg = filtered_df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.pie(data_agg, names=x_col, values=y_col, title=title, template="plotly_dark")
            elif x_col:
                data_agg = filtered_df[x_col].value_counts().reset_index()
                data_agg.columns = [x_col, 'Count']
                fig = px.pie(data_agg, names=x_col, values='Count', title=title, template="plotly_dark")

        elif chart_type == "Histogram" and y_col:
            fig = px.histogram(filtered_df, x=y_col, color=x_col, title=title, template="plotly_dark", nbins=20)

        elif chart_type == "Bar Chart" and x_col and y_col:
            data_agg = filtered_df.groupby(x_col)[y_col].mean().reset_index()
            fig = px.bar(data_agg, x=x_col, y=y_col, color=x_col, title=title, template="plotly_dark")

        elif chart_type == "Line Chart" and x_col and y_col:
            fig = px.line(filtered_df, x=x_col, y=y_col,
                          color=x_col if x_col in get_column_types(filtered_df)[1] else None, title=title,
                          template="plotly_dark")

        elif chart_type == "Scatter Plot" and x_col and y_col:
            fig = px.scatter(filtered_df, x=x_col, y=y_col, title=title, template="plotly_dark")

        if fig:
            fig.update_layout(title_x=0.5)
        return fig

    except Exception as e:
        st.error(f"Error creating Plotly chart: {e}")
        return None


# Main Processing Function


def process_user_input(user_input):
    """Determines intent (Chart vs. Q&A) and executes the corresponding flow."""

    chart_keywords = ["chart", "plot", "bar", "pie", "histogram", "line", "trend", "compare", "scatter"]

    if any(word in user_input.lower() for word in chart_keywords):

        # 1. CHART / INSIGHTS FLOW

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            # Use a placeholder for the output
            output_placeholder = st.empty()

            with st.spinner("Analyzing request and generating chart..."):
                filtered_df = filter_data(user_input, st.session_state.df)
                chart_type, x_col, y_col = parse_chart_request(user_input, filtered_df)

                if chart_type and x_col and y_col:
                    fig = create_plotly_chart(chart_type, x_col, y_col, filtered_df)

                    if fig:
                        output_placeholder.markdown(f"**📊 {chart_type}:** {fig.layout.title.text}")
                        st.plotly_chart(fig, use_container_width=True)

                        # --- Download Button ---
                        col1, col2 = st.columns([1, 4])
                        img_bytes = fig.to_image(format="png")
                        col1.download_button(
                            label="Download Chart (PNG)",
                            data=img_bytes,
                            file_name=f"{chart_type.replace(' ', '_')}.png",
                            mime="image/png"
                        )

                        # --- AI Insights Generation ---
                        fig_desc = f"Chart type: {chart_type}, X-axis: {x_col}, Y-axis: {y_col}. Data includes: {filtered_df.shape[0]} rows."
                        insights = generate_ai_insights(fig_desc, filtered_df, user_input)

                        st.markdown("---")
                        st.markdown("**💡 AI Insights:**")
                        st.info(insights)

                        log_query(user_input, f"Generated {chart_type}. Insights: {insights[:40]}...")
                        st.session_state.messages.append({"role": "assistant",
                                                          "content": f"Generated {chart_type}. Here are the insights:\n\n{insights}"})
                    else:
                        error_msg = "Could not generate a meaningful chart. Try being more specific (e.g., 'Bar chart of Region by Sales')."
                        output_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                else:
                    error_msg = f"Chart request detected but could not identify key columns. Please specify a chart type and at least one relevant column from your data."
                    output_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


    else:

        # 2. DIRECT Q&A FLOW (LangChain Agent)

        if not OPENAI_API_KEY:
            st.error("OpenAI API key not configured. Please check your `.env` file.")
            return

        # 1. Create a temporary container for the noisy thought process (HIDDEN)
        thought_container = st.empty()

        # 2. Start the chat message for the final answer
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            # 3. Create a placeholder for the final response
            final_response_placeholder = st.empty()

            # Show a brief spinning message while waiting for the LLM
            with st.spinner("Analyzing data..."):
                try:
                    llm = ChatOpenAI(
                        temperature=0,
                        model=DEFAULT_AI_MODEL,
                        openai_api_key=OPENAI_API_KEY,
                        streaming=True
                    )

                    # Instantiate the callback handler to stream thoughts to the HIDDEN container
                    st_cb = StreamlitCallbackHandler(thought_container, expand_new_thoughts=False)

                    agent = create_pandas_dataframe_agent(
                        llm=llm,
                        df=st.session_state.df,
                        verbose=False,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        handle_parsing_errors=True,
                        allow_dangerous_code=True,
                    )

                    # Execute the agent, streaming the output (thoughts go to thought_container)
                    response = agent.run(user_input, callbacks=[st_cb])

                    # 4. Once finished, display the clean final response
                    final_response_placeholder.markdown(response)

                    st.session_state.messages.append({"role": "assistant", "content": response})
                    log_query(user_input, response[:80])

                except Exception as e:
                    error_msg = f"⚠️ Error from AI Agent: {e}. Please try a different question."
                    final_response_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# Run App

if __name__ == "__main__":
    setup_sidebar()
    display_main_content()