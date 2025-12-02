import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import datetime
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
import json


# -----------------------
# Helper: Copyable Code Block
# -----------------------
def copyable_code(code: str, language: str = "python"):
    code_id = f"code_{abs(hash(code)) % 1000000}"
    display_code = code.replace("<", "&lt;").replace(">", "&gt;")
    js_safe_code = json.dumps(code)

    html_code = f"""
    <style>
        .code-container {{
            position: relative;
            margin-top: 10px;
            border-radius: 8px;
            overflow: hidden;
            background: #282c34;
            border: 1px solid #3d424b;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }}
        .copy-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 6px 12px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            z-index: 10;
        }}
        .copy-btn:hover {{
            background-color: #45a049;
        }}
        pre {{
            margin: 0;
            padding: 14px;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 13.5px;
            line-height: 1.45;
            color: #abb2bf;
            background: #282c34;
            overflow-x: auto;
        }}
    </style>

    <div class="code-container">
        <button class="copy-btn" id="btn_{code_id}">Copy Code</button>
        <pre><code class="language-{language}" id="codeblock_{code_id}">{display_code}</code></pre>
    </div>

    <script>
    (function() {{
        const btn = document.getElementById('btn_{code_id}');
        const codeContent = {js_safe_code};

        btn.onclick = function() {{
            navigator.clipboard.writeText(codeContent).then(() => {{
                btn.innerText = 'Copied!';
                btn.style.backgroundColor = '#4BB543';
                setTimeout(() => {{
                    btn.innerText = 'Copy Code';
                    btn.style.backgroundColor = '#4CAF50';
                }}, 2000);
            }}).catch(err => {{
                const range = document.createRange();
                const codeEl = document.getElementById('codeblock_{code_id}');
                range.selectNodeContents(codeEl);
                const selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
                document.execCommand('copy');
                selection.removeAllRanges();

                btn.innerText = 'Copied!';
                btn.style.backgroundColor = '#4BB543';
                setTimeout(() => {{
                    btn.innerText = 'Copy Code';
                    btn.style.backgroundColor = '#4CAF50';
                }}, 2000);
            }});
        }};

        if (typeof hljs === 'undefined') {{
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
            script.onload = () => {{
                const link = document.createElement('link');
                link.rel = 'stylesheet';
                link.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css';
                document.head.appendChild(link);
                hljs.highlightAll();
            }};
            document.head.appendChild(script);
        }} else {{
            hljs.highlightAll();
        }}
    }})();
    </script>
    """
    estimated_height = 60 + len(code.split('\n')) * 22
    components.html(html_code, height=min(estimated_height, 450), scrolling=True)


# -----------------------
# Load .env
# -----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(page_title="DataSense AI Analyst", layout="wide")
st.title("📊 DataSense AI Analyst")

# -----------------------
# Session State Initialization
# -----------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# -----------------------
# Sidebar: CSV upload
# -----------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"CSV loaded: {uploaded_file.name}")

    st.markdown("---")
    st.header("📜 Query History")
    if st.session_state.query_history:
        for q in reversed(st.session_state.query_history[-5:]):
            st.markdown(f"**[{q['time']}]** `{q['prompt']}`")
            with st.expander("Summary"):
                st.caption(q["summary"])
    else:
        st.info("No queries yet.")

# -----------------------
# Dataset Preview
# -----------------------
if st.session_state.df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    st.markdown("---")


# -----------------------
# Chart Keyword Detection
# -----------------------
def is_chart_request(prompt: str) -> bool:
    chart_words = {"chart", "plot", "graph", "bar", "pie", "line", "histogram", "scatter", "trend", "visualize",
                   "visualization"}
    return any(word in prompt.lower() for word in chart_words)


# -----------------------
# Execute Code and Capture Output
# -----------------------
def execute_code(code: str, df: pd.DataFrame):
    local_ns = {
        'pd': pd,
        'df': df,
        'px': px,
        'go': go,
    }

    try:
        exec(code, {"__builtins__": __builtins__}, local_ns)

        # Check for Plotly figure
        fig = None
        for val in local_ns.values():
            if isinstance(val, go.Figure):
                fig = val
                break

        # Get textual result
        output = None
        if 'result' in local_ns:
            output = local_ns['result']
        else:
            candidates = [
                v for k, v in local_ns.items()
                if k not in ('pd', 'df', 'px', 'go', '__builtins__') and not callable(v)
            ]
            if candidates:
                output = candidates[-1]

        return output, fig

    except Exception as e:
        raise e


# -----------------------
# Main Input & Query Section
# -----------------------
if st.session_state.df is not None and OPENAI_API_KEY:
    user_prompt = st.text_input("Ask anything about your ")

    if st.button("Run") and user_prompt.strip():
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # ✅ ONLY add Plotly instruction if it's a chart request
        if is_chart_request(user_prompt):
            system_prompt = (
                "You are a data analyst. Use ONLY Plotly (plotly.express or plotly.graph_objects). "
                "Do NOT use matplotlib. Do NOT call fig.show(). Create the figure and ensure it is the final result. "
                "The DataFrame is named 'df'."
            )
            final_prompt = f"{system_prompt}\n\nQuestion: {user_prompt}"
        else:
            # Plain question → no chart instruction
            final_prompt = user_prompt

        st.info("🧠 Processing with AI...")

        try:
            llm = OpenAI(api_token=OPENAI_API_KEY)
            pandas_ai = PandasAI(
                llm,
                enable_cache=False,
                verbose=False,
                save_charts=False,
                custom_whitelisted_dependencies=[
                    "plotly", "plotly.express", "plotly.graph_objects"
                ]
            )

            _ = pandas_ai.run(st.session_state.df, prompt=final_prompt)
            generated_code = getattr(pandas_ai, 'last_code_executed', None)

            if not generated_code or not generated_code.strip():
                st.error("No code generated.")
                summary = "No code"
            else:
                output, fig = execute_code(generated_code, st.session_state.df)

                # Display result
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    summary = "Chart displayed"
                elif output is not None:
                    st.write(output)
                    summary = str(output)[:100]
                else:
                    st.write("(No output)")
                    summary = "No result"

                with st.expander("🔍 View AI-Generated Code"):
                    copyable_code(generated_code.strip(), language="python")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            summary = f"Error: {str(e)[:80]}"
            if 'pandas_ai' in locals():
                code = getattr(pandas_ai, 'last_code_executed', None)
                if code:
                    with st.expander("🔍 AI Code (Error)"):
                        copyable_code(code, language="python")

        st.session_state.query_history.append({
            "time": timestamp,
            "prompt": user_prompt,
            "summary": summary
        })

# -----------------------
# Recent Queries
# -----------------------
if st.session_state.query_history:
    st.subheader("Recent Queries")
    for q in reversed(st.session_state.query_history[-5:]):
        st.markdown(f"**[{q['time']}]** `{q['prompt']}` → {q['summary']}")

# -----------------------
# Clear Session
# -----------------------
if st.button("Clear Session"):
    st.session_state.df = None
    st.session_state.query_history = []
    st.success("Session cleared!")