"""Streamlit UI: pick a dataset, ask a question, see the answer table and the SQL that produced it.

Run locally (API on http://localhost:8000, or set API_URL):
    streamlit run src/datacopilot/ui/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `streamlit run` from a checkout

from datacopilot.ui import client  # noqa: E402

EXAMPLES = {
    "crop": "Which state had the highest total rice production?",
    "ipl": "Which batter scored the most runs in IPL history?",
    "f1": "How many races were held in 2021?",
    "olist": "How many orders were delivered?",
}

st.set_page_config(page_title="DataCopilot", layout="wide")
st.title("DataCopilot")
st.caption("Ask a question in English or Hinglish. The LLM only writes SQL; the database computes the answer.")

try:
    info = client.health()
    names = client.datasets()
except client.APIError as err:
    st.error(str(err))
    st.stop()

with st.sidebar:
    st.subheader("Server")
    llm = info.get("llm", {})
    st.write(f"LLM: `{llm.get('provider', '?')}` / `{llm.get('model', '?')}`" if "error" not in llm
             else f"LLM not configured: {llm['error']}")
    st.write(f"API: `{client.api_url()}`")

if not names:
    st.warning("No datasets are available on the server.")
    st.stop()

dataset = st.selectbox("Dataset", names)
question = st.text_area("Question", placeholder=EXAMPLES.get(dataset, ""), max_chars=500)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Writing and checking SQL..."):
        try:
            result = client.ask(dataset, question)
        except client.APIError as err:
            st.error(str(err))
            st.stop()

    status = result["status"]
    if status == "refused":
        st.info(result["message"])
    elif status == "failed":
        st.warning(f"No answer: {result['error']}")
    else:
        st.success(f"{result['row_count']} row(s)" + (" (showing the first rows)" if result["truncated"] else ""))
        st.dataframe(pd.DataFrame(result["rows"], columns=result["columns"]), hide_index=True)
    if result.get("sql"):
        st.code(result["sql"], language="sql")
    gate = result["gate"]
    t = result.get("timings_ms", {})
    st.caption(f"Gate: P(answerable) = {gate['p_answerable']:.2f} (threshold {gate['threshold']}) · "
               f"LLM {t.get('llm', 0) / 1000:.1f} s · total {t.get('total', 0) / 1000:.1f} s · "
               f"request {result['request_id']}")
