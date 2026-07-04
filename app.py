"""
Streamlit UI for the E-Commerce Product Q&A Agent.

Run with:
    streamlit run app.py

Requires the pipeline to have been run first:
    python src/etl_pyspark.py
    python src/build_index.py
"""

import sys
import os

import streamlit as st

# Make src/ importable when Streamlit runs this file from the project root.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from agent import ProductQAAgent  # noqa: E402


@st.cache_resource
def load_agent():
    """Load the agent once per server session (loads the embedding model + index)."""
    return ProductQAAgent()


st.set_page_config(page_title="Product Q&A Agent", page_icon="🛒")
st.title("🛒 E-Commerce Product Q&A Agent")
st.caption("A RAG agent that answers questions using product descriptions, reviews, inventory, and support tickets.")

try:
    agent = load_agent()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

products = agent.list_products()
product_options = ["All products / general question"] + [f"{pid} - {name}" for pid, name in products]
selected = st.selectbox("Which product is your question about?", product_options)

selected_product_id = None
if selected != "All products / general question":
    selected_product_id = selected.split(" - ")[0]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question, e.g. 'Is this product available?'")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.answer(question, product_id=selected_product_id)
        st.markdown(result["answer"])
        with st.expander("Sources used for this answer"):
            for chunk in result["sources"]:
                st.markdown(f"**[{chunk['doc_type']}]** {chunk['product_name']}: {chunk['text']}")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
