import streamlit as st

st.set_page_config(
    page_title="Hello World",
    layout="centered",
)
st.title("Cats and Dogs Classifier")
file = st.file_uploader("Pick a file")