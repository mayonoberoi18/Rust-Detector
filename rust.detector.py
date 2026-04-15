import streamlit as st
import streamlit.components.v1 as components

# Load your HTML file
with open("rust-detector.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Render it in the app
components.html(html_content, height=600, scrolling=True)
