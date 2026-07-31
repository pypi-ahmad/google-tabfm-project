import streamlit as st

from tabfm_workbench.config import Settings
from tabfm_workbench.ui import render_model_context

st.header("Model")
render_model_context(Settings())
