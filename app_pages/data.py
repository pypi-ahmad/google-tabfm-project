import streamlit as st

from tabfm_workbench.config import Settings
from tabfm_workbench.ui import render_data_loading

st.header("Data")
render_data_loading(Settings())
