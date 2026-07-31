import streamlit as st

from tabfm_workbench.config import Settings
from tabfm_workbench.ui import render_history

st.header("History")
render_history(Settings())
