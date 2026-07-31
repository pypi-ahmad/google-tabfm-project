"""Streamlit entrypoint for the local TabFM research workbench."""

import streamlit as st

from tabfm_workbench.config import Settings
from tabfm_workbench.integrations import ProviderError
from tabfm_workbench.loader import DataFormatError
from tabfm_workbench.predictor import InferenceError
from tabfm_workbench.remote import RemoteFetchError
from tabfm_workbench.ui import initialize_session_state, render_sidebar

st.set_page_config(
    page_title="TabFM Research Workbench", page_icon=":material/table_chart:", layout="wide"
)
settings = Settings()
initialize_session_state(settings)
page = st.navigation(
    [
        st.Page("app_pages/data.py", title="Data", icon=":material/database:"),
        st.Page("app_pages/model.py", title="Model", icon=":material/model_training:"),
        st.Page("app_pages/predictions.py", title="Predictions", icon=":material/query_stats:"),
        st.Page("app_pages/eda_reports.py", title="EDA & Reports", icon=":material/analytics:"),
        st.Page("app_pages/history.py", title="History", icon=":material/history:"),
    ],
    position="top",
)
st.title("TabFM Research Workbench")
st.caption("Local-first in-context classification and regression")
st.warning("Research use only. TabFM weights prohibit commercial and production use.")
render_sidebar(settings)
try:
    page.run()
except (DataFormatError, RemoteFetchError, ProviderError, InferenceError, RuntimeError) as exc:
    st.error(str(exc))
