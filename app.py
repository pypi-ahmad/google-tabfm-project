"""Streamlit entrypoint for the local TabFM research workbench."""

import streamlit as st

from tabfm_workbench.config import Settings
from tabfm_workbench.integrations import ProviderError
from tabfm_workbench.loader import DataFormatError
from tabfm_workbench.predictor import InferenceError
from tabfm_workbench.remote import RemoteFetchError
from tabfm_workbench.ui import (
    render_batch_predictions,
    render_data_loading,
    render_model_context,
    render_sidebar,
    render_single_prediction,
)

st.set_page_config(page_title="TabFM Research Workbench", page_icon="▦", layout="wide")


def main() -> None:
    settings = Settings()
    st.title("TabFM Research Workbench")
    st.caption("Local-first in-context classification and regression")
    st.warning("Research use only. TabFM weights prohibit commercial and production use.")
    render_sidebar(settings)
    data_tab, model_tab, batch_tab, single_tab = st.tabs(
        ["Data Loading", "Model & Context", "Batch Predictions", "Single Test Case"]
    )
    try:
        with data_tab:
            render_data_loading(settings)
        with model_tab:
            render_model_context(settings)
        with batch_tab:
            render_batch_predictions()
        with single_tab:
            render_single_prediction()
    except (DataFormatError, RemoteFetchError, ProviderError, InferenceError, RuntimeError) as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
