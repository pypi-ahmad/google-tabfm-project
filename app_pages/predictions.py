import streamlit as st

from tabfm_workbench.config import Settings
from tabfm_workbench.ui import (
    render_batch_predictions,
    render_single_prediction,
    task_widget_key,
)

st.header("Predictions")
mode = st.segmented_control(
    "Prediction mode",
    ["Batch", "Single"],
    default="Batch",
    key=task_widget_key("prediction_mode", st.session_state.task_generation),
)
st.session_state.prediction_mode = mode
if mode == "Batch":
    render_batch_predictions(Settings())
else:
    render_single_prediction()
