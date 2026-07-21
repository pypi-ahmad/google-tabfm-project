from streamlit.testing.v1 import AppTest


def test_app_renders_license_gate_and_upload_source() -> None:
    app = AppTest.from_file("../app.py", default_timeout=10).run()
    assert not app.exception
    assert "TabFM Research Workbench" in [title.value for title in app.title]
    assert any("commercial" in warning.value.lower() for warning in app.warning)
    assert [tab.label for tab in app.tabs[:4]] == [
        "Data Loading",
        "Model & Context",
        "Batch Predictions",
        "Single Test Case",
    ]
