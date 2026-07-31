from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_license_gate_and_upload_source() -> None:
    app = AppTest.from_file(APP, default_timeout=10).run()
    assert not app.exception
    assert "TabFM Research Workbench" in [title.value for title in app.title]
    assert any("commercial" in warning.value.lower() for warning in app.warning)
    assert "Data" in app.markdown[0].value or app.file_uploader
    assert any(button.label == "Start new task" for button in app.button)
