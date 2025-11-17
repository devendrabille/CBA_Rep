import pytest
from src.ui.layout import create_layout
from src.ui.controls import create_controls
from src.ui.chat import create_chat_interface

def test_create_layout():
    layout = create_layout()
    assert layout is not None
    assert 'header' in layout
    assert 'sidebar' in layout

def test_create_controls():
    controls = create_controls()
    assert controls is not None
    assert 'upload_button' in controls
    assert 'run_analysis_button' in controls

def test_create_chat_interface():
    chat_interface = create_chat_interface()
    assert chat_interface is not None
    assert 'chat_input' in chat_interface
    assert 'chat_output' in chat_interface

def test_ui_components_render():
    # This test would require a Streamlit testing framework or mock
    # Here we would check if the UI components render without errors
    try:
        create_layout()
        create_controls()
        create_chat_interface()
    except Exception as e:
        pytest.fail(f"UI components failed to render: {e}")