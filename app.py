import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from meddx.serve import ui_app  # this imports and runs the Streamlit script
